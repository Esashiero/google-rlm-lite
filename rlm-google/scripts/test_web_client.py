#!/usr/bin/env python3
"""
CLI tool to test the ADK-RLM web server via WebSocket.

This tool connects to a running web server and allows you to send queries
and receive streaming responses without opening the browser UI.

Usage:
    # First, start the web server
    python -m adk_rlm.web

    # Then in another terminal, send a test query
    python scripts/test_web_client.py query "What is 2 + 2?"

    # List all sessions
    python scripts/test_web_client.py sessions

    # Use a specific session
    python scripts/test_web_client.py query "Tell me more" --session-id <id>

    # Send query with files
    python scripts/test_web_client.py query "Summarize this" --files "*.md"

    # Interactive mode
    python scripts/test_web_client.py interactive
"""

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import websockets
from websockets.exceptions import ConnectionClosed


class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_event(event: dict, verbose: bool = False):
    """Print a WebSocket event in a formatted way."""
    event_type = event.get("type", "unknown")

    if event_type == "event":
        # RLM event (iteration, LLM call, etc.)
        icon = event.get("icon", "•")
        label = event.get("label", event.get("event_type", "Unknown"))
        iteration = event.get("iteration", 0)
        timestamp = event.get("timestamp", 0)
        color = event.get("color", "#FFFFFF")

        # Map hex colors to ANSI
        color_map = {
            "#7AA2F7": Colors.BLUE,  # primary
            "#BB9AF7": Colors.HEADER,  # secondary
            "#9ECE6A": Colors.GREEN,  # success
            "#E0AF68": Colors.YELLOW,  # warning
            "#F7768E": Colors.RED,  # error
        }
        ansi_color = color_map.get(color, Colors.CYAN)

        print(f"{ansi_color}[{iteration}] {icon} {label}{Colors.ENDC}")

        if verbose and event.get("metadata"):
            for key, value in event["metadata"].items():
                if value and key not in ("event_type",):
                    print(f"    {key}: {value}")

    elif event_type == "query_start":
        print(f"\n{Colors.BOLD}{Colors.CYAN}🚀 Query started:{Colors.ENDC}")
        print(f"  Prompt: {event.get('prompt', 'N/A')}")

    elif event_type == "query_complete":
        print(f"\n{Colors.BOLD}{Colors.GREEN}✓ Query complete!{Colors.ENDC}")
        print(f"  Elapsed: {event.get('elapsed_seconds', 0):.2f}s")
        print(f"  Events: {event.get('total_events', 0)}")
        if event.get("title"):
            print(f"  Title: {event['title']}")
        if event.get("final_answer"):
            print(f"\n{Colors.BOLD}Answer:{Colors.ENDC}")
            print(event["final_answer"])

    elif event_type == "error":
        print(f"\n{Colors.BOLD}{Colors.RED}✗ Error:{Colors.ENDC}")
        print(f"  {event.get('message', 'Unknown error')}")

    elif event_type == "status":
        print(f"{Colors.YELLOW}ℹ {event.get('message', '')}{Colors.ENDC}")

    elif event_type == "sessions_list":
        sessions = event.get("sessions", [])
        if not sessions:
            print(f"{Colors.YELLOW}No sessions found{Colors.ENDC}")
            return

        print(f"\n{Colors.BOLD}Sessions:{Colors.ENDC}")
        print("-" * 80)
        for s in sessions:
            print(f"  ID: {s.get('session_id', 'N/A')[:8]}...")
            print(f"  Title: {s.get('title', 'Untitled')}")
            print(f"  Messages: {s.get('message_count', 0)}")
            if s.get("updated_at"):
                print(f"  Updated: {s['updated_at']}")
            print("-" * 40)

    elif event_type == "session_created":
        print(f"{Colors.GREEN}✓ Session created:{Colors.ENDC} {event.get('session_id', 'N/A')}")
        if event.get("title"):
            print(f"  Title: {event['title']}")

    elif event_type == "session_deleted":
        print(f"{Colors.GREEN}✓ Session deleted:{Colors.ENDC} {event.get('session_id', 'N/A')}")

    elif event_type == "files_added":
        print(f"{Colors.GREEN}✓ Files added:{Colors.ENDC}")
        print(f"  Patterns: {', '.join(event.get('patterns', []))}")
        print(f"  Count: {event.get('count', 0)} files")
        if event.get("names"):
            print(f"  Files: {', '.join(event['names'][:5])}")
            if event.get("total", 0) > 5:
                print(f"  ... and {event['total'] - 5} more")

    elif event_type == "status_response":
        print(f"\n{Colors.BOLD}Session Status:{Colors.ENDC}")
        print(f"  ID: {event.get('session_id', 'N/A')}")
        print(f"  Title: {event.get('title', 'Untitled')}")
        print(f"  Model: {event.get('model', 'N/A')}")
        print(f"  Max Iterations: {event.get('max_iterations', 'N/A')}")
        if event.get("files"):
            print(f"  Files: {', '.join(event['files'])}")
        print(f"  Messages: {len(event.get('conversation', []))}")

    elif verbose:
        # Print other event types only in verbose mode
        print(f"{Colors.CYAN}[{event_type}]{Colors.ENDC} {json.dumps(event, indent=2)}")


class WebSocketClient:
    """WebSocket client for testing the ADK-RLM web server."""

    def __init__(self, base_url: str = "ws://localhost:8000"):
        self.base_url = base_url
        self.session_id = str(uuid.uuid4())

    def get_ws_url(self, session_id: str | None = None) -> str:
        """Get WebSocket URL for a session."""
        sid = session_id or self.session_id
        parsed = urlparse(self.base_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        return f"{ws_scheme}://{parsed.netloc}/ws/{sid}"

    async def send_query(
        self,
        prompt: str,
        session_id: str | None = None,
        files: list[str] | None = None,
        verbose: bool = False,
        wait_for_completion: bool = True,
    ) -> dict:
        """Send a query and wait for completion."""
        ws_url = self.get_ws_url(session_id)

        print(f"{Colors.CYAN}Connecting to {ws_url}...{Colors.ENDC}")

        async with websockets.connect(ws_url) as ws:
            print(f"{Colors.GREEN}✓ Connected{Colors.ENDC}\n")

            # Add files if specified
            if files:
                print(f"{Colors.YELLOW}Adding files...{Colors.ENDC}")
                await ws.send(
                    json.dumps(
                        {
                            "action": "add_files",
                            "patterns": files,
                        }
                    )
                )

                # Wait for files_added response
                while True:
                    try:
                        response = json.loads(await ws.recv())
                        print_event(response, verbose)
                        if response.get("type") in ("files_added", "error"):
                            break
                    except asyncio.TimeoutError:
                        break

            # Send query
            print(f"{Colors.YELLOW}Sending query...{Colors.ENDC}")
            await ws.send(
                json.dumps(
                    {
                        "action": "query",
                        "prompt": prompt,
                    }
                )
            )

            # Collect events
            events = []
            final_answer = None
            query_complete = False

            try:
                while not query_complete:
                    message = await ws.recv()
                    event = json.loads(message)
                    events.append(event)
                    print_event(event, verbose)

                    if event.get("type") == "query_complete":
                        query_complete = True
                        final_answer = event.get("final_answer")
                    elif event.get("type") == "error":
                        if wait_for_completion:
                            raise Exception(event.get("message", "Unknown error"))
                        else:
                            query_complete = True

            except ConnectionClosed:
                print(f"\n{Colors.YELLOW}Connection closed{Colors.ENDC}")

            return {
                "events": events,
                "final_answer": final_answer,
                "session_id": session_id or self.session_id,
            }

    async def list_sessions(self) -> list[dict]:
        """List all sessions."""
        ws_url = self.get_ws_url()

        print(f"{Colors.CYAN}Connecting to {ws_url}...{Colors.ENDC}")

        async with websockets.connect(ws_url) as ws:
            print(f"{Colors.GREEN}✓ Connected{Colors.ENDC}\n")

            await ws.send(json.dumps({"action": "list_sessions"}))

            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    event = json.loads(message)
                    print_event(event)

                    if event.get("type") == "sessions_list":
                        return event.get("sessions", [])
                    elif event.get("type") == "error":
                        raise Exception(event.get("message", "Unknown error"))

                except asyncio.TimeoutError:
                    print(f"{Colors.RED}Timeout waiting for response{Colors.ENDC}")
                    return []

    async def create_session(self, title: str | None = None) -> str:
        """Create a new session."""
        ws_url = self.get_ws_url()

        print(f"{Colors.CYAN}Connecting to {ws_url}...{Colors.ENDC}")

        async with websockets.connect(ws_url) as ws:
            print(f"{Colors.GREEN}✓ Connected{Colors.ENDC}\n")

            data = {"action": "new_session"}
            if title:
                data["title"] = title

            await ws.send(json.dumps(data))

            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    event = json.loads(message)
                    print_event(event)

                    if event.get("type") == "session_created":
                        return event.get("session_id", "")
                    elif event.get("type") == "error":
                        raise Exception(event.get("message", "Unknown error"))

                except asyncio.TimeoutError:
                    print(f"{Colors.RED}Timeout waiting for response{Colors.ENDC}")
                    return ""

    async def delete_session(self, session_id: str):
        """Delete a session."""
        ws_url = self.get_ws_url()

        print(f"{Colors.CYAN}Connecting to {ws_url}...{Colors.ENDC}")

        async with websockets.connect(ws_url) as ws:
            print(f"{Colors.GREEN}✓ Connected{Colors.ENDC}\n")

            await ws.send(
                json.dumps(
                    {
                        "action": "delete_session",
                        "session_id": session_id,
                    }
                )
            )

            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    event = json.loads(message)
                    print_event(event)

                    if event.get("type") == "session_deleted":
                        return
                    elif event.get("type") == "error":
                        raise Exception(event.get("message", "Unknown error"))

                except asyncio.TimeoutError:
                    print(f"{Colors.RED}Timeout waiting for response{Colors.ENDC}")
                    return

    async def get_status(self, session_id: str | None = None) -> dict:
        """Get session status."""
        ws_url = self.get_ws_url(session_id)

        print(f"{Colors.CYAN}Connecting to {ws_url}...{Colors.ENDC}")

        async with websockets.connect(ws_url) as ws:
            print(f"{Colors.GREEN}✓ Connected{Colors.ENDC}\n")

            await ws.send(json.dumps({"action": "get_status"}))

            while True:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    event = json.loads(message)
                    print_event(event)

                    if event.get("type") == "status_response":
                        return event
                    elif event.get("type") == "error":
                        raise Exception(event.get("message", "Unknown error"))

                except asyncio.TimeoutError:
                    print(f"{Colors.RED}Timeout waiting for response{Colors.ENDC}")
                    return {}

    async def interactive_mode(self):
        """Run in interactive mode."""
        ws_url = self.get_ws_url()

        print(f"\n{Colors.BOLD}{Colors.CYAN}ADK-RLM Web Client - Interactive Mode{Colors.ENDC}")
        print(f"{Colors.CYAN}Connecting to {ws_url}...{Colors.ENDC}")

        async with websockets.connect(ws_url) as ws:
            print(f"{Colors.GREEN}✓ Connected{Colors.ENDC}")
            print(f"\n{Colors.YELLOW}Session ID: {self.session_id}{Colors.ENDC}")
            print(f"\nType your queries below. Commands:")
            print(f"  /quit - Exit")
            print(f"  /status - Show session status")
            print(f"  /sessions - List all sessions")
            print(f"  /clear - Clear conversation")
            print(f"  /files <pattern> - Add files")
            print(f"  /config key=value - Update config")
            print()

            while True:
                try:
                    # Get user input
                    user_input = input(f"{Colors.BOLD}You:{Colors.ENDC} ").strip()

                    if not user_input:
                        continue

                    if user_input == "/quit":
                        print(f"{Colors.YELLOW}Goodbye!{Colors.ENDC}")
                        break

                    if user_input == "/status":
                        await ws.send(json.dumps({"action": "get_status"}))
                        response = await ws.recv()
                        print_event(json.loads(response))
                        continue

                    if user_input == "/sessions":
                        await ws.send(json.dumps({"action": "list_sessions"}))
                        response = await ws.recv()
                        print_event(json.loads(response))
                        continue

                    if user_input == "/clear":
                        await ws.send(json.dumps({"action": "clear"}))
                        response = await ws.recv()
                        print_event(json.loads(response))
                        continue

                    if user_input.startswith("/files "):
                        pattern = user_input[7:].strip()
                        await ws.send(
                            json.dumps(
                                {
                                    "action": "add_files",
                                    "patterns": [pattern],
                                }
                            )
                        )
                        response = await ws.recv()
                        print_event(json.loads(response))
                        continue

                    if user_input.startswith("/config "):
                        config_str = user_input[8:].strip()
                        if "=" in config_str:
                            key, value = config_str.split("=", 1)
                            await ws.send(
                                json.dumps(
                                    {
                                        "action": "config",
                                        key: value,
                                    }
                                )
                            )
                            response = await ws.recv()
                            print_event(json.loads(response))
                        continue

                    # Send query
                    print(f"\n{Colors.CYAN}Sending query...{Colors.ENDC}\n")
                    await ws.send(
                        json.dumps(
                            {
                                "action": "query",
                                "prompt": user_input,
                            }
                        )
                    )

                    # Receive events until completion
                    while True:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            event = json.loads(message)
                            print_event(event)

                            if event.get("type") in ("query_complete", "error"):
                                break
                        except asyncio.TimeoutError:
                            # Continue waiting
                            continue

                    print()  # Empty line after response

                except KeyboardInterrupt:
                    print(f"\n{Colors.YELLOW}Goodbye!{Colors.ENDC}")
                    break
                except Exception as e:
                    print(f"{Colors.RED}Error: {e}{Colors.ENDC}")


def main():
    parser = argparse.ArgumentParser(
        description="Test ADK-RLM web server via WebSocket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--url",
        "-u",
        default="ws://localhost:8000",
        help="Web server URL (default: ws://localhost:8000)",
    )
    parser.add_argument(
        "--session-id",
        "-s",
        help="Session ID to use (creates new if not specified)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including all event metadata",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Query command
    query_parser = subparsers.add_parser("query", help="Send a query")
    query_parser.add_argument("prompt", help="The query prompt")
    query_parser.add_argument(
        "--files",
        "-f",
        nargs="+",
        help="File patterns to add",
    )

    # Sessions command
    subparsers.add_parser("sessions", help="List all sessions")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new session")
    create_parser.add_argument("--title", "-t", help="Session title")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a session")
    delete_parser.add_argument("session_id", help="Session ID to delete")

    # Status command
    status_parser = subparsers.add_parser("status", help="Get session status")

    # Interactive command
    subparsers.add_parser("interactive", help="Run in interactive mode")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    client = WebSocketClient(base_url=args.url)

    # Set session ID if provided
    if args.session_id:
        client.session_id = args.session_id

    try:
        if args.command == "query":
            result = asyncio.run(
                client.send_query(
                    prompt=args.prompt,
                    session_id=args.session_id,
                    files=args.files,
                    verbose=args.verbose,
                )
            )

            if result.get("final_answer"):
                print(f"\n{Colors.BOLD}{Colors.GREEN}Final Answer:{Colors.ENDC}")
                print(result["final_answer"])

        elif args.command == "sessions":
            asyncio.run(client.list_sessions())

        elif args.command == "create":
            session_id = asyncio.run(client.create_session(title=args.title))
            if session_id:
                print(f"\n{Colors.GREEN}Session ID: {session_id}{Colors.ENDC}")

        elif args.command == "delete":
            asyncio.run(client.delete_session(args.session_id))

        elif args.command == "status":
            asyncio.run(client.get_status(args.session_id))

        elif args.command == "interactive":
            asyncio.run(client.interactive_mode())

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    main()
