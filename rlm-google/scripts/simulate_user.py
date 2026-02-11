"""
Simulate a user interacting with the ADK-RLM web interface via WebSocket.

Usage:
    python scripts/simulate_user.py "Your prompt here"
"""

import asyncio
import json
import uuid
import sys
import websockets

async def simulate_user(prompt, url="ws://127.0.0.1:8000/ws/"):
    session_id = str(uuid.uuid4())
    full_url = f"{url}{session_id}"

    print(f"Connecting to {full_url}...")
    try:
        async with websockets.connect(full_url) as websocket:
            # Send query action
            query_action = {
                "action": "query",
                "prompt": prompt
            }
            await websocket.send(json.dumps(query_action))
            print(f"Sent query: {prompt}")

            # Listen for events
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)

                    msg_type = data.get("type")
                    if msg_type == "event":
                        label = data.get("label", "Unknown Event")
                        iteration = data.get("iteration", "-")
                        print(f"[{iteration}] {label}")

                        # Print some metadata if useful
                        metadata = data.get("metadata", {})
                        if "error" in metadata:
                            print(f"  ERROR: {metadata['error']}")
                        if "answer" in metadata:
                            print(f"\nFINAL ANSWER:\n{metadata['answer']}\n")
                        if "code" in metadata:
                            print(f"  Code: {metadata['code']}...")

                    elif msg_type == "query_start":
                        print("Query started...")
                    elif msg_type == "query_complete":
                        print(f"Query complete in {data.get('elapsed_seconds', 0):.2f}s")
                        break
                    elif msg_type == "error":
                        print(f"SERVER ERROR: {data.get('message')}")
                        break
                    elif msg_type == "status":
                        print(f"STATUS: {data.get('message')}")
                    elif msg_type == "status_response":
                        # Ignore status responses
                        pass
                    else:
                        # Other messages (like icons/colors for UI)
                        pass

                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed by server.")
                    break
    except Exception as e:
        print(f"Failed to connect or communicate: {e}")
        print("\nMake sure the web server is running: python -m adk_rlm.web")

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What is 2+2?"
    asyncio.run(simulate_user(prompt))
