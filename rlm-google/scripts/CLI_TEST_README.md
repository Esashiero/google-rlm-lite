# CLI WebSocket Test Client

This tool allows you to test the ADK-RLM web server via WebSocket without opening the browser UI. Perfect for debugging and automated testing.

## Installation

The tool is included in the web dependencies. Make sure you have installed the web extras:

```bash
pip install -e ".[web]"
```

Or install websockets directly:

```bash
pip install websockets>=12.0.0
```

## Usage

### 1. Start the Web Server

```bash
python -m adk_rlm.web
# Server starts on http://localhost:8000
```

### 2. Send Test Queries

In another terminal:

```bash
# Simple query
python scripts/test_web_client.py query "What is 2 + 2?"

# Query with files
python scripts/test_web_client.py query "Summarize these files" --files "*.md"

# Query in a specific session
python scripts/test_web_client.py query "Tell me more" --session-id <session-id>

# Verbose output (show all event metadata)
python scripts/test_web_client.py query "Explain Python" -v
```

### 3. Session Management

```bash
# List all sessions
python scripts/test_web_client.py sessions

# Create a new session
python scripts/test_web_client.py create --title "My Test Session"

# Get session status
python scripts/test_web_client.py status --session-id <session-id>

# Delete a session
python scripts/test_web_client.py delete <session-id>
```

### 4. Interactive Mode

```bash
python scripts/test_web_client.py interactive
```

This opens an interactive shell where you can:
- Type queries directly
- Use `/status` to see session info
- Use `/sessions` to list sessions
- Use `/clear` to clear conversation
- Use `/files <pattern>` to add files
- Use `/quit` to exit

## Examples

### Example 1: Quick Test

```bash
# Terminal 1: Start server
python -m adk_rlm.web

# Terminal 2: Send query
python scripts/test_web_client.py query "What is the capital of France?"
```

### Example 2: With Files

```bash
python scripts/test_web_client.py query \
  "Summarize the main points" \
  --files "./docs/**/*.md"
```

### Example 3: Multi-turn Conversation

```bash
# Create a session
python scripts/test_web_client.py create --title "Code Review"

# First query
python scripts/test_web_client.py query \
  "Review this code" \
  --files "./src/*.py" \
  --session-id <id-from-create>

# Follow-up query (same session)
python scripts/test_web_client.py query \
  "What issues did you find?" \
  --session-id <id-from-create>
```

### Example 4: Debug Mode

```bash
# Show all events and metadata
python scripts/test_web_client.py query "Test" -v
```

## Output Format

The CLI shows:
- **Connection status** - When connecting to the server
- **Event stream** - Real-time events (iterations, LLM calls, etc.)
- **Final answer** - The completed response
- **Timing info** - Elapsed time and event count

### Event Types

Events are color-coded:
- 🔵 Blue - Run/Iteration start
- 🟣 Purple - LLM calls
- 🟢 Green - Success/completion
- 🟡 Yellow - Warnings/status
- 🔴 Red - Errors

## Troubleshooting

### Connection Refused

Make sure the web server is running:

```bash
python -m adk_rlm.web
```

### Module Not Found

Install the web dependencies:

```bash
pip install -e ".[web]"
```

### Timeout Errors

If queries timeout, the LLM might be slow. The client waits for the complete response by default.

## Advanced Usage

### Custom Server URL

```bash
python scripts/test_web_client.py \
  --url ws://localhost:8080 \
  query "Test"
```

### Save Session ID

```bash
# Create and capture session ID
SESSION_ID=$(python scripts/test_web_client.py create --title "Test" | grep "Session ID:" | awk '{print $3}')
echo "Session: $SESSION_ID"

# Use in subsequent queries
python scripts/test_web_client.py query "Hello" --session-id $SESSION_ID
```

## Integration with Debugging

Combine with log analysis tools:

```bash
# Send query and analyze logs
python scripts/test_web_client.py query "Test"
python scripts/analyze_log.py logs/rlm_*.jsonl
```

## See Also

- `scripts/run_query.py` - Direct RLM query (no web server needed)
- `scripts/analyze_log.py` - Analyze RLM execution logs
- `scripts/diagnose_run.py` - Diagnose RLM issues
