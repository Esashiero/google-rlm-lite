#!/bin/bash
# Setup script for ADK-RLM testing
# Usage: ./setup_and_test.sh

set -e  # Exit on error

echo "=== ADK-RLM Setup & Test ==="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Must run from rlm-google directory"
    exit 1
fi

# Step 1: Create virtual environment
echo "[1/5] Creating virtual environment with uv..."
if ! command -v uv &> /dev/null; then
    echo "Error: uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

uv venv
source .venv/bin/activate

# Step 2: Install package
echo "[2/5] Installing package..."
uv pip install -e ".[all]"

# Step 3: Check for .env file
echo "[3/5] Checking environment configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "Creating .env from .env.example..."
        cp .env.example .env
        echo "⚠️  Please edit .env and add your API keys before running!"
    else
        echo "Error: No .env or .env.example found"
        exit 1
    fi
else
    echo "✓ .env file exists"
fi

# Step 4: Validate API keys
echo "[4/5] Validating configuration..."
if grep -q "your_.*_api_key_here" .env; then
    echo "⚠️  Warning: API keys not configured in .env"
    echo "Please edit .env and add your actual API keys"
fi

# Step 5: Test instructions
echo ""
echo "=== Setup Complete ==="
echo ""
echo "To test the application:"
echo ""
echo "Terminal 1 - Start the web server:"
echo "  source .venv/bin/activate"
echo "  python -m adk_rlm.web"
echo ""
echo "Terminal 2 - Send a test query:"
echo "  source .venv/bin/activate"
echo "  python scripts/test_web_client.py query \"What's the best way to get rich through polymarket?\""
echo ""
echo "Or use the simple simulator:"
echo "  python scripts/simulate_user.py \"What is 2+2?\""
echo ""
