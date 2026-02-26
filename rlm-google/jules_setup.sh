#!/bin/bash
# Jules Setup Script for ADK-RLM
# Runs in /app directory after repo clone

set -e

cd /app/rlm-google

# Create venv and install
echo "Setting up environment..."
uv venv
source .venv/bin/activate
uv pip install -e ".[all]"

echo "✓ Setup complete"
