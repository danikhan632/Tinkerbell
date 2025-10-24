#!/bin/bash
# Simple script to run the Flask server with HuggingFace backend

PYTHON=/home/green/py312/bin/python3

echo "Starting Flask server with HuggingFace backend..."
echo "Server will be available at http://localhost:8000"
echo ""
echo "To use Megatron backend instead, run:"
echo "  export USE_MEGATRON=true && $PYTHON app.py"
echo ""

cd "$(dirname "$0")"
$PYTHON app.py
