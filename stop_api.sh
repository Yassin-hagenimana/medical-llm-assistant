#!/bin/bash
# Stop the Medical LLM API server

echo "Stopping Medical LLM Assistant API..."

# Find and kill uvicorn processes
pkill -f "uvicorn deployment.api:app" 2>/dev/null
pkill -f "python start_api.py" 2>/dev/null

# Wait a moment
sleep 1

# Check if processes are still running
if pgrep -f "uvicorn deployment.api:app" > /dev/null; then
    echo "Force killing remaining processes..."
    pkill -9 -f "uvicorn deployment.api:app"
    pkill -9 -f "python start_api.py"
fi

echo "✓ API server stopped"
