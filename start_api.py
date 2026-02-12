"""
Medical LLM Assistant API Startup Script
Starts the FastAPI backend server
"""

import os
import sys
import subprocess

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Change to project root
os.chdir(project_root)

print("=" * 80)
print("Starting Medical LLM Assistant API...")
print("=" * 80)
print(f"API URL: http://localhost:8000")
print(f"API Docs: http://localhost:8000/docs")
print(f"Project Root: {project_root}")
print("=" * 80)
print("\nPress Ctrl+C to stop the server\n")

# Start uvicorn server
try:
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "deployment.api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ], check=True)
except KeyboardInterrupt:
    print("\n\nServer stopped by user")
except subprocess.CalledProcessError as e:
    print(f"\nError starting server: {e}")
    sys.exit(1)
