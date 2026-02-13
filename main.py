"""
Server Entry Point
Run with: py -3.11 main.py
"""
import os
import sys

# Add local python_packages to path (installed on F: to avoid C: disk space issues)
_local_pkgs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_packages")
if os.path.isdir(_local_pkgs) and _local_pkgs not in sys.path:
    sys.path.insert(0, _local_pkgs)

import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from app import app  # noqa: E402  (import after load_dotenv)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "3000"))
    print(f"[Server] Starting on http://localhost:{port}")
    print(f"[Server] API docs at http://localhost:{port}/docs")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Disabled — reload spawns 2 Python processes (doubles RAM)
    )
