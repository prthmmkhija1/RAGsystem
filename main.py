"""
Server Entry Point
Run with: python main.py
"""
import os

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
        reload=False,
    )
