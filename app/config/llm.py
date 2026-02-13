"""
LLM Configuration (Groq — OpenAI-compatible API)
Embeddings are handled locally via ONNX runtime (no API needed).
"""
import os

API_KEY = os.getenv("GROQ_API_KEY", "")
API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Local embedding model (ONNX runtime — runs locally, free)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))  # seconds
