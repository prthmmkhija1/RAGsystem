"""
ChromaDB Configuration
Uses PersistentClient — no separate ChromaDB server needed.
"""
import os

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_documents")
DISTANCE_FUNCTION = "cosine"  # cosine | l2 | ip
BATCH_SIZE = 100  # max chunks to insert per batch
