"""
Vector Store Service
Manages ChromaDB collection: store, search, delete embeddings.
Uses PersistentClient — no separate ChromaDB server process needed.
"""
import chromadb
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.config import database as db_config
from app.utils.error_handler import ExternalServiceError

_client: Optional[chromadb.ClientAPI] = None
_collection = None


def initialize():
    """Initialize ChromaDB PersistentClient and ensure collection exists."""
    global _client, _collection
    try:
        _client = chromadb.PersistentClient(path=db_config.CHROMA_PERSIST_DIR)
        _collection = _client.get_or_create_collection(
            name=db_config.CHROMA_COLLECTION,
            metadata={"hnsw:space": db_config.DISTANCE_FUNCTION},
        )
        count = _collection.count()
        print(f'[VectorStore] Connected to ChromaDB. Collection "{db_config.CHROMA_COLLECTION}" has {count} vectors.')
        return _collection
    except Exception as e:
        raise ExternalServiceError("ChromaDB", f"Initialization failed: {e}")


def _get_collection():
    """Lazy-initialize and return the collection."""
    global _collection
    if _collection is None:
        initialize()
    return _collection


# ─── Store ────────────────────────────────────────────────

def store_chunks(chunk_metadata: List[Dict], embeddings: List[List[float]]) -> None:
    """Store document chunk embeddings in ChromaDB (batched)."""
    col = _get_collection()
    batch_size = db_config.BATCH_SIZE

    for i in range(0, len(chunk_metadata), batch_size):
        batch_meta = chunk_metadata[i : i + batch_size]
        batch_emb = embeddings[i : i + batch_size]

        try:
            col.add(
                ids=[c["chunk_id"] for c in batch_meta],
                embeddings=batch_emb,
                documents=[c["text"] for c in batch_meta],
                metadatas=[
                    {
                        "document_id": c["document_id"],
                        "filename": c["filename"],
                        "chunk_index": c["chunk_index"],
                        "total_chunks": c["total_chunks"],
                        "word_count": c["word_count"],
                        "uploaded_at": datetime.now(timezone.utc).isoformat(),
                    }
                    for c in batch_meta
                ],
            )
        except Exception as e:
            raise ExternalServiceError("ChromaDB", f"Failed to store chunks: {e}")

    doc_id = chunk_metadata[0]["document_id"] if chunk_metadata else "?"
    print(f"[VectorStore] Stored {len(chunk_metadata)} chunks for document {doc_id}")


# ─── Search ───────────────────────────────────────────────

def search_similar(
    query_embedding: List[float],
    top_k: int = 5,
    where_filter: Optional[Dict] = None,
) -> List[Dict]:
    """Search for similar chunks across all documents."""
    col = _get_collection()
    try:
        params: Dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            params["where"] = where_filter

        results = col.query(**params)

        if not results["ids"] or not results["ids"][0]:
            return []

        return [
            {
                "chunk_id": results["ids"][0][idx],
                "text": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "distance": results["distances"][0][idx],
                "similarity_score": 1 - results["distances"][0][idx],
            }
            for idx in range(len(results["ids"][0]))
        ]
    except Exception as e:
        raise ExternalServiceError("ChromaDB", f"Search failed: {e}")


def search_by_document(
    query_embedding: List[float],
    document_id: str,
    top_k: int = 5,
) -> List[Dict]:
    """Search chunks scoped to a specific document."""
    return search_similar(query_embedding, top_k, {"document_id": document_id})


# ─── Delete ───────────────────────────────────────────────

def delete_document(document_id: str) -> None:
    """Delete all chunks belonging to a document."""
    col = _get_collection()
    try:
        col.delete(where={"document_id": document_id})
        print(f"[VectorStore] Deleted all chunks for document {document_id}")
    except Exception as e:
        raise ExternalServiceError("ChromaDB", f"Delete failed: {e}")


# ─── List / Stats ─────────────────────────────────────────

def list_documents() -> List[Dict]:
    """List all unique documents in the collection."""
    col = _get_collection()
    try:
        all_data = col.get(include=["metadatas"])
        if not all_data["ids"]:
            return []

        doc_map: Dict[str, Dict] = {}
        for meta in all_data["metadatas"]:
            did = meta["document_id"]
            if did not in doc_map:
                doc_map[did] = {
                    "document_id": did,
                    "filename": meta.get("filename"),
                    "chunk_count": 0,
                    "uploaded_at": meta.get("uploaded_at"),
                }
            doc_map[did]["chunk_count"] += 1

        return list(doc_map.values())
    except Exception as e:
        raise ExternalServiceError("ChromaDB", f"List failed: {e}")


def get_stats() -> Dict:
    """Get collection statistics."""
    col = _get_collection()
    count = col.count()
    docs = list_documents()
    return {
        "total_chunks": count,
        "total_documents": len(docs),
        "documents": docs,
    }
