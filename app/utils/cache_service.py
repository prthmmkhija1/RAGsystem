"""
Cache Service
In-memory TTL caching for embeddings, query results, and document metadata.

Cache tiers:
  - Embeddings : 24 h TTL  (deterministic — text always maps to same vector)
  - Queries    :  1 h TTL  (invalidated when corpus changes)
  - Documents  : 30 min TTL
"""
import hashlib
import time
import os
from typing import Any, Dict, List, Optional, Tuple

# ─── Configuration ─────────────────────────────────────────

EMBEDDING_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", 86400))    # 24 hours
QUERY_TTL = int(os.getenv("QUERY_CACHE_TTL", 3600))             # 1 hour
DOC_TTL = int(os.getenv("DOC_CACHE_TTL", 1800))                 # 30 min
MAX_EMBEDDING_KEYS = int(os.getenv("EMBEDDING_CACHE_MAX", 10000))
MAX_QUERY_KEYS = int(os.getenv("QUERY_CACHE_MAX", 1000))


class TTLCache:
    """Simple dict-backed cache with per-key TTL and max-size eviction."""

    def __init__(self, default_ttl: int, max_keys: int = 10000):
        self._store: Dict[str, Tuple[Any, float]] = {}  # key → (value, expires_at)
        self.default_ttl = default_ttl
        self.max_keys = max_keys

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if len(self._store) >= self.max_keys:
            self._evict_expired()
        if len(self._store) >= self.max_keys:
            # Evict oldest entry
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        self._store[key] = (value, time.time() + (ttl or self.default_ttl))

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def flush(self) -> None:
        self._store.clear()

    def keys(self) -> List[str]:
        self._evict_expired()
        return list(self._store.keys())

    @property
    def size(self) -> int:
        self._evict_expired()
        return len(self._store)

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]


# ─── Cache Instances ───────────────────────────────────────

_embedding_cache = TTLCache(EMBEDDING_TTL, MAX_EMBEDDING_KEYS)
_query_cache = TTLCache(QUERY_TTL, MAX_QUERY_KEYS)
_document_cache = TTLCache(DOC_TTL, 500)


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _build_query_key(query: str, opts: dict) -> str:
    parts = "|".join([
        query,
        str(opts.get("top_k", 5)),
        opts.get("document_id") or "all",
        "rerank" if opts.get("rerank") else "norerank",
    ])
    return f"q:{_hash(parts)}"


# ─── Embedding Cache ──────────────────────────────────────

def get_embedding(text: str):
    return _embedding_cache.get(f"emb:{_hash(text)}")

def set_embedding(text: str, embedding, ttl: Optional[int] = None):
    _embedding_cache.set(f"emb:{_hash(text)}", embedding, ttl)

def get_embeddings_batch(texts: List[str]):
    hits, misses = {}, []
    for t in texts:
        cached = get_embedding(t)
        if cached is not None:
            hits[t] = cached
        else:
            misses.append(t)
    return hits, misses

def set_embeddings_batch(mapping: dict):
    for text, emb in mapping.items():
        set_embedding(text, emb)


# ─── Query Cache ──────────────────────────────────────────

def get_query_result(query: str, opts: dict = {}):
    return _query_cache.get(_build_query_key(query, opts))

def set_query_result(query: str, opts: dict, result: dict, ttl: Optional[int] = None):
    _query_cache.set(_build_query_key(query, opts), result, ttl)

def invalidate_queries():
    _query_cache.flush()

def invalidate_document_queries(document_id: str):
    # Flush all query cache when a document changes
    _query_cache.flush()


# ─── Document Cache ───────────────────────────────────────

def get_document(document_id: str):
    return _document_cache.get(f"doc:{document_id}")

def set_document(document_id: str, data: dict):
    _document_cache.set(f"doc:{document_id}", data)


# ─── Stats ────────────────────────────────────────────────

def get_stats() -> dict:
    return {
        "embeddings": _embedding_cache.size,
        "queries": _query_cache.size,
        "documents": _document_cache.size,
    }

def flush_all():
    _embedding_cache.flush()
    _query_cache.flush()
    _document_cache.flush()
