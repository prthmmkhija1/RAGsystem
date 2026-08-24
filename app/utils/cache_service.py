"""
Cache Service
In-memory TTL caching for embeddings, query results, and document metadata.

Cache tiers:
  - Embeddings : 24 h TTL  (deterministic — text always maps to same vector)
  - Queries    :  1 h TTL  (invalidated when the underlying documents change)
  - Documents  : 30 min TTL

Query/compare entries are indexed by the documents they touch, so deleting a
single document only evicts the entries that actually depend on it.
"""
import hashlib
import time
import os
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# ─── Configuration ─────────────────────────────────────────

EMBEDDING_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", 86400))    # 24 hours
QUERY_TTL = int(os.getenv("QUERY_CACHE_TTL", 3600))             # 1 hour
DOC_TTL = int(os.getenv("DOC_CACHE_TTL", 1800))                 # 30 min
MAX_EMBEDDING_KEYS = int(os.getenv("EMBEDDING_CACHE_MAX", 10000))
MAX_QUERY_KEYS = int(os.getenv("QUERY_CACHE_MAX", 1000))

# Sentinel bucket for queries that search the whole corpus (no document filter).
_GLOBAL_SCOPE = "__all__"


class TTLCache:
    """LRU cache with per-key TTL and max-size eviction."""

    def __init__(self, default_ttl: int, max_keys: int = 10000):
        # OrderedDict gives us true LRU: most-recently-used lives at the end.
        self._store: "OrderedDict[str, Tuple[Any, float]]" = OrderedDict()
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
        self._store.move_to_end(key)  # mark as recently used
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if key in self._store:
            # Refresh in place so the entry moves to the MRU end below.
            del self._store[key]
        elif len(self._store) >= self.max_keys:
            self._evict_expired()
            while len(self._store) >= self.max_keys:
                self._store.popitem(last=False)  # drop least-recently-used
        self._store[key] = (value, time.time() + (ttl or self.default_ttl))
        self._store.move_to_end(key)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def flush(self) -> None:
        self._store.clear()

    def keys(self) -> List[str]:
        self._evict_expired()
        return list(self._store.keys())

    @property
    def size(self) -> int:
        # Cheap: count only unexpired entries without mutating the store.
        # (Expired entries are reaped lazily on get/set.)
        now = time.time()
        return sum(1 for _, exp in self._store.values() if now <= exp)

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if now > exp]
        for k in expired:
            del self._store[k]


# ─── Cache Instances ───────────────────────────────────────

_embedding_cache = TTLCache(EMBEDDING_TTL, MAX_EMBEDDING_KEYS)
_query_cache = TTLCache(QUERY_TTL, MAX_QUERY_KEYS)
_document_cache = TTLCache(DOC_TTL, 500)

# document_id (or _GLOBAL_SCOPE) → set of query-cache keys that depend on it.
_doc_index: Dict[str, Set[str]] = {}


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _build_key(prefix: str, subject: str, opts: dict) -> str:
    """
    Build a cache key from *every* option supplied.

    Options are serialized in sorted order, so adding a new retrieval knob
    automatically participates in the key instead of being silently dropped
    (which previously let a verify=True request hit a verify=False entry).
    """
    normalized = "|".join(f"{k}={opts[k]!r}" for k in sorted(opts))
    return f"{prefix}:{_hash(subject + '||' + normalized)}"


def _register(key: str, doc_ids: Iterable[Optional[str]]) -> None:
    """Index a cache key under each document it depends on."""
    scopes = {d or _GLOBAL_SCOPE for d in doc_ids} or {_GLOBAL_SCOPE}
    for scope in scopes:
        _doc_index.setdefault(scope, set()).add(key)


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
    return _query_cache.get(_build_key("q", query, opts))

def set_query_result(query: str, opts: dict, result: dict, ttl: Optional[int] = None):
    key = _build_key("q", query, opts)
    _query_cache.set(key, result, ttl)
    # A query with no document_id searches everything, so it lands in the
    # global bucket and is evicted whenever any document changes.
    _register(key, [opts.get("document_id")])


# ─── Compare Cache ────────────────────────────────────────

def get_compare_result(topic: str, opts: dict = {}):
    return _query_cache.get(_build_key("c", topic, opts))

def set_compare_result(topic: str, opts: dict, result: dict, ttl: Optional[int] = None):
    key = _build_key("c", topic, opts)
    _query_cache.set(key, result, ttl)
    _register(key, opts.get("document_ids") or [])


# ─── Invalidation ─────────────────────────────────────────

def invalidate_queries():
    """Drop every cached query/compare result (used when the corpus grows)."""
    _query_cache.flush()
    _doc_index.clear()

def invalidate_document_queries(document_id: str):
    """
    Evict only the entries that depend on *document_id*.

    That means results scoped to this document, compares involving it, and
    whole-corpus queries (which may have retrieved its chunks). Results scoped
    to other specific documents are left intact.
    """
    keys = _doc_index.pop(document_id, set()) | _doc_index.get(_GLOBAL_SCOPE, set())
    for key in keys:
        _query_cache.delete(key)
    _doc_index.pop(_GLOBAL_SCOPE, None)
    # Drop the evicted keys from any remaining document buckets.
    for scope in list(_doc_index):
        _doc_index[scope] -= keys
        if not _doc_index[scope]:
            del _doc_index[scope]


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
    _doc_index.clear()
