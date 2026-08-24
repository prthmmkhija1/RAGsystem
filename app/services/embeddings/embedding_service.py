"""
Embedding Service
Generates vector embeddings using ChromaDB's built-in ONNX embedding model.
Same model as sentence-transformers all-MiniLM-L6-v2 (384-dim) but runs via
onnxruntime instead of PyTorch — uses ~100 MB RAM instead of ~2 GB.
No API key needed, no cost, no rate limits.
"""
import asyncio
import gc
from typing import Dict, List, Optional

from app.utils import cache_service
from app.utils.error_handler import ExternalServiceError

# ─── Model (lazy-loaded on first use) ─────────────────────

_embed_fn = None


def _get_embed_fn():
    """Lazy-load ChromaDB's built-in ONNX embedding function."""
    global _embed_fn
    if _embed_fn is None:
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            print("[Embeddings] Loading ONNX embedding model (all-MiniLM-L6-v2)...")
            _embed_fn = DefaultEmbeddingFunction()
            # Warm-up call to verify it works
            test = _embed_fn(["test"])
            dim = len(test[0])
            print(f"[Embeddings] Model loaded — dimension: {dim}, backend: onnxruntime")
            gc.collect()  # Free any temp allocations from model loading
        except Exception as e:
            raise ExternalServiceError(
                "Embeddings",
                f"Failed to load embedding model: {e}"
            )
    return _embed_fn


# ─── Single Embedding ─────────────────────────────────────

async def generate_embedding(text: str, skip_cache: bool = False) -> List[float]:
    """Generate an embedding for a single text string (with caching)."""
    if not skip_cache:
        cached = cache_service.get_embedding(text)
        if cached is not None:
            return cached

    embeddings = await generate_embeddings([text])
    emb = embeddings[0]
    cache_service.set_embedding(text, emb)
    return emb


# ─── Batch Embeddings ────────────────────────────────────

async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    Runs in a thread pool to avoid blocking the async event loop.
    """
    if not texts:
        return []

    # Replace empty / whitespace-only texts
    valid_texts = [t if t and t.strip() else "[empty]" for t in texts]

    try:
        embed_fn = _get_embed_fn()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: [list(map(float, e)) for e in embed_fn(valid_texts)]
        )
        return embeddings
    except Exception as e:
        raise ExternalServiceError("Embeddings", f"Embedding generation failed: {e}")


# ─── Batched Embedding Generation ─────────────────────────

async def generate_embeddings_batched(
    texts: List[str],
    batch_size: int = 32,
    skip_cache: bool = False,
) -> List[List[float]]:
    """
    Generate embeddings in batches with caching.
    Checks cache first, only generates for misses.
    """
    if not texts:
        return []

    results: List[Optional[List[float]]] = [None] * len(texts)

    # Map each text still needing an embedding to every position it occupies.
    # Deduplicating here means a repeated chunk (boilerplate headers, footers,
    # a duplicated paragraph) is embedded once instead of once per occurrence.
    pending: Dict[str, List[int]] = {}

    if not skip_cache:
        hits, _ = cache_service.get_embeddings_batch(texts)
        for idx, text in enumerate(texts):
            if text in hits:
                results[idx] = hits[text]
            else:
                pending.setdefault(text, []).append(idx)

        if not pending:
            print(f"[Embeddings] All {len(texts)} embeddings served from cache")
            return results  # type: ignore
        print(f"[Embeddings] Cache: {len(hits)} hits, {len(texts) - len(hits)} misses "
              f"({len(pending)} unique to embed)")
    else:
        for idx, text in enumerate(texts):
            pending.setdefault(text, []).append(idx)

    unique_texts = list(pending)

    # Generate in batches
    generated: List[List[float]] = []
    for i in range(0, len(unique_texts), batch_size):
        batch = unique_texts[i : i + batch_size]
        embs = await generate_embeddings(batch)
        generated.extend(embs)

    # Fan each embedding back out to every position its text occupied
    to_cache = {}
    for text, emb in zip(unique_texts, generated):
        for idx in pending[text]:
            results[idx] = emb
        to_cache[text] = emb

    cache_service.set_embeddings_batch(to_cache)
    return results  # type: ignore
