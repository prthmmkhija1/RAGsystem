"""
Embedding Service
Generates vector embeddings using ChromaDB's built-in ONNX embedding model.
Same model as sentence-transformers all-MiniLM-L6-v2 (384-dim) but runs via
onnxruntime instead of PyTorch — uses ~100 MB RAM instead of ~2 GB.
No API key needed, no cost, no rate limits.
"""
import asyncio
import gc
from typing import List, Optional

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
    texts_to_generate: List[str] = []
    indices_to_fill: List[int] = []

    if not skip_cache:
        hits, misses = cache_service.get_embeddings_batch(texts)
        for idx, text in enumerate(texts):
            if text in hits:
                results[idx] = hits[text]
            else:
                texts_to_generate.append(text)
                indices_to_fill.append(idx)

        if not texts_to_generate:
            print(f"[Embeddings] All {len(texts)} embeddings served from cache")
            return results  # type: ignore
        print(f"[Embeddings] Cache: {len(hits)} hits, {len(misses)} misses")
    else:
        texts_to_generate = list(texts)
        indices_to_fill = list(range(len(texts)))

    # Generate in batches
    generated: List[List[float]] = []
    for i in range(0, len(texts_to_generate), batch_size):
        batch = texts_to_generate[i : i + batch_size]
        embs = await generate_embeddings(batch)
        generated.extend(embs)

    # Fill results and cache new embeddings
    to_cache = {}
    for j, emb in enumerate(generated):
        idx = indices_to_fill[j]
        results[idx] = emb
        to_cache[texts_to_generate[j]] = emb

    cache_service.set_embeddings_batch(to_cache)
    return results  # type: ignore
