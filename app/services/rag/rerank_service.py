"""
Re-ranking Service
Keyword-overlap re-ranker that combines vector similarity scores
with BM25-like term-frequency relevance for better retrieval quality.

This lightweight approach avoids heavy ML model dependencies while
still providing meaningful re-ranking of candidate chunks.
"""
import math
import re
from collections import Counter
from typing import Dict, List, Optional


def rerank_chunks(
    query: str,
    chunks: List[Dict],
    top_n: Optional[int] = None,
    vector_weight: float = 0.5,
    keyword_weight: float = 0.5,
) -> List[Dict]:
    """
    Re-rank chunks using a combined score of vector similarity and keyword overlap.

    Args:
        query: The search query.
        chunks: List of dicts with 'text', 'metadata', 'similarity_score'.
        top_n: Return only the top N results (default: all).
        vector_weight: Weight for vector similarity [0-1].
        keyword_weight: Weight for keyword score [0-1].

    Returns:
        Re-ranked list of chunks with 'rerank_score' and 'original_rank' added.
    """
    if not chunks:
        return []
    if len(chunks) == 1:
        chunks[0]["rerank_score"] = chunks[0].get("similarity_score", 1.0)
        chunks[0]["original_rank"] = 1
        return chunks

    query_terms = _tokenize(query)
    if not query_terms:
        return chunks

    # Compute IDF across the candidate set
    doc_freq: Counter = Counter()
    for chunk in chunks:
        tokens = set(_tokenize(chunk.get("text", "")))
        for t in tokens:
            doc_freq[t] += 1
    n_docs = len(chunks)

    scored = []
    for rank, chunk in enumerate(chunks):
        # Vector similarity (already 0-1 from ChromaDB)
        vec_score = chunk.get("similarity_score", 0.0)

        # BM25-like keyword score
        kw_score = _bm25_score(query_terms, chunk.get("text", ""), doc_freq, n_docs)

        combined = (vector_weight * vec_score) + (keyword_weight * kw_score)
        scored.append({
            **chunk,
            "rerank_score": round(combined, 4),
            "keyword_score": round(kw_score, 4),
            "original_rank": rank + 1,
        })

    # Sort by combined score descending
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    if top_n and top_n < len(scored):
        scored = scored[:top_n]

    return scored


# ─── Helpers ──────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    return [w.lower() for w in re.findall(r"\b\w+\b", text) if len(w) > 1]


def _bm25_score(
    query_terms: List[str],
    doc_text: str,
    doc_freq: Counter,
    n_docs: int,
    k1: float = 1.5,
    b: float = 0.75,
    avg_dl: float = 200.0,
) -> float:
    """
    Simplified BM25 relevance score.
    """
    doc_tokens = _tokenize(doc_text)
    dl = len(doc_tokens)
    if dl == 0:
        return 0.0

    tf_counter = Counter(doc_tokens)
    score = 0.0

    for term in query_terms:
        tf = tf_counter.get(term, 0)
        df = doc_freq.get(term, 0)
        if tf == 0 or df == 0:
            continue
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avg_dl)))
        score += idf * tf_norm

    # Normalize to 0-1 range (approximate)
    max_possible = len(query_terms) * 3.0  # rough upper bound
    return min(score / max_possible, 1.0) if max_possible > 0 else 0.0
