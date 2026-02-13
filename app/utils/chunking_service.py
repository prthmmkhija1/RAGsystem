"""
Chunking Service
Splits documents into overlapping chunks with sentence-boundary awareness.
"""
import os
import re
from typing import Dict, List, Optional

DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))        # characters
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # characters


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[str]:
    """
    Split *text* into overlapping, sentence-aware chunks.

    Args:
        text: Full document text.
        chunk_size: Max characters per chunk (default from env / 500).
        chunk_overlap: Overlap in characters (default from env / 100).

    Returns:
        List of text chunks.
    """
    size = chunk_size or DEFAULT_CHUNK_SIZE
    overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP

    if not text or not text.strip():
        return []

    # Validate: overlap must be smaller than chunk size
    if overlap >= size:
        print(f"[Chunking] Overlap ({overlap}) >= chunkSize ({size}). "
              f"Clamping to {int(size * 0.2)}.")
        overlap = int(size * 0.2)

    sentences = _split_into_sentences(text)

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for sentence in sentences:
        s_len = len(sentence)

        # Single sentence exceeds chunk size → force-split by characters
        if s_len > size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = _get_overlap_sentences(current_chunk, overlap)
                current_len = sum(len(s) for s in current_chunk) + max(len(current_chunk) - 1, 0)
            chunks.extend(_force_split(sentence, size, overlap))
            current_chunk = []
            current_len = 0
            continue

        # Adding this sentence would exceed chunk size → start new chunk
        sep_len = 1 if current_chunk else 0  # space separator
        if current_len + sep_len + s_len > size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = _get_overlap_sentences(current_chunk, overlap)
            current_len = sum(len(s) for s in current_chunk) + max(len(current_chunk) - 1, 0)

        current_chunk.append(sentence)
        current_len += (1 if current_len > 0 else 0) + s_len

    # Flush remaining
    if current_chunk:
        remaining = " ".join(current_chunk).strip()
        if remaining:
            if chunks and len(remaining) < overlap:
                chunks[-1] += " " + remaining
            else:
                chunks.append(remaining)

    return chunks


def create_chunk_metadata(
    document_id: str,
    filename: str,
    chunks: List[str],
) -> List[Dict]:
    """Build metadata dicts for each chunk."""
    total = len(chunks)
    return [
        {
            "chunk_id": f"{document_id}_chunk_{i}",
            "document_id": document_id,
            "filename": filename,
            "chunk_index": i,
            "total_chunks": total,
            "text": chunk,
            "word_count": _count_words(chunk),
        }
        for i, chunk in enumerate(chunks)
    ]


# ─── Helpers ──────────────────────────────────────────────

def _split_into_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def _count_words(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(text.strip().split())


def _get_overlap_sentences(sentences: List[str], overlap_chars: int) -> List[str]:
    result: List[str] = []
    char_count = 0
    for s in reversed(sentences):
        sl = len(s)
        if char_count + sl > overlap_chars:
            break
        result.insert(0, s)
        char_count += sl
    return result


def _force_split(text: str, size: int, overlap: int) -> List[str]:
    # Safety: prevent infinite loop if bad params slip through
    if size <= 0:
        size = 500
    if overlap >= size:
        overlap = int(size * 0.2)

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Advance by (size - overlap), minimum 1 char to guarantee progress
        step = max(size - overlap, 1)
        start += step
    return chunks
