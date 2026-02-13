"""
RAG Orchestration Service
Ties together: parse → chunk → embed → store → search → rerank → LLM answer.
"""
import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

from app.services.embeddings import embedding_service
from app.services.llm import llm_service
from app.services.rag import rerank_service
from app.vectorstore import vector_store_service
from app.utils import cache_service
from app.utils.document_parser import parse_document
from app.utils.chunking_service import (
    chunk_text,
    create_chunk_metadata,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)


# ──────────────────────────────────────────────────────────
# Document Ingestion Pipeline
# ──────────────────────────────────────────────────────────

async def ingest_document(
    filename: str,
    content: bytes,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full upload pipeline: parse → chunk → embed → store.
    """
    document_id = str(uuid.uuid4())
    start = time.time()

    print(f'[RAG] Ingesting "{filename}" (id: {document_id})')

    # 1. Parse the document into raw text
    raw_text = await parse_document(filename, content)
    print(f"[RAG]   Parsed: {len(raw_text)} characters")

    # 2. Chunk the text
    chunks = chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"[RAG]   Chunked: {len(chunks)} chunks")

    if not chunks:
        raise ValueError("Document produced zero chunks after parsing")

    # 3. Create chunk metadata
    chunk_meta = create_chunk_metadata(document_id, filename, chunks)

    # 4. Generate embeddings for all chunks (batched)
    embeddings = await embedding_service.generate_embeddings_batched(chunks, batch_size=20)
    print(f"[RAG]   Embedded: {len(embeddings)} vectors")

    # 5. Store in ChromaDB
    vector_store_service.store_chunks(chunk_meta, embeddings)

    # 6. Invalidate query cache since corpus changed
    cache_service.invalidate_queries()

    elapsed = f"{time.time() - start:.2f}s"
    print(f"[RAG]   Ingestion complete in {elapsed}")

    return {
        "document_id": document_id,
        "filename": filename,
        "chunk_count": len(chunks),
        "character_count": len(raw_text),
        "chunk_config": {
            "chunk_size": chunk_size or DEFAULT_CHUNK_SIZE,
            "chunk_overlap": chunk_overlap or DEFAULT_CHUNK_OVERLAP,
        },
        "processing_time": elapsed,
        "message": f'Successfully ingested "{filename}" into {len(chunks)} chunks',
    }


# ──────────────────────────────────────────────────────────
# Query Pipeline
# ──────────────────────────────────────────────────────────

async def query_documents(
    query: str,
    top_k: int = 5,
    document_id: Optional[str] = None,
    temperature: Optional[float] = None,
    include_metadata: bool = True,
    verify: bool = False,
    rerank: bool = False,
    skip_cache: bool = False,
) -> Dict[str, Any]:
    """
    Full query pipeline: embed query → search → rerank → LLM answer.
    """
    skip_cache = skip_cache or verify
    start = time.time()

    # Check cache
    if not skip_cache:
        cache_opts = {"top_k": top_k, "document_id": document_id, "rerank": rerank}
        cached = cache_service.get_query_result(query, cache_opts)
        if cached:
            print(f'[RAG] Query served from cache: "{query[:50]}..."')
            cached["processing_time"] = "0.00s (cached)"
            return cached

    print(f'[RAG] Query: "{query[:80]}..." (topK={top_k}, verify={verify}, rerank={rerank})')

    # 1. Embed the query
    query_embedding = await embedding_service.generate_embedding(query)

    # 2. Search for relevant chunks (fetch extra if reranking)
    fetch_k = min(top_k * 3, 20) if rerank else top_k
    if document_id:
        results = vector_store_service.search_by_document(query_embedding, document_id, fetch_k)
    else:
        results = vector_store_service.search_similar(query_embedding, fetch_k)

    if not results:
        return {
            "answer": "No relevant documents found. Please upload documents first.",
            "confidence": {"score": 0, "reason": "No documents found", "level": "none"},
            "sources": [], "query": query, "reranked": False,
            "processing_time": f"{time.time() - start:.2f}s",
        }

    # 3. Optionally re-rank
    if rerank and len(results) > 1:
        print(f"[RAG]   Re-ranking {len(results)} chunks...")
        results = rerank_service.rerank_chunks(query, results, top_n=top_k)
    elif len(results) > top_k:
        results = results[:top_k]

    # 4. Generate answer
    raw_answer = await llm_service.generate_answer(query, results, temperature=temperature)

    # 5. Parse confidence
    answer, confidence = llm_service.parse_confidence(raw_answer)

    # 6. Optionally verify
    verification = None
    if verify:
        print("[RAG]   Running answer verification...")
        verification = await llm_service.verify_answer(answer, results)
        v_status = "PASSED" if verification.get("isVerified") else "FAILED"
        print(f"[RAG]   Verification: {v_status} ({verification.get('overallScore', 0)}/10)")

    # 7. Build source citations
    sources = []
    for r in results:
        meta = r.get("metadata", {})
        src = {
            "filename": meta.get("filename"),
            "chunk_index": meta.get("chunk_index"),
            "document_id": meta.get("document_id"),
            "similarity_score": round(r.get("similarity_score", 0), 4),
        }
        if r.get("rerank_score") is not None:
            src["rerank_score"] = r["rerank_score"]
            src["original_rank"] = r.get("original_rank")
        if include_metadata:
            text = r.get("text", "")
            src["preview"] = text[:200] + ("..." if len(text) > 200 else "")
        sources.append(src)

    elapsed = f"{time.time() - start:.2f}s"
    print(f"[RAG]   Answered in {elapsed} using {len(results)} chunks (confidence: {confidence.get('score')}/10)")

    response: Dict[str, Any] = {
        "answer": answer,
        "confidence": confidence,
        "sources": sources,
        "query": query,
        "top_k": top_k,
        "reranked": rerank,
        "chunks_used": len(results),
        "processing_time": elapsed,
    }
    if verification:
        response["verification"] = verification

    # Cache result
    if not skip_cache:
        cache_opts = {"top_k": top_k, "document_id": document_id, "rerank": rerank}
        cache_service.set_query_result(query, cache_opts, response)

    return response


# ──────────────────────────────────────────────────────────
# Compare Pipeline
# ──────────────────────────────────────────────────────────

async def compare_documents(
    document_ids: List[str],
    topic: str,
    top_k: int = 5,
    structured: bool = False,
) -> Dict[str, Any]:
    """
    Compare two documents on a given topic.
    """
    start = time.time()
    print(f'[RAG] Comparing docs [{", ".join(document_ids)}] on: "{topic[:60]}" (structured={structured})')

    # 1. Embed topic
    topic_embedding = await embedding_service.generate_embedding(topic)

    # 2. Search each document
    doc1_results = vector_store_service.search_by_document(topic_embedding, document_ids[0], top_k)
    doc2_results = vector_store_service.search_by_document(topic_embedding, document_ids[1], top_k)

    if not doc1_results and not doc2_results:
        empty = {
            "similarities": [], "differences": [],
            "uniqueToDoc1": [], "uniqueToDoc2": [],
            "summary": {
                "overallAssessment": "No relevant content found in either document.",
                "agreementLevel": "none",
                "keyTakeaway": "Upload documents with relevant content and try again.",
            },
            "metadata": {"doc1ChunksAnalyzed": 0, "doc2ChunksAnalyzed": 0, "topic": topic},
        }
        return {
            "comparison": empty if structured else "No relevant content found.",
            "structured": structured,
            "doc1_sources": [], "doc2_sources": [],
            "topic": topic, "documents_compared": document_ids,
            "processing_time": f"{time.time() - start:.2f}s",
        }

    # 3. Generate comparison
    if structured:
        comparison = await llm_service.generate_structured_comparison(topic, doc1_results, doc2_results)
    else:
        comparison = await llm_service.generate_comparison(topic, doc1_results, doc2_results)

    # 4. Build citations
    def build_sources(results):
        return [
            {
                "filename": r.get("metadata", {}).get("filename"),
                "chunk_index": r.get("metadata", {}).get("chunk_index"),
                "document_id": r.get("metadata", {}).get("document_id"),
                "similarity_score": round(r.get("similarity_score", 0), 4),
                "preview": r.get("text", "")[:200],
            }
            for r in results
        ]

    elapsed = f"{time.time() - start:.2f}s"
    print(f"[RAG]   Comparison generated in {elapsed}")

    return {
        "comparison": comparison,
        "structured": structured,
        "doc1_sources": build_sources(doc1_results),
        "doc2_sources": build_sources(doc2_results),
        "topic": topic,
        "documents_compared": document_ids,
        "processing_time": elapsed,
    }


# ──────────────────────────────────────────────────────────
# Document Management
# ──────────────────────────────────────────────────────────

def delete_document(document_id: str) -> Dict:
    vector_store_service.delete_document(document_id)
    cache_service.invalidate_document_queries(document_id)
    return {"document_id": document_id, "message": "Document deleted successfully"}


def list_documents() -> List[Dict]:
    return vector_store_service.list_documents()


def get_stats() -> Dict:
    return vector_store_service.get_stats()
