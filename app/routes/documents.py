"""
Document Routes
POST /api/documents/upload  — Upload a document
GET  /api/documents         — List all documents
GET  /api/documents/stats   — Collection statistics
DELETE /api/documents/{id}  — Delete a document
"""
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from typing import Optional

from app.services.rag import rag_service

router = APIRouter(prefix="/api/documents", tags=["Documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(..., description="PDF, DOCX, TXT, or Markdown file"),
    chunk_size: Optional[int] = Form(None, description="Characters per chunk (100-10000). Leave empty for default 500."),
    chunk_overlap: Optional[int] = Form(None, description="Overlap in characters (0-500). Leave empty for default 100."),
):
    """Upload a document (PDF, DOCX, TXT, Markdown), chunk, embed, and store."""
    # Treat 0 as "not provided" (Swagger sends 0 for empty integer fields)
    if chunk_size is not None and chunk_size == 0:
        chunk_size = None
    if chunk_overlap is not None and chunk_overlap == 0:
        chunk_overlap = None

    # Validate extension
    import os
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    # Validate chunk params
    if chunk_size is not None and not (100 <= chunk_size <= 10000):
        raise HTTPException(400, "chunk_size must be between 100 and 10000")
    if chunk_overlap is not None and not (0 <= chunk_overlap <= 500):
        raise HTTPException(400, "chunk_overlap must be between 0 and 500")

    # Read file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "File size exceeds 50 MB limit")
    if len(content) == 0:
        raise HTTPException(400, "Uploaded file is empty")

    try:
        result = await rag_service.ingest_document(
            filename=file.filename,
            content=content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return {"success": True, "data": result}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        detail = str(e) or repr(e)
        raise HTTPException(500, f"Ingestion failed: {detail}")


@router.get("")
async def list_documents():
    """List all uploaded documents."""
    docs = rag_service.list_documents()
    return {"success": True, "data": docs, "count": len(docs)}


@router.get("/stats")
async def get_stats():
    """Get vector store statistics."""
    stats = rag_service.get_stats()
    return {"success": True, "data": stats}


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks."""
    import uuid as _uuid
    try:
        _uuid.UUID(document_id)
    except ValueError:
        raise HTTPException(400, "Invalid document ID (must be UUID)")

    try:
        result = rag_service.delete_document(document_id)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {e}")
