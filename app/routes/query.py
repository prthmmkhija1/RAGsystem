"""
Query Routes
POST /api/query — Ask a question against the document corpus
"""
from fastapi import APIRouter, HTTPException

from app.utils.validators import QueryRequest
from app.services.rag import rag_service

router = APIRouter(prefix="/api", tags=["Query"])


@router.post("/query")
async def query(req: QueryRequest):
    """
    Query the document corpus. Returns an LLM-generated answer
    grounded in retrieved context chunks with citations.
    """
    try:
        result = await rag_service.query_documents(
            query=req.query,
            top_k=req.top_k,
            document_id=req.document_id,
            temperature=req.temperature,
            include_metadata=req.include_metadata,
            verify=req.verify,
            rerank=req.rerank,
        )
        return {"success": True, "data": result}
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")
