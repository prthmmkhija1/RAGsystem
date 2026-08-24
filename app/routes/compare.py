"""
Compare Routes
POST /api/compare — Compare two documents on a given topic
"""
from fastapi import APIRouter, HTTPException

from app.utils.validators import CompareRequest
from app.utils.error_handler import NotFoundError
from app.services.rag import rag_service

router = APIRouter(prefix="/api", tags=["Compare"])


@router.post("/compare")
async def compare(req: CompareRequest):
    """
    Compare two documents on a given topic.
    Returns similarities, differences, and structured analysis.
    """
    try:
        result = await rag_service.compare_documents(
            document_ids=req.document_ids,
            topic=req.topic,
            top_k=req.top_k,
            structured=req.structured,
            verify=req.verify,
            rerank=req.rerank,
        )
        return {"success": True, "data": result}
    except NotFoundError as e:
        raise HTTPException(404, e.message)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Comparison failed: {e}")
