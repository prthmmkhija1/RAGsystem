"""
Request Validators
Pydantic models for validating incoming requests.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import uuid


class UploadParams(BaseModel):
    """Optional body fields sent alongside file upload."""
    chunk_size: Optional[int] = Field(None, ge=100, le=10000)
    chunk_overlap: Optional[int] = Field(None, ge=0, le=500)


class QueryRequest(BaseModel):
    """POST /api/query body."""
    query: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(5, ge=1, le=20)
    document_id: Optional[str] = None
    include_metadata: bool = True
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
    verify: bool = False
    rerank: bool = False

    @field_validator("document_id")
    @classmethod
    def validate_uuid(cls, v):
        if v is None:
            return None
        # Treat Swagger placeholders and empty strings as "not provided"
        if not v or not v.strip() or v.strip().lower() == "string":
            return None
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError("document_id must be a valid UUID")
        return v


class CompareRequest(BaseModel):
    """POST /api/compare body."""
    document_ids: List[str] = Field(..., min_length=2, max_length=2)
    topic: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    structured: bool = False

    @field_validator("document_ids")
    @classmethod
    def validate_doc_ids(cls, v):
        if len(v) != 2:
            raise ValueError("Exactly two document IDs are required for comparison")
        for doc_id in v:
            try:
                uuid.UUID(doc_id)
            except ValueError:
                raise ValueError(f"Invalid UUID: {doc_id}")
        return v
