"""
Centralized Error Handling
Custom exception classes + FastAPI exception handlers.
"""
from fastapi import Request
from fastapi.responses import JSONResponse


# ─── Custom Exception Classes ──────────────────────────────

class AppError(Exception):
    """Base application error."""
    def __init__(self, message: str, status_code: int = 500, details=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


class ValidationError(AppError):
    """Input validation failed."""
    def __init__(self, message: str, details=None):
        super().__init__(message, 400, details)


class NotFoundError(AppError):
    """Requested resource not found."""
    def __init__(self, resource: str = "Resource"):
        super().__init__(f"{resource} not found", 404)


class ExternalServiceError(AppError):
    """Third-party service (Grok, ChromaDB) failure."""
    def __init__(self, service: str, message: str):
        super().__init__(f"{service} error: {message}", 502)


class RateLimitError(AppError):
    """API rate limit exceeded."""
    def __init__(self, retry_after: int = 60):
        super().__init__("Rate limit exceeded, please try again later", 429)
        self.retry_after = retry_after


# ─── FastAPI Exception Handlers ────────────────────────────

async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Handle all custom AppError subclasses."""
    body: dict = {
        "success": False,
        "error": {"message": exc.message},
    }
    if exc.details:
        body["error"]["details"] = exc.details
    if isinstance(exc, RateLimitError):
        body["error"]["retryAfter"] = exc.retry_after
    return JSONResponse(status_code=exc.status_code, content=body)


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {"message": "Internal server error"},
        },
    )
