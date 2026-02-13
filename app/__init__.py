"""
FastAPI Application
Main entry point — registers routes, middleware, and exception handlers.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.utils.error_handler import AppError, app_error_handler, generic_error_handler
from app.utils import cache_service
from app.vectorstore import vector_store_service

from app.routes.documents import router as documents_router
from app.routes.query import router as query_router
from app.routes.compare import router as compare_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup / shutdown logic."""
    # ── Startup ──
    print("[Server] Initializing vector store...")
    vector_store_service.initialize()
    print("[Server] RAG system ready!")
    yield
    # ── Shutdown ──
    print("[Server] Shutting down — clearing caches...")
    cache_service.flush_all()
    print("[Server] Goodbye.")


app = FastAPI(
    title="AI RAG System",
    description="Production-grade RAG system using Groq LLM and ChromaDB",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Exception Handlers ──────────────────────────────────
app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(Exception, generic_error_handler)

# ─── Routes ───────────────────────────────────────────────
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(compare_router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    stats = vector_store_service.get_stats()
    return {
        "status": "healthy",
        "service": "AI RAG System",
        "vector_store": {
            "total_chunks": stats["total_chunks"],
            "total_documents": stats["total_documents"],
        },
        "cache": cache_service.get_stats(),
    }