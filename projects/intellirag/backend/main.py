"""
IntelliRAG — FastAPI Server
Production API with document ingestion, querying, evaluation, and health monitoring.
"""

import os
import time
import logging
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from backend.config import settings
from backend.rag.pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Pipeline singleton
pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG pipeline on startup."""
    global pipeline
    logger.info("🚀 Initializing IntelliRAG pipeline...")
    pipeline = RAGPipeline()
    logger.info("✅ IntelliRAG pipeline ready")
    yield
    logger.info("👋 Shutting down IntelliRAG")


app = FastAPI(
    title="IntelliRAG API",
    description="Production-grade RAG pipeline with hybrid search, reranking, and evaluation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track startup time
START_TIME = time.time()


# --- Request/Response Models ---
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The question to ask")
    stream: bool = Field(default=True, description="Whether to stream the response")


class TextIngestRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text content to ingest")
    source_name: str = Field(default="direct_input", description="Name for the source")


class EvaluateRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]


# --- Endpoints ---
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    uptime = int(time.time() - START_TIME)
    stats = pipeline.ingestion.get_collection_stats() if pipeline else {}
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "collection": stats,
        "model": settings.openai_model,
    }


@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a document file (PDF, TXT, MD)."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Validate file type
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".txt", ".md"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: .pdf, .txt, .md",
        )

    # Save to temp file and ingest
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await pipeline.ingestion.ingest_file(tmp_path)
        return result
    finally:
        os.unlink(tmp_path)


@app.post("/ingest/text")
async def ingest_text(request: TextIngestRequest):
    """Ingest raw text directly."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return await pipeline.ingestion.ingest_text(request.text, request.source_name)


@app.post("/query")
async def query(request: QueryRequest):
    """Query the RAG pipeline. Supports streaming and non-streaming responses."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if request.stream:
        return StreamingResponse(
            pipeline.query_stream(request.question),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        result = await pipeline.query(request.question)
        return result


@app.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    """Run RAGAS evaluation on a query-answer pair."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return await pipeline.evaluator.evaluate(
        request.question, request.answer, request.contexts
    )


@app.get("/collection/stats")
async def collection_stats():
    """Get document collection statistics."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline.ingestion.get_collection_stats()


@app.delete("/collection")
async def clear_collection():
    """Clear all documents from the collection."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    pipeline.ingestion.clear_collection()
    return {"status": "collection cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
