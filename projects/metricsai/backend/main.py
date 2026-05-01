"""
MetricsAI — FastAPI Server
Real-time analytics dashboard API with WebSocket streaming.
"""

import json
import time
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import settings
from backend.analytics.engine import DataEngine
from backend.analytics.ai_insights import AIInsights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

data_engine: DataEngine | None = None
ai_insights: AIInsights | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global data_engine, ai_insights
    logger.info("🚀 MetricsAI starting up...")
    data_engine = DataEngine()
    ai_insights = AIInsights()
    logger.info("✅ MetricsAI ready")
    yield
    logger.info("👋 MetricsAI shutting down")


app = FastAPI(
    title="MetricsAI API",
    description="Real-time AI analytics dashboard with WebSocket streaming",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "uptime_seconds": int(time.time() - START_TIME),
    }


@app.get("/api/metrics/latest")
async def get_latest_metrics():
    """Get the latest metrics snapshot."""
    return data_engine.get_latest()


@app.get("/api/metrics/history")
async def get_metrics_history(points: int = 60):
    """Get historical metrics data."""
    return data_engine.get_history(points)


@app.get("/api/metrics/summary")
async def get_metrics_summary():
    """Get aggregated metrics summary."""
    return data_engine.get_summary()


@app.get("/api/insights")
async def get_insights():
    """Get AI-powered insights on current metrics."""
    summary = data_engine.get_summary()
    history = data_engine.get_history(30)
    return await ai_insights.analyze_metrics(summary, history)


@app.post("/api/query")
async def query_data(request: QueryRequest):
    """Ask natural language questions about the data."""
    summary = data_engine.get_summary()
    history = data_engine.get_history(30)
    answer = await ai_insights.query_data(request.question, summary, history)
    return {"answer": answer}


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics streaming."""
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            # Push latest metrics
            latest = data_engine.get_latest()
            await websocket.send_json({"type": "metrics", "data": latest})
            await asyncio.sleep(settings.stream_interval_ms / 1000)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# Init files
@app.on_event("startup")
async def create_init_files():
    """Ensure package init files exist."""
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=True)
