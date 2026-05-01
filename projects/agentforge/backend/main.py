"""
AgentForge — FastAPI Server
API for the multi-agent research assistant.
"""

import json
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 AgentForge starting up...")
    yield
    logger.info("👋 AgentForge shutting down")


app = FastAPI(
    title="AgentForge API",
    description="Multi-Agent Research Assistant powered by LangGraph",
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


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    stream: bool = Field(default=False)
    max_revisions: int = Field(default=3, ge=1, le=5)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "uptime_seconds": int(time.time() - START_TIME),
        "model": settings.openai_model,
        "agents": ["Router", "Researcher", "Writer", "Critic"],
    }


@app.post("/research")
async def research(request: ResearchRequest):
    """Run the multi-agent research pipeline."""
    from backend.agents.graph import agent

    initial_state = {
        "query": request.query,
        "query_type": "",
        "research_results": [],
        "draft_report": "",
        "quality_score": 0.0,
        "critique": "",
        "revision_count": 0,
        "max_revisions": request.max_revisions,
        "requires_approval": False,
        "approved": True,
        "final_output": "",
        "agent_log": [],
        "error": "",
    }

    if request.stream:
        return StreamingResponse(
            _stream_research(agent, initial_state),
            media_type="text/event-stream",
        )

    # Non-streaming execution
    start = time.time()
    result = await agent.ainvoke(initial_state)
    elapsed = time.time() - start

    return {
        "output": result.get("final_output", ""),
        "quality_score": result.get("quality_score", 0.0),
        "revisions": result.get("revision_count", 0),
        "agent_log": result.get("agent_log", []),
        "timing_ms": int(elapsed * 1000),
    }


async def _stream_research(agent, initial_state: dict):
    """Stream agent activity as Server-Sent Events."""
    start = time.time()
    
    async for event in agent.astream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            # Stream agent log entries
            for log in node_output.get("agent_log", []):
                yield f"data: {json.dumps({'type': 'agent_activity', 'node': node_name, 'log': log})}\n\n"

            # Stream final output if available
            if node_output.get("final_output"):
                yield f"data: {json.dumps({'type': 'output', 'content': node_output['final_output']})}\n\n"

            # Stream quality scores
            if node_output.get("quality_score"):
                yield f"data: {json.dumps({'type': 'quality', 'score': node_output['quality_score'], 'critique': node_output.get('critique', '')})}\n\n"

    elapsed = time.time() - start
    yield f"data: {json.dumps({'type': 'done', 'timing_ms': int(elapsed * 1000)})}\n\n"


@app.post("/quick-answer")
async def quick_answer(request: ResearchRequest):
    """Direct answer without the full agent pipeline (for simple questions)."""
    from backend.agents.nodes import direct_answer_node

    state = {"query": request.query, "agent_log": []}
    result = await direct_answer_node(state)
    return {"output": result.get("final_output", "")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.host, port=settings.port, reload=True)
