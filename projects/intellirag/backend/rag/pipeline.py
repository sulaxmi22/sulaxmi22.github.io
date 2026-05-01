"""
IntelliRAG — RAG Pipeline Orchestrator
Coordinates the full RAG pipeline: retrieve → evaluate → generate.
Implements Corrective RAG pattern.
"""

import logging
import time
from typing import AsyncGenerator

from backend.rag.ingestion import DocumentIngestion
from backend.rag.retriever import HybridRetriever
from backend.rag.generator import Generator
from backend.rag.evaluator import RAGEvaluator
from backend.config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Production RAG pipeline with Corrective RAG pattern:
    1. Retrieve (hybrid search + rerank)
    2. Evaluate retrieval quality
    3. If quality is low → fall back / warn
    4. Generate with citations
    """

    def __init__(self):
        self.ingestion = DocumentIngestion()
        self.retriever = HybridRetriever()
        self.generator = Generator()
        self.evaluator = RAGEvaluator()

    async def query(self, question: str) -> dict:
        """
        Full non-streaming RAG pipeline execution.
        Returns answer, citations, metrics, and timing.
        """
        start_time = time.time()

        # Step 1: Retrieve
        t0 = time.time()
        chunks = await self.retriever.retrieve(question)
        retrieval_time = time.time() - t0

        if not chunks:
            return {
                "answer": "I couldn't find any relevant documents to answer your question. Please try uploading some documents first.",
                "citations": [],
                "metrics": {"retrieval_quality": 0.0},
                "timing": {"total_ms": int((time.time() - start_time) * 1000)},
                "corrective_rag": {"triggered": False},
            }

        # Step 2: Corrective RAG — evaluate retrieval quality
        relevance_score = self.retriever.evaluate_relevance(question, chunks)
        corrective_triggered = relevance_score < settings.relevance_threshold

        if corrective_triggered:
            logger.warning(
                f"Corrective RAG triggered: relevance={relevance_score:.3f} "
                f"< threshold={settings.relevance_threshold}. "
                f"Context may be insufficient for a reliable answer."
            )

        # Step 3: Generate
        t1 = time.time()
        result = await self.generator.generate(question, chunks)
        generation_time = time.time() - t1

        total_time = time.time() - start_time

        return {
            "answer": result["answer"],
            "citations": result["citations"],
            "model": result["model"],
            "tokens_used": result["tokens_used"],
            "corrective_rag": {
                "triggered": corrective_triggered,
                "relevance_score": round(relevance_score, 3),
                "threshold": settings.relevance_threshold,
            },
            "timing": {
                "retrieval_ms": int(retrieval_time * 1000),
                "generation_ms": int(generation_time * 1000),
                "total_ms": int(total_time * 1000),
            },
            "chunks_used": len(chunks),
        }

    async def query_stream(self, question: str) -> AsyncGenerator[str, None]:
        """
        Streaming RAG pipeline — yields SSE events.
        """
        import json

        start_time = time.time()

        # Step 1: Retrieve
        chunks = await self.retriever.retrieve(question)

        if not chunks:
            yield f"data: {json.dumps({'type': 'error', 'content': 'No relevant documents found. Please upload documents first.'})}\n\n"
            return

        # Step 2: Check relevance (Corrective RAG)
        relevance_score = self.retriever.evaluate_relevance(question, chunks)
        if relevance_score < settings.relevance_threshold:
            yield f"data: {json.dumps({'type': 'warning', 'content': f'Low retrieval confidence ({relevance_score:.2f}). Answer may be less reliable.'})}\n\n"

        # Step 3: Stream generation
        async for event in self.generator.generate_stream(question, chunks):
            yield event

        # Final metrics
        total_time = time.time() - start_time
        yield f"data: {json.dumps({'type': 'metrics', 'timing_ms': int(total_time * 1000), 'chunks_used': len(chunks), 'relevance_score': round(relevance_score, 3)})}\n\n"

    async def evaluate_pipeline(self, question: str, answer: str, chunks: list) -> dict:
        """Run RAGAS evaluation on a query-answer pair."""
        contexts = [c.content for c in chunks]
        return await self.evaluator.evaluate(question, answer, contexts)
