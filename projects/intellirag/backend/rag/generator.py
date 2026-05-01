"""
IntelliRAG — LLM Generator Module
Handles LLM generation with streaming and citation enforcement.
"""

import json
import logging
from typing import AsyncGenerator
from openai import AsyncOpenAI

from backend.config import settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are IntelliRAG, an expert AI assistant that answers questions based on provided context documents.

RULES:
1. Answer ONLY based on the provided context. If the context doesn't contain the answer, say "I don't have enough information in the provided documents to answer this question."
2. ALWAYS cite your sources using [Source: filename, Page X] format after each claim.
3. Be concise but thorough. Use bullet points for complex answers.
4. If multiple sources agree, mention all of them.
5. Never make up information not in the context.

CONTEXT DOCUMENTS:
{context}
"""


class Generator:
    """LLM generation with streaming responses and citation enforcement."""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.openai_model

    def _format_context(self, chunks: list) -> str:
        """Format retrieved chunks into a numbered context string."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source_info = f"[Source: {chunk.source}, Page {chunk.page}]"
            context_parts.append(
                f"--- Document {i} {source_info} ---\n{chunk.content}\n"
            )
        return "\n".join(context_parts)

    def _extract_citations(self, text: str, chunks: list) -> list[dict]:
        """Extract citation references from the generated text."""
        citations = []
        seen = set()
        for chunk in chunks:
            source_key = f"{chunk.source}:p{chunk.page}"
            if source_key not in seen:
                seen.add(source_key)
                citations.append({
                    "source": chunk.source,
                    "page": chunk.page,
                    "relevance_score": round(chunk.rerank_score, 3),
                })
        return citations

    async def generate(self, query: str, chunks: list) -> dict:
        """Generate a complete response (non-streaming)."""
        context = self._format_context(chunks)
        system_message = SYSTEM_PROMPT.format(context=context)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content
        citations = self._extract_citations(answer, chunks)

        return {
            "answer": answer,
            "citations": citations,
            "model": self.model,
            "tokens_used": {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            },
        }

    async def generate_stream(self, query: str, chunks: list) -> AsyncGenerator[str, None]:
        """Generate a streaming response using Server-Sent Events format."""
        context = self._format_context(chunks)
        system_message = SYSTEM_PROMPT.format(context=context)

        # First, send the sources
        citations = self._extract_citations("", chunks)
        yield f"data: {json.dumps({'type': 'sources', 'citations': citations})}\n\n"

        # Stream the LLM response
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": query},
            ],
            temperature=0.1,
            max_tokens=1024,
            stream=True,
        )

        full_response = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        # Send completion signal
        yield f"data: {json.dumps({'type': 'done', 'full_response': full_response})}\n\n"
