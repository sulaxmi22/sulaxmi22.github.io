"""
IntelliRAG — RAGAS-style Evaluation Module
Measures RAG pipeline quality: Faithfulness, Answer Relevancy, Context Precision.
"""

import json
import logging
from openai import AsyncOpenAI

from backend.config import settings


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality using LLM-as-judge approach
    inspired by RAGAS metrics.
    """

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.openai_model

    async def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Faithfulness: Are all claims in the answer supported by the context?
        Score 0.0-1.0
        """
        prompt = f"""Evaluate the faithfulness of the following answer based on the provided context.
Faithfulness measures whether all claims in the answer can be traced back to the context.

Context: {context[:3000]}

Answer: {answer}

Score the faithfulness from 0.0 to 1.0 where:
- 1.0 = Every claim is fully supported by the context
- 0.5 = Some claims are supported, some are not
- 0.0 = The answer contains mostly unsupported claims

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,

            )
            result = _parse_json(response.choices[0].message.content)
            return min(max(float(result.get("score", 0.0)), 0.0), 1.0)
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return 0.0

    async def evaluate_relevancy(self, query: str, answer: str) -> float:
        """
        Answer Relevancy: How relevant is the answer to the question?
        Score 0.0-1.0
        """
        prompt = f"""Evaluate how relevant the following answer is to the question.

Question: {query}

Answer: {answer}

Score the relevancy from 0.0 to 1.0 where:
- 1.0 = The answer directly and completely addresses the question
- 0.5 = The answer partially addresses the question
- 0.0 = The answer is completely irrelevant to the question

Respond with ONLY a JSON object: {{"score": <float>, "reasoning": "<brief explanation>"}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,

            )
            result = _parse_json(response.choices[0].message.content)
            return min(max(float(result.get("score", 0.0)), 0.0), 1.0)
        except Exception as e:
            logger.error(f"Relevancy evaluation failed: {e}")
            return 0.0

    async def evaluate_context_precision(self, query: str, contexts: list[str]) -> float:
        """
        Context Precision: How precise is the retrieved context?
        Measures if the relevant chunks appear early in the ranking.
        """
        if not contexts:
            return 0.0

        prompt = f"""For the given question, evaluate each context chunk for relevance.
        
Question: {query}

Contexts:
{chr(10).join(f'Context {i+1}: {ctx[:500]}' for i, ctx in enumerate(contexts[:5]))}

For each context, respond with 1 (relevant) or 0 (irrelevant).
Respond with ONLY a JSON object: {{"relevance": [1, 0, 1, ...]}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,

            )
            result = _parse_json(response.choices[0].message.content)
            relevance = result.get("relevance", [])

            if not relevance:
                return 0.0

            # Calculate precision@k (weighted by position)
            precision_sum = 0.0
            relevant_count = 0
            for i, rel in enumerate(relevance):
                if rel == 1:
                    relevant_count += 1
                    precision_sum += relevant_count / (i + 1)

            if relevant_count == 0:
                return 0.0
            return precision_sum / relevant_count

        except Exception as e:
            logger.error(f"Context precision evaluation failed: {e}")
            return 0.0

    async def evaluate(self, query: str, answer: str, contexts: list[str]) -> dict:
        """Run all evaluation metrics and return a comprehensive report."""
        context_combined = "\n\n".join(contexts)

        faithfulness = await self.evaluate_faithfulness(answer, context_combined)
        relevancy = await self.evaluate_relevancy(query, answer)
        precision = await self.evaluate_context_precision(query, contexts)

        return {
            "faithfulness": round(faithfulness, 3),
            "answer_relevancy": round(relevancy, 3),
            "context_precision": round(precision, 3),
            "overall": round((faithfulness + relevancy + precision) / 3, 3),
        }
