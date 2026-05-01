"""
MetricsAI — AI Insights Engine
Uses LLM to analyze data patterns, detect anomalies, and generate insights.
"""

import json
import logging
from openai import AsyncOpenAI

from backend.config import settings

logger = logging.getLogger(__name__)


class AIInsights:
    """LLM-powered data analysis and anomaly detection."""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    async def analyze_metrics(self, summary: dict, recent_data: list[dict]) -> dict:
        """Analyze current metrics and generate insights."""
        prompt = f"""You are a senior data analyst AI. Analyze these business metrics and provide actionable insights.

SUMMARY (last 30 data points):
{json.dumps(summary, indent=2)}

RECENT DATA (last 5 points):
{json.dumps(recent_data[-5:], indent=2)}

Provide your analysis as JSON:
{{
    "status": "healthy|warning|critical",
    "insights": [
        {{"title": "...", "description": "...", "type": "trend|anomaly|recommendation", "severity": "info|warning|critical"}}
    ],
    "anomalies_detected": true/false,
    "key_metric_summary": "One sentence summary of the most important finding"
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {
                "status": "unknown",
                "insights": [{"title": "Analysis unavailable", "description": str(e), "type": "info", "severity": "info"}],
                "anomalies_detected": False,
                "key_metric_summary": "AI analysis is currently unavailable.",
            }

    async def query_data(self, question: str, summary: dict, recent_data: list[dict]) -> str:
        """Answer natural language questions about the data."""
        prompt = f"""You are a data analyst. Answer the user's question based on this business data.

DATA SUMMARY:\n{json.dumps(summary, indent=2)}

RECENT DATA:\n{json.dumps(recent_data[-10:], indent=2)}

USER QUESTION: {question}

Provide a clear, data-backed answer. Include specific numbers and trends."""

        try:
            response = await self.client.chat.completions.create(
                model=settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Unable to analyze: {str(e)}"
