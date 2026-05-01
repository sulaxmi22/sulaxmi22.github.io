"""
AgentForge — Tool Definitions
Tools available to the agents: web search, text analysis, etc.
"""

import os
import json
import logging
import httpx
from datetime import datetime

logger = logging.getLogger(__name__)


async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using Tavily API (or fallback to DuckDuckGo).
    Returns a list of search results with title, url, and snippet.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY", "")
    
    if tavily_api_key:
        return await _tavily_search(query, tavily_api_key, max_results)
    else:
        return await _duckduckgo_search(query, max_results)


async def _tavily_search(query: str, api_key: str, max_results: int) -> list[dict]:
    """Search using Tavily API."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": True,
                    "search_depth": "advanced",
                },
            )
            data = response.json()
            results = []
            for r in data.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")[:500],
                    "source": "tavily",
                })
            return results
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return await _duckduckgo_search(query, max_results)


async def _duckduckgo_search(query: str, max_results: int) -> list[dict]:
    """Fallback search using DuckDuckGo Instant Answer API."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1},
            )
            data = response.json()
            results = []

            # Extract from related topics
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:100],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", "")[:500],
                        "source": "duckduckgo",
                    })

            # If we have an abstract, add it first
            if data.get("Abstract"):
                results.insert(0, {
                    "title": data.get("Heading", query),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data.get("Abstract", "")[:500],
                    "source": "duckduckgo",
                })

            return results[:max_results]
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return [{
            "title": "Search unavailable",
            "url": "",
            "snippet": f"Could not search for: {query}. Error: {str(e)}",
            "source": "error",
        }]


def analyze_text(text: str) -> dict:
    """Analyze a text passage — word count, key stats, readability estimate."""
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    paragraphs = text.count('\n\n') + 1
    
    avg_word_length = sum(len(w) for w in words) / max(len(words), 1)
    avg_sentence_length = len(words) / max(sentences, 1)
    
    return {
        "word_count": len(words),
        "sentence_count": sentences,
        "paragraph_count": paragraphs,
        "avg_word_length": round(avg_word_length, 1),
        "avg_sentence_length": round(avg_sentence_length, 1),
        "estimated_read_time_minutes": round(len(words) / 200, 1),
    }


def get_current_datetime() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
