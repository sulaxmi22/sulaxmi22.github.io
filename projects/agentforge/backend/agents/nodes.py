"""
AgentForge — Agent Node Implementations
Each function is a node in the LangGraph state machine.
"""

import json
import logging
import time
from openai import AsyncOpenAI

from backend.agents.state import AgentState
from backend.agents.tools import web_search, analyze_text, get_current_datetime
from backend.config import settings

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
)


def _log_entry(agent: str, action: str, details: str = "") -> dict:
    """Create a log entry for the agent activity visualization."""
    return {
        "agent": agent,
        "action": action,
        "details": details,
        "timestamp": time.time(),
    }


async def _llm_call(system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
    """Helper to make an LLM call."""
    kwargs = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 2048,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    
    response = await _client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# NODE: Router Agent
# ─────────────────────────────────────────────
async def router_node(state: AgentState) -> dict:
    """
    Router Agent: Classifies the query and determines the workflow.
    Routes to: research, document_analysis, or direct_answer.
    """
    logger.info(f"[Router] Processing: {state['query'][:80]}")

    system_prompt = """You are a Query Router. Classify the user's query into one of these categories:

1. "research" — requires searching the web for current information, comparing multiple sources, or investigating a topic
2. "document_analysis" — requires analyzing provided text, data, or documents
3. "direct_answer" — a simple question that can be answered from general knowledge

Respond with ONLY JSON: {"query_type": "research|document_analysis|direct_answer", "reasoning": "brief explanation"}"""

    result = await _llm_call(system_prompt, state["query"], json_mode=True)
    parsed = json.loads(result)
    query_type = parsed.get("query_type", "research")

    return {
        "query_type": query_type,
        "agent_log": [_log_entry("Router", f"Classified as: {query_type}", parsed.get("reasoning", ""))],
    }


# ─────────────────────────────────────────────
# NODE: Researcher Agent
# ─────────────────────────────────────────────
async def researcher_node(state: AgentState) -> dict:
    """
    Researcher Agent: Searches the web and gathers information.
    Generates multiple search queries for comprehensive coverage.
    """
    logger.info(f"[Researcher] Researching: {state['query'][:80]}")

    # Step 1: Generate search queries
    system_prompt = """You are a Research Strategist. Given a user query, generate 2-3 focused search queries 
that would find comprehensive, diverse information on the topic. 
Respond with JSON: {"queries": ["query1", "query2", "query3"]}"""

    queries_result = await _llm_call(system_prompt, state["query"], json_mode=True)
    queries = json.loads(queries_result).get("queries", [state["query"]])

    # Step 2: Execute searches
    all_results = []
    for q in queries[:3]:
        results = await web_search(q, max_results=3)
        all_results.extend(results)

    # Step 3: Synthesize findings
    findings_text = "\n\n".join(
        f"**{r['title']}** ({r['url']})\n{r['snippet']}"
        for r in all_results if r.get("snippet")
    )

    system_prompt = """You are a Research Analyst. Synthesize the following search results into structured research notes.
Include key findings, data points, and source attribution. Organize by theme."""

    synthesis = await _llm_call(
        system_prompt,
        f"Original question: {state['query']}\n\nSearch Results:\n{findings_text}",
    )

    research_entry = {
        "type": "web_research",
        "queries_used": queries,
        "num_results": len(all_results),
        "synthesis": synthesis,
        "sources": [{"title": r["title"], "url": r["url"]} for r in all_results if r.get("url")],
    }

    return {
        "research_results": [research_entry],
        "agent_log": [
            _log_entry("Researcher", f"Searched {len(queries)} queries", f"Found {len(all_results)} results"),
            _log_entry("Researcher", "Synthesized findings", f"{len(synthesis)} chars"),
        ],
    }


# ─────────────────────────────────────────────
# NODE: Writer Agent
# ─────────────────────────────────────────────
async def writer_node(state: AgentState) -> dict:
    """
    Writer Agent: Synthesizes research into a well-structured report.
    Handles both initial drafts and revisions based on critic feedback.
    """
    logger.info(f"[Writer] Drafting report (revision {state.get('revision_count', 0)})")

    research_text = ""
    for r in state.get("research_results", []):
        if isinstance(r, dict):
            research_text += r.get("synthesis", str(r)) + "\n\n"

    is_revision = state.get("revision_count", 0) > 0
    critique = state.get("critique", "")

    if is_revision and critique:
        system_prompt = f"""You are an expert technical writer revising a report based on feedback.

PREVIOUS DRAFT:
{state.get('draft_report', '')}

CRITIC FEEDBACK:
{critique}

Revise the report addressing ALL feedback points. Maintain a professional, well-structured format.
Use markdown formatting with clear headers, bullet points, and citations."""
    else:
        system_prompt = """You are an expert technical writer. Create a comprehensive, well-structured report 
based on the research findings below. 

Requirements:
- Clear executive summary
- Organized sections with headers
- Key findings highlighted
- Source citations
- Actionable conclusions
- Use markdown formatting

Write in a professional, authoritative tone."""

    report = await _llm_call(
        system_prompt,
        f"Query: {state['query']}\n\nResearch:\n{research_text}",
    )

    action = "Revised report" if is_revision else "Drafted initial report"
    return {
        "draft_report": report,
        "revision_count": state.get("revision_count", 0) + 1,
        "agent_log": [_log_entry("Writer", action, f"{len(report)} chars, revision #{state.get('revision_count', 0) + 1}")],
    }


# ─────────────────────────────────────────────
# NODE: Critic Agent
# ─────────────────────────────────────────────
async def critic_node(state: AgentState) -> dict:
    """
    Critic Agent: Evaluates the report quality and provides feedback.
    Returns a quality score (0-1) and detailed critique.
    """
    logger.info(f"[Critic] Evaluating report quality")

    system_prompt = """You are a rigorous Quality Critic. Evaluate the following report on these criteria:

1. **Accuracy** - Are claims supported by cited sources?
2. **Completeness** - Does it thoroughly address the original query?
3. **Structure** - Is it well-organized with clear sections?
4. **Clarity** - Is the writing clear and professional?
5. **Actionability** - Does it provide useful, actionable insights?

Respond with JSON:
{
    "quality_score": <float 0.0-1.0>,
    "criteria_scores": {
        "accuracy": <float>,
        "completeness": <float>,
        "structure": <float>,
        "clarity": <float>,
        "actionability": <float>
    },
    "strengths": ["..."],
    "improvements_needed": ["..."],
    "critique": "Detailed feedback for the writer"
}"""

    result = await _llm_call(
        system_prompt,
        f"Original Query: {state['query']}\n\nReport to Evaluate:\n{state['draft_report']}",
        json_mode=True,
    )

    parsed = json.loads(result)
    quality_score = min(max(float(parsed.get("quality_score", 0.5)), 0.0), 1.0)

    return {
        "quality_score": quality_score,
        "critique": parsed.get("critique", ""),
        "agent_log": [_log_entry(
            "Critic",
            f"Quality: {quality_score:.2f}",
            f"{'✅ Approved' if quality_score >= 0.8 else '🔄 Needs revision'} | {', '.join(parsed.get('improvements_needed', [])[:2])}",
        )],
    }


# ─────────────────────────────────────────────
# NODE: Direct Answer
# ─────────────────────────────────────────────
async def direct_answer_node(state: AgentState) -> dict:
    """Simple direct answer for straightforward questions."""
    system_prompt = """You are a knowledgeable AI assistant. Provide a clear, accurate, and concise answer.
Use markdown formatting. Include relevant context and examples where helpful."""

    answer = await _llm_call(system_prompt, state["query"])

    return {
        "final_output": answer,
        "agent_log": [_log_entry("Direct", "Generated answer", f"{len(answer)} chars")],
    }


# ─────────────────────────────────────────────
# NODE: Finalize
# ─────────────────────────────────────────────
async def finalize_node(state: AgentState) -> dict:
    """Finalize the report as the output."""
    return {
        "final_output": state.get("draft_report", "No report generated."),
        "agent_log": [_log_entry("System", "Report finalized", f"Quality: {state.get('quality_score', 0):.2f}")],
    }
