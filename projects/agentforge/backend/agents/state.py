"""
AgentForge — Typed State Definitions
Defines the state that flows through the LangGraph agent graph.
"""

from typing import TypedDict, Annotated, Literal
from operator import add


class AgentState(TypedDict):
    """State that flows through the multi-agent graph."""

    # User's original query
    query: str

    # Router classification
    query_type: Literal["research", "document_analysis", "direct_answer", ""] 

    # Research findings from Researcher agent
    research_results: Annotated[list[dict], add]

    # Written report from Writer agent
    draft_report: str

    # Critic evaluation
    quality_score: float
    critique: str

    # Revision tracking
    revision_count: int
    max_revisions: int

    # Human-in-the-loop
    requires_approval: bool
    approved: bool

    # Final output
    final_output: str

    # Agent activity log for visualization
    agent_log: Annotated[list[dict], add]

    # Error tracking
    error: str
