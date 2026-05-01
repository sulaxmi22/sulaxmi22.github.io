"""
AgentForge — LangGraph State Machine
Defines the multi-agent graph with conditional routing and retry loops.
"""

import logging
from langgraph.graph import StateGraph, END

from backend.agents.state import AgentState
from backend.agents.nodes import (
    router_node,
    researcher_node,
    writer_node,
    critic_node,
    direct_answer_node,
    finalize_node,
)

logger = logging.getLogger(__name__)

# Quality threshold for the Critic review loop
QUALITY_THRESHOLD = 0.8
MAX_REVISIONS = 3


def route_by_query_type(state: AgentState) -> str:
    """Conditional edge: route based on query classification."""
    query_type = state.get("query_type", "research")
    logger.info(f"[Graph] Routing to: {query_type}")

    if query_type == "direct_answer":
        return "direct_answer"
    elif query_type == "document_analysis":
        return "researcher"  # Use researcher for doc analysis too
    else:
        return "researcher"


def should_revise(state: AgentState) -> str:
    """
    Conditional edge after Critic: decide whether to revise or finalize.
    Implements the retry loop.
    """
    quality = state.get("quality_score", 0.0)
    revisions = state.get("revision_count", 0)
    max_rev = state.get("max_revisions", MAX_REVISIONS)

    if quality >= QUALITY_THRESHOLD:
        logger.info(f"[Graph] Quality {quality:.2f} >= {QUALITY_THRESHOLD} → Finalize")
        return "finalize"
    elif revisions >= max_rev:
        logger.info(f"[Graph] Max revisions ({max_rev}) reached → Finalize anyway")
        return "finalize"
    else:
        logger.info(f"[Graph] Quality {quality:.2f} < {QUALITY_THRESHOLD}, revision {revisions}/{max_rev} → Revise")
        return "revise"


def build_graph() -> StateGraph:
    """
    Build the multi-agent LangGraph state machine.
    
    Graph structure:
        Router → [research | direct_answer]
        research → Researcher → Writer → Critic → [finalize | revise→Writer]
        direct_answer → END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("finalize", finalize_node)

    # Set entry point
    graph.set_entry_point("router")

    # Conditional routing from Router
    graph.add_conditional_edges(
        "router",
        route_by_query_type,
        {
            "researcher": "researcher",
            "direct_answer": "direct_answer",
        },
    )

    # Linear flow: Researcher → Writer → Critic
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "critic")

    # Conditional loop: Critic → [finalize | revise(writer)]
    graph.add_conditional_edges(
        "critic",
        should_revise,
        {
            "finalize": "finalize",
            "revise": "writer",
        },
    )

    # Terminal nodes
    graph.add_edge("direct_answer", END)
    graph.add_edge("finalize", END)

    return graph


def create_agent():
    """Create and compile the agent graph."""
    graph = build_graph()
    return graph.compile()


# Pre-built agent instance
agent = create_agent()
