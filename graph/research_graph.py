"""
LangGraph workflow orchestration.

Defines the state machine: Researcher → Analyzer → Comparator → Synthesizer
"""

from langgraph.graph import StateGraph, END
from loguru import logger

from graph.state import ResearchState
from agents.researcher import researcher_node
from agents.analyzer import analyzer_node
from agents.comparator import comparator_node
from agents.synthesizer import synthesizer_node


def create_research_graph() -> StateGraph:
    """
    Create LangGraph workflow for autonomous research.

    Flow:
        START → Researcher → Analyzer → Comparator → Synthesizer → END

    Returns:
        Compiled StateGraph ready to invoke
    """
    logger.info("Building LangGraph workflow...")

    # Initialize graph
    workflow = StateGraph(ResearchState)

    # Add agent nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("comparator", comparator_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Define linear flow
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "analyzer")
    workflow.add_edge("analyzer", "comparator")
    workflow.add_edge("comparator", "synthesizer")
    workflow.add_edge("synthesizer", END)

    logger.info("✓ LangGraph workflow built")

    return workflow.compile()
