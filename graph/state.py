"""
State definitions for the research workflow.

Defines TypedDict schemas for LangGraph state management.
"""

from typing import TypedDict, List, Dict, Optional, Any
from typing_extensions import Annotated
from operator import add


class ResearchPaper(TypedDict):
    """Individual paper metadata"""
    arxiv_id: str
    title: str
    authors: List[str]
    year: int
    summary: str
    pdf_path: Optional[str]
    citation: str  # Pre-formatted: [AuthorName et al., Year, arXiv:ID]
    extraction_status: str  # "success", "failed_download", "failed_parse"


class Analysis(TypedDict):
    """Analysis results from analyzer agent"""
    key_findings: List[Dict[str, Any]]  # [{finding: str, sources: List[str]}]
    contradictions: List[Dict[str, str]]  # [{point: str, paper1: str, paper2: str, explanation: str}]
    complementary_findings: List[Dict[str, str]]  # [{finding_1: str, finding_2: str, relationship: str}]
    trends: List[str]
    consensus_points: List[str]
    gaps: List[str]


class Comparison(TypedDict):
    """Comparison results from comparator agent"""
    metrics_table: List[Dict[str, Any]]  # [{paper: str, metric_name: value, ...}]
    metric_names: List[str]  # List of detected metric names (e.g., "Energy Density", "Cycle Life")
    comparison_summary: str  # Textual summary of key differences/similarities


class ResearchState(TypedDict):
    """Main state shared across all agents in the research workflow"""

    # Job identification
    job_id: str  # Unique identifier for job isolation and tracking

    # Input
    topic: str
    max_papers: int

    # Researcher outputs
    documents: Annotated[List[ResearchPaper], add]  # Allows appending
    faiss_index_path: Optional[str]
    error_log: Annotated[List[Dict[str, str]], add]  # Track failures

    # Analyzer outputs
    analysis: Optional[Analysis]

    # Comparator outputs
    comparison: Optional[Comparison]

    # Synthesizer outputs
    final_report: Optional[str]
    insights_json: Optional[Dict[str, Any]]

    # Metadata
    timestamp: str
    processing_stage: str  # "researching", "analyzing", "synthesizing", "complete"
