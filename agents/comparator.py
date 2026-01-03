"""
Comparator Agent - Extracts and compares quantitative metrics across papers.

Responsibilities:
- Extract structured metrics (energy density, ionic conductivity, cycle life, etc.)
- Build comparison table across all papers
- Identify which papers report specific metrics
- Generate comparison summary highlighting key differences
"""

from typing import Dict, List
from loguru import logger

from graph.state import ResearchState, Comparison
from memory.vector_store import FAISSVectorStore
from langchain_ollama import ChatOllama
from config import settings


# Metric extraction prompt
METRIC_EXTRACTION_PROMPT = """You are extracting quantitative metrics from a research paper excerpt.

Extract ANY numerical measurements, performance metrics, or quantitative results mentioned.

Common metric types (but not limited to):
- Energy density (Wh/kg, Wh/L)
- Ionic conductivity (S/cm, mS/cm)
- Cycle life (cycles, charge/discharge cycles)
- Capacity (mAh/g, Ah)
- Voltage (V)
- Temperature (°C, K)
- Efficiency (%)
- Power density (W/kg)
- Resolution (µm, nm)
- Accuracy (%)

Paper excerpt:
{text}

CRITICAL INSTRUCTIONS:
- Extract ONLY metrics that are explicitly stated with numbers and units
- Format: "Metric Name: Value Unit"
- If NO quantitative metrics found, respond: "NO_METRICS"
- Do NOT make up values
- Do NOT extract metrics from citations or references to other papers

Example outputs:
- Energy density: 475 Wh/kg
- Ionic conductivity: 1.2 mS/cm at 25°C
- Cycle life: 1000 cycles
- NO_METRICS

Extracted metrics (one per line):
"""


def comparator_node(state: ResearchState) -> Dict:
    """
    Comparator agent node for LangGraph.

    Args:
        state: Current research state

    Returns:
        State updates (comparison)
    """
    job_id = state["job_id"]
    logger.info(f"\n═══ Comparator Agent Starting ═══")
    logger.info(f"Job ID: {job_id}")

    if not state.get("faiss_index_path"):
        logger.error("❌ No FAISS index available - cannot compare")
        return {
            "comparison": {
                "metrics_table": [],
                "metric_names": [],
                "comparison_summary": "No metrics extracted - FAISS index unavailable"
            },
            "processing_stage": "synthesizing"
        }

    # Initialize tools
    vector_store = FAISSVectorStore()
    vector_store.load_index(state["faiss_index_path"])

    llm = ChatOllama(
        model=settings.OLLAMA_MODEL,
        temperature=0.3,  # Lower temperature for factual extraction
        base_url=settings.OLLAMA_BASE_URL
    )

    logger.info(f"Loaded FAISS index with {len(vector_store.metadata)} chunks")

    # Extract metrics from each paper
    logger.info(f"\n📊 Extracting metrics from papers...")
    paper_metrics = _extract_metrics_by_paper(vector_store, llm, state["documents"])

    # Build comparison table
    logger.info(f"\n📋 Building comparison table...")
    metrics_table, metric_names = _build_comparison_table(paper_metrics)

    # Generate comparison summary
    logger.info(f"\n📝 Generating comparison summary...")
    comparison_summary = _generate_comparison_summary(metrics_table, metric_names, llm)

    # Build comparison object
    comparison: Comparison = {
        "metrics_table": metrics_table,
        "metric_names": metric_names,
        "comparison_summary": comparison_summary
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Comparator Agent Summary:")
    logger.info(f"  Papers with metrics: {len(metrics_table)}")
    logger.info(f"  Unique metrics found: {len(metric_names)}")
    logger.info(f"  Metric types: {', '.join(metric_names[:5])}{'...' if len(metric_names) > 5 else ''}")
    logger.info(f"{'='*60}\n")

    return {
        "comparison": comparison,
        "processing_stage": "synthesizing"
    }


def _extract_metrics_by_paper(
    vector_store: FAISSVectorStore,
    llm: ChatOllama,
    documents: List[Dict]
) -> Dict[str, List[str]]:
    """
    Extract metrics for each paper.

    Returns:
        Dict mapping paper citation to list of metric strings
    """
    paper_metrics = {}

    # Only process successfully extracted papers
    successful_papers = [d for d in documents if d.get("extraction_status") == "success"]

    for paper in successful_papers:
        arxiv_id = paper["arxiv_id"]
        citation = paper["citation"]

        logger.info(f"  Processing {citation}...")

        # Search for results/measurements sections in this paper
        queries = [
            "results measurements performance",
            "experimental data observed measured",
            "achieved demonstrated reported"
        ]

        paper_chunks = []
        for query in queries:
            # Get chunks from this specific paper
            all_results = vector_store.similarity_search(query, k=50)

            # Filter to only this paper's chunks
            for text, metadata in all_results:
                if metadata["arxiv_id"] == arxiv_id:
                    paper_chunks.append((text, metadata))
                    if len(paper_chunks) >= 5:  # Max 5 chunks per paper
                        break

            if len(paper_chunks) >= 5:
                break

        # Extract metrics from chunks
        metrics = []
        for text, metadata in paper_chunks[:3]:  # Top 3 most relevant chunks
            prompt = METRIC_EXTRACTION_PROMPT.format(text=text[:600])

            try:
                response = llm.invoke(prompt)
                response_text = response.content.strip()

                # Skip if no metrics found
                if "NO_METRICS" in response_text:
                    continue

                # Parse metrics (one per line)
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                for line in lines:
                    # Skip header lines or non-metric lines
                    if ':' in line and not line.startswith('Extracted') and not line.startswith('Example'):
                        metrics.append(line)

            except Exception as e:
                logger.warning(f"Failed to extract metrics from {citation}: {e}")

        if metrics:
            # Deduplicate metrics
            unique_metrics = list(set(metrics))
            paper_metrics[citation] = unique_metrics
            logger.info(f"    ✓ Found {len(unique_metrics)} metrics")
        else:
            logger.info(f"    - No metrics found")

    return paper_metrics


def _build_comparison_table(paper_metrics: Dict[str, List[str]]) -> tuple[List[Dict], List[str]]:
    """
    Build a structured comparison table from extracted metrics.

    Returns:
        (metrics_table, metric_names)
    """
    if not paper_metrics:
        return [], []

    # Extract all unique metric names
    all_metric_names = set()
    for metrics in paper_metrics.values():
        for metric in metrics:
            # Extract metric name (before colon)
            if ':' in metric:
                metric_name = metric.split(':')[0].strip()
                all_metric_names.add(metric_name)

    metric_names = sorted(list(all_metric_names))

    # Build table rows
    metrics_table = []
    for citation, metrics in paper_metrics.items():
        row = {"paper": citation}

        # Parse metrics into dict
        for metric in metrics:
            if ':' in metric:
                name, value = metric.split(':', 1)
                row[name.strip()] = value.strip()

        metrics_table.append(row)

    return metrics_table, metric_names


def _generate_comparison_summary(
    metrics_table: List[Dict],
    metric_names: List[str],
    llm: ChatOllama
) -> str:
    """
    Generate a textual summary of the comparison.
    """
    if not metrics_table:
        return "No quantitative metrics were extracted from the papers."

    # Build summary prompt
    table_text = "\n".join([
        f"- {row['paper']}: {', '.join([f'{k}={v}' for k, v in row.items() if k != 'paper'])}"
        for row in metrics_table[:5]  # Limit to first 5 papers
    ])

    prompt = f"""Based on the following metrics extracted from research papers, write a 2-3 sentence summary highlighting key differences or similarities.

Metrics:
{table_text}

Summary (2-3 sentences):"""

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.warning(f"Failed to generate comparison summary: {e}")
        return f"Extracted {len(metric_names)} unique metric types across {len(metrics_table)} papers."
