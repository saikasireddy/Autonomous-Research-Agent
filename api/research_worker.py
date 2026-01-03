"""
Background worker for executing LangGraph research workflows

This module handles asynchronous execution of research jobs with progress tracking.
"""

from typing import Dict, Any
from datetime import datetime
from loguru import logger

from graph.research_graph import create_research_graph
from graph.state import ResearchState
from api.job_store import JobStore


def run_research_job(
    job_id: str,
    topic: str,
    max_papers: int,
    job_store: JobStore
) -> None:
    """
    Background task that executes the LangGraph research workflow

    This function:
    1. Initializes ResearchState with job_id
    2. Streams through the LangGraph workflow
    3. Updates job_store after each node completion
    4. Stores final results in database

    Args:
        job_id: Unique job identifier
        topic: Research topic
        max_papers: Number of papers to analyze
        job_store: JobStore instance for status updates
    """
    logger.info(f"Starting research job {job_id}: {topic} ({max_papers} papers)")

    try:
        # Initialize state with job_id
        initial_state: ResearchState = {
            "job_id": job_id,
            "topic": topic,
            "max_papers": max_papers,
            "documents": [],
            "faiss_index_path": None,
            "error_log": [],
            "analysis": None,
            "comparison": None,  # NEW: Comparator output
            "final_report": None,
            "insights_json": None,
            "timestamp": datetime.now().isoformat(),
            "processing_stage": "researching"
        }

        # Build LangGraph workflow
        graph = create_research_graph()

        # Update: Starting research phase
        job_store.update_job_status(
            job_id=job_id,
            status="researching",
            processing_stage="researching",
            progress_percentage=10,
            current_message="Fetching papers from arXiv and building vector store..."
        )

        # Stream through graph to capture intermediate state updates
        final_state = {**initial_state}  # Start with initial state

        for event in graph.stream(initial_state):
            logger.debug(f"Job {job_id} - Graph event: {list(event.keys())}")

            # Event format: {node_name: state_update}
            for node_name, state_update in event.items():
                current_stage = state_update.get("processing_stage", "unknown")
                logger.info(f"Job {job_id} - Node '{node_name}' completed, stage: {current_stage}")

                # CRITICAL: Merge state updates from EVERY node, not just the last one
                # This ensures documents from researcher, analysis from analyzer, etc. are all preserved
                final_state = {**final_state, **state_update}

                # Update job status based on processing stage
                if current_stage == "analyzing":
                    job_store.update_job_status(
                        job_id=job_id,
                        status="analyzing",
                        processing_stage="analyzing",
                        progress_percentage=40,
                        current_message="Analyzing papers: detecting patterns, contradictions, and trends..."
                    )

                elif current_stage == "comparing":
                    job_store.update_job_status(
                        job_id=job_id,
                        status="comparing",
                        processing_stage="comparing",
                        progress_percentage=60,
                        current_message="Comparing metrics: extracting quantitative data across papers..."
                    )

                elif current_stage == "synthesizing":
                    job_store.update_job_status(
                        job_id=job_id,
                        status="synthesizing",
                        processing_stage="synthesizing",
                        progress_percentage=80,
                        current_message="Synthesizing insights and generating final report..."
                    )

        # Validate completion
        if final_state and final_state.get("processing_stage") == "complete":
            logger.info(f"Job {job_id} completed successfully")

            # Clean up final_state before storing to reduce database size
            # Remove "text" field from documents (only needed during FAISS indexing, not storage)
            if "documents" in final_state:
                cleaned_documents = []
                for doc in final_state["documents"]:
                    clean_doc = {k: v for k, v in doc.items() if k != "text"}
                    cleaned_documents.append(clean_doc)
                final_state["documents"] = cleaned_documents

            # Update job store with final results
            job_store.update_job_status(
                job_id=job_id,
                status="complete",
                processing_stage="complete",
                progress_percentage=100,
                current_message="Research complete! Report and insights ready.",
                final_state=final_state
            )

        else:
            # Graph execution completed but didn't reach 'complete' stage
            error_msg = f"Graph execution did not reach 'complete' stage (final stage: {final_state.get('processing_stage') if final_state else 'unknown'})"
            logger.error(f"Job {job_id} - {error_msg}")

            job_store.update_job_status(
                job_id=job_id,
                error=error_msg,
                progress_percentage=0
            )

    except ValueError as e:
        # User input validation errors (e.g., no papers found for topic)
        logger.warning(f"Job {job_id} validation error: {e}")
        job_store.update_job_status(
            job_id=job_id,
            error=f"Validation error: {str(e)}",
            progress_percentage=0
        )

    except ConnectionError as e:
        # Network errors (arXiv API, Ollama)
        logger.error(f"Job {job_id} connection error: {e}")
        job_store.update_job_status(
            job_id=job_id,
            error=f"Connection failed: {str(e)}. Check arXiv/Ollama availability.",
            progress_percentage=0
        )

    except Exception as e:
        # Catch-all for unexpected errors
        logger.exception(f"Job {job_id} unexpected error: {e}")
        job_store.update_job_status(
            job_id=job_id,
            error=f"Unexpected error: {str(e)}",
            progress_percentage=0
        )
