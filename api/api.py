"""
FastAPI application for Autonomous Research Agent

This module provides REST API endpoints for asynchronous research job management.
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uuid
from loguru import logger

from api.schemas import (
    ResearchJobRequest,
    ResearchJobResponse,
    JobStatusResponse,
    JobResultsResponse,
    JobListResponse,
    ErrorResponse,
    HealthResponse
)
from api.job_store import JobStore
from api.research_worker import run_research_job
from config import Settings

# Initialize FastAPI app
app = FastAPI(
    title="Autonomous Research Agent API",
    description="REST API for asynchronous research paper analysis using LangGraph + Ollama",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit default
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize job store and settings
job_store = JobStore(db_path="jobs.db")
settings = Settings()


def check_ollama_health() -> bool:
    """
    Check if Ollama is accessible

    Returns:
        True if Ollama is reachable, False otherwise
    """
    try:
        import requests as req
        # Quick health check - just verify Ollama API is responding
        response = req.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"Ollama health check failed: {e}")
        return False


@app.post("/research", response_model=ResearchJobResponse, status_code=202)
async def create_research_job(
    request: ResearchJobRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a new research job and execute it in the background

    Args:
        request: Research job parameters (topic, max_papers)
        background_tasks: FastAPI background tasks manager

    Returns:
        ResearchJobResponse with job_id for status tracking

    Raises:
        HTTPException: If Ollama is unavailable or job creation fails
    """
    # Check Ollama availability before accepting job
    if not check_ollama_health():
        raise HTTPException(
            status_code=503,
            detail="Ollama service unavailable. Please ensure Ollama is running at http://localhost:11434"
        )

    # Generate unique job ID
    job_id = str(uuid.uuid4())
    logger.info(f"Creating research job {job_id}: {request.topic} ({request.max_papers} papers)")

    try:
        # Create job-specific output directory
        job_output_dir = settings.get_job_output_dir(job_id)
        logger.debug(f"Created job directory: {job_output_dir}")

        # Initialize job in database
        job_store.create_job(
            job_id=job_id,
            topic=request.topic,
            max_papers=request.max_papers
        )

        # Schedule background task to run LangGraph workflow
        background_tasks.add_task(
            run_research_job,
            job_id=job_id,
            topic=request.topic,
            max_papers=request.max_papers,
            job_store=job_store
        )

        return ResearchJobResponse(
            job_id=job_id,
            status="queued",
            topic=request.topic,
            max_papers=request.max_papers,
            created_at=datetime.now(),
            message="Research job queued successfully. Use /status/{job_id} to track progress."
        )

    except Exception as e:
        logger.error(f"Failed to create job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get current status and progress of a research job

    This endpoint is designed for polling from the frontend every 2-3 seconds.

    Args:
        job_id: Unique job identifier

    Returns:
        JobStatusResponse with current status, progress, and messages

    Raises:
        HTTPException: If job not found
    """
    job = job_store.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        processing_stage=job["processing_stage"],
        topic=job["topic"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        progress_percentage=job["progress_percentage"],
        current_message=job.get("current_message"),
        error=job.get("error")
    )


@app.get("/results/{job_id}", response_model=JobResultsResponse)
async def get_job_results(job_id: str):
    """
    Retrieve final results of a completed research job

    Only call this endpoint after /status indicates job is complete.

    Args:
        job_id: Unique job identifier

    Returns:
        JobResultsResponse with report, insights, and metadata

    Raises:
        HTTPException: If job not found, not complete, or results missing
    """
    job = job_store.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )

    if job["status"] != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not complete yet. Current status: {job['status']}"
        )

    if not job.get("final_state"):
        raise HTTPException(
            status_code=500,
            detail="Job marked as complete but results are missing"
        )

    final_state = job["final_state"]

    # Count papers
    papers_analyzed = len([
        d for d in final_state.get("documents", [])
        if d.get("extraction_status") == "success"
    ])
    papers_failed = len([
        d for d in final_state.get("documents", [])
        if d.get("extraction_status") != "success"
    ])

    return JobResultsResponse(
        job_id=job_id,
        status="complete",
        topic=job["topic"],
        final_report=final_state.get("final_report", ""),
        insights_json=final_state.get("insights_json", {}),
        papers_analyzed=papers_analyzed,
        papers_failed=papers_failed,
        created_at=job["created_at"],
        completed_at=job["updated_at"]
    )


@app.get("/jobs", response_model=JobListResponse)
async def list_all_jobs():
    """
    List all research jobs with summaries

    Returns lightweight job summaries for the history tab, including:
    - Job metadata (id, topic, status, timestamps)
    - Paper counts for completed jobs
    - Ordered by creation time (newest first)

    Returns:
        JobListResponse with list of job summaries
    """
    summaries = job_store.get_job_summaries()

    return JobListResponse(
        jobs=summaries,
        total_count=len(summaries)
    )


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job from the database

    Useful for cleanup after results have been downloaded.

    Args:
        job_id: Unique job identifier

    Returns:
        Success message

    Raises:
        HTTPException: If job not found
    """
    success = job_store.delete_job(job_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )

    logger.info(f"Deleted job {job_id} via API request")

    return {
        "message": f"Job {job_id} deleted successfully",
        "job_id": job_id
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns API status, active job count, and Ollama connectivity.

    Returns:
        HealthResponse with system health information
    """
    active_jobs = job_store.get_active_jobs_count()
    ollama_connected = check_ollama_health()

    return HealthResponse(
        status="healthy" if ollama_connected else "degraded",
        active_jobs=active_jobs,
        timestamp=datetime.now(),
        ollama_connected=ollama_connected
    )


@app.get("/")
async def root():
    """
    API root endpoint

    Returns basic information and links to documentation.
    """
    return {
        "name": "Autonomous Research Agent API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "create_job": "POST /research",
            "get_status": "GET /status/{job_id}",
            "get_results": "GET /results/{job_id}",
            "list_jobs": "GET /jobs",
            "delete_job": "DELETE /jobs/{job_id}",
            "health": "GET /health"
        }
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Actions to perform on API startup
    """
    logger.info("Starting Autonomous Research Agent API")
    logger.info(f"Ollama URL: {settings.OLLAMA_BASE_URL}")
    logger.info(f"Ollama Model: {settings.OLLAMA_MODEL}")

    # Check Ollama on startup
    if check_ollama_health():
        logger.info("Ollama health check passed")
    else:
        logger.warning("Ollama health check failed - service may be unavailable")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup actions on API shutdown
    """
    logger.info("Shutting down Autonomous Research Agent API")
