"""
Pydantic schemas for FastAPI request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class ResearchJobRequest(BaseModel):
    """Request schema for creating a research job"""
    topic: str = Field(..., min_length=3, max_length=200, description="Research topic to investigate")
    max_papers: int = Field(default=5, ge=1, le=10, description="Number of papers to analyze")

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "quantum computing error correction",
                "max_papers": 5
            }
        }


class ResearchJobResponse(BaseModel):
    """Response schema when a job is created"""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status (queued)")
    topic: str = Field(..., description="Research topic")
    max_papers: int = Field(..., description="Number of papers to analyze")
    created_at: datetime = Field(..., description="Job creation timestamp")
    message: str = Field(default="Research job queued successfully", description="Status message")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "queued",
                "topic": "quantum computing error correction",
                "max_papers": 5,
                "created_at": "2026-01-03T10:30:00",
                "message": "Research job queued successfully"
            }
        }


class JobStatusResponse(BaseModel):
    """Response schema for job status checks"""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status (queued, researching, analyzing, synthesizing, complete, failed)")
    processing_stage: str = Field(..., description="Current processing stage")
    topic: str = Field(..., description="Research topic")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    progress_percentage: int = Field(..., ge=0, le=100, description="Progress percentage (0-100)")
    current_message: Optional[str] = Field(None, description="Current processing message")
    error: Optional[str] = Field(None, description="Error message if status is failed")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "analyzing",
                "processing_stage": "analyzing",
                "topic": "quantum computing error correction",
                "created_at": "2026-01-03T10:30:00",
                "updated_at": "2026-01-03T10:35:00",
                "progress_percentage": 40,
                "current_message": "Detecting patterns and contradictions...",
                "error": None
            }
        }


class JobResultsResponse(BaseModel):
    """Response schema for completed job results"""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status (should be 'complete')")
    topic: str = Field(..., description="Research topic")
    final_report: str = Field(..., description="Final research report in markdown format")
    insights_json: Dict[str, Any] = Field(..., description="Structured insights in JSON format")
    papers_analyzed: int = Field(..., description="Number of papers successfully analyzed")
    papers_failed: int = Field(..., description="Number of papers that failed to process")
    created_at: datetime = Field(..., description="Job creation timestamp")
    completed_at: datetime = Field(..., description="Job completion timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "complete",
                "topic": "quantum computing error correction",
                "final_report": "# Research Report\n\n...",
                "insights_json": {
                    "key_findings": [],
                    "contradictions": [],
                    "trends": []
                },
                "papers_analyzed": 5,
                "papers_failed": 0,
                "created_at": "2026-01-03T10:30:00",
                "completed_at": "2026-01-03T10:45:00"
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    detail: str = Field(..., description="Error description")
    job_id: Optional[str] = Field(None, description="Job ID if applicable")

    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Job not found",
                "job_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema"""
    status: str = Field(..., description="API health status")
    active_jobs: int = Field(..., description="Number of currently active jobs")
    timestamp: datetime = Field(..., description="Health check timestamp")
    ollama_connected: bool = Field(..., description="Whether Ollama is accessible")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "active_jobs": 2,
                "timestamp": "2026-01-03T10:30:00",
                "ollama_connected": True
            }
        }


class JobSummary(BaseModel):
    """Summary of a job for history listing"""
    job_id: str = Field(..., description="Unique job identifier")
    topic: str = Field(..., description="Research topic")
    status: str = Field(..., description="Job status")
    processing_stage: str = Field(..., description="Current processing stage")
    progress_percentage: int = Field(..., ge=0, le=100, description="Progress percentage")
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    papers_analyzed: Optional[int] = Field(None, description="Number of papers analyzed (if complete)")
    papers_failed: Optional[int] = Field(None, description="Number of papers failed (if complete)")
    error: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "topic": "quantum computing error correction",
                "status": "complete",
                "processing_stage": "complete",
                "progress_percentage": 100,
                "created_at": "2026-01-03T10:30:00",
                "updated_at": "2026-01-03T10:45:00",
                "papers_analyzed": 5,
                "papers_failed": 0,
                "error": None
            }
        }


class JobListResponse(BaseModel):
    """Response schema for listing all jobs"""
    jobs: list[JobSummary] = Field(..., description="List of job summaries")
    total_count: int = Field(..., description="Total number of jobs")

    class Config:
        json_schema_extra = {
            "example": {
                "jobs": [
                    {
                        "job_id": "550e8400-e29b-41d4-a716-446655440000",
                        "topic": "quantum computing",
                        "status": "complete",
                        "processing_stage": "complete",
                        "progress_percentage": 100,
                        "created_at": "2026-01-03T10:30:00",
                        "updated_at": "2026-01-03T10:45:00",
                        "papers_analyzed": 5,
                        "papers_failed": 0,
                        "error": None
                    }
                ],
                "total_count": 1
            }
        }
