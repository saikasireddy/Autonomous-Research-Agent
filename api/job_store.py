"""
SQLite-based job persistence for tracking research job status

This module provides thread-safe job storage that persists across API restarts.
"""

import sqlite3
import json
from typing import Dict, Optional, Any
from datetime import datetime
from threading import Lock
from pathlib import Path
from loguru import logger


class JobStore:
    """
    Thread-safe SQLite-based job tracking system

    Stores job state, progress, and results with persistence across restarts.
    """

    def __init__(self, db_path: str = "jobs.db"):
        """
        Initialize the job store

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._lock = Lock()
        self._init_database()
        logger.info(f"JobStore initialized with database: {self.db_path}")

    def _init_database(self) -> None:
        """Create the jobs table if it doesn't exist"""
        with self._lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    max_papers INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    processing_stage TEXT NOT NULL,
                    progress_percentage INTEGER NOT NULL DEFAULT 0,
                    current_message TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    final_state_json TEXT
                )
            """)

            conn.commit()
            conn.close()
            logger.debug("Database schema initialized")

    def create_job(self, job_id: str, topic: str, max_papers: int) -> None:
        """
        Create a new job entry

        Args:
            job_id: Unique job identifier
            topic: Research topic
            max_papers: Number of papers to analyze
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()

            now = datetime.now().isoformat()

            cursor.execute("""
                INSERT INTO jobs (
                    job_id, topic, max_papers, status, processing_stage,
                    progress_percentage, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id, topic, max_papers, "queued", "queued",
                0, now, now
            ))

            conn.commit()
            conn.close()
            logger.info(f"Created job {job_id}: {topic} ({max_papers} papers)")

    def update_job_status(
        self,
        job_id: str,
        status: Optional[str] = None,
        processing_stage: Optional[str] = None,
        progress_percentage: Optional[int] = None,
        current_message: Optional[str] = None,
        error: Optional[str] = None,
        final_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update job status and progress

        Args:
            job_id: Job identifier
            status: Job status (queued, researching, analyzing, synthesizing, complete, failed)
            processing_stage: Current processing stage
            progress_percentage: Progress (0-100)
            current_message: Current processing message
            error: Error message if failed
            final_state: Final ResearchState dict (will be JSON serialized)
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()

            # Build dynamic UPDATE query
            updates = []
            params = []

            if status is not None:
                updates.append("status = ?")
                params.append(status)

            if processing_stage is not None:
                updates.append("processing_stage = ?")
                params.append(processing_stage)

            if progress_percentage is not None:
                updates.append("progress_percentage = ?")
                params.append(progress_percentage)

            if current_message is not None:
                updates.append("current_message = ?")
                params.append(current_message)

            if error is not None:
                updates.append("error = ?")
                params.append(error)
                updates.append("status = ?")
                params.append("failed")

            if final_state is not None:
                # Serialize final_state to JSON string for SQLite storage
                updates.append("final_state_json = ?")
                params.append(json.dumps(final_state))

            # Always update timestamp
            updates.append("updated_at = ?")
            params.append(datetime.now().isoformat())

            # Add job_id for WHERE clause
            params.append(job_id)

            query = f"UPDATE jobs SET {', '.join(updates)} WHERE job_id = ?"

            cursor.execute(query, params)
            conn.commit()
            conn.close()

            logger.debug(f"Updated job {job_id}: {', '.join([u.split(' = ')[0] for u in updates])}")

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve job details

        Args:
            job_id: Job identifier

        Returns:
            Job dictionary or None if not found
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            # Convert Row to dict
            job = dict(row)

            # Parse timestamps
            job["created_at"] = datetime.fromisoformat(job["created_at"])
            job["updated_at"] = datetime.fromisoformat(job["updated_at"])

            # Deserialize final_state_json if present
            if job.get("final_state_json"):
                try:
                    job["final_state"] = json.loads(job["final_state_json"])
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to deserialize final_state for job {job_id}: {e}")
                    job["final_state"] = None
            else:
                job["final_state"] = None

            # Remove the raw JSON field (we have final_state now)
            job.pop("final_state_json", None)

            return job

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job from the database

        Args:
            job_id: Job identifier

        Returns:
            True if job was deleted, False if not found
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            deleted = cursor.rowcount > 0

            conn.commit()
            conn.close()

            if deleted:
                logger.info(f"Deleted job {job_id}")

            return deleted

    def get_all_jobs(self) -> list[Dict[str, Any]]:
        """
        Get all jobs (for debugging/monitoring)

        Returns:
            List of job dictionaries
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
            rows = cursor.fetchall()
            conn.close()

            jobs = []
            for row in rows:
                job = dict(row)
                job["created_at"] = datetime.fromisoformat(job["created_at"])
                job["updated_at"] = datetime.fromisoformat(job["updated_at"])

                # We don't deserialize final_state for bulk queries (performance)
                job.pop("final_state_json", None)
                job["final_state"] = None

                jobs.append(job)

            return jobs

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """
        Delete jobs older than max_age_hours

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of jobs deleted
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()

            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
            cutoff_iso = datetime.fromtimestamp(cutoff_time).isoformat()

            cursor.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff_iso,))
            deleted_count = cursor.rowcount

            conn.commit()
            conn.close()

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old jobs (>{max_age_hours}h)")

            return deleted_count

    def get_active_jobs_count(self) -> int:
        """
        Get count of active (non-complete, non-failed) jobs

        Returns:
            Count of active jobs
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM jobs
                WHERE status NOT IN ('complete', 'failed')
            """)
            count = cursor.fetchone()[0]
            conn.close()

            return count

    def get_job_summaries(self) -> list[Dict[str, Any]]:
        """
        Get lightweight summaries of all jobs for history listing

        Returns job metadata with papers_analyzed/papers_failed counts
        for completed jobs (extracted from final_state_json).

        Returns:
            List of job summary dictionaries
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    job_id, topic, status, processing_stage,
                    progress_percentage, created_at, updated_at,
                    error, final_state_json
                FROM jobs
                ORDER BY created_at DESC
            """)
            rows = cursor.fetchall()
            conn.close()

            summaries = []
            for row in rows:
                summary = {
                    "job_id": row["job_id"],
                    "topic": row["topic"],
                    "status": row["status"],
                    "processing_stage": row["processing_stage"],
                    "progress_percentage": row["progress_percentage"],
                    "created_at": datetime.fromisoformat(row["created_at"]),
                    "updated_at": datetime.fromisoformat(row["updated_at"]),
                    "error": row["error"],
                    "papers_analyzed": None,
                    "papers_failed": None
                }

                # For complete jobs, extract paper counts from final_state
                if row["status"] == "complete" and row["final_state_json"]:
                    try:
                        final_state = json.loads(row["final_state_json"])
                        documents = final_state.get("documents", [])

                        papers_analyzed = len([
                            d for d in documents
                            if d.get("extraction_status") == "success"
                        ])
                        papers_failed = len([
                            d for d in documents
                            if d.get("extraction_status") != "success"
                        ])

                        summary["papers_analyzed"] = papers_analyzed
                        summary["papers_failed"] = papers_failed

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse final_state for job {row['job_id']}: {e}")

                summaries.append(summary)

            return summaries
