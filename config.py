"""
Configuration settings for the autonomous research system.

Uses Pydantic for validation and environment variable loading.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings"""

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent
    MEMORY_DIR: Path = PROJECT_ROOT / "memory"
    PDF_DIR: Path = MEMORY_DIR / "pdfs"
    FAISS_DIR: Path = MEMORY_DIR / "faiss"
    OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # Ollama LLM settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1"
    LLM_TEMPERATURE: float = 0.7

    # Research settings
    MAX_PAPERS: int = 7
    CHUNK_SIZE: int = 700  # Sweet spot for Llama 3
    CHUNK_OVERLAP: int = 250  # ~36% overlap - ensures sentences aren't split
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # arXiv settings
    ARXIV_TIMEOUT: int = 30  # seconds
    ARXIV_RATE_LIMIT_DELAY: float = 3.0  # seconds between requests

    # Backward compatibility alias
    @property
    def OLLAMA_TEMPERATURE(self) -> float:
        """Alias for LLM_TEMPERATURE for backward compatibility"""
        return self.LLM_TEMPERATURE

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_job_output_dir(self, job_id: str) -> Path:
        """
        Get output directory for a specific job

        Args:
            job_id: Unique job identifier

        Returns:
            Path to job-specific output directory
        """
        path = self.OUTPUT_DIR / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_job_pdf_dir(self, job_id: str) -> Path:
        """
        Get PDF directory for a specific job

        Args:
            job_id: Unique job identifier

        Returns:
            Path to job-specific PDF directory
        """
        path = self.PDF_DIR / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_job_faiss_dir(self, job_id: str) -> Path:
        """
        Get FAISS index directory for a specific job

        Args:
            job_id: Unique job identifier

        Returns:
            Path to job-specific FAISS directory
        """
        path = self.FAISS_DIR / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global settings instance
settings = Settings()
