"""
MCP-compatible arXiv tool for searching and downloading research papers.

Follows Model Context Protocol philosophy:
- Self-contained with clear inputs/outputs
- Stateless operation
- Error handling built-in
- Observable/loggable
"""

from typing import List, Dict, Any
from pathlib import Path
import time

import arxiv
from arxiv import Client, Search, SortCriterion
import requests
from loguru import logger

from config import settings


class ArxivMCPTool:
    """
    MCP-compatible tool for interacting with arXiv.

    Provides stateless search and download operations.
    """

    def __init__(self):
        self.client = Client()
        self.name = "arxiv_search"
        self.description = "Search arXiv for academic papers"

    def search(
        self,
        query: str,
        max_results: int = 10,
        sort_by: SortCriterion = SortCriterion.Relevance
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers.

        Args:
            query: Search query string
            max_results: Maximum number of papers to return
            sort_by: Sort criterion (Relevance, LastUpdatedDate, SubmittedDate)

        Returns:
            List of paper metadata dicts with keys:
                - arxiv_id, title, authors, published, summary, pdf_url
        """
        try:
            logger.info(f"Searching arXiv for: '{query}' (max_results={max_results})")

            search = Search(
                query=query,
                max_results=max_results,
                sort_by=sort_by
            )

            results = []
            for paper in self.client.results(search):
                results.append({
                    "arxiv_id": paper.entry_id.split("/")[-1].split("v")[0],  # Remove version
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors],
                    "published": paper.published,
                    "summary": paper.summary,
                    "pdf_url": paper.pdf_url
                })

                # Respect rate limits
                time.sleep(settings.ARXIV_RATE_LIMIT_DELAY)

            logger.info(f"ArxivMCPTool: Found {len(results)} papers for query '{query}'")
            return results

        except Exception as e:
            logger.error(f"ArxivMCPTool search error: {e}")
            return []

    def download_pdf(self, pdf_url: str, save_path: Path) -> bool:
        """
        Download PDF from arXiv.

        Args:
            pdf_url: URL of the PDF to download
            save_path: Path where PDF should be saved

        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading PDF from {pdf_url}")

            response = requests.get(pdf_url, timeout=settings.ARXIV_TIMEOUT)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Downloaded PDF to {save_path}")
            return True

        except Exception as e:
            logger.error(f"PDF download failed for {pdf_url}: {e}")
            return False
