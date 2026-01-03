"""
PDF text extraction utilities.

Handles extracting text from arXiv PDFs with error handling.
"""

from pathlib import Path
from typing import Optional

from pypdf import PdfReader
from loguru import logger


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from PDF using PyPDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text as string

    Raises:
        Exception: If PDF extraction fails
    """
    try:
        logger.info(f"Extracting text from {pdf_path}")

        reader = PdfReader(pdf_path)
        text = ""

        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
            except Exception as e:
                logger.warning(f"Failed to extract page {page_num} from {pdf_path}: {e}")
                continue

        # Clean up
        text = text.replace('\x00', '')  # Remove null bytes
        text = ' '.join(text.split())  # Normalize whitespace

        if len(text) < 100:
            raise ValueError(f"Extracted text too short ({len(text)} chars)")

        logger.info(f"Extracted {len(text)} characters from {pdf_path}")
        return text

    except Exception as e:
        logger.error(f"PDF extraction failed for {pdf_path}: {e}")
        raise


def format_citation(authors: list[str], year: int, arxiv_id: str) -> str:
    """
    Format citation string in simple inline format.

    Args:
        authors: List of author names
        year: Publication year
        arxiv_id: arXiv ID

    Returns:
        Formatted citation: [FirstAuthor et al., Year, arXiv:ID]
    """
    if not authors:
        return f"[Unknown, {year}, arXiv:{arxiv_id}]"

    # Extract last name from first author
    first_author = authors[0].split()[-1]  # Last name

    if len(authors) == 1:
        return f"[{first_author}, {year}, arXiv:{arxiv_id}]"
    else:
        return f"[{first_author} et al., {year}, arXiv:{arxiv_id}]"
