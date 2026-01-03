"""
Researcher Agent - Fetches papers and builds FAISS index.

Responsibilities:
- Search arXiv for papers
- Download PDFs (with caching)
- Extract text
- Build FAISS vector index
"""

from pathlib import Path
from typing import Dict
from loguru import logger

from graph.state import ResearchState
from mcp.arxiv_tool import ArxivMCPTool
from memory.pdf_extractor import extract_text_from_pdf, format_citation
from memory.vector_store import FAISSVectorStore
from config import settings


def researcher_node(state: ResearchState) -> Dict:
    """
    Researcher agent node for LangGraph.

    Args:
        state: Current research state

    Returns:
        State updates (documents, faiss_index_path, error_log)
    """
    job_id = state["job_id"]
    logger.info(f"═══ Researcher Agent Starting ═══")
    logger.info(f"Job ID: {job_id}")
    logger.info(f"Topic: '{state['topic']}'")
    logger.info(f"Target papers: {state['max_papers']}")

    # Get job-specific PDF directory
    pdf_dir = settings.get_job_pdf_dir(job_id)
    logger.info(f"PDF directory: {pdf_dir}")

    # Initialize tools
    arxiv_tool = ArxivMCPTool()
    vector_store = FAISSVectorStore()

    # Search arXiv
    logger.info(f"Searching arXiv...")
    papers = arxiv_tool.search(
        query=state["topic"],
        max_results=state["max_papers"]
    )

    if not papers:
        logger.error("❌ No papers found on arXiv")
        return {
            "documents": [],
            "error_log": [{"error": "No papers found", "stage": "arxiv_search"}],
            "processing_stage": "failed"
        }

    logger.info(f"✓ Found {len(papers)} papers on arXiv")

    # Process each paper
    documents = []
    error_log = []

    for idx, paper_meta in enumerate(papers, 1):
        arxiv_id = paper_meta["arxiv_id"]
        logger.info(f"\n[{idx}/{len(papers)}] Processing {arxiv_id}")

        try:
            # Check if PDF already exists (caching) in job-specific directory
            pdf_path = pdf_dir / f"{arxiv_id}.pdf"

            if pdf_path.exists():
                logger.info(f"  ✓ PDF already cached at {pdf_path}")
            else:
                logger.info(f"  ⬇ Downloading PDF...")
                success = arxiv_tool.download_pdf(paper_meta["pdf_url"], pdf_path)
                if not success:
                    raise ValueError("PDF download failed")
                logger.info(f"  ✓ PDF downloaded")

            # Extract text
            logger.info(f"  📄 Extracting text from PDF...")
            text = extract_text_from_pdf(pdf_path)
            logger.info(f"  ✓ Extracted {len(text)} characters")

            # Format citation
            citation = format_citation(
                authors=paper_meta["authors"],
                year=paper_meta["published"].year,
                arxiv_id=arxiv_id
            )

            # Add successful document with ResearchPaper structure
            documents.append({
                "arxiv_id": arxiv_id,
                "title": paper_meta["title"],
                "authors": paper_meta["authors"],
                "year": paper_meta["published"].year,
                "summary": paper_meta["summary"],
                "pdf_path": str(pdf_path),
                "citation": citation,
                "extraction_status": "success",
                "text": text  # For FAISS indexing
            })

            logger.info(f"  ✅ Successfully processed {arxiv_id}")

        except Exception as e:
            logger.error(f"  ❌ Failed to process {arxiv_id}: {e}")

            # Track error
            error_log.append({
                "arxiv_id": arxiv_id,
                "error": str(e),
                "stage": "pdf_extraction"
            })

            # Add minimal metadata for failed paper
            documents.append({
                "arxiv_id": arxiv_id,
                "title": paper_meta.get("title", "Unknown"),
                "authors": paper_meta.get("authors", []),
                "year": paper_meta.get("published", None).year if paper_meta.get("published") else 0,
                "summary": paper_meta.get("summary", ""),
                "pdf_path": None,
                "citation": f"[{arxiv_id}]",
                "extraction_status": f"failed: {str(e)[:50]}"
            })

    # Build FAISS index from successful papers
    logger.info(f"\n{'='*60}")
    faiss_index_path = None
    successful_docs = [d for d in documents if d["extraction_status"] == "success"]

    if successful_docs:
        try:
            logger.info(f"Building FAISS index from {len(successful_docs)} successful papers...")
            # Pass job_id to save index in job-specific directory
            faiss_index_path = vector_store.build_index(successful_docs, job_id=job_id)
            logger.info(f"✅ FAISS index built successfully")
            logger.info(f"   Index saved to: {faiss_index_path}")
        except Exception as e:
            logger.error(f"❌ Failed to build FAISS index: {e}")
            error_log.append({
                "error": str(e),
                "stage": "faiss_building"
            })
    else:
        logger.warning("⚠️  No successful papers to index")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Researcher Agent Summary:")
    logger.info(f"  Total papers: {len(documents)}")
    logger.info(f"  Successful: {len(successful_docs)}")
    logger.info(f"  Failed: {len(documents) - len(successful_docs)}")
    logger.info(f"  FAISS index: {'✓ Built' if faiss_index_path else '✗ Not built'}")
    logger.info(f"{'='*60}\n")

    return {
        "documents": documents,
        "faiss_index_path": faiss_index_path,
        "error_log": error_log,
        "processing_stage": "analyzing"
    }
