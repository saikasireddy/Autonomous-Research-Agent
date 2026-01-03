"""
FAISS vector store for research paper embeddings.

Uses RecursiveCharacterTextSplitter for arXiv PDF chunking.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import pickle
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger

from config import settings


class FAISSVectorStore:
    """
    Persistent FAISS vector store with metadata management.

    Uses local embeddings (no API calls) and recursive text chunking
    optimized for arXiv PDFs.
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize vector store.

        Args:
            base_path: Base path for storage (defaults to settings.MEMORY_DIR)
        """
        self.base_path = Path(base_path) if base_path else settings.MEMORY_DIR
        self.index_dir = self.base_path / "faiss"
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Local embeddings (no API calls)
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL)

        self.index: Optional[faiss.IndexFlatL2] = None
        self.metadata: List[Dict] = []  # Maps index position to metadata

        # Text splitter for arXiv PDFs
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,  # 700 tokens
            chunk_overlap=settings.CHUNK_OVERLAP,  # 100 tokens (~14%)
            separators=["\n\n", "\n", ". ", " ", ""],  # Recursive fallback
            length_function=len  # Character-based (can use tiktoken for precise tokens)
        )

    def build_index(self, documents: List[Dict], job_id: Optional[str] = None) -> str:
        """
        Build FAISS index from research papers.

        Args:
            documents: List of dicts with keys:
                - text: str (full paper text)
                - arxiv_id: str
                - title: str
                - citation: str
                - extraction_status: str
            job_id: Optional job identifier for isolated storage

        Returns:
            Path to saved FAISS index
        """
        logger.info(f"Building FAISS index from {len(documents)} documents")
        if job_id:
            logger.info(f"Using job-specific directory for job_id: {job_id}")

        chunks_with_metadata = []

        # Chunk each successful document
        for doc in documents:
            if doc.get("extraction_status") != "success":
                logger.warning(f"Skipping {doc.get('arxiv_id')} - extraction failed")
                continue

            text = doc.get("text", "")
            if len(text) < 100:
                logger.warning(f"Skipping {doc.get('arxiv_id')} - text too short")
                continue

            # Chunk with recursive splitter
            chunks = self._chunk_text(
                text=text,
                arxiv_id=doc["arxiv_id"],
                title=doc.get("title", "Unknown"),
                citation=doc["citation"]
            )

            chunks_with_metadata.extend(chunks)
            logger.info(f"Created {len(chunks)} chunks for {doc['arxiv_id']}")

        if not chunks_with_metadata:
            raise ValueError("No valid chunks created - cannot build index")

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks_with_metadata)} chunks")
        texts = [c["text"] for c in chunks_with_metadata]
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        self.metadata = chunks_with_metadata

        # Persist to disk (job-specific if job_id provided)
        index_path = self._save_index(job_id=job_id)
        logger.info(f"Built FAISS index with {len(chunks_with_metadata)} chunks")

        return index_path

    def _chunk_text(
        self,
        text: str,
        arxiv_id: str,
        title: str,
        citation: str
    ) -> List[Dict]:
        """
        Chunk text using RecursiveCharacterTextSplitter.

        Critical for arXiv PDFs with multi-column layouts, formulas, bibliographies.

        Args:
            text: Full paper text
            arxiv_id: arXiv ID
            title: Paper title
            citation: Formatted citation

        Returns:
            List of chunk dicts with metadata
        """
        chunks = self.text_splitter.split_text(text)

        return [
            {
                "text": chunk,
                "arxiv_id": arxiv_id,
                "title": title,
                "citation": citation,
                "chunk_id": i
            }
            for i, chunk in enumerate(chunks)
        ]

    def _save_index(self, job_id: Optional[str] = None) -> str:
        """
        Save FAISS index and metadata to disk.

        Args:
            job_id: Optional job identifier for isolated storage

        Returns:
            Path to saved index file
        """
        # Use job-specific directory if job_id provided
        if job_id:
            save_dir = settings.get_job_faiss_dir(job_id)
        else:
            save_dir = self.index_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        index_path = save_dir / f"index_{timestamp}.faiss"
        metadata_path = save_dir / f"metadata_{timestamp}.pkl"

        # Save index
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Saved index to {index_path}")
        return str(index_path)

    def load_index(self, index_path: str):
        """
        Load existing FAISS index from disk.

        Args:
            index_path: Path to .faiss file
        """
        logger.info(f"Loading FAISS index from {index_path}")

        self.index = faiss.read_index(index_path)

        # Load corresponding metadata
        metadata_path = index_path.replace(".faiss", ".pkl").replace("index_", "metadata_")
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        logger.info(f"Loaded index with {len(self.metadata)} chunks")

    def similarity_search(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[str, Dict]]:
        """
        Search for similar chunks.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (text, metadata) tuples
        """
        if self.index is None:
            raise ValueError("Index not built or loaded")

        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True).astype('float32')

        # Search
        distances, indices = self.index.search(query_embedding, k)

        # Return results with metadata
        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append((meta["text"], meta))

        return results

    def get_all_chunks(self) -> List[Dict]:
        """
        Get all chunks with metadata.

        Returns:
            List of all chunk metadata dicts
        """
        return self.metadata
