"""
vector_store.py — Dual vector storage: FAISS (fast) + ChromaDB (persistent).
Manages separate indexes for PDF and Web content.
"""
import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import uuid


class FAISSVectorStore:
    """
    FAISS-based vector store with metadata management.
    Uses separate indexes for PDF and Web content.
    Persists to disk as .index + .meta files.
    """

    def __init__(self, db_path: str, dimension: int = 768):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension

        # Separate indexes per source type
        self.indexes: Dict[str, faiss.IndexFlatIP] = {}
        self.metadata: Dict[str, List[Dict]] = {}

        # Load existing indexes
        self._load_indexes()

    def _get_index_path(self, source_type: str) -> Tuple[Path, Path]:
        """Get paths for FAISS index and metadata files."""
        idx_path = self.db_path / f"{source_type}.index"
        meta_path = self.db_path / f"{source_type}.meta"
        return idx_path, meta_path

    def _load_indexes(self):
        """Load all existing FAISS indexes from disk."""
        for source_type in ["pdf", "web", "all"]:
            idx_path, meta_path = self._get_index_path(source_type)
            if idx_path.exists() and meta_path.exists():
                self.indexes[source_type] = faiss.read_index(str(idx_path))
                with open(meta_path, "rb") as f:
                    self.metadata[source_type] = pickle.load(f)
                logger.info(
                    f"Loaded {source_type} index: "
                    f"{self.indexes[source_type].ntotal} vectors"
                )

    def _save_index(self, source_type: str):
        """Save FAISS index and metadata to disk."""
        if source_type not in self.indexes:
            return
        idx_path, meta_path = self._get_index_path(source_type)
        faiss.write_index(self.indexes[source_type], str(idx_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata[source_type], f)

    def _get_or_create_index(self, source_type: str) -> faiss.IndexFlatIP:
        """Get existing index or create new one."""
        if source_type not in self.indexes:
            # Inner product index (works with L2-normalized vectors as cosine sim)
            self.indexes[source_type] = faiss.IndexFlatIP(self.dimension)
            self.metadata[source_type] = []
        return self.indexes[source_type]

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Add text chunks and their embeddings to the vector store.

        Args:
            chunks: List of chunk dicts with metadata
            embeddings: numpy array of shape (len(chunks), dimension)
        """
        if len(chunks) == 0:
            return

        # Group by source type
        groups: Dict[str, List[Tuple[int, Dict]]] = {}
        for i, chunk in enumerate(chunks):
            src_type = chunk.get("type", "all")
            # Normalize: pdf_table -> pdf
            if src_type.startswith("pdf"):
                src_type = "pdf"
            elif src_type == "web":
                src_type = "web"
            else:
                src_type = "all"

            if src_type not in groups:
                groups[src_type] = []
            groups[src_type].append((i, chunk))

        # Add to each index
        for src_type, items in groups.items():
            indices = [i for i, _ in items]
            chunk_metas = [c for _, c in items]
            vecs = embeddings[indices].astype(np.float32)

            index = self._get_or_create_index(src_type)
            index.add(vecs)
            self.metadata[src_type].extend(chunk_metas)
            self._save_index(src_type)

        # Also add to "all" combined index
        all_index = self._get_or_create_index("all")
        all_index.add(embeddings.astype(np.float32))
        self.metadata["all"].extend(chunks)
        self._save_index("all")

        logger.info(f"Added {len(chunks)} chunks to vector store")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        source_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using cosine similarity.

        Args:
            query_embedding: 1D numpy array
            top_k: Number of results
            source_type: Filter by 'pdf', 'web', or None (search all)

        Returns:
            List of chunk dicts with 'score' added
        """
        index_key = source_type or "all"
        if index_key not in self.indexes or self.indexes[index_key].ntotal == 0:
            logger.warning(f"No vectors in index: {index_key}")
            return []

        index = self.indexes[index_key]
        meta = self.metadata[index_key]

        query_vec = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(top_k, index.ntotal)

        scores, indices = index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            chunk = dict(meta[idx])
            chunk["score"] = float(score)
            chunk["retrieval_method"] = "semantic"
            results.append(chunk)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about stored vectors."""
        return {
            index_key: {
                "total_vectors": self.indexes[index_key].ntotal,
                "metadata_count": len(self.metadata[index_key]),
            }
            for index_key in self.indexes
        }

    def clear(self, source_type: Optional[str] = None):
        """Clear the vector store (optionally by source type)."""
        keys = [source_type] if source_type else list(self.indexes.keys())
        for key in keys:
            self.indexes.pop(key, None)
            self.metadata.pop(key, None)
            idx_path, meta_path = self._get_index_path(key)
            idx_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        logger.info(f"Cleared vector store: {keys}")