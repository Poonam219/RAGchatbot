"""
embeddings.py — Sentence embedding using BAAI/bge-base-en-v1.5.
Handles batch encoding with progress tracking and caching.
"""
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Union, Tuple, Dict, Any
import numpy as np
from loguru import logger
import torch


class EmbeddingModel:
    """
    Wrapper around BAAI/bge-base-en-v1.5 for generating dense embeddings.

    This model excels at:
    - Semantic similarity for retrieval
    - Both query and passage encoding
    - Multi-domain text (technical docs, web content)
    """

    # BGE models require a query prefix for optimal retrieval performance
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model: {model_name} on {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()

        logger.info(f"Embedding model loaded. Dimension: {self.dimension}")

    def encode_documents(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode document/passage texts for indexing.
        Documents are encoded without query prefix.

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([])

        logger.info(f"Encoding {len(texts)} documents...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            convert_to_numpy=True,
        )

        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a user query with BGE query prefix.
        The prefix instructs the model to optimize for retrieval.

        Returns:
            numpy array of shape (dimension,)
        """
        prefixed_query = self.QUERY_PREFIX + query

        embedding = self.model.encode(
            [prefixed_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embedding[0].astype(np.float32)

    def encode_batch(
        self,
        texts: List[str],
        is_query: bool = False,
        batch_size: int = 32,
    ) -> np.ndarray:
        """General batch encoding with query/document mode."""
        if is_query:
            texts = [self.QUERY_PREFIX + t for t in texts]

        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        ).astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        return self.dimension


class CrossEncoderReranker:
    """
    Sentence Transformers Cross-Encoder for precise reranking.
    A Cross Encoder takes (query, passage) pairs and outputs a relevance score.
    It is much more accurate than Bi-Encoders but slower.
    """
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading reranker model: {model_name} on {self.device}")
        
        # Load model only if needed (heavy resource usage)
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank a list of chunks based on a query.
        Returns the top_k most relevant chunks.
        """
        if not chunks:
            return []
            
        passages = [c.get("text", "") for c in chunks]
        pairs = [[query, p] for p in passages]
        
        # Cross-encoder prediction (Batch mode)
        logger.info(f"Reranking {len(chunks)} candidates...")
        scores = self.model.predict(pairs)
        
        # Assemble results
        for idx, score in enumerate(scores):
            chunks[idx]["rerank_score"] = float(score)
            chunks[idx]["original_score"] = chunks[idx].get("score", 0)
            # Use rerank score as the primary score for final sorting
            chunks[idx]["score"] = float(score)
            chunks[idx]["retrieval_method"] = (
                f"{chunks[idx].get('retrieval_method', 'unknown')} + reranked"
            )
            
        # Sort by rerank score (descending)
        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]