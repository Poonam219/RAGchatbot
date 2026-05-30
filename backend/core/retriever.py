import os
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from backend.config import settings
from backend.core.embeddings import CrossEncoderReranker

class HybridRetriever:
    """
    Combines FAISS semantic search and BM25 text-based search.
    Provides alpha-weighted scoring for precision and recall.
    """
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25: Optional[BM25Okapi] = None
        self.metadata: List[Dict] = []
        self.corpus: List[str] = []
        self.alpha = settings.semantic_weight  # Default 0.7 for semantic, 0.3 for BM25

        # Initialize Cross-Encoder Reranker if enabled
        self.reranker = None
        if settings.enable_rerank:
            self.reranker = CrossEncoderReranker(model_name=settings.rerank_model)

        # Initialize BM25 from existing metadata if possible
        self._initialize_bm25()

    def _initialize_bm25(self):
        """Loads all text content from vector store metadata to build BM25 index."""
        try:
            # Note: We take metadata from the vector store's "all" index
            if "all" in self.vector_store.metadata:
                self.metadata = self.vector_store.metadata["all"]
                self.corpus = [m.get("text", "") for m in self.metadata]
                
                if self.corpus:
                    tokenized_corpus = [doc.split() for doc in self.corpus]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                    logger.info(f"HybridRetriever: BM25 initialized with {len(self.corpus)} docs.")
                else:
                    logger.warning("HybridRetriever: Empty corpus during BM25 initialization.")
        except Exception as e:
            logger.error(f"HybridRetriever: BM25 Init Error: {e}")

    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        source_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search (BM25 + Semantic)."""
        logger.info(f"HybridRetriever: Searching for '{query}' (alpha={self.alpha})")
        
        # 1. Semantic Search
        query_embedding = self.embedder.encode_query(query)
        semantic_results = self.vector_store.search(query_embedding, top_k=top_k * 2, source_type=source_type)
        
        # 2. BM25 Search
        bm25_results = []
        if self.bm25:
            tokenized_query = query.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Normalize BM25 scores to [0,1] for hybrid scoring
            if len(bm25_scores) > 0:
                max_score = np.max(bm25_scores) if np.max(bm25_scores) > 0 else 1.0
                normalized_scores = bm25_scores / max_score
                
                # Use source_type filter if provided
                for idx, score in enumerate(normalized_scores):
                    meta = self.metadata[idx]
                    if source_type and meta.get("type", "all") != source_type:
                        continue
                    if score > 0.05: # Simple threshold
                        res = dict(meta)
                        res["score"] = float(score)
                        res["retrieval_method"] = "keyword"
                        bm25_results.append(res)
        
        # 3. Hybrid Reranking (Reciprocal Rank Fusion or Alpha Weighting)
        # For simplicity, we use weighted score sum
        # Normalize and merge
        combined = {}
        
        # Add semantic scores (weight = alpha)
        for res in semantic_results:
            text = res["text"]
            combined[text] = {
                "chunk": res,
                "score": res["score"] * self.alpha
            }
        
        # Add BM25 scores (weight = 1 - alpha)
        for res in bm25_results:
            text = res["text"]
            if text in combined:
                combined[text]["score"] += res["score"] * (1 - self.alpha)
                combined[text]["chunk"]["retrieval_method"] = "hybrid"
            else:
                combined[text] = {
                    "chunk": res,
                    "score": res["score"] * (1 - self.alpha)
                }

        # Sort combined results
        sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        initial_candidates = [r["chunk"] for r in sorted_results]
        
        # 4. Step 2: Reranking (Cross-Encoder)
        # If enabled, we take the top ~20 candidates and rerank them for extreme precision
        if self.reranker and initial_candidates:
            rerank_pool = initial_candidates[:20] # Take a broader pool
            logger.info(f"HybridRetriever: Reranking top {len(rerank_pool)} candidates.")
            return self.reranker.rerank(query, rerank_pool, top_k=top_k)

        # Fallback to top K from hybrid scoring
        return initial_candidates[:top_k]