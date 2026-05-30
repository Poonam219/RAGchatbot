from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from loguru import logger
from backend.core.rag_pipeline import RAGPipeline

router = APIRouter()

# Share the same core pipeline if we want combined usage
# rag_pipeline = RAGPipeline()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    source_type: Optional[str] = None # 'pdf' or 'web' or None

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str

@router.post("/search/hybrid", response_model=SearchResponse)
async def hybrid_search_api(request: SearchRequest):
    """Retrieve raw chunks using hybrid search (BM25 + Semantic)."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
    
    try:
        from backend.core.rag_pipeline import RAGPipeline
        p = RAGPipeline()
        logger.info(f"Search API: Received '{request.query}' for retrieval.")
        results = p.retriever.search(request.query, top_k=request.top_k, source_type=request.source_type)
        
        return SearchResponse(
            results=results,
            query=request.query
        )
    except Exception as e:
        logger.error(f"Search API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))