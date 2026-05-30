from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from loguru import logger
from backend.core.rag_pipeline import RAGPipeline

router = APIRouter()

# Share the core pipeline instance if needed, but for now we focus on manual scripts
# pipeline = RAGPipeline()

class WebIngestRequest(BaseModel):
    url: str
    depth: int = 2
    max_pages: int = 50

class ResetDBRequest(BaseModel):
    source_type: Optional[str] = None # 'pdf' or 'web' or None for all

@router.get("/ingest/status")
async def get_ingestion_status():
    """Return brief stats about currently ingested content."""
    # Note: We need access to the vector store stats
    try:
        from backend.core.rag_pipeline import RAGPipeline
        p = RAGPipeline()
        stats = p.vector_store.get_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error(f"Ingest Status API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Placeholder for async ingestion tasks via Celery or BackgroundTasks
# @router.post("/ingest/website")
# async def ingest_website_via_api(request: WebIngestRequest):
#     ...