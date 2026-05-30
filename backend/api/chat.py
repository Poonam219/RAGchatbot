from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from loguru import logger
from backend.core.rag_pipeline import RAGPipeline

router = APIRouter()

# Single session instance for simple demo or multiple sessions in memory
rag_pipeline = RAGPipeline()

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    source_type: Optional[str] = None # 'pdf' or 'web' or None

class Citation(BaseModel):
    source: str
    page: Optional[int]
    url: Optional[str]
    score: float
    method: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    session_id: str

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint for RAG queries."""
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        logger.info(f"Chat API: Received '{request.query}' for session '{request.session_id}'")
        results = await rag_pipeline.answer(request.query, session_id=request.session_id)
        
        return ChatResponse(
            answer=results["answer"],
            citations=results["citations"],
            session_id=request.session_id
        )
    except Exception as e:
        logger.error(f"Chat API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/chat/clear/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear memory for a specific chat session."""
    try:
        rag_pipeline.clear_session(session_id)
        return {"status": "success", "message": f"Cleared session '{session_id}'"}
    except Exception as e:
        logger.error(f"Clear History API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))