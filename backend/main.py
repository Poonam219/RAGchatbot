from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from backend.config import settings
from backend.api import chat, ingest, search
from loguru import logger
import os

# 1. Initialize FastAPI
app = FastAPI(
    title="VPA RAG Chatbot",
    description="High-performance RAG pipeline for Visakhapatnam Port Authority documentation.",
    version="1.0.0"
)

# 2. CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Include Routers
# NOTE: Using a single router for Chatbot Logic and hybrid search
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(search.router, prefix="/api", tags=["Search"])
app.include_router(ingest.router, prefix="/api", tags=["Ingest"])

# 4. Mount Frontend (Static Files)
# This will serve index.html at the root if it exists in the frontend folder
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

# 5. Root Health Check
@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

# 6. Global Error Handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred.", "detail": str(exc)},
    )

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)