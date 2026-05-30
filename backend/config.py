"""
config.py — Centralized configuration using pydantic-settings.
Loads from .env file automatically.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os


class Settings(BaseSettings):
    # ── Groq ──────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    default_model: str = Field("llama-3.1-8b-instant", env="DEFAULT_MODEL")

    available_models: List[str] = [
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "llama3-70b-8192",
        "llama3-8b-8192",
    ]

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = Field("BAAI/bge-base-en-v1.5", env="EMBEDDING_MODEL")

    # ── Vector DB ─────────────────────────────────────────────────────────────
    vector_db_type: str = Field("faiss", env="VECTOR_DB_TYPE")
    vector_db_path: str = Field("./data/vector_db", env="VECTOR_DB_PATH")
    chroma_collection_pdf: str = Field("pdf_docs", env="CHROMA_COLLECTION_PDF")
    chroma_collection_web: str = Field("web_docs", env="CHROMA_COLLECTION_WEB")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = Field(5, env="TOP_K")
    chunk_size: int = Field(400, env="CHUNK_SIZE")
    chunk_overlap: int = Field(80, env="CHUNK_OVERLAP")
    hybrid_search: bool = Field(True, env="HYBRID_SEARCH")
    bm25_weight: float = Field(0.3, env="BM25_WEIGHT")
    semantic_weight: float = Field(0.7, env="SEMANTIC_WEIGHT")

    # ── Scraping ──────────────────────────────────────────────────────────────
    scrape_max_depth: int = Field(2, env="SCRAPE_MAX_DEPTH")
    scrape_max_pages: int = Field(50, env="SCRAPE_MAX_PAGES")
    scrape_delay: float = Field(1.0, env="SCRAPE_DELAY")

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    cors_origins: str = Field("*", env="CORS_ORIGINS")

    # ── Reranking ─────────────────────────────────────────────────────────────
    enable_rerank: bool = Field(True, env="ENABLE_RERANK")
    rerank_model: str = Field("BAAI/bge-reranker-base", env="RERANK_MODEL")
    rerank_top_k: int = Field(5, env="RERANK_TOP_K")

    # ── Memory ────────────────────────────────────────────────────────────────
    max_history_turns: int = Field(10, env="MAX_HISTORY_TURNS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton settings instance
settings = Settings()

# Ensure data directories exist
for path in [
    settings.vector_db_path,
    "./data/pdfs",
    "./data/scraped",
]:
    os.makedirs(path, exist_ok=True)