import os
import sys
import argparse
import asyncio
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from backend.config import settings
from backend.core.embeddings import EmbeddingModel
from backend.core.vector_store import FAISSVectorStore
from backend.utils.web_scraper import WebScraper
from backend.utils.chunker import TextChunker
from loguru import logger

async def ingest_website(url: str, depth: int, max_pages: int):
    """
    Pipeline to ingest a website:
    1. Scrape URLs recursively
    2. Chunk content
    3. Generate embeddings
    4. Store in FAISS
    """
    logger.info(f"Starting website ingestion: {url} (depth={depth})")
    
    # 1. Scrape
    scraper = WebScraper(
        max_depth=depth or settings.scrape_max_depth,
        max_pages=max_pages or settings.scrape_max_pages,
        delay=settings.scrape_delay
    )
    pages = await scraper.scrape(url)
    
    if not pages:
        logger.warning(f"No pages scraped from {url}")
        return

    # 2. Chunk
    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    chunks = chunker.chunk_blocks(pages)
    
    if not chunks:
        logger.warning(f"No chunks created from website {url}")
        return

    # 3. Embed
    embedder = EmbeddingModel(model_name=settings.embedding_model)
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode_documents(texts)

    # 4. Store
    vector_store = FAISSVectorStore(
        db_path=settings.vector_db_path,
        dimension=embedder.embedding_dim
    )
    vector_store.add_chunks(chunks, embeddings)
    
    logger.success(f"Successfully ingested {len(chunks)} chunks from {url}")

def main():
    parser = argparse.ArgumentParser(description="Ingest a website into the RAG system.")
    parser.add_argument("--url", type=str, required=True, help="Base URL of the website to scrape.")
    parser.add_argument("--depth", type=int, default=2, help="Crawling depth.")
    parser.add_argument("--max-pages", type=int, default=50, help="Maximum number of pages to scrape.")
    
    args = parser.parse_args()
    
    # Run the async ingestion
    asyncio.run(ingest_website(args.url, args.depth, args.max_pages))

if __name__ == "__main__":
    main()
