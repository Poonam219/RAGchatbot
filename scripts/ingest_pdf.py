import os
import sys
import argparse
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from backend.config import settings
from backend.core.embeddings import EmbeddingModel
from backend.core.vector_store import FAISSVectorStore
from backend.utils.pdf_parser import PDFParser
from backend.utils.chunker import TextChunker
from loguru import logger

def ingest_pdf(pdf_path: str):
    """
    Full pipeline to ingest a PDF:
    1. Parse PDF into blocks
    2. Chunk blocks into manageable pieces
    3. Generate embeddings
    4. Store in FAISS
    """
    logger.info(f"Starting ingestion for: {pdf_path}")
    
    # 1. Parse
    parser = PDFParser(extract_tables=True)
    blocks = parser.parse(pdf_path)
    
    if not blocks:
        logger.warning(f"No content extracted from {pdf_path}")
        return

    # 2. Chunk
    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    chunks = chunker.chunk_blocks(blocks)
    
    if not chunks:
        logger.warning(f"No chunks created from {pdf_path}")
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
    
    logger.success(f"Successfully ingested {len(chunks)} chunks from {pdf_path}")

def main():
    parser = argparse.ArgumentParser(description="Ingest PDF files into the RAG system.")
    parser.add_argument("--file", type=str, help="Path to the PDF file to ingest.")
    parser.add_argument("--dir", type=str, help="Path to a directory of PDFs to ingest.")
    
    args = parser.parse_args()
    
    if args.file:
        ingest_pdf(args.file)
    elif args.dir:
        pdf_files = list(Path(args.dir).glob("*.pdf"))
        for pdf in pdf_files:
            ingest_pdf(str(pdf))
    else:
        # Default behavior: ingest everything in data/pdfs
        pdf_dir = Path("data/pdfs")
        if pdf_dir.exists():
            pdf_files = list(pdf_dir.glob("*.pdf"))
            if not pdf_files:
                logger.warning("No PDF files found in data/pdfs")
            for pdf in pdf_files:
                ingest_pdf(str(pdf))
        else:
            logger.error("No input specified and data/pdfs directory does not exist.")

if __name__ == "__main__":
    main()
