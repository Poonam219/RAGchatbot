RAG Chatbot
===========

Retrieval-Augmented Generation (RAG) chatbot that ingests PDFs and websites, builds embeddings into a persistent vector store, and answers conversational queries using a lightweight Python backend and a minimal web frontend.

**Overview**
- The project ingests documents (PDFs, web pages), slices them into chunks, computes embeddings, stores them in an on-disk vector database, and answers user questions by retrieving relevant passages and composing responses with a language model.

**Key Features**
- **Ingestion:** PDF and website ingestion scripts: [scripts/ingest_pdf.py](scripts/ingest_pdf.py), [scripts/ingest_website.py](scripts/ingest_website.py).
- **Embeddings & Vector DB:** Embeddings pipeline and persistent vector store at [data/vector_db/](data/vector_db/).
- **Backend API & RAG pipeline:** Python backend and API endpoints: [backend/main.py](backend/main.py), [backend/api/chat.py](backend/api/chat.py), [backend/core/retriever.py](backend/core/retriever.py), [backend/core/rag_pipeline.py](backend/core/rag_pipeline.py), [backend/core/llm.py](backend/core/llm.py), [backend/core/embeddings.py](backend/core/embeddings.py).
- **Conversational memory:** Session/short-term memory implementation: [backend/core/memory.py](backend/core/memory.py).
- **Frontend:** Simple interactive UI: [frontend/index.html](frontend/index.html), [frontend/app.js](frontend/app.js).

**Quick Start**
1. Install Python dependencies:

```bash
pip install -r backend/requirements.txt
```

2. Run the backend API:

```bash
python backend/main.py
```

3. (Optional) Ingest documents to populate the vector DB:

```bash
python scripts/ingest_pdf.py path/to/file.pdf
python scripts/ingest_website.py https://example.com
```

4. Open the frontend in a browser by opening [frontend/index.html](frontend/index.html) or point your browser to the backend's UI URL if served.

**API Usage (example)**
- Send a chat query (example HTTP request):

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

**Project Layout (high level)**
- backend/: Python API, core RAG code, embeddings and retrieval logic.
- scripts/: ingestion helpers and utilities for seeding the DB.
- data/vector_db/: persisted vector index and metadata.
- frontend/: minimal web UI for chatting.

**Notes & Next Steps**
- You can swap the LLM backend in `backend/core/llm.py` to use a different provider or a local model.
- Consider adding a small Dockerfile or a `docker-compose.yml` to simplify deployment.

**License**
- MIT (update as needed).
