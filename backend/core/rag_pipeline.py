from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from backend.config import settings
from backend.core.embeddings import EmbeddingModel
from backend.core.vector_store import FAISSVectorStore
from backend.core.retriever import HybridRetriever
from backend.core.llm import GroqLLM
from backend.core.memory import ConversationMemory

class RAGPipeline:
    """
    Orchestrator for the RAG chatbot.
    Links vector storage, hybrid retrieval, and LLM generation.
    """
    def __init__(self):
        # 1. Initialize Components
        self.embedder = EmbeddingModel(model_name=settings.embedding_model)
        self.vector_store = FAISSVectorStore(
            db_path=settings.vector_db_path,
            dimension=self.embedder.embedding_dim
        )
        self.retriever = HybridRetriever(self.vector_store, self.embedder)
        self.llm = GroqLLM()
        self.memory = ConversationMemory()

        # 2. RAG Prompt Template
        self.system_prompt = (
            "You are a helpful, assistant for the Visakhapatnam Port Authority (VPT/VPA).\n"
            "Use the provided context to answer user queries accurately.\n"
            "If the answer is NOT in the context, say that you don't know but try to provide helpful general info if relevant.\n"
            "Focus on being professional and respectful.\n"
            "Context:\n{context}"
        )

    async def answer(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """Entry point for user queries."""
        logger.info(f"RAGPipeline: Processing query for session '{session_id}'")
        
        # 1. Retrieve
        relevant_chunks = self.retriever.search(query, top_k=settings.top_k)
        
        # 2. Build Context
        context_text = "\n\n".join([f"Source: {c['source']}\nContent: {c['text']}" for c in relevant_chunks])
        
        # 3. Construct Messages (History + Context)
        history = self.memory.get_history(session_id)
        
        # System Message with Context
        messages = [{"role": "system", "content": self.system_prompt.format(context=context_text)}]
        
        # Add History
        messages.extend(history)
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        # 4. Generate
        response = self.llm.generate_chat_completion(messages)
        
        # 5. Store in Memory
        self.memory.add_message(session_id, "user", query)
        self.memory.add_message(session_id, "assistant", response)
        
        # 6. Return response with citations
        return {
            "answer": response,
            "citations": [
                {
                    "source": c["source"], 
                    "page": c.get("page"), 
                    "url": c.get("url"),
                    "score": round(c.get("score", 0), 3),
                    "method": c.get("retrieval_method")
                } 
                for c in relevant_chunks
            ]
        }

    def clear_session(self, session_id: str):
        self.memory.clear(session_id)