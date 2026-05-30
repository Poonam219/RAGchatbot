from groq import Groq
from backend.config import settings
from typing import List, Dict, Any, Optional
from loguru import logger

class GroqLLM:
    """
    Wrapper for Groq API integration.
    Handles chat completions with context injection.
    """
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.groq_api_key
        if not self.api_key:
            logger.error("GROQ_API_KEY is missing!")
            raise ValueError("Groq API Key must be provided either in settings or as an argument.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model or settings.default_model

    def generate_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7, 
        max_tokens: int = 2048
    ) -> str:
        """Generate response using Groq's high-speed inference."""
        try:
            logger.info(f"Generating completion with model: {self.model}")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                stream=False,
                stop=None,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq Completion Error: {e}")
            return f"Error: {str(e)}"

    async def generate_chat_completion_async(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.7, 
        max_tokens: int = 2048
    ) -> str:
        """Async variant of chat completion (uses sync client for simplicity unless client is async)."""
        # Note: groq-python's current version (0.9+) has AsyncGroq, but for now we wrap the sync one.
        return self.generate_chat_completion(messages, temperature, max_tokens)