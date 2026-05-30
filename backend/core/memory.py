from typing import List, Dict, Any, Optional
from collections import deque
from backend.config import settings
from loguru import logger

class ConversationMemory:
    """
    Manages chat history for multiple sessions using in-memory storage.
    Supports fixed-window history for maintaining context within token limits.
    """
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns or settings.max_history_turns
        self.sessions: Dict[str, deque] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return history for a specific session ID."""
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.max_turns * 2)
        return list(self.sessions[session_id])

    def add_message(self, session_id: str, role: str, content: str):
        """Add user or assistant message to session history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.max_turns * 2)
        self.sessions[session_id].append({"role": role, "content": content})
        logger.debug(f"Memory: Added {role} to {session_id}. History size: {len(self.sessions[session_id])}")

    def clear(self, session_id: str):
        """Clear session history."""
        if session_id in self.sessions:
            self.sessions[session_id].clear()
            logger.info(f"Memory: Cleared history for {session_id}")

    def get_context_string(self, session_id: str) -> str:
        """Convert history into a single string for LLM prompting (if needed)."""
        history = self.get_history(session_id)
        context = ""
        for msg in history:
            role = "Human" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content']}\n"
        return context