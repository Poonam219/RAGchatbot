"""
chunker.py — Smart text chunking with overlap preservation.
Supports token-based and sentence-based chunking strategies.
"""
import re
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class TextChunker:
    """
    Splits long text into overlapping chunks for RAG retrieval.

    Strategy:
    1. Split by paragraphs / sentence boundaries
    2. Group into chunks of ~chunk_size tokens
    3. Add overlap between adjacent chunks
    4. Preserve metadata from source block
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        min_chunk_size: int = 50,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        if TIKTOKEN_AVAILABLE:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        else:
            self.encoder = None

    def chunk_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a list of parsed text blocks.

        Args:
            blocks: List of {text, source, page, url, ...} dicts

        Returns:
            List of chunk dicts with chunk_index added
        """
        all_chunks = []

        for block in blocks:
            text = block.get("text", "").strip()
            if not text:
                continue

            chunks = self._chunk_text(text)

            for i, chunk_text in enumerate(chunks):
                chunk = {
                    **block,  # Inherit all metadata
                    "text": chunk_text,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "token_count": self._count_tokens(chunk_text),
                }
                all_chunks.append(chunk)

        logger.info(f"Created {len(all_chunks)} chunks from {len(blocks)} blocks")
        return all_chunks

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        # Step 1: Split into sentences/paragraphs
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        # Step 2: Group sentences into chunks
        chunks = []
        current_chunk_sentences = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # If single sentence exceeds chunk size, split it further
            if sentence_tokens > self.chunk_size:
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = []
                    current_tokens = 0
                # Force-split long sentence by words
                word_chunks = self._split_long_text(sentence)
                chunks.extend(word_chunks)
                continue

            # If adding this sentence would exceed limit, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))

                # Keep last N sentences for overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences
                )
                current_chunk_sentences = overlap_sentences
                current_tokens = sum(
                    self._count_tokens(s) for s in current_chunk_sentences
                )

            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Save the last chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        # Filter out tiny chunks
        return [c for c in chunks if self._count_tokens(c) >= self.min_chunk_size]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Split on sentence boundaries, preserving structure
        text = re.sub(r'\n{2,}', ' [PARA] ', text)  # Mark paragraphs

        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Restore paragraph breaks as separate items
        result = []
        for sentence in sentences:
            parts = sentence.split(' [PARA] ')
            result.extend(p.strip() for p in parts if p.strip())

        return result

    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get trailing sentences for overlap."""
        if not sentences:
            return []

        overlap_sentences = []
        overlap_tokens = 0

        for sentence in reversed(sentences):
            tokens = self._count_tokens(sentence)
            if overlap_tokens + tokens <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += tokens
            else:
                break

        return overlap_sentences

    def _split_long_text(self, text: str) -> List[str]:
        """Split a very long text by words into chunks."""
        words = text.split()
        chunks = []
        current_words = []
        current_tokens = 0

        for word in words:
            word_tokens = self._count_tokens(word)
            if current_tokens + word_tokens > self.chunk_size and current_words:
                chunks.append(" ".join(current_words))
                current_words = []
                current_tokens = 0
            current_words.append(word)
            current_tokens += word_tokens

        if current_words:
            chunks.append(" ".join(current_words))

        return chunks

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoder:
            return len(self.encoder.encode(text))
        # Fallback: approximate by word count * 1.3
        return int(len(text.split()) * 1.3)