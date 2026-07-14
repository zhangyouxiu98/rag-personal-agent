"""MemoryManager: unified facade for session memory and vector store."""

from pathlib import Path
from typing import List, Optional

from agent.core.state import DocumentChunk, Citation
from agent.memory.base import BaseVectorStore, BaseSessionMemory


class MemoryManager:
    """Unified memory interface that combines session memory and vector store.

    Provides a single entry point for:
    - Conversation history management (session memory)
    - Document retrieval and ingestion (vector store)
    - Context building from retrieved chunks
    """

    def __init__(
        self,
        session_memory: BaseSessionMemory,
        vector_store: BaseVectorStore,
    ):
        self.session = session_memory
        self.vector = vector_store

    # ---- Session memory delegation ----

    def get_history(self, session_id: str) -> List[dict]:
        """Get conversation history for a session."""
        return self.session.get_history(session_id)

    def add_turn(
        self,
        session_id: str,
        query: str,
        answer: str,
        citations: Optional[List[Citation]] = None,
    ):
        """Add a complete Q&A turn to session history."""
        self.session.add_entry(session_id, "user", query)
        answer_with_citations = answer
        if citations:
            sources = "\n".join(
                f"  [{i+1}] {c.source}" for i, c in enumerate(citations)
            )
            answer_with_citations = f"{answer}\n\nSources:\n{sources}"
        self.session.add_entry(session_id, "assistant", answer_with_citations)

    def clear_session(self, session_id: str):
        """Clear the conversation history for a session."""
        self.session.clear(session_id)

    def get_recent_history(self, session_id: str, n: int = 5) -> List[dict]:
        """Get the most recent n conversation turns."""
        return self.session.get_recent(session_id, n)

    # ---- Vector store delegation ----

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict] = None,
    ) -> List[DocumentChunk]:
        """Retrieve relevant document chunks from the vector store."""
        return await self.vector.similarity_search(query=query, k=k, filter=filter)

    async def delete_document(self, source: str) -> int:
        """Remove a document from the knowledge base."""
        return await self.vector.delete_document(source)

    async def count_documents(self) -> int:
        """Get the total number of ingested documents."""
        return await self.vector.count_documents()

    # ---- Context building ----

    def build_context(
        self,
        chunks: List[DocumentChunk],
        max_tokens: int = 3000,
    ) -> str:
        """Build a context string from retrieved chunks for LLM prompting.

        Chunks are sorted by score descending. Context is truncated
        to approximately max_tokens by removing the lowest-scoring
        chunks first. A rough estimate of 4 chars per token is used.

        Args:
            chunks: Retrieved document chunks with scores.
            max_tokens: Maximum tokens to include in the context.

        Returns:
            Formatted context string with source annotations.
        """
        if not chunks:
            return ""

        # Sort by score descending
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

        # Build context with source annotations
        parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # rough estimate: 4 chars per token

        for i, chunk in enumerate(sorted_chunks):
            source = chunk.metadata.get("source", "unknown")
            heading = chunk.metadata.get("heading", "")
            header = f"[Chunk {i+1}"
            if source:
                header += f" | {Path(source).name}" if isinstance(source, str) else f" | {source}"
            if heading:
                header += f" | {heading}"
            header += "]"

            part = f"{header}\n{chunk.content}"

            # Check if adding this chunk exceeds the limit
            if total_chars + len(part) > max_chars and parts:
                break

            parts.append(part)
            total_chars += len(part)

        return "\n\n".join(parts)
