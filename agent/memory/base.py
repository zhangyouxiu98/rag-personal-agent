"""Abstract interfaces for vector store and session memory."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ChunkMetadata:
    """Metadata attached to each document chunk in the vector store."""
    source: str
    file_type: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    page_number: Optional[int] = None
    heading: Optional[str] = None
    timestamp: float = 0.0


class BaseVectorStore(ABC):
    """Abstract interface for vector-based document storage and retrieval."""

    @abstractmethod
    async def initialize(self):
        """Initialize or load the vector store."""
        ...

    @abstractmethod
    async def add_texts(
        self,
        texts: List[str],
        metadatas: List[ChunkMetadata],
    ) -> List[str]:
        """Embed and add texts to the store. Returns chunk IDs."""
        ...

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict] = None,
    ) -> list:
        """Search for chunks similar to the query. Returns list of (DocumentChunk, score)."""
        ...

    @abstractmethod
    async def delete_document(self, source: str) -> int:
        """Delete all chunks from a given source document. Returns count deleted."""
        ...

    @abstractmethod
    async def count_documents(self) -> int:
        """Return the total number of unique documents in the store."""
        ...

    @abstractmethod
    async def clear(self):
        """Remove all data from the store."""
        ...


class BaseSessionMemory(ABC):
    """Abstract interface for conversation session memory."""

    @abstractmethod
    def get_history(self, session_id: str) -> List[dict]:
        """Retrieve conversation history for a session."""
        ...

    @abstractmethod
    def add_entry(self, session_id: str, role: str, content: str):
        """Add a conversation turn to the session."""
        ...

    @abstractmethod
    def clear(self, session_id: str):
        """Clear the history for a session."""
        ...
