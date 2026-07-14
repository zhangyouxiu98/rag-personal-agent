"""ChromaDB vector store implementation with Ollama embedding support."""

import hashlib
import os
import time
from typing import List, Optional

# Disable ChromaDB telemetry to avoid posthog import (Python 3.8 compat)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np

from agent.memory.base import BaseVectorStore, ChunkMetadata
from agent.core.state import DocumentChunk


class OllamaEmbeddingFunction:
    """ChromaDB-compatible embedding function that uses Ollama's embedding API.

    Wraps the OpenAI-compatible embedding endpoint exposed by Ollama,
    with automatic fallback to sentence-transformers if unavailable.
    """

    def __init__(self, base_url: str, model: str):
        import openai
        self.client = openai.OpenAI(
            base_url=f"{base_url}/v1",
            api_key="ollama",
        )
        self.model = model
        self._fallback_fn = None

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embed a list of texts. Returns list of embedding vectors."""
        try:
            resp = self.client.embeddings.create(
                model=self.model,
                input=input,
            )
            return [item.embedding for item in resp.data]
        except Exception:
            return self._fallback_embed(input)

    def _fallback_embed(self, input: List[str]) -> List[List[float]]:
        """Fallback to sentence-transformers when Ollama embedding fails."""
        if self._fallback_fn is None:
            from sentence_transformers import SentenceTransformer
            self._fallback_fn = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self._fallback_fn.encode(input, convert_to_numpy=True)
        return embeddings.tolist()


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-backed vector store for document chunk storage and retrieval."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_fn: OllamaEmbeddingFunction,
    ):
        self._persist_dir = persist_directory
        self._collection_name = collection_name
        self._embedding_fn = embedding_fn
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None

    async def initialize(self):
        """Create or load the ChromaDB collection."""
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        return self._collection

    @staticmethod
    def _make_chunk_id(source: str, chunk_index: int, text: str = "") -> str:
        """Generate a deterministic chunk ID from source, index, and content."""
        raw = f"{source}::{chunk_index}::{len(text)}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    async def add_texts(
        self,
        texts: List[str],
        metadatas: List[ChunkMetadata],
    ) -> List[str]:
        """Embed and add text chunks to the store."""
        ids = []
        chroma_metadatas = []
        documents = []

        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            chunk_id = self._make_chunk_id(meta.source, meta.chunk_index, text[:100])
            ids.append(chunk_id)

            meta_dict = {
                "source": str(meta.source),
                "file_type": meta.file_type,
                "chunk_index": meta.chunk_index,
                "total_chunks": meta.total_chunks,
                "page_number": meta.page_number or -1,
                "heading": meta.heading or "",
                "timestamp": meta.timestamp or time.time(),
            }
            chroma_metadatas.append(meta_dict)
            documents.append(text)

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=chroma_metadatas,
        )
        return ids

    async def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict] = None,
    ) -> List[DocumentChunk]:
        """Search for chunks similar to the query."""
        where_filter = {}
        if filter:
            where_filter = filter

        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where_filter if where_filter else None,
        )

        chunks = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                content = results["documents"][0][i] if results["documents"] else ""
                distance = results["distances"][0][i] if results["distances"] else 0.0

                # Convert cosine distance to similarity score (1 - distance for cosine)
                score = 1.0 - float(distance)

                chunks.append(DocumentChunk(
                    id=chunk_id,
                    content=content,
                    metadata=metadata,
                    score=score,
                ))

        return chunks

    async def delete_document(self, source: str) -> int:
        """Delete all chunks belonging to a source document."""
        results = self.collection.get(
            where={"source": source},
        )
        if results and results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    async def count_documents(self) -> int:
        """Count unique documents in the store."""
        results = self.collection.get()
        if results and results["metadatas"]:
            sources = {m.get("source") for m in results["metadatas"] if m}
            return len(sources)
        return 0

    async def clear(self):
        """Delete all data from the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
