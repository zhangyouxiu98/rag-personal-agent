"""Document retriever skill: ingestion and retrieval pipeline."""

import time
from pathlib import Path
from typing import Dict, List, Optional

from agent.core.state import DocumentChunk
from agent.memory.base import BaseVectorStore, ChunkMetadata
from agent.skills.weather_skill import DocumentLoader
from agent.skills.text_chunker import TextChunker


class DocumentRetrieverSkill:
    """Core RAG skill combining document ingestion and retrieval.

    Ingestion pipeline: load -> chunk -> embed -> store
    Retrieval pipeline: embed query -> search -> return relevant chunks

    Usage:
        skill = DocumentRetrieverSkill(vector_store, chunker, loader)
        # Ingest
        count = await skill.ingest("docs/report.pdf")
        # Retrieve
        chunks = await skill.retrieve("What is the revenue?")
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        chunker: TextChunker,
        loader: Optional[DocumentLoader] = None,
    ):
        self.vector_store = vector_store
        self.chunker = chunker
        self.loader = loader or DocumentLoader()

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        filter: Optional[dict] = None,
    ) -> List[DocumentChunk]:
        """Retrieve relevant document chunks for a query."""
        return await self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )

    async def ingest(self, file_path: str) -> int:
        """Ingest a single document into the knowledge base.

        Pipeline: load text → chunk → embed → store.

        Args:
            file_path: Path to the document file.

        Returns:
            Number of chunks created and stored.
        """
        path = Path(file_path)

        # Load document
        text, file_meta = self.loader.load_with_metadata(str(path))

        if not text.strip():
            return 0

        # Build chunk metadata
        base_meta = ChunkMetadata(
            source=str(path.absolute()),
            file_type=path.suffix.lower(),
            chunk_index=0,
            total_chunks=0,
            timestamp=time.time(),
        )

        # Chunk the document
        if path.suffix.lower() == ".md":
            chunks = self.chunker.split_by_headings(text, base_meta)
        else:
            chunks = self.chunker.split(text, base_meta)

        if not chunks:
            return 0

        # Prepare for vector store
        texts = [c.content for c in chunks]
        metadatas = [
            ChunkMetadata(
                source=str(path.absolute()),
                file_type=path.suffix.lower(),
                chunk_index=c.metadata.get("chunk_index", i),
                total_chunks=len(chunks),
                page_number=c.metadata.get("page_number"),
                heading=c.metadata.get("heading"),
                timestamp=time.time(),
            )
            for i, c in enumerate(chunks)
        ]

        # Embed and store
        await self.vector_store.add_texts(texts=texts, metadatas=metadatas)

        return len(chunks)

    async def ingest_directory(self, dir_path: str) -> Dict[str, int]:
        """Batch ingest all supported documents in a directory.

        Args:
            dir_path: Path to the directory.

        Returns:
            Dictionary mapping file paths to chunk counts.
        """
        from tqdm import tqdm

        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        supported = self.loader.supported_formats()
        files = [
            p for p in path.rglob("*")
            if p.suffix.lower() in supported
        ]

        results = {}
        for f in tqdm(files, desc="Ingesting documents"):
            try:
                count = await self.ingest(str(f))
                results[str(f)] = count
            except Exception as e:
                results[str(f)] = -1  # mark error
                print(f"Error ingesting {f}: {e}")

        return results

    async def delete(self, file_path: str) -> int:
        """Remove a document from the knowledge base.

        Returns:
            Number of chunks deleted.
        """
        abs_path = str(Path(file_path).absolute())
        return await self.vector_store.delete_document(abs_path)

    async def count_documents(self) -> int:
        """Get the total number of ingested documents."""
        return await self.vector_store.count_documents()
