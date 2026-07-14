"""Recursive text chunker for document splitting."""

import re
from typing import List

from agent.memory.base import ChunkMetadata
from agent.core.state import DocumentChunk


class TextChunker:
    """Split documents into overlapping chunks for embedding.

    Uses recursive splitting with configuraable chunk_size and overlap.
    For Markdown files, splits preferentially at heading boundaries
    to maintain semantic coherence.

    Splitting order: headings -> paragraphs (\\n\\n) -> sentences (。) -> words ( )
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Separators in descending priority
        self._separators = [
            # Markdown headings
            r"\n(?=#{1,6}\s)",
            # Paragraph breaks
            r"\n\n+",
            # Line breaks
            r"\n",
            # Sentence endings (Chinese + English)
            r"(?<=[。！？.!?])\s*",
            # Clause breaks
            r"(?<=[，,；;：:])\s*",
            # Word boundaries
            r"\s+",
            # Character fallback
            "",
        ]

    def split(self, text: str, metadata: ChunkMetadata) -> List[DocumentChunk]:
        """Split a document into overlapping chunks.

        Args:
            text: The full document text.
            metadata: Source document metadata to attach to each chunk.

        Returns:
            List of DocumentChunk objects ready for embedding.
        """
        if not text.strip():
            return []

        splits = self._recursive_split(text, self._separators)
        chunks = self._merge_splits(splits)

        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk_meta = ChunkMetadata(
                source=metadata.source,
                file_type=metadata.file_type,
                chunk_index=i,
                total_chunks=len(chunks),
                page_number=metadata.page_number,
                heading=metadata.heading,
                timestamp=metadata.timestamp,
            )
            chunk_objects.append(DocumentChunk(
                id="",  # Assigned by vector store
                content=chunk_text,
                metadata={
                    "source": str(chunk_meta.source),
                    "file_type": chunk_meta.file_type,
                    "chunk_index": chunk_meta.chunk_index,
                    "total_chunks": chunk_meta.total_chunks,
                    "page_number": chunk_meta.page_number,
                    "heading": chunk_meta.heading,
                    "timestamp": chunk_meta.timestamp,
                },
            ))

        return chunk_objects

    def split_by_headings(self, text: str, metadata: ChunkMetadata) -> List[DocumentChunk]:
        """Markdown-aware splitting: each top-level heading becomes a chunk boundary."""
        # Split by ## or # headings
        sections = re.split(r"\n(?=#{1,2}\s)", text)
        all_chunks = []
        for section in sections:
            if section.strip():
                # Extract heading for metadata
                heading_match = re.match(r"^#{1,2}\s+(.+)", section)
                heading = heading_match.group(1) if heading_match else None
                section_meta = ChunkMetadata(
                    source=metadata.source,
                    file_type=metadata.file_type,
                    chunk_index=0,
                    total_chunks=0,
                    heading=heading,
                    timestamp=metadata.timestamp,
                )
                all_chunks.extend(self.split(section, section_meta))
        return all_chunks

    def _recursive_split(self, text: str, separators: list) -> List[str]:
        """Recursively split text using the first effective separator."""
        # If text fits in one chunk, don't split it
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            return [text]

        sep_pattern = separators[0]

        if not sep_pattern:
            # Last resort: no more separators, keep as single piece
            return [text]

        splits = re.split(sep_pattern, text)
        # Filter out empty strings
        splits = [s.strip() for s in splits if s.strip()]

        # If splitting didn't help, try the next separator
        if len(splits) <= 1:
            return self._recursive_split(text, separators[1:])

        # Check each split piece against chunk_size
        result = []
        for piece in splits:
            if len(piece) > self.chunk_size:
                # Piece is still too large, recurse deeper
                result.extend(self._recursive_split(piece, separators[1:]))
            else:
                result.append(piece)

        return result

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge small splits into chunks that fit within chunk_size.

        Uses overlap: each chunk reuses the last chunk_overlap characters
        from the previous chunk for context continuity.
        """
        if not splits:
            return []

        chunks = []
        current_chunk = ""

        for piece in splits:
            if not piece:
                continue

            # If adding this piece exceeds chunk_size, finalize current chunk
            if len(current_chunk) + len(piece) + 1 > self.chunk_size and current_chunk:
                chunks.append(current_chunk)

                # Start new chunk with overlap from previous
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    current_chunk = current_chunk[-self.chunk_overlap:] + "\n" + piece
                else:
                    current_chunk = piece
            else:
                if current_chunk:
                    current_chunk += "\n" + piece
                else:
                    current_chunk = piece

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def estimate_chunks(self, text: str) -> int:
        """Estimate the number of chunks a document will produce."""
        if not text.strip():
            return 0
        splits = self._recursive_split(text, self._separators)
        chunks = self._merge_splits(splits)
        return len(chunks)
