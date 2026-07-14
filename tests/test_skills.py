"""Tests for skills: loading, chunking, and session memory."""

import os
import tempfile

import pytest

from agent.memory.session_memory import SessionMemory
from agent.skills.weather_skill import DocumentLoader
from agent.skills.text_chunker import TextChunker
from agent.memory.base import ChunkMetadata


class TestSessionMemory:
    """Tests for the SessionMemory class."""

    def test_add_and_get_history(self):
        mem = SessionMemory(max_history=10)
        mem.add_entry("sess_1", "user", "Hello")
        mem.add_entry("sess_1", "assistant", "Hi there!")
        history = mem.get_history("sess_1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_sliding_window(self):
        mem = SessionMemory(max_history=3)
        for i in range(5):
            mem.add_entry("sess_1", "user", f"msg_{i}")
        history = mem.get_history("sess_1")
        assert len(history) == 3
        assert history[0]["content"] == "msg_2"
        assert history[-1]["content"] == "msg_4"

    def test_clear(self):
        mem = SessionMemory()
        mem.add_entry("sess_1", "user", "test")
        mem.clear("sess_1")
        assert mem.get_history("sess_1") == []

    def test_get_recent(self):
        mem = SessionMemory()
        for i in range(10):
            mem.add_entry("sess_1", "user", f"msg_{i}")
        recent = mem.get_recent("sess_1", 3)
        assert len(recent) == 3
        assert recent[-1]["content"] == "msg_9"


class TestDocumentLoader:
    """Tests for the DocumentLoader class."""

    def test_load_txt(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Hello, world!")
            tmp_path = f.name

        try:
            loader = DocumentLoader()
            content = loader.load(tmp_path)
            assert content == "Hello, world!"
        finally:
            os.unlink(tmp_path)

    def test_load_md(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write("# Title\nContent here.")
            tmp_path = f.name

        try:
            loader = DocumentLoader()
            content = loader.load(tmp_path)
            assert "# Title" in content
            assert "Content here." in content
        finally:
            os.unlink(tmp_path)

    def test_load_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write('{"key": "value"}')
            tmp_path = f.name

        try:
            loader = DocumentLoader()
            content = loader.load(tmp_path)
            assert '"key"' in content
            assert '"value"' in content
        finally:
            os.unlink(tmp_path)

    def test_unsupported_format(self):
        loader = DocumentLoader()
        with pytest.raises(ValueError):
            loader.load("file.xyz")

    def test_supported_formats(self):
        loader = DocumentLoader()
        formats = loader.supported_formats()
        assert ".pdf" in formats
        assert ".md" in formats
        assert ".txt" in formats
        assert ".json" in formats


class TestTextChunker:
    """Tests for the TextChunker class."""

    def test_split_empty(self):
        chunker = TextChunker()
        meta = ChunkMetadata(source="test.txt", file_type=".txt")
        chunks = chunker.split("", meta)
        assert len(chunks) == 0

    def test_split_short_text(self):
        chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        meta = ChunkMetadata(source="test.txt", file_type=".txt")
        text = "This is a short sentence."
        chunks = chunker.split(text, meta)
        assert len(chunks) == 1
        assert "short sentence" in chunks[0].content

    def test_split_long_text(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        meta = ChunkMetadata(source="test.txt", file_type=".txt")
        # Create text that will require multiple chunks
        text = ". ".join(["Sentence number " + str(i) for i in range(50)])
        chunks = chunker.split(text, meta)
        assert len(chunks) > 1

    def test_estimate_chunks(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = ". ".join(["Sentence number " + str(i) for i in range(50)])
        estimated = chunker.estimate_chunks(text)
        assert estimated > 0
