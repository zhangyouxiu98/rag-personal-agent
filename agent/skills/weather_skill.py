"""Document loader supporting multiple file formats."""

from pathlib import Path
from typing import Callable, Dict, Tuple


class DocumentLoader:
    """Load documents from PDF, Markdown, TXT, and JSON files.

    Supports registering custom format handlers via register_handler().

    Usage:
        loader = DocumentLoader()
        text = loader.load("path/to/document.pdf")
        text, metadata = loader.load_with_metadata("path/to/document.md")
    """

    def __init__(self):
        self._handlers: Dict[str, Callable[[str], str]] = {
            ".pdf": self._load_pdf,
            ".md": self._load_markdown,
            ".txt": self._load_text,
            ".json": self._load_json,
        }

    def supported_formats(self) -> list:
        """Return a list of supported file extensions."""
        return list(self._handlers.keys())

    def register_handler(self, ext: str, handler: Callable[[str], str]):
        """Register a custom file format handler.

        Args:
            ext: File extension including the dot, e.g. '.csv'.
            handler: Callable that takes a file path and returns text content.
        """
        if not ext.startswith("."):
            ext = f".{ext}"
        self._handlers[ext.lower()] = handler

    def load(self, file_path: str) -> str:
        """Load and extract text from a document file.

        Raises:
            ValueError: If the file extension is not supported.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        # Check extension first for better error messages
        handler = self._handlers.get(ext)
        if handler is None:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {self.supported_formats()}"
            )

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return handler(str(path))

    def load_with_metadata(self, file_path: str) -> Tuple[str, dict]:
        """Load text content and extract basic metadata.

        Returns:
            Tuple of (text_content, metadata_dict).
        """
        text = self.load(file_path)
        path = Path(file_path)
        stat = path.stat()

        metadata = {
            "source": str(path.absolute()),
            "filename": path.name,
            "file_type": path.suffix.lower(),
            "file_size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
        }
        return text, metadata

    # ---- Internal format handlers ----

    @staticmethod
    def _load_pdf(file_path: str) -> str:
        """Extract text from a PDF using PyMuPDF (fitz)."""
        import fitz
        doc = fitz.open(file_path)
        pages = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                pages.append(text.strip())
        doc.close()
        return "\n\n".join(pages)

    @staticmethod
    def _load_markdown(file_path: str) -> str:
        """Read a Markdown file as-is."""
        return Path(file_path).read_text(encoding="utf-8")

    @staticmethod
    def _load_text(file_path: str) -> str:
        """Read a plain text file."""
        return Path(file_path).read_text(encoding="utf-8")

    @staticmethod
    def _load_json(file_path: str) -> str:
        """Pretty-print JSON content for embedding."""
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return json.dumps(data, ensure_ascii=False, indent=2)
