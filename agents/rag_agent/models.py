"""
models.py
=========
Plain dataclasses shared across the RAG pipeline (parsing -> chunking ->
embedding -> vector store -> retrieval -> answering). Kept separate from
FastAPI's request/response Pydantic models in app.py - those are the HTTP
contract, these are the internal shape everything else here passes around.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ParsedSection:
    """One extracted piece of text from a source document, before
    chunking. For PDFs this is one page; for everything else it's the
    whole document as a single section (page=None)."""

    text: str
    page: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """One chunk ready to embed and store, with metadata that survives
    all the way to the citation shown to the user."""

    chunk_id: str
    document_id: str
    file_name: str
    file_type: str
    chunk_index: int
    text: str
    page: Optional[int] = None

    def metadata(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "document_id": self.document_id,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "chunk_index": self.chunk_index,
            "source": self.file_name,
        }
        # Chroma metadata values can't be None - omit page instead of
        # storing a null when this chunk has no page number.
        if self.page is not None:
            meta["page"] = self.page
        return meta


@dataclass
class RetrievedChunk:
    """A chunk returned from vector search, with its similarity score."""

    chunk_id: str
    document_id: str
    file_name: str
    file_type: str
    chunk_index: int
    text: str
    score: float
    page: Optional[int] = None

    def to_source_dict(self) -> Dict[str, Any]:
        preview = self.text.strip().replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220].rsplit(" ", 1)[0] + "..."
        return {
            "document_id": self.document_id,
            "file_name": self.file_name,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
            "preview": preview,
        }

    def to_context_dict(self) -> Dict[str, Any]:
        """Full (not truncated) shape kept for internal use - e.g. a
        future RAGAS evaluation needs the exact context text, not the
        preview shown to the user."""
        return {
            "document_id": self.document_id,
            "file_name": self.file_name,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
            "text": self.text,
        }


@dataclass
class DocumentRecord:
    """One row in the document registry (storage/rag/metadata/documents.json)."""

    document_id: str
    file_name: str
    file_type: str
    sha256: str
    size_bytes: int
    uploaded_at: str
    status: str = "processing"  # processing | ready | failed
    chunk_count: int = 0
    page_count: Optional[int] = None
    error: Optional[str] = None
    stored_path: Optional[str] = None  # relative to RAG_DOCUMENTS_DIR

    def to_public_dict(self) -> Dict[str, Any]:
        """What the frontend gets - never the absolute local file path."""
        return {
            "document_id": self.document_id,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "size_bytes": self.size_bytes,
            "uploaded_at": self.uploaded_at,
            "status": self.status,
            "chunk_count": self.chunk_count,
            "page_count": self.page_count,
            "error": self.error,
        }

    def to_storage_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "uploaded_at": self.uploaded_at,
            "status": self.status,
            "chunk_count": self.chunk_count,
            "page_count": self.page_count,
            "error": self.error,
            "stored_path": self.stored_path,
        }

    @staticmethod
    def from_storage_dict(data: Dict[str, Any]) -> "DocumentRecord":
        return DocumentRecord(**data)


class RAGError(Exception):
    """Raised for any RAG pipeline failure (parsing, embedding, vector
    store, retrieval) that should be surfaced as a clean API error rather
    than a 500 stack trace."""


class UnsupportedFileTypeError(RAGError):
    """Raised when an uploaded file's extension isn't supported."""


class EmptyDocumentError(RAGError):
    """Raised when a parsed document has no extractable text."""
