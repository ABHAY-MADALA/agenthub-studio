"""
ingestion.py
============
The pipeline described in the RAG spec, in one orchestrating function:

    file bytes -> validate -> hash (dedup check) -> save to disk
               -> parse -> chunk -> embed -> vector store
               -> register in the document registry

Nothing here talks to chromadb, OpenAI, or the filesystem layout directly
except through the other rag_agent modules - this file is glue, not
implementation, so it stays short and easy to follow end to end.

Security notes (see also app.py's upload endpoint):
- The on-disk filename is always `{document_id}{extension}`, never the
  user-supplied name, so a malicious filename (e.g. containing `../`)
  can't cause a path traversal - the original name is only ever used as
  a *label* (sanitized for display/metadata), never as a path component.
- Extension allow-list and a size cap are enforced before any parsing.
- Uploaded content is only ever read as text/parsed by a library - never
  executed.
"""

import hashlib
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import config
from . import chunking, document_store, embeddings, parsers
from .models import DocumentRecord, RAGError, UnsupportedFileTypeError

logger = config.get_logger(__name__)

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._\- ]+")


def sanitize_display_name(original_filename: str) -> str:
    """Strip any path components and unsafe characters. This is a label
    only - it is never used to build a filesystem path."""
    name = Path(original_filename).name  # drops any directory components, incl. ../
    name = _SAFE_NAME_RE.sub("_", name).strip()
    return name[:200] or "document"


def extension_of(file_name: str) -> str:
    return Path(file_name).suffix.lower()


def file_type_of(file_name: str) -> str:
    ext = extension_of(file_name)
    return ext.lstrip(".") if ext else ""


def validate_upload(file_name: str, size_bytes: int) -> None:
    ext = extension_of(file_name)
    if ext not in config.RAG_ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.RAG_ALLOWED_EXTENSIONS))
        raise UnsupportedFileTypeError(f"'{ext or '(no extension)'}' isn't supported. Allowed types: {allowed}")

    max_bytes = config.RAG_MAX_FILE_SIZE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise RAGError(f"File is {size_bytes / 1024 / 1024:.1f} MB, over the {config.RAG_MAX_FILE_SIZE_MB} MB limit.")
    if size_bytes == 0:
        raise RAGError("Uploaded file is empty.")


def compute_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def ingest_file(content: bytes, original_filename: str) -> Tuple[DocumentRecord, bool]:
    """Full pipeline for one uploaded file. Returns the final
    DocumentRecord (status "ready" or "failed" - failures are captured in
    the record, not raised, except for validation errors which reject the
    upload outright before anything is written to disk).

    Returns (record, is_duplicate)."""
    display_name = sanitize_display_name(original_filename)
    validate_upload(display_name, len(content))

    sha256 = compute_sha256(content)
    existing = document_store.find_by_hash(sha256)
    if existing:
        logger.info("Duplicate upload detected for '%s' -> existing document %s.", display_name, existing.document_id)
        return existing, True

    file_type = file_type_of(display_name)
    document_id = f"doc_{uuid.uuid4().hex[:16]}"
    ext = extension_of(display_name)
    stored_path = config.RAG_DOCUMENTS_DIR / f"{document_id}{ext}"
    stored_path.write_bytes(content)

    record = DocumentRecord(
        document_id=document_id,
        file_name=display_name,
        file_type=file_type,
        sha256=sha256,
        size_bytes=len(content),
        uploaded_at=datetime.now().isoformat(timespec="seconds"),
        status="processing",
        stored_path=stored_path.name,
    )
    document_store.add_document(record)
    logger.info("Upload accepted: %s (%s, %d bytes) -> %s", display_name, file_type, len(content), document_id)

    try:
        _process_document(record, stored_path)
    except RAGError as exc:
        logger.exception("Ingestion failed for %s (%s): %s", document_id, display_name, exc)
        document_store.update_document(document_id, status="failed", error=str(exc))
        return document_store.get_document(document_id), False
    except Exception as exc:  # noqa: BLE001 - any unexpected failure must not crash the request
        logger.exception("Unexpected ingestion failure for %s (%s): %s", document_id, display_name, exc)
        document_store.update_document(document_id, status="failed", error=f"Unexpected error: {exc}")
        return document_store.get_document(document_id), False

    return document_store.get_document(document_id), False


def _process_document(record: DocumentRecord, stored_path: Path) -> None:
    started = time.perf_counter()

    sections = parsers.parse_document(stored_path, record.file_type)
    page_count = max((s.page for s in sections if s.page), default=None)

    chunks = chunking.chunk_document(sections, record.document_id, record.file_name, record.file_type)
    if not chunks:
        raise RAGError("Document produced no chunks after splitting.")

    from . import vector_store

    vectors = embeddings.embed_texts([c.text for c in chunks])
    vector_store.add_chunks(chunks, vectors)

    document_store.update_document(
        record.document_id,
        status="ready",
        chunk_count=len(chunks),
        page_count=page_count,
        error=None,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "Indexed '%s' (%s): %d section(s) -> %d chunk(s) in %d ms.",
        record.file_name, record.document_id, len(sections), len(chunks), elapsed_ms,
    )


def delete_document(document_id: str) -> bool:
    record = document_store.get_document(document_id)
    if not record:
        return False

    from . import vector_store

    vector_store.delete_document(document_id)

    if record.stored_path:
        path = config.RAG_DOCUMENTS_DIR / record.stored_path
        path.unlink(missing_ok=True)

    document_store.delete_document(document_id)
    logger.info("Deleted document %s ('%s') - vectors and source file removed.", document_id, record.file_name)
    return True
