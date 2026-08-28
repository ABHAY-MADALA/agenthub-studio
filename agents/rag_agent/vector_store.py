"""
vector_store.py
================
Persistent Chroma wrapper - the only module that talks to chromadb
directly. Everything else (retrieval.py, ingestion.py) calls the plain
functions below, so the vector database could be swapped later (e.g. for
a hosted Chroma, Qdrant, pgvector) without touching the rest of the RAG
pipeline.

One collection holds chunks from every document, distinguished by a
`document_id` metadata field. That makes "query all documents" the
default case (no filter) and "query specific documents" a metadata filter
($in), and it makes deleting a document's vectors a single
`delete(where={"document_id": ...})` call - no risk of stale per-document
collections being left behind.

Storage lives under storage/rag/chroma/ (see config.py) and is a
PersistentClient, so it survives a FastAPI restart - this is deliberately
not an in-memory client.
"""

from typing import Any, Dict, List, Optional

import chromadb

import config
from .models import Chunk, RAGError

logger = config.get_logger(__name__)

_COLLECTION_NAME = "agenthub_documents"

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection
    try:
        config.RAG_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.RAG_CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return _collection
    except Exception as exc:  # noqa: BLE001
        logger.exception("Vector store could not be opened at %s: %s", config.RAG_CHROMA_DIR, exc)
        raise RAGError(
            "Vector store is unavailable. Close any other running AgentHub server/process "
            f"using '{config.RAG_CHROMA_DIR}', then try again. Details: {exc}"
        ) from exc


def add_chunks(chunks: List[Chunk], embeddings: List[List[float]]) -> None:
    if not chunks:
        return
    if len(chunks) != len(embeddings):
        raise RAGError("Chunk/embedding count mismatch - refusing to write to the vector store.")
    try:
        collection = _get_collection()
        collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Vector store add() failed for %d chunk(s): %s", len(chunks), exc)
        raise RAGError(f"Vector store write failed: {exc}") from exc


def query(
    query_embedding: List[float],
    n_results: int,
    document_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    # None => search all docs. [] => search nothing (scoped chat with no attachments).
    if document_ids is not None and len(document_ids) == 0:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "embeddings": [[]],
        }
    where = {"document_id": {"$in": document_ids}} if document_ids is not None else None
    try:
        collection = _get_collection()
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Vector store query() failed: %s", exc)
        raise RAGError(f"Vector search failed: {exc}") from exc


def get_document_chunks(document_id: str) -> Dict[str, Any]:
    """All chunks for one document, in storage order - used by
    summarization.py (map-reduce needs the whole document, not a top-k
    similarity search)."""
    try:
        collection = _get_collection()
        result = collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Vector store get() failed for document %s: %s", document_id, exc)
        raise RAGError(f"Fetching document chunks failed: {exc}") from exc

    # Chroma doesn't guarantee ordering - sort by chunk_index so
    # map-reduce summarization reads the document front-to-back.
    rows = list(zip(result["metadatas"], result["documents"]))
    rows.sort(key=lambda pair: pair[0].get("chunk_index", 0))
    return {
        "metadatas": [r[0] for r in rows],
        "documents": [r[1] for r in rows],
    }


def delete_document(document_id: str) -> None:
    try:
        collection = _get_collection()
        collection.delete(where={"document_id": document_id})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Vector store delete() failed for document %s: %s", document_id, exc)
        raise RAGError(f"Deleting vectors failed: {exc}") from exc
    logger.info("Deleted vectors for document %s.", document_id)


def count() -> int:
    return _get_collection().count()
