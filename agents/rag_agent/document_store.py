"""
document_store.py
==================
Persistent registry of uploaded/indexed documents (storage/rag/metadata/
documents.json). This is metadata only - the actual vectors live in Chroma
(vector_store.py) and the source files live in storage/rag/documents/.

Same in-process-dict-backed-by-a-file pattern as agents/sql_agent/memory.py
and agents/database_builder/builder.py's pending-build dict, except this
one needs to survive a restart (documents must still be listed/queryable
after the server restarts), so every mutation is written straight through
to disk. Fine at this scale - a real deployment would swap this for a
proper database without touching the four functions below.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import config
from .models import DocumentRecord

logger = config.get_logger(__name__)

_REGISTRY_PATH: Path = config.RAG_METADATA_DIR / "documents.json"


def _load() -> Dict[str, dict]:
    if not _REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(_REGISTRY_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        logger.warning("Could not read document registry at %s - starting empty.", _REGISTRY_PATH)
        return {}


def _save(data: Dict[str, dict]) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))


def add_document(record: DocumentRecord) -> None:
    data = _load()
    data[record.document_id] = record.to_storage_dict()
    _save(data)
    logger.info("Document registered: %s (%s, status=%s)", record.document_id, record.file_name, record.status)


def update_document(document_id: str, **fields) -> Optional[DocumentRecord]:
    data = _load()
    row = data.get(document_id)
    if row is None:
        return None
    row.update(fields)
    data[document_id] = row
    _save(data)
    return DocumentRecord.from_storage_dict(row)


def get_document(document_id: str) -> Optional[DocumentRecord]:
    row = _load().get(document_id)
    return DocumentRecord.from_storage_dict(row) if row else None


def list_documents(status: Optional[str] = None) -> List[DocumentRecord]:
    records = [DocumentRecord.from_storage_dict(row) for row in _load().values()]
    records.sort(key=lambda r: r.uploaded_at, reverse=True)
    if status:
        records = [r for r in records if r.status == status]
    return records


def find_by_hash(sha256: str) -> Optional[DocumentRecord]:
    for row in _load().values():
        if row.get("sha256") == sha256 and row.get("status") != "failed":
            return DocumentRecord.from_storage_dict(row)
    return None


def delete_document(document_id: str) -> bool:
    data = _load()
    if document_id not in data:
        return False
    del data[document_id]
    _save(data)
    logger.info("Document unregistered: %s", document_id)
    return True


def ready_document_ids() -> List[str]:
    return [r.document_id for r in list_documents(status="ready")]


def has_ready_documents() -> bool:
    return len(ready_document_ids()) > 0
