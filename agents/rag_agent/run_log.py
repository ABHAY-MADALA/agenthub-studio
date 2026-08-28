"""
run_log.py
==========
Persists the exact data from every real RAG execution so later evaluation
(e.g. RAGAS) can score the original answer + retrieved contexts without
re-running retrieval.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

logger = config.get_logger(__name__)

RAG_RUNS_DIR = config.RAG_DIR / "runs"
RAG_RUNS_DIR.mkdir(parents=True, exist_ok=True)


def persist_run(
    *,
    question: str,
    answer: str,
    retrieved_contexts: List[Dict[str, Any]],
    sources: List[Dict[str, Any]],
    document_ids: Optional[List[str]],
    conversation_id: Optional[str],
    debug: Dict[str, Any],
) -> str:
    run_id = uuid.uuid4().hex
    payload = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "document_ids": document_ids,
        "conversation_id": conversation_id,
        "retrieved_contexts": retrieved_contexts,
        "sources": sources,
        "debug": debug,
    }
    path = RAG_RUNS_DIR / f"{run_id}.json"
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to persist RAG run log: %s", exc)
    return run_id
