"""
service.py
==========
Clean callable surface for the RAG capability.

Auto / planners should call `rag_query(...)` rather than assembling
memory + agent.run themselves. This module owns conversation memory
lookup/update and exact-run persistence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import agent, memory, run_log


def rag_query(
    question: str,
    document_ids: Optional[List[str]] = None,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a grounded RAG query.

    Args:
        question: User question (or summarization request).
        document_ids: Optional scope. ``None`` searches every ready
            document; an empty list searches nothing; a non-empty list
            searches only those ids.
        conversation_id: Optional thread id for follow-up memory.

    Returns:
        ``{answer, sources, retrieved_contexts, debug}``
    """
    conversation_context = ""
    if conversation_id:
        conversation_context = memory.conversation_memory.get_context(conversation_id)

    result = agent.run(
        question=question,
        document_ids=document_ids,
        conversation_context=conversation_context,
    )

    if conversation_id:
        memory.conversation_memory.add_turn(conversation_id, question, result["answer"])

    retrieved_contexts = result.get("retrieved_chunks") or []
    sources = result.get("sources") or []
    debug = dict(result.get("debug") or {})
    debug["run_id"] = run_log.persist_run(
        question=question,
        answer=result["answer"],
        retrieved_contexts=retrieved_contexts,
        sources=sources,
        document_ids=document_ids,
        conversation_id=conversation_id,
        debug=debug,
    )

    return {
        "answer": result["answer"],
        "sources": sources,
        "retrieved_contexts": retrieved_contexts,
        "retrieved_chunks": retrieved_contexts,  # backward-compatible alias
        "debug": debug,
        "intent": result.get("intent"),
    }
