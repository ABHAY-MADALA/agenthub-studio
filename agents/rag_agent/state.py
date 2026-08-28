"""
state.py
========
STATE for one RAG graph run (compare to MEMORY in memory.py, which is
conversation context carried across runs - same distinction the SQL
agent draws in its own agent.py docstring).
"""

from typing import Any, Dict, List, Optional, TypedDict


class RAGState(TypedDict):
    question: str
    conversation_context: str
    document_ids: Optional[List[str]]

    intent: str  # "qa" | "summarize"
    target_document: Optional[Dict[str, Any]]  # set when intent == "summarize"

    retrieved_chunks: List[Any]  # List[RetrievedChunk]
    answer: str
    sources: List[Dict[str, Any]]
    context_used: List[Dict[str, Any]]  # full (untruncated) context, for future RAGAS eval

    # Debug / observability
    retrieval_count: int
    documents_searched: int
    top_scores: List[float]
    latency_ms: Dict[str, int]
