"""
agent.py
========
The LangGraph part of the RAG Agent: the State, the nodes, and the
compiled graph. Same shape as agents/sql_agent/agent.py (State ->
StateGraph -> linear nodes), scaled down: RAG's control flow doesn't need
sql_agent's validate/retry loop, so there are no conditional edges here -
just prepare -> retrieve -> generate -> format.

Flow:

    START
      |
      v
  prepare_question   (detect "summarize this document" vs normal Q&A)
      |
      v
  retrieve_context    (vector search, or full-document fetch for summarize)
      |
      v
  generate_answer     (grounded LLM answer, or map-reduce summary)
      |
      v
  format_sources       (build the structured sources/debug the API returns)
      |
      v
     END

No checkpointer: unlike the SQL agent this graph never needs to resume
mid-run, and cross-turn memory is handled explicitly by memory.py (passed
in as conversation_context) - a checkpointer here would be ceremony with
no behavior behind it.
"""

import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

import config
from . import document_store, prompts, retrieval, summarization, vector_store
from .models import RetrievedChunk
from .state import RAGState

logger = config.get_logger(__name__)

llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=config.LLM_TEMPERATURE,
    api_key=config.OPENAI_API_KEY,
)

# Document-overview asks should use the full-document summarization path,
# not a single top-k chunk that may miss the real heading/title.
_SUMMARIZE_RE = re.compile(
    r"(?:"
    r"\bsummar(y|ies|ize|ise|ising|izing|isation|ization)\b"
    r"|\b(?:give me|what's|whats|what is|provide|suggest)\b.{0,40}\b(?:heading|title|headline)\b"
    r"|\b(?:heading|title|headline)\b.{0,40}\b(?:for|of|to)\b.{0,20}\b(?:this|the|it|file|document|pdf)\b"
    r"|\bwhat (?:is|are) this\b.{0,40}\b(?:file|document|pdf|about)\b"
    r"|\bwhat is this (?:file|document|pdf)\b"
    r"|\bwhat is this about\b"
    r"|\boverview of (?:this|the)\b"
    r")",
    re.IGNORECASE,
)


def _ask_llm(prompt: str) -> str:
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def prepare_question(state: RAGState) -> Dict[str, Any]:
    question = state["question"]
    document_ids = state.get("document_ids") or None

    if not _SUMMARIZE_RE.search(question):
        return {"intent": "qa", "target_document": None}

    target = _resolve_summary_target(document_ids)
    if target is None:
        # Ambiguous target (zero or multiple candidate documents) - fall
        # back to normal retrieval rather than guessing which document.
        logger.info("Summarize intent detected but target document is ambiguous - falling back to Q&A retrieval.")
        return {"intent": "qa", "target_document": None}

    return {"intent": "summarize", "target_document": target}


def _resolve_summary_target(document_ids: Optional[List[str]]) -> Optional[Dict[str, str]]:
    candidate_ids = document_ids or document_store.ready_document_ids()
    if len(candidate_ids) != 1:
        return None
    record = document_store.get_document(candidate_ids[0])
    if not record or record.status != "ready":
        return None
    return {"document_id": record.document_id, "file_name": record.file_name}


def retrieve_context(state: RAGState) -> Dict[str, Any]:
    started = time.perf_counter()

    if state["intent"] == "summarize":
        target = state["target_document"]
        stored = vector_store.get_document_chunks(target["document_id"])
        chunks = [
            RetrievedChunk(
                chunk_id=f"{target['document_id']}:{i}",
                document_id=meta.get("document_id", ""),
                file_name=meta.get("file_name", target["file_name"]),
                file_type=meta.get("file_type", ""),
                chunk_index=meta.get("chunk_index", i),
                text=text,
                score=1.0,  # not a similarity score - the whole document was used
                page=meta.get("page"),
            )
            for i, (meta, text) in enumerate(zip(stored["metadatas"], stored["documents"]))
        ]
    else:
        document_ids = state.get("document_ids") or None
        chunks = retrieval.retrieve(
            question=state["question"],
            document_ids=document_ids,
            conversation_context=state.get("conversation_context", ""),
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    documents_searched = len(state.get("document_ids") or document_store.ready_document_ids())

    return {
        "retrieved_chunks": chunks,
        "retrieval_count": len(chunks),
        "documents_searched": documents_searched,
        "top_scores": [round(c.score, 4) for c in chunks[:5]],
        "latency_ms": {**state.get("latency_ms", {}), "retrieve": elapsed_ms},
    }


def generate_answer(state: RAGState) -> Dict[str, Any]:
    started = time.perf_counter()
    chunks: List[RetrievedChunk] = state["retrieved_chunks"]

    if state["intent"] == "summarize":
        target = state["target_document"]
        try:
            answer = summarization.summarize_document(target["document_id"], target["file_name"])
        except Exception as exc:  # noqa: BLE001
            logger.exception("Summarization failed: %s", exc)
            answer = f"Sorry, I couldn't summarize '{target['file_name']}': {exc}"

    elif not chunks:
        answer = prompts.build_no_context_message(document_store.has_ready_documents())

    else:
        context_dicts = [c.to_context_dict() for c in chunks]
        prompt = prompts.build_rag_answer_prompt(
            question=state["question"],
            context_chunks=context_dicts,
            conversation_context=state.get("conversation_context", ""),
        )
        try:
            answer = _ask_llm(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG answer generation failed; returning retrieved excerpts: %s", exc)
            answer = _fallback_chunk_answer(chunks)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "answer": answer,
        "latency_ms": {**state.get("latency_ms", {}), "generate": elapsed_ms},
    }


def _fallback_chunk_answer(chunks: List[RetrievedChunk]) -> str:
    lines = [
        "I found relevant source text, but the answer-generation model is unavailable right now. "
        "Here are the strongest retrieved excerpts:"
    ]
    for chunk in chunks[:3]:
        source = chunk.file_name
        if chunk.page:
            source += f", page {chunk.page}"
        preview = chunk.text.strip().replace("\n", " ")
        if len(preview) > 500:
            preview = preview[:500].rsplit(" ", 1)[0] + "..."
        lines.append(f"\n**{source}**\n{preview}")
    return "\n".join(lines)


def format_sources(state: RAGState) -> Dict[str, Any]:
    chunks: List[RetrievedChunk] = state["retrieved_chunks"]
    return {
        "sources": [c.to_source_dict() for c in chunks],
        "context_used": [c.to_context_dict() for c in chunks],
    }


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

graph = StateGraph(RAGState)

graph.add_node("prepare_question", prepare_question)
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("generate_answer", generate_answer)
graph.add_node("format_sources", format_sources)

graph.add_edge(START, "prepare_question")
graph.add_edge("prepare_question", "retrieve_context")
graph.add_edge("retrieve_context", "generate_answer")
graph.add_edge("generate_answer", "format_sources")
graph.add_edge("format_sources", END)

compiled_app = graph.compile()


# ---------------------------------------------------------------------------
# Public entrypoint - what app.py calls
# ---------------------------------------------------------------------------


def run(
    question: str,
    document_ids: Optional[List[str]] = None,
    conversation_context: str = "",
) -> Dict[str, Any]:
    """Runs the graph and returns a plain dict ready to serialize into the
    API response. Kept separate from the FastAPI route so it's easy to
    call from a script/test without spinning up the server."""
    started = time.perf_counter()

    initial_state: RAGState = {
        "question": question,
        "conversation_context": conversation_context,
        "document_ids": document_ids,
        "intent": "qa",
        "target_document": None,
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "context_used": [],
        "retrieval_count": 0,
        "documents_searched": 0,
        "top_scores": [],
        "latency_ms": {},
    }

    result = compiled_app.invoke(initial_state)
    total_ms = int((time.perf_counter() - started) * 1000)

    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "retrieved_chunks": result["context_used"],
        "intent": result["intent"],
        "debug": {
            "retrieval_count": result["retrieval_count"],
            "documents_searched": result["documents_searched"],
            "top_scores": result["top_scores"],
            "latency_ms": {**result["latency_ms"], "total": total_ms},
            "intent": result["intent"],
        },
    }
