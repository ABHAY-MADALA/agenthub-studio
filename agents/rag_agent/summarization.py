"""
summarization.py
=================
"Summarize this document" needs a different path than normal Q&A
retrieval: fetching the top-k similar chunks to some generic "summarize"
query would just return a handful of arbitrary chunks and produce a
summary of a fifth of the document, not the whole thing.

Instead this reads every indexed chunk for the document (already stored
in Chroma, in order - see vector_store.get_document_chunks) and does a
map-reduce summary:

    chunks -> batch into ~RAG_SUMMARY_MAP_BATCH_TOKENS-sized groups
           -> summarize each batch (map)
           -> combine the batch summaries into one final summary (reduce)

Small documents that fit in a single batch skip straight to a one-pass
summary (still grounded, just without the extra map step).
"""

from typing import List

import tiktoken
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

import config
from . import prompts, vector_store
from .models import RAGError

logger = config.get_logger(__name__)

llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=config.LLM_TEMPERATURE,
    api_key=config.OPENAI_API_KEY,
)

_encoding = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_encoding.encode(text))


def _ask_llm(prompt: str) -> str:
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as exc:  # noqa: BLE001
        logger.exception("Summarization LLM call failed: %s", exc)
        raise RAGError(f"Summarization failed: {exc}") from exc


def _batch_chunks(documents: List[str], batch_token_budget: int) -> List[str]:
    """Group consecutive chunk texts into batches under the token budget,
    so each map call sends one coherent piece of the document."""
    batches: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for text in documents:
        text_tokens = _token_count(text)
        if current and current_tokens + text_tokens > batch_token_budget:
            batches.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(text)
        current_tokens += text_tokens

    if current:
        batches.append("\n\n".join(current))
    return batches


def summarize_document(document_id: str, file_name: str) -> str:
    stored = vector_store.get_document_chunks(document_id)
    documents = stored["documents"]
    if not documents:
        raise RAGError("This document has no indexed content to summarize.")

    full_text = "\n\n".join(documents)
    total_tokens = _token_count(full_text)

    if total_tokens <= config.RAG_SUMMARY_MAP_BATCH_TOKENS:
        logger.info("Summarizing '%s' in a single pass (%d tokens).", file_name, total_tokens)
        return _ask_llm(prompts.build_single_pass_summary_prompt(file_name, full_text))

    batches = _batch_chunks(documents, config.RAG_SUMMARY_MAP_BATCH_TOKENS)
    logger.info("Summarizing '%s' via map-reduce: %d batch(es), %d total tokens.", file_name, len(batches), total_tokens)

    partial_summaries = [
        _ask_llm(prompts.build_summary_map_prompt(file_name, batch))
        for batch in batches
    ]
    return _ask_llm(prompts.build_summary_reduce_prompt(file_name, partial_summaries))
