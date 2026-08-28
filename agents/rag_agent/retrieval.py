"""
retrieval.py
============
Question -> embedding -> vector search -> (optional MMR rerank) -> top K
RetrievedChunk objects.

Structured as small composable steps on purpose (embed -> search -> rerank
-> build results) so hybrid retrieval can be added later without a
rewrite:

- BM25 / sparse retrieval -> another candidate-fetch step merged in
  _fetch_candidates()
- reranking (cross-encoder, cohere, etc.) -> swap _apply_mmr() for a
  different reorder step, or chain both
- query expansion / multi-query -> multiple calls into _fetch_candidates()
  before dedup
- metadata filtering -> already supported via document_ids

MMR (Maximal Marginal Relevance) is included since Chroma already returns
embeddings for the candidate set, so it's nearly free to add: it re-orders
the over-fetched candidates to balance relevance against redundancy, which
matters when a document repeats the same fact in several chunks.
"""

from typing import List, Optional

import numpy as np

import config
from . import vector_store
from .embeddings import embed_query
from .models import RetrievedChunk

logger = config.get_logger(__name__)


def build_retrieval_query(question: str, conversation_context: str) -> str:
    """Fold a little conversation context into the text that gets
    embedded, so a short follow-up like "What about sick leave?" doesn't
    retrieve on those four words alone. This does NOT replace retrieval -
    every question still runs a fresh vector search (see module docstring
    in agent.py) - it only improves what gets searched for.

    Only applied to short/pronoun-heavy questions, so a normal
    self-contained question isn't diluted with irrelevant prior context.
    """
    if not conversation_context:
        return question

    looks_like_followup = len(question.split()) <= 8 or any(
        word in question.lower().split() for word in ("it", "that", "this", "they", "them", "those")
    )
    if not looks_like_followup:
        return question

    return f"{conversation_context}\n\nCurrent question: {question}"


def retrieve(
    question: str,
    document_ids: Optional[List[str]] = None,
    k: Optional[int] = None,
    conversation_context: str = "",
) -> List[RetrievedChunk]:
    if document_ids is not None and len(document_ids) == 0:
        return []

    k = k or config.RAG_TOP_K
    retrieval_query = build_retrieval_query(question, conversation_context)
    query_embedding = embed_query(retrieval_query)

    fetch_n = k * config.RAG_MMR_FETCH_MULTIPLIER if config.RAG_USE_MMR else k
    raw = vector_store.query(query_embedding, n_results=fetch_n, document_ids=document_ids)

    candidates = _to_retrieved_chunks(raw)
    if not candidates:
        return []

    if config.RAG_USE_MMR and len(candidates) > k:
        embeddings = raw["embeddings"][0]
        ranked = _apply_mmr(query_embedding, candidates, embeddings, k, config.RAG_MMR_LAMBDA)
    else:
        ranked = candidates[:k]

    logger.info(
        "Retrieved %d chunk(s) for question (documents=%s, mmr=%s).",
        len(ranked), document_ids or "all", config.RAG_USE_MMR,
    )
    return ranked


def _to_retrieved_chunks(raw: dict) -> List[RetrievedChunk]:
    ids = raw["ids"][0]
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    chunks = []
    for chunk_id, text, meta, distance in zip(ids, documents, metadatas, distances):
        # Cosine distance in Chroma is 1 - cosine_similarity; clamp so a
        # tiny floating point overshoot never prints a negative score.
        score = max(0.0, min(1.0, 1.0 - distance))
        chunks.append(RetrievedChunk(
            chunk_id=chunk_id,
            document_id=meta.get("document_id", ""),
            file_name=meta.get("file_name", "unknown"),
            file_type=meta.get("file_type", ""),
            chunk_index=meta.get("chunk_index", 0),
            text=text,
            score=score,
            page=meta.get("page"),
        ))
    return chunks


def _apply_mmr(
    query_embedding: List[float],
    candidates: List[RetrievedChunk],
    candidate_embeddings: List[List[float]],
    k: int,
    lambda_mult: float,
) -> List[RetrievedChunk]:
    """Standard greedy MMR: repeatedly pick the candidate that maximizes
    (lambda * relevance_to_query) - ((1 - lambda) * max_similarity_to_already_picked)."""
    query_vec = np.array(query_embedding)
    cand_vecs = [np.array(e) for e in candidate_embeddings]

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        return float(np.dot(a, b) / denom)

    relevance = [cosine(query_vec, v) for v in cand_vecs]

    selected_idx: List[int] = []
    remaining_idx = list(range(len(candidates)))

    while remaining_idx and len(selected_idx) < k:
        best_idx, best_score = None, -1e9
        for i in remaining_idx:
            diversity_penalty = 0.0
            if selected_idx:
                diversity_penalty = max(cosine(cand_vecs[i], cand_vecs[j]) for j in selected_idx)
            mmr_score = lambda_mult * relevance[i] - (1 - lambda_mult) * diversity_penalty
            if mmr_score > best_score:
                best_idx, best_score = i, mmr_score
        selected_idx.append(best_idx)
        remaining_idx.remove(best_idx)

    return [candidates[i] for i in selected_idx]
