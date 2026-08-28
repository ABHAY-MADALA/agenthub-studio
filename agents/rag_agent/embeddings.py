"""
embeddings.py
=============
Thin, reusable wrapper around the project's OpenAI embedding model - one
place that owns the OpenAIEmbeddings client, instead of every caller
constructing its own (same pattern as the ChatOpenAI instances in
general_chat.py / sql_agent/agent.py / schema_generator.py).
"""

import hashlib
import math
import re
from typing import List

from langchain_openai import OpenAIEmbeddings

import config

logger = config.get_logger(__name__)

_embeddings = OpenAIEmbeddings(
    model=config.RAG_EMBEDDING_MODEL,
    api_key=config.OPENAI_API_KEY,
)

_LOCAL_EMBEDDING_DIM = 1536
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of chunk texts (ingestion path)."""
    if not texts:
        return []
    if not config.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY is not set; using local deterministic RAG embeddings.")
        return [_local_embedding(text) for text in texts]
    try:
        return _embeddings.embed_documents(texts)
    except Exception as exc:  # noqa: BLE001 - keep uploads usable when embeddings are unavailable
        logger.warning("Embedding %d text(s) failed; falling back to local embeddings: %s", len(texts), exc)
        return [_local_embedding(text) for text in texts]


def embed_query(text: str) -> List[float]:
    """Embed a single user question (retrieval path)."""
    if not config.OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY is not set; using local deterministic RAG query embedding.")
        return _local_embedding(text)
    try:
        return _embeddings.embed_query(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embedding query failed; falling back to local embedding: %s", exc)
        return _local_embedding(text)


def _local_embedding(text: str) -> List[float]:
    """Deterministic hashed bag-of-words fallback.

    This is not as semantically strong as OpenAI embeddings, but it keeps
    uploads/query retrieval working in offline demos and preserves the same
    vector dimension as text-embedding-3-small.
    """
    vector = [0.0] * _LOCAL_EMBEDDING_DIM
    tokens = _TOKEN_RE.findall((text or "").lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % _LOCAL_EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]
