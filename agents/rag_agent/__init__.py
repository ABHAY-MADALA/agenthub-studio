"""RAG capability package.

The query service imports Chroma-backed retrieval, so expose it lazily.
That keeps lightweight imports such as ``document_store`` from crashing app
startup if the vector database is temporarily locked or unavailable.
"""

__all__ = ["rag_query"]


def rag_query(*args, **kwargs):
    from .service import rag_query as _rag_query

    return _rag_query(*args, **kwargs)
