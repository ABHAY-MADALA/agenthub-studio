"""
chunking.py
===========
List[ParsedSection] -> List[Chunk]. Uses LangChain's recursive character
splitter with a tiktoken-based length function, so chunk_size/overlap are
measured in tokens (matches the embedding model's own units) rather than
raw characters. This is the "good default" strategy - it splits on
paragraph/sentence boundaries first and only falls back to harder splits
when a piece is still too big, so chunks stay coherent instead of cutting
mid-sentence at a fixed character count.

Chunk index is global across the whole document (not reset per page/
section) so citations and debug output have one stable ordering.
"""

import uuid
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from .models import Chunk, ParsedSection

logger = config.get_logger(__name__)

_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=config.RAG_CHUNK_SIZE_TOKENS,
    chunk_overlap=config.RAG_CHUNK_OVERLAP_TOKENS,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Chunks shorter than this (in characters) are almost never useful on
# their own - fold them into the previous chunk instead of storing a
# near-empty vector.
_MIN_CHUNK_CHARS = 40


def chunk_document(sections: List[ParsedSection], document_id: str, file_name: str, file_type: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunk_index = 0

    for section in sections:
        pieces = _splitter.split_text(section.text)
        pieces = _merge_tiny_trailing_piece(pieces)

        for piece in pieces:
            if not piece.strip():
                continue
            chunks.append(Chunk(
                chunk_id=f"{document_id}:{uuid.uuid4().hex[:12]}",
                document_id=document_id,
                file_name=file_name,
                file_type=file_type,
                chunk_index=chunk_index,
                text=piece.strip(),
                page=section.page,
            ))
            chunk_index += 1

    logger.info("Chunked '%s' into %d chunk(s).", file_name, len(chunks))
    return chunks


def _merge_tiny_trailing_piece(pieces: List[str]) -> List[str]:
    """Avoid a lone tiny fragment at the end of a section (e.g. one short
    leftover sentence) becoming its own chunk - fold it into the previous
    piece instead."""
    if len(pieces) < 2:
        return pieces
    if len(pieces[-1].strip()) < _MIN_CHUNK_CHARS:
        merged = pieces[:-2] + [pieces[-2] + "\n" + pieces[-1]]
        return merged
    return pieces
