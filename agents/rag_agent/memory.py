"""
memory.py
=========
Bounded per-thread conversation memory for RAG Documents mode - same
pattern as agents/sql_agent/memory.py (see that file's docstring for the
State vs Memory vs Cache distinction). Kept as its own instance rather
than sharing the SQL agent's, since a "previous question" in Query
Database mode and a "previous question" in RAG mode aren't the same
conversation and shouldn't resolve each other's pronouns.

In-process only, like the rest of this project's chat memory - it does
not need to survive a restart (only the vector index and document
registry do).
"""

from dataclasses import dataclass, field
from typing import Dict, List

import config

logger = config.get_logger(__name__)


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class RAGConversationMemory:
    _threads: Dict[str, List[Turn]] = field(default_factory=dict)

    def add_turn(self, thread_id: str, question: str, answer: str) -> None:
        turns = self._threads.setdefault(thread_id, [])
        turns.append(Turn(question=question, answer=answer))
        if len(turns) > config.RAG_MAX_MEMORY_TURNS:
            del turns[: len(turns) - config.RAG_MAX_MEMORY_TURNS]
        logger.info("RAG memory updated for thread '%s' (%d turn(s) kept).", thread_id, len(turns))

    def get_context(self, thread_id: str) -> str:
        turns = self._threads.get(thread_id, [])
        if not turns:
            return ""
        lines = []
        for turn in turns:
            lines.append(f"Previous question: {turn.question}")
            lines.append(f"Previous answer: {turn.answer}")
        return "\n".join(lines)

    def clear(self, thread_id: str) -> None:
        self._threads.pop(thread_id, None)
        logger.info("RAG memory cleared for thread '%s'.", thread_id)


conversation_memory = RAGConversationMemory()
