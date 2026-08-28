"""
memory.py
=========
MEMORY = conversation context carried across multiple user turns within one
chat session. (Compare to STATE, which is only the data used during one
graph run - see the State TypedDict in agent.py.)

A dict of thread_id -> list of past turns (question, sql_query, final
answer), kept to the last N turns (config.MAX_MEMORY_TURNS) so prompts
don't grow forever. This is what makes follow-up questions like
"How many employees are in that department?" resolve correctly.

Swappable later for SQLite/Postgres/Redis without touching agent.py, since
agent.py (and app.py) only ever call the four methods on
ConversationMemory below.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import config

logger = config.get_logger(__name__)


@dataclass
class Turn:
    question: str
    sql_query: str
    final_answer: str


@dataclass
class ConversationMemory:
    """Bounded per-thread conversation history, kept entirely in-process."""

    _threads: Dict[str, List[Turn]] = field(default_factory=dict)

    def add_turn(self, thread_id: str, question: str, sql_query: str, final_answer: str) -> None:
        turns = self._threads.setdefault(thread_id, [])
        turns.append(Turn(question=question, sql_query=sql_query, final_answer=final_answer))

        # Bounded strategy: only keep the most recent MAX_MEMORY_TURNS turns.
        if len(turns) > config.MAX_MEMORY_TURNS:
            del turns[: len(turns) - config.MAX_MEMORY_TURNS]

        logger.info("Memory updated for thread '%s' (%d turn(s) kept).", thread_id, len(turns))

    def get_context(self, thread_id: str) -> str:
        """Format the last few turns into plain text the SQL-generation
        prompt can use to resolve references like 'that department'."""
        turns = self._threads.get(thread_id, [])
        if not turns:
            return ""

        lines = []
        for turn in turns:
            lines.append(f"Previous question: {turn.question}")
            lines.append(f"Previous SQL: {turn.sql_query}")
            lines.append(f"Previous answer: {turn.final_answer}")
        return "\n".join(lines)

    def clear(self, thread_id: str) -> None:
        self._threads.pop(thread_id, None)
        logger.info("Memory cleared for thread '%s'.", thread_id)


# One shared instance used by the whole app (per-process, in-memory).
# Swappable later for a persistent backend without changing its interface.
conversation_memory = ConversationMemory()
