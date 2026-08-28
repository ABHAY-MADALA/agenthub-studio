"""
cache.py
========
CACHE = reusable computation we don't want to redo on every request.

This is deliberately tiny: a dictionary with an optional time-to-live (TTL).
The main thing we cache is database schema text, keyed by db_path, since it
rarely changes and re-inspecting SQLite (sqlite_master + PRAGMA table_info
for every table) on every single question is wasted work.

This is intentionally NOT Redis or anything external - just an in-process
dict, scoped per database file. If you later want a shared/multi-process
cache (e.g. Redis), you can swap the internals of SimpleCache without
touching agent.py, because agent.py only calls get_cached_schema() /
set_cached_schema() / clear_schema_cache().
"""

import time
from typing import Any, Dict, Optional, Tuple

import config

logger = config.get_logger(__name__)


class SimpleCache:
    """A minimal in-process cache: dict + optional TTL per entry."""

    def __init__(self, ttl_seconds: Optional[float] = None):
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None

        stored_at, value = entry
        if self.ttl_seconds is not None and (time.time() - stored_at) > self.ttl_seconds:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)

    def clear(self, key: Optional[str] = None) -> None:
        if key is None:
            self._store.clear()
        else:
            self._store.pop(key, None)


# ---------------------------------------------------------------------------
# Schema cache: keyed by db_path, never expires automatically (schema
# rarely changes), but is explicitly cleared whenever a database is
# (re)created by the Database Builder, or manually via clear_schema_cache().
# ---------------------------------------------------------------------------
_schema_cache = SimpleCache(ttl_seconds=None)


def get_cached_schema(db_path: str) -> Optional[str]:
    return _schema_cache.get(db_path)


def set_cached_schema(db_path: str, schema_text: str) -> None:
    _schema_cache.set(db_path, schema_text)


def clear_schema_cache(db_path: Optional[str] = None) -> None:
    _schema_cache.clear(db_path)
    logger.info("Schema cache cleared%s.", f" for {db_path}" if db_path else " (all databases)")


# ---------------------------------------------------------------------------
# NOTE on query-result caching (deliberately NOT implemented):
# The database can change between questions (especially right after the
# Database Builder creates or seeds one), so caching *query results* would
# risk returning stale answers. If this is ever added, it should use its
# own SimpleCache instance with a short TTL, a cache key of
# (db_path, sql_query), and an easy on/off switch - never share the schema
# cache instance above for that purpose.
#
# NOTE on swapping in Redis later:
# Replace SimpleCache's internals (get/set/clear) with redis-py calls
# behind the exact same three functions above. Nothing in agent.py,
# app.py, or the database_builder package would need to change, because
# they only ever import this module's functions, never SimpleCache
# directly.
# ---------------------------------------------------------------------------
