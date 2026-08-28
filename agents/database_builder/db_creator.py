"""
db_creator.py
=============
Turns a validated schema dict (see schema_generator.py) into an actual
SQLite .db file: CREATE TABLE statements, then optional sample-data
INSERTs. Plain sqlite3 - no LLM calls in this file.

Foreign-key-aware: tables are created (and seeded) in dependency order via
a simple topological sort, so a "enrollments" table referencing "students"
never gets created/seeded before "students" does.
"""

import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, List

import config

logger = config.get_logger(__name__)

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_]+")


def sanitize_db_name(raw_name: str) -> str:
    """Turn free-text (e.g. 'College DB!') into a safe file name
    ('college_db.db'), always ending in .db and never containing path
    separators - this file name is used directly under storage/databases/."""
    base = os.path.basename(raw_name or "database")
    base = base[:-3] if base.lower().endswith(".db") else base
    base = _SAFE_NAME_RE.sub("_", base).strip("_").lower()
    if not base:
        base = "database"
    return f"{base}.db"


def unique_db_name(desired_name: str) -> str:
    """Avoid clobbering an existing database - append _2, _3, ... if the
    name is already taken."""
    name = sanitize_db_name(desired_name)
    candidate = name
    counter = 2
    while (config.DATABASES_DIR / candidate).exists():
        candidate = name[:-3] + f"_{counter}.db"
        counter += 1
    return candidate


# ---------------------------------------------------------------------------
# Dependency ordering
# ---------------------------------------------------------------------------


def topological_order(schema: dict) -> List[dict]:
    """Order tables so a table only appears after every table its foreign
    keys point to. Falls back to the original order for any cycle (rare
    for LLM-generated demo schemas, but must never crash the build)."""
    tables = schema["tables"]
    by_name = {t["name"]: t for t in tables}

    visited: Dict[str, str] = {}  # name -> "visiting" | "done"
    ordered: List[dict] = []

    def visit(name: str) -> None:
        state = visited.get(name)
        if state == "done":
            return
        if state == "visiting":
            return  # cycle - break it, table will still be added below
        table = by_name.get(name)
        if table is None:
            return
        visited[name] = "visiting"
        for fk in table.get("foreign_keys", []):
            ref = fk.get("ref_table")
            if ref and ref != name:
                visit(ref)
        visited[name] = "done"
        ordered.append(table)

    for table in tables:
        visit(table["name"])

    return ordered


# ---------------------------------------------------------------------------
# CREATE TABLE generation
# ---------------------------------------------------------------------------


def build_create_statements(schema: dict) -> List[str]:
    statements = []
    for table in topological_order(schema):
        col_lines = []
        pk_columns = [c["name"] for c in table["columns"] if c.get("primary_key")]
        single_inline_pk = len(pk_columns) == 1

        for col in table["columns"]:
            parts = [col["name"], col.get("type", "TEXT")]
            if single_inline_pk and col["name"] == pk_columns[0]:
                parts.append("PRIMARY KEY")
            elif col.get("not_null"):
                parts.append("NOT NULL")
            col_lines.append(" ".join(parts))

        if not single_inline_pk and pk_columns:
            col_lines.append(f"PRIMARY KEY ({', '.join(pk_columns)})")

        for fk in table.get("foreign_keys", []):
            col_lines.append(
                f'FOREIGN KEY ({fk["column"]}) REFERENCES {fk["ref_table"]}({fk["ref_column"]})'
            )

        columns_sql = ",\n    ".join(col_lines)
        statements.append(f'CREATE TABLE {table["name"]} (\n    {columns_sql}\n)')

    return statements


# ---------------------------------------------------------------------------
# Database creation
# ---------------------------------------------------------------------------


def create_database(schema: dict, db_name: str) -> str:
    """Create the .db file with all tables. Returns the full path.
    Raises if a file with that name already exists (call unique_db_name
    first if you want auto-renaming instead of an error)."""
    db_path = str(config.DATABASES_DIR / db_name)
    if os.path.exists(db_path):
        raise FileExistsError(f"A database named '{db_name}' already exists.")

    statements = build_create_statements(schema)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for stmt in statements:
            cur.execute(stmt)
        conn.commit()
    except sqlite3.Error:
        conn.close()
        # Don't leave a half-created, broken .db file behind.
        if os.path.exists(db_path):
            os.remove(db_path)
        raise
    else:
        conn.close()

    logger.info("Created database '%s' with %d table(s).", db_name, len(schema["tables"]))
    return db_path


def insert_sample_data(db_path: str, schema: dict, data: Dict[str, list]) -> Dict[str, int]:
    """Insert LLM-generated sample rows, table by table in dependency
    order. Skips (and logs) any table whose rows fail to insert rather
    than aborting the whole database - a demo database with 4/5 tables
    seeded is far better than none at all."""
    inserted_counts: Dict[str, int] = {}
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for table in topological_order(schema):
            table_name = table["name"]
            rows = data.get(table_name, [])
            if not rows:
                inserted_counts[table_name] = 0
                continue

            column_names = [c["name"] for c in table["columns"]]
            placeholders = ", ".join(["?"] * len(column_names))
            columns_sql = ", ".join(column_names)
            insert_sql = f"INSERT INTO {table_name} ({columns_sql}) VALUES ({placeholders})"

            values = [tuple(row.get(col) for col in column_names) for row in rows]
            try:
                cur.executemany(insert_sql, values)
                inserted_counts[table_name] = len(values)
            except sqlite3.Error as exc:
                logger.warning("Sample data insert failed for table '%s': %s", table_name, exc)
                inserted_counts[table_name] = 0
        conn.commit()
    finally:
        conn.close()

    return inserted_counts


def delete_database(db_name: str) -> str:
    """Delete one application-owned SQLite database and return its path.

    Only plain .db files directly inside storage/databases/ are eligible.
    """
    if not db_name:
        raise ValueError("Database name is required.")
    if Path(db_name).name != db_name or not db_name.lower().endswith(".db"):
        raise ValueError("Only application database files can be deleted.")

    databases_dir = config.DATABASES_DIR.resolve()
    db_path = (config.DATABASES_DIR / db_name).resolve()
    if db_path.parent != databases_dir:
        raise ValueError("Only application database files can be deleted.")
    if not db_path.exists():
        raise FileNotFoundError(f"Database '{db_name}' no longer exists.")
    if not db_path.is_file():
        raise ValueError("Only application database files can be deleted.")

    db_path.unlink()
    logger.info("Deleted database '%s'.", db_name)
    return str(db_path)
