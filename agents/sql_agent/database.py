"""
database.py
============
Everything that touches SQLite directly for the Query Database feature:

- list_available_databases() -> every .db file under storage/databases/
- get_schema_text()          -> inspects a DB and returns a readable schema
- strip_sql_markdown()       -> removes ```sql fences the LLM sometimes adds
- validate_sql_query()       -> read-only safety + "does it even parse" checks
- execute_readonly_query()   -> actually runs a SELECT and returns rows

None of this file knows about LangGraph or the LLM - it is plain,
testable database code. agent.py calls into it from graph nodes.

This module is intentionally multi-database aware: every function takes an
explicit db_path instead of relying on one global path, since the platform
lets users create and switch between several SQLite databases.
"""

import os
import re
import sqlite3
from typing import List, Tuple

import config

logger = config.get_logger(__name__)

# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = None  # rows come back as plain tuples
    return conn


# ---------------------------------------------------------------------------
# Database discovery (for the UI's database selector)
# ---------------------------------------------------------------------------


def list_available_databases() -> List[str]:
    """Return the file names (not full paths) of every .db file in
    storage/databases/, sorted alphabetically."""
    if not config.DATABASES_DIR.exists():
        return []
    names = [f.name for f in config.DATABASES_DIR.iterdir() if f.suffix == ".db"]
    return sorted(names)


def db_name_to_path(db_name: str) -> str:
    return str(config.DATABASES_DIR / db_name)


# ---------------------------------------------------------------------------
# Bundled demo database (so Query Database mode has something to query on
# first launch, before the user has built anything with Database Builder)
# ---------------------------------------------------------------------------


def ensure_sample_database(db_path: str = None) -> None:
    """Create storage/databases/sample_company.db with a departments /
    employees schema and sample rows, but ONLY if it doesn't already
    exist. Mirrors the reference SQL_LANGGRAPH_PROJECT's demo dataset."""
    db_path = db_path or config.DEFAULT_DB_PATH
    if os.path.exists(db_path):
        return

    logger.info("No sample database found at '%s' - creating one.", db_path)

    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE departments (
            department_id   INTEGER PRIMARY KEY,
            department_name TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE employees (
            employee_id        INTEGER PRIMARY KEY,
            employee_name       TEXT NOT NULL,
            age                 INTEGER,
            gender              TEXT,
            job_title           TEXT,
            salary              INTEGER,
            bonus               INTEGER,
            hire_date           TEXT,
            city                TEXT,
            performance_rating  REAL,
            department_id       INTEGER,
            FOREIGN KEY (department_id) REFERENCES departments (department_id)
        )
        """
    )

    departments = [
        (1, "Data Science"), (2, "Engineering"), (3, "Sales"),
        (4, "Marketing"), (5, "Human Resources"), (6, "Finance"),
    ]
    cur.executemany("INSERT INTO departments VALUES (?, ?)", departments)

    employees = [
        (1, "Alice Johnson", 29, "Female", "Data Scientist", 95000, 12000, "2021-03-14", "Austin", 4.8, 1),
        (2, "Brian Lee", 34, "Male", "ML Engineer", 88000, 9000, "2020-07-01", "Austin", 4.5, 1),
        (3, "Carla Gomez", 41, "Female", "Data Science Manager", 120000, 20000, "2018-01-22", "Seattle", 4.9, 1),
        (4, "David Kim", 26, "Male", "Data Analyst", 71000, 5000, "2022-05-10", "Austin", 4.1, 1),
        (5, "Ethan Brooks", 31, "Male", "Software Engineer", 98000, 11000, "2019-09-15", "Seattle", 4.6, 2),
        (6, "Fiona Chen", 28, "Female", "Software Engineer", 96000, 10000, "2021-11-01", "Denver", 4.4, 2),
        (7, "George Patel", 45, "Male", "Engineering Manager", 135000, 25000, "2016-02-20", "Seattle", 4.7, 2),
        (8, "Hannah White", 24, "Female", "Junior Developer", 65000, 3000, "2023-01-09", "Denver", 3.9, 2),
        (9, "Ian Rodriguez", 37, "Male", "Sales Executive", 62000, 18000, "2019-06-18", "Chicago", 4.2, 3),
        (10, "Julia Nguyen", 33, "Female", "Sales Manager", 90000, 22000, "2017-08-30", "Chicago", 4.6, 3),
        (11, "Kevin Ortiz", 29, "Male", "Sales Rep", 55000, 9000, "2022-02-14", "Chicago", 3.8, 3),
        (12, "Laura Silva", 40, "Female", "Marketing Manager", 92000, 15000, "2018-04-05", "New York", 4.5, 4),
        (13, "Marcus Wright", 27, "Male", "Marketing Specialist", 61000, 6000, "2021-10-11", "New York", 4.0, 4),
        (14, "Nina Adams", 35, "Female", "HR Manager", 84000, 8000, "2019-01-07", "New York", 4.4, 5),
        (15, "Oscar Diaz", 30, "Male", "HR Coordinator", 58000, 4000, "2022-09-19", "Denver", 3.7, 5),
        (16, "Priya Sharma", 38, "Female", "Finance Manager", 110000, 17000, "2017-12-01", "Chicago", 4.8, 6),
        (17, "Quentin Moore", 32, "Male", "Financial Analyst", 79000, 7000, "2020-03-23", "Chicago", 4.3, 6),
        (18, "Rachel Kim", 26, "Female", "Data Analyst", 70000, 5500, "2022-06-06", "Seattle", 4.0, 1),
        (19, "Samuel Turner", 44, "Male", "Principal Engineer", 145000, 28000, "2015-05-19", "Seattle", 4.9, 2),
        (20, "Tara Wilson", 31, "Female", "Sales Executive", 63000, 19000, "2020-11-25", "Austin", 4.3, 3),
    ]
    cur.executemany(
        """
        INSERT INTO employees
            (employee_id, employee_name, age, gender, job_title, salary,
             bonus, hire_date, city, performance_rating, department_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        employees,
    )

    conn.commit()
    conn.close()
    logger.info(
        "Sample database created with %d departments and %d employees.",
        len(departments), len(employees),
    )


# ---------------------------------------------------------------------------
# Schema inspection (raw introspection - caching lives in cache.py)
# ---------------------------------------------------------------------------


def get_schema_text(db_path: str) -> str:
    """Inspect the SQLite database and return a human/LLM readable schema
    description, e.g.:

        Table: departments
        - department_id (INTEGER) (PRIMARY KEY)
        - department_name (TEXT)

        Table: employees
        - employee_id (INTEGER) (PRIMARY KEY)
        ...
        Foreign keys:
        - employees.department_id -> departments.department_id
    """
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    table_names = [row[0] for row in cur.fetchall()]

    lines: List[str] = []
    foreign_key_lines: List[str] = []

    for table_name in table_names:
        lines.append(f"Table: {table_name}")

        cur.execute(f"PRAGMA table_info({table_name})")
        for _cid, col_name, col_type, _notnull, _default, pk in cur.fetchall():
            pk_marker = " (PRIMARY KEY)" if pk else ""
            lines.append(f"- {col_name} ({col_type}){pk_marker}")

        cur.execute(f"PRAGMA foreign_key_list({table_name})")
        for fk in cur.fetchall():
            # fk columns: (id, seq, table, from, to, on_update, on_delete, match)
            ref_table, from_col, to_col = fk[2], fk[3], fk[4]
            foreign_key_lines.append(f"- {table_name}.{from_col} -> {ref_table}.{to_col}")

        lines.append("")  # blank line between tables

    if foreign_key_lines:
        lines.append("Foreign keys:")
        lines.extend(foreign_key_lines)

    conn.close()
    return "\n".join(lines).strip()


def get_schema_summary(db_path: str) -> dict:
    """Structured version of the schema for the Schema Viewer panel:
    {"tables": [{"name", "columns": [{"name","type","pk","notnull"}], "foreign_keys": [...]}]}."""
    if not os.path.exists(db_path):
        return {"tables": []}

    conn = get_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    table_names = [row[0] for row in cur.fetchall()]

    tables = []
    for table_name in table_names:
        cur.execute(f"PRAGMA table_info({table_name})")
        columns = [
            {"name": col_name, "type": col_type, "pk": bool(pk), "notnull": bool(notnull)}
            for _cid, col_name, col_type, notnull, _default, pk in cur.fetchall()
        ]

        cur.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = [
            {"column": fk[3], "ref_table": fk[2], "ref_column": fk[4]} for fk in cur.fetchall()
        ]

        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cur.fetchone()[0]

        tables.append({
            "name": table_name, "columns": columns, "foreign_keys": foreign_keys,
            "row_count": row_count,
        })

    conn.close()
    return {"tables": tables}


# ---------------------------------------------------------------------------
# Cleaning LLM output
# ---------------------------------------------------------------------------


def strip_sql_markdown(text: str) -> str:
    """Remove ```sql ... ``` (or plain ``` ... ```) fences the LLM sometimes
    wraps its answer in, and trim whitespace / trailing semicolons issues."""
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# SQL safety validation (Query Database mode: read-only, always)
# ---------------------------------------------------------------------------

# Keywords that would modify the database or its schema. Blocked outright.
_FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE",
    "TRUNCATE", "ATTACH", "DETACH", "VACUUM", "REINDEX", "GRANT", "REVOKE",
]

_FORBIDDEN_PATTERN = re.compile(
    r"\b(" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE
)


def validate_sql_query(sql_query: str, db_path: str) -> Tuple[bool, str]:
    """Read-only safety + sanity checks for a generated SQL query.

    Returns (is_valid, error_message). error_message is "" when valid.
    This is a plain Python function (not an LLM call) so it is fast,
    deterministic, and cannot be talked out of its rules.
    """
    if not sql_query or not sql_query.strip():
        return False, "Generated SQL is empty."

    query = sql_query.strip()

    # Only SELECT / WITH ... SELECT statements are allowed to start a query.
    # (This also rejects PRAGMA outright - the LLM never needs it since the
    # schema is already provided in the prompt.)
    if not re.match(r"^\s*(SELECT|WITH)\b", query, re.IGNORECASE):
        return False, "Only SELECT (or WITH ... SELECT) queries are allowed in Query Database mode."

    # Block multiple statements (e.g. "SELECT ...; DROP ...").
    # A single trailing semicolon is fine; anything after it is not.
    body = query.rstrip(";").strip()
    if ";" in body:
        return False, "Multiple SQL statements are not allowed."

    # Block dangerous/modifying keywords anywhere in the query.
    match = _FORBIDDEN_PATTERN.search(body)
    if match:
        return False, f"Query contains a disallowed keyword: {match.group(1).upper()}."

    # Ask SQLite to plan the query without executing it. This catches most
    # syntax errors AND invalid table/column references, without ever
    # running the query or touching data.
    try:
        conn = get_connection(db_path)
        conn.execute(f"EXPLAIN QUERY PLAN {body}")
        conn.close()
    except sqlite3.Error as exc:
        return False, f"SQLite could not validate this query: {exc}"

    return True, ""


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def execute_readonly_query(sql_query: str, db_path: str) -> Tuple[bool, object]:
    """Run a validated read-only query.

    Returns (success, result). On success, result is
    {"columns": [...], "rows": [...]}. On failure, result is the SQLite
    error message.
    """
    try:
        conn = get_connection(db_path)
        cur = conn.cursor()
        cur.execute(sql_query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        conn.close()
        return True, {"columns": columns, "rows": rows}
    except sqlite3.Error as exc:
        return False, str(exc)
