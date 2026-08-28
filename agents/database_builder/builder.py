"""
builder.py
==========
Orchestrates the Database Builder flow end to end:

    English description
          |
          v
    schema_generator.generate_schema()   -> schema dict
          |
          v
    format_schema_preview()              -> markdown shown to the user
          |
          v
    (user clicks Approve in the UI)
          |
          v
    db_creator.create_database()         -> writes the .db file
          |
          v (if "include sample data" was checked)
    schema_generator.generate_sample_data() -> db_creator.insert_sample_data()

This module holds no UI code and no LangGraph - app.py calls straight into
these functions and keeps the "pending schema" for one chat thread in a
plain dict (see PendingBuild below), since this is a short two-step
wizard, not a multi-turn agent loop.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import config
from . import db_creator, schema_generator
from .schema_generator import SchemaGenerationError

logger = config.get_logger(__name__)


@dataclass
class PendingBuild:
    """One in-progress Database Builder proposal, awaiting user approval.
    Kept per chat thread so different tabs/sessions never collide."""

    schema: dict
    description: str
    suggested_db_name: str


# thread_id -> PendingBuild, in-process only (matches the rest of the MVP's
# in-memory state - see agents/sql_agent/memory.py for the same pattern).
_pending_builds: Dict[str, PendingBuild] = {}


def format_schema_preview(schema: dict) -> str:
    """Render the schema as readable markdown for the chat panel:
    tables, columns, PK/FK markers."""
    lines = [f"**Proposed database:** `{schema.get('database_name', 'database')}.db`\n"]

    for table in schema["tables"]:
        lines.append(f"### {table['name']}")
        fk_by_column = {fk["column"]: fk for fk in table.get("foreign_keys", [])}

        for col in table["columns"]:
            markers = []
            if col.get("primary_key"):
                markers.append("PK")
            if col["name"] in fk_by_column:
                fk = fk_by_column[col["name"]]
                markers.append(f'FK -> {fk["ref_table"]}.{fk["ref_column"]}')
            if col.get("not_null") and not col.get("primary_key"):
                markers.append("NOT NULL")
            # Backtick-styled rather than underscore-italics: schema marker text
            # like "FK -> members.member_id" contains underscores itself, which
            # would collide with markdown's underscore-emphasis syntax.
            marker_text = f" `{', '.join(markers)}`" if markers else ""
            lines.append(f"- `{col['name']}` ({col.get('type', 'TEXT')}){marker_text}")
        lines.append("")

    lines.append(
        "Reply **approve** to create this database, describe a change "
        "(e.g. \"also add a library table\") to revise it, or **cancel** to discard it."
    )
    return "\n".join(lines)


def start_build(thread_id: str, description: str) -> str:
    """First step: description -> schema -> preview text. Stores the
    pending proposal for this thread and returns the preview markdown."""
    schema = schema_generator.generate_schema(description)
    suggested_name = db_creator.sanitize_db_name(schema.get("database_name", "database"))

    _pending_builds[thread_id] = PendingBuild(
        schema=schema, description=description, suggested_db_name=suggested_name
    )
    return format_schema_preview(schema)


def refine_build(thread_id: str, description: str) -> str:
    """Follow-up step: apply a change to the pending schema and return the
    updated preview. Requires start_build() to have been called first."""
    pending = _pending_builds.get(thread_id)
    if pending is None:
        return start_build(thread_id, description)

    schema = schema_generator.refine_schema(description, pending.schema)
    suggested_name = db_creator.sanitize_db_name(schema.get("database_name", pending.suggested_db_name))
    _pending_builds[thread_id] = PendingBuild(
        schema=schema, description=description, suggested_db_name=suggested_name
    )
    return format_schema_preview(schema)


def has_pending_build(thread_id: str) -> bool:
    return thread_id in _pending_builds


def get_pending_schema(thread_id: str) -> Optional[dict]:
    pending = _pending_builds.get(thread_id)
    return pending.schema if pending else None


def cancel_build(thread_id: str) -> None:
    _pending_builds.pop(thread_id, None)


def approve_build(thread_id: str, include_sample_data: bool, rows_per_table: int = 5) -> Dict[str, object]:
    """Second step, called after the user approves: actually create the
    .db file (and optionally seed it). Returns a summary dict for the UI:
    {"db_name", "db_path", "table_count", "sample_data_note"}."""
    pending = _pending_builds.get(thread_id)
    if pending is None:
        raise SchemaGenerationError("There is no pending database proposal to approve.")

    db_name = db_creator.unique_db_name(pending.suggested_db_name)
    db_path = db_creator.create_database(pending.schema, db_name)

    sample_data_note = "No sample data was added."
    if include_sample_data:
        try:
            data = schema_generator.generate_sample_data(pending.schema, rows_per_table)
            counts = db_creator.insert_sample_data(db_path, pending.schema, data)
            total_rows = sum(counts.values())
            sample_data_note = f"Inserted {total_rows} sample row(s) across {len(counts)} table(s)."
        except SchemaGenerationError as exc:
            sample_data_note = f"Database created, but sample data generation failed: {exc}"

    _pending_builds.pop(thread_id, None)

    return {
        "db_name": db_name,
        "db_path": db_path,
        "table_count": len(pending.schema["tables"]),
        "sample_data_note": sample_data_note,
    }
