"""
prompts.py
==========
Prompt text for the Database Builder feature: turning an English
description into a structured SQLite schema, and (optionally) into small
sample data for that schema. Kept as plain functions, same style as
agents/sql_agent/prompts.py.
"""

import json

SCHEMA_JSON_INSTRUCTIONS = """
Respond with ONLY a single JSON object (no markdown fences, no commentary)
in exactly this shape:

{
  "database_name": "short_snake_case_name",
  "tables": [
    {
      "name": "table_name",
      "columns": [
        {"name": "column_name", "type": "INTEGER|TEXT|REAL", "primary_key": true, "not_null": true}
      ],
      "foreign_keys": [
        {"column": "column_name", "ref_table": "other_table", "ref_column": "other_table_pk"}
      ]
    }
  ]
}

Rules:
- "database_name" should be a short (2-4 word) snake_case name describing
  the domain, e.g. "college_db" or "ecommerce_store".
- Use snake_case for table and column names.
- Every table needs exactly one primary key column, typically an
  auto-incrementing "<table>_id" INTEGER column.
- Only use SQLite types: INTEGER, TEXT, REAL.
- Add foreign_keys entries whenever a table logically references another
  (e.g. an "enrollments" table referencing "students" and "courses").
- Design a normalized relational schema appropriate for the description -
  create as many tables as make sense (typically 3-8 for a demo).
- Do not include any explanation - JSON only.
""".strip()


def build_schema_generation_prompt(description: str) -> str:
    return f"""
You are a database architect. Design a normalized SQLite schema for the
following request from a user.

User request:
{description}

{SCHEMA_JSON_INSTRUCTIONS}
""".strip()


def build_schema_refinement_prompt(description: str, current_schema: dict) -> str:
    return f"""
You are a database architect. A user already has this SQLite schema (as
JSON) and wants to change it based on new instructions. Apply the change
and return the FULL updated schema (not just the diff).

Current schema:
{json.dumps(current_schema, indent=2)}

User's requested change:
{description}

{SCHEMA_JSON_INSTRUCTIONS}
""".strip()


def build_sample_data_prompt(schema: dict, rows_per_table: int = 5) -> str:
    return f"""
You are generating small, realistic sample data for a demo SQLite database
with this schema (JSON):

{json.dumps(schema, indent=2)}

Generate about {rows_per_table} rows per table. Respond with ONLY a single
JSON object (no markdown fences, no commentary) mapping each table name to
a list of row objects, e.g.:

{{
  "table_name": [
    {{"column_name": "value", "...": "..."}}
  ]
}}

Rules:
- Include every column for every row.
- Primary key columns should be sequential integers starting at 1, unless
  the column type is TEXT (then use a short readable unique id/code).
- Foreign key columns MUST reference primary key values that you actually
  generated for the parent table (valid, in-range ids).
- Use realistic, varied, human-readable sample values (names, dates as
  'YYYY-MM-DD' text, etc.) - this is for a live demo, so keep it tidy and
  plausible, not random gibberish.
- JSON only, no explanation.
""".strip()
