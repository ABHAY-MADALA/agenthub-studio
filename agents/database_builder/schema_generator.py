"""
schema_generator.py
====================
English description -> structured SQLite schema (as plain Python
dicts/lists, matching the JSON shape documented in prompts.py).

This is a single LLM call plus strict JSON parsing/validation - no
LangGraph here, the Database Builder flow is simple enough (generate ->
human approves -> create) that a graph would add ceremony without adding
clarity. See builder.py for the approval flow.
"""

import json
import re

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

import config
from . import prompts

logger = config.get_logger(__name__)

llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=config.LLM_TEMPERATURE,
    api_key=config.OPENAI_API_KEY,
)


class SchemaGenerationError(Exception):
    """Raised when the LLM's schema (or sample data) response can't be
    parsed or doesn't match the expected shape."""


def _ask_llm(prompt: str) -> str:
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json_object(raw_text: str, context: str) -> dict:
    cleaned = _strip_json_fences(raw_text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s JSON: %s\nRaw text: %s", context, exc, raw_text)
        raise SchemaGenerationError(
            f"The AI's {context} response wasn't valid JSON. Try rephrasing your request."
        ) from exc


def _validate_schema_shape(schema: dict) -> None:
    if not isinstance(schema, dict) or "tables" not in schema:
        raise SchemaGenerationError("Schema response is missing a top-level 'tables' list.")

    tables = schema["tables"]
    if not isinstance(tables, list) or not tables:
        raise SchemaGenerationError("Schema response must contain at least one table.")

    for table in tables:
        if "name" not in table or "columns" not in table:
            raise SchemaGenerationError("Every table needs a 'name' and 'columns'.")
        if not table["columns"]:
            raise SchemaGenerationError(f"Table '{table.get('name')}' has no columns.")
        pk_count = sum(1 for col in table["columns"] if col.get("primary_key"))
        if pk_count == 0:
            # Be lenient: promote the first column to primary key rather
            # than failing the whole build over a minor LLM omission.
            table["columns"][0]["primary_key"] = True
            logger.warning(
                "Table '%s' had no primary key - defaulting first column to PK.", table["name"]
            )
        table.setdefault("foreign_keys", [])

    if not schema.get("database_name"):
        schema["database_name"] = "database"


def generate_schema(description: str) -> dict:
    """English description -> validated schema dict."""
    logger.info("Generating schema for description: %s", description)
    prompt = prompts.build_schema_generation_prompt(description)
    raw = _ask_llm(prompt)
    schema = _parse_json_object(raw, "schema")
    _validate_schema_shape(schema)
    logger.info("Generated schema with %d table(s).", len(schema["tables"]))
    return schema


def refine_schema(description: str, current_schema: dict) -> dict:
    """Apply a follow-up change ('also add a library table') to an
    existing pending schema and return the full updated schema."""
    logger.info("Refining schema with instruction: %s", description)
    prompt = prompts.build_schema_refinement_prompt(description, current_schema)
    raw = _ask_llm(prompt)
    schema = _parse_json_object(raw, "schema")
    _validate_schema_shape(schema)
    return schema


def generate_sample_data(schema: dict, rows_per_table: int = 5) -> dict:
    """schema dict -> {table_name: [row_dict, ...]}."""
    logger.info("Generating sample data (~%d rows/table).", rows_per_table)
    prompt = prompts.build_sample_data_prompt(schema, rows_per_table)
    raw = _ask_llm(prompt)
    data = _parse_json_object(raw, "sample data")
    if not isinstance(data, dict):
        raise SchemaGenerationError("Sample data response must be a JSON object keyed by table name.")
    return data
