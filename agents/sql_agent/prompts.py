"""
prompts.py
==========
All SQL-agent prompt text lives here, kept as plain functions that return a
single string. agent.py just does:

    prompt = build_sql_generation_prompt(...)
    response = llm.invoke(prompt)

If you want to tweak wording, this is the only file you need to open.
"""

DIALECT_NOTES = """
This is a SQLite database. Keep these SQLite-specific rules in mind:
- Use standard SQL. SQLite supports SELECT, WITH, JOIN, GROUP BY, HAVING,
  ORDER BY, LIMIT, subqueries, and window functions.
- Dates are stored as TEXT (e.g. '2021-03-14'). Use SQLite date functions
  such as date(), strftime(), julianday() when working with dates.
- There is no BOOLEAN type; 0/1 integers are used instead.
- Use double quotes or square brackets only if a column name needs
  escaping; otherwise use plain identifiers.
""".strip()


def build_sql_generation_prompt(question: str, schema: str, conversation_context: str = "") -> str:
    context_block = ""
    if conversation_context:
        context_block = f"""
Conversation so far (for resolving references like "that department" or "them"):
{conversation_context}
""".rstrip()

    return f"""
You are an expert SQL analyst. Write a single SQLite query that answers the
user's question, using ONLY the tables and columns listed in the schema
below. Never invent tables or columns that are not listed.

Database schema:
{schema}

{DIALECT_NOTES}
{context_block}

Rules:
- Return ONLY the SQL query. No explanations, no markdown fences, no comments.
- The query must be a single read-only SELECT (or WITH ... SELECT) statement.
- Use JOINs correctly when a question needs data from more than one table.
- Use WHERE to filter rows before aggregation, and HAVING to filter after
  aggregation (e.g. filtering on an AVG() or COUNT() result).
- Use GROUP BY whenever aggregate functions (AVG, SUM, COUNT, MIN, MAX) are
  mixed with non-aggregated columns.
- Add ORDER BY / LIMIT when the question implies "highest", "lowest",
  "top N", etc.

User question:
{question}

SQL query:
""".strip()


def build_sql_repair_prompt(question: str, schema: str, failed_sql: str, error: str) -> str:
    return f"""
You are an expert SQL analyst. A SQLite query you previously wrote failed.
Fix it using ONLY the tables and columns listed in the schema below.

Database schema:
{schema}

{DIALECT_NOTES}

Original user question:
{question}

Query that failed:
{failed_sql}

Error message:
{error}

Rules:
- Return ONLY the corrected SQL query. No explanations, no markdown fences.
- The query must be a single read-only SELECT (or WITH ... SELECT) statement.
- Fix the specific problem described in the error message - do not rewrite
  unrelated parts of the query.

Corrected SQL query:
""".strip()


def build_final_answer_prompt(question: str, sql_query: str, sql_result: dict) -> str:
    columns = sql_result.get("columns", [])
    rows = sql_result.get("rows", [])

    return f"""
You are a helpful data analyst. Convert the database result below into a
clear, natural-English answer to the user's question. Be specific and use
the actual numbers from the result. If the result is empty, say plainly
that no matching data was found. Do not mention SQL, databases, tables, or
any internal process - just answer the question directly.

IMPORTANT: This system is strictly read-only - only SELECT/WITH queries are
ever run, and no data can be inserted, updated, or deleted through this
chat. If the user's question asked for a write/delete/update action (e.g.
"delete all the students"), do not offer to perform it, do not ask for
confirmation to modify or remove data, and never imply that a write is
pending or possible. Instead, simply present the matching data that was
found and, if relevant, note plainly that this mode can only read data, not
change it.

User question:
{question}

SQL query that was run:
{sql_query}

Result columns:
{columns}

Result rows:
{rows}

Answer:
""".strip()


def build_failure_message(question: str, error: str, retry_count: int) -> str:
    """Used when we run out of retries - no LLM call needed, just a clear,
    honest message instead of looping forever."""
    return (
        f"I couldn't find a working SQL query for that question after "
        f"{retry_count} attempt(s). The last error was: {error}. "
        "Try rephrasing the question, or check that it refers to data that "
        "actually exists in the selected database."
    )
