"""
agent.py
========
The LangGraph part of the SQL Query Agent: the State, the nodes, the
conditional edges, and the compiled graph.

Flow:

    START
      |
      v
   get_schema  (schema cache hit/miss - see cache.py)
      |
      v
  generate_sql  (LLM)
      |
      v
  validate_sql  (plain Python, no LLM - see database.py)
      |
      +-- invalid & retries left --> fix_sql (LLM) --> validate_sql (loop)
      |
      v (valid)
    run_sql
      |
      +-- error & retries left --> fix_sql (LLM) --> validate_sql (loop)
      |
      v (success, or retries exhausted)
  generate_answer  (LLM)
      |
      v
     END

STATE (this file) is only the data used during one graph run - including
which database (db_path) this run targets, since the platform supports
multiple databases.
MEMORY (memory.py) is conversation context carried across multiple runs.
CACHE (cache.py) is reused computation (the schema) shared across all runs,
keyed per database.
"""

from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

import config
from . import cache, database, prompts

logger = config.get_logger(__name__)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=config.LLM_TEMPERATURE,
    api_key=config.OPENAI_API_KEY,
)


def _ask_llm(prompt: str) -> str:
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class State(TypedDict):
    question: str
    db_path: str  # which SQLite file this run targets

    schema: str
    sql_query: str
    sql_result: Dict[str, Any]  # {"columns": [...], "rows": [...]}
    final_answer: str

    # Validation / execution / retry tracking
    sql_error: str
    validation_error: str
    retry_count: int
    query_valid: bool
    execution_success: bool

    # Conversation memory (built from memory.py before the graph runs)
    conversation_context: str

    # Small debug trail shown in the UI ("retry/fix information")
    retry_history: List[Dict[str, str]]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def get_schema(state: State) -> Dict[str, Any]:
    db_path = state["db_path"]

    cached_schema = cache.get_cached_schema(db_path)
    if cached_schema is not None:
        logger.info("Schema cache hit for '%s'.", db_path)
        return {"schema": cached_schema}

    logger.info("Schema cache miss for '%s' - inspecting database.", db_path)
    schema_text = database.get_schema_text(db_path)
    cache.set_cached_schema(db_path, schema_text)
    return {"schema": schema_text}


def generate_sql(state: State) -> Dict[str, Any]:
    logger.info("Generating SQL for question: %s", state["question"])

    prompt = prompts.build_sql_generation_prompt(
        question=state["question"],
        schema=state["schema"],
        conversation_context=state.get("conversation_context", ""),
    )
    raw_sql = _ask_llm(prompt)
    sql_query = database.strip_sql_markdown(raw_sql)

    logger.info("Generated SQL: %s", sql_query)
    return {
        "sql_query": sql_query,
        "retry_count": 0,
        "sql_error": "",
        "validation_error": "",
        "retry_history": [],
    }


def validate_sql(state: State) -> Dict[str, Any]:
    is_valid, error = database.validate_sql_query(state["sql_query"], state["db_path"])
    logger.info("Validation result: %s%s", "VALID" if is_valid else "INVALID", f" - {error}" if error else "")
    return {"query_valid": is_valid, "validation_error": error}


def run_sql(state: State) -> Dict[str, Any]:
    success, result = database.execute_readonly_query(state["sql_query"], state["db_path"])
    if success:
        logger.info("SQL executed successfully, %d row(s) returned.", len(result["rows"]))
        return {"sql_result": result, "execution_success": True, "sql_error": ""}

    logger.info("SQL execution failed: %s", result)
    return {"sql_result": {"columns": [], "rows": []}, "execution_success": False, "sql_error": result}


def fix_sql(state: State) -> Dict[str, Any]:
    new_retry_count = state.get("retry_count", 0) + 1
    error = state.get("validation_error") or state.get("sql_error") or "Unknown error"
    failed_sql = state["sql_query"]

    logger.info("Repair attempt %d/%d for error: %s", new_retry_count, config.MAX_RETRIES, error)

    prompt = prompts.build_sql_repair_prompt(
        question=state["question"],
        schema=state["schema"],
        failed_sql=failed_sql,
        error=error,
    )
    raw_sql = _ask_llm(prompt)
    fixed_sql = database.strip_sql_markdown(raw_sql)

    retry_history = list(state.get("retry_history", []))
    retry_history.append(
        {
            "attempt": str(new_retry_count),
            "failed_sql": failed_sql,
            "error": error,
            "fixed_sql": fixed_sql,
        }
    )

    return {
        "sql_query": fixed_sql,
        "retry_count": new_retry_count,
        "sql_error": "",
        "validation_error": "",
        "retry_history": retry_history,
    }


def generate_answer(state: State) -> Dict[str, Any]:
    if not state.get("execution_success", False):
        error = state.get("validation_error") or state.get("sql_error") or "Unknown error"
        answer = prompts.build_failure_message(state["question"], error, state.get("retry_count", 0))
        logger.info("Final result: FAILURE after %d retries.", state.get("retry_count", 0))
        return {"final_answer": answer}

    prompt = prompts.build_final_answer_prompt(
        question=state["question"],
        sql_query=state["sql_query"],
        sql_result=state["sql_result"],
    )
    answer = _ask_llm(prompt)
    logger.info("Final result: SUCCESS.")
    return {"final_answer": answer}


# ---------------------------------------------------------------------------
# Conditional edges (the routing logic)
# ---------------------------------------------------------------------------


def route_after_validation(state: State) -> Literal["run_sql", "fix_sql", "generate_answer"]:
    if state["query_valid"]:
        return "run_sql"
    if state.get("retry_count", 0) >= config.MAX_RETRIES:
        return "generate_answer"
    return "fix_sql"


def route_after_execution(state: State) -> Literal["generate_answer", "fix_sql"]:
    if state["execution_success"]:
        return "generate_answer"
    if state.get("retry_count", 0) >= config.MAX_RETRIES:
        return "generate_answer"
    return "fix_sql"


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

graph = StateGraph(State)

graph.add_node("get_schema", get_schema)
graph.add_node("generate_sql", generate_sql)
graph.add_node("validate_sql", validate_sql)
graph.add_node("fix_sql", fix_sql)
graph.add_node("run_sql", run_sql)
graph.add_node("generate_answer", generate_answer)

graph.add_edge(START, "get_schema")
graph.add_edge("get_schema", "generate_sql")
graph.add_edge("generate_sql", "validate_sql")

graph.add_conditional_edges(
    "validate_sql",
    route_after_validation,
    {
        "run_sql": "run_sql",
        "fix_sql": "fix_sql",
        "generate_answer": "generate_answer",
    },
)

# fix_sql always re-validates the query it just produced before running it.
graph.add_edge("fix_sql", "validate_sql")

graph.add_conditional_edges(
    "run_sql",
    route_after_execution,
    {
        "generate_answer": "generate_answer",
        "fix_sql": "fix_sql",
    },
)

graph.add_edge("generate_answer", END)

# A LangGraph-native checkpointer, keyed by thread_id, so the graph itself
# is resumable per conversation. The conversation *context* fed into
# prompts is still built explicitly in memory.py (kept as a separate,
# easy-to-read layer - see the note at the top of that file).
checkpointer = MemorySaver()

compiled_app = graph.compile(checkpointer=checkpointer)
