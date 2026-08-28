"""
app.py
======
AgentHub Studio's HTTP API and web UI host (FastAPI). This file is the
"product" - a professional AI workspace with a sidebar (navigation,
recent chats), a center chat workspace, and a right-hand inspector panel
(generated SQL, query results, schema, debug/retry, sources). The UI
itself is a static single-page app under static/ that talks to the JSON
endpoints defined here.

No agent logic lives here. This file only:
- keeps a stable thread/session id per browser tab
- resolves the active mode (explicit, or via router.py in Auto mode)
- dispatches to the right module for that mode:
    Build Database  -> agents/database_builder/builder.py
    Query Database  -> agents/sql_agent/agent.py (compiled LangGraph app)
    General Chat    -> agents/general_chat.py
- serializes results for the frontend (chat reply, SQL, result table,
  schema, timing/validation, sources, debug info)
"""

import ast
import calendar
import operator
import os
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, List, Optional

import holidays as holidays_lib

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import orchestrator
import router
from agents import general_chat
from agents.database_builder import builder, db_creator
from agents.database_builder.schema_generator import SchemaGenerationError
from agents.rag_agent import document_store as rag_documents
from agents.rag_agent import ingestion as rag_ingestion
from agents.rag_agent import rag_query as execute_rag_query
from agents.rag_agent.models import RAGError
from agents.sql_agent import database as sql_database
from agents.sql_agent import cache as sql_cache
from agents.sql_agent import memory as sql_memory
from agents.sql_agent.agent import compiled_app
from tools.mcp import oauth
from tools.mcp.registry import ALL_TOOLS, sidebar_lines

logger = config.get_logger(__name__)

STATIC_DIR = config.BASE_DIR / "static"

# Make sure there's a database to query on first launch.
sql_database.ensure_sample_database()

if not config.OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set - LLM calls will fail until it's configured in .env")

MODES = ["Auto", "Build Database", "Query Database", "General Chat", "RAG"]
# Modes listed in MODES but not yet implemented would go here (disabled in
# the UI's mode dropdown but still visible as a "coming soon" option). RAG
# now has a real backend (agents/rag_agent/) so it's no longer gated.
UNAVAILABLE_MODES = set()

BUILD_EXAMPLES = [
    "Create a database for a college with students, professors, courses, enrollments, and grades.",
    "Build a database for an e-commerce store with customers, products, orders, and order items.",
]
QUERY_EXAMPLES = [
    "Which department has the highest average salary?",
    "Who are the top 3 highest paid employees?",
    "How many employees work in Engineering?",
]

# Tool label -> (MCPTool, ToolAction), for the MCP Tools page's "try it" demo.
_MCP_ACTION_MAP = {
    f"{tool.display_name}: {action.description}": (tool, action)
    for tool in ALL_TOOLS
    for action in tool.actions
}
_PENDING_MCP_ACTIONS = {}
_THREAD_AUTO_CONTEXT = {}
# Auto waits for missing info (database / document / email recipient) without
# asking the user to switch modes. Shape includes conversation_id, goal,
# missing_fields, field_values, observations, pending_write, and status.
_THREAD_PENDING_GOAL: dict = {}
# Per-thread memory of name/relation labels the user has already mapped to a
# concrete contact (e.g. "dad" -> "drmadala99@gmail.com"). General on purpose:
# any label the user supplies works, not just a fixed list like "dad"/"mom".
# Persists for the rest of the conversation (until /api/new-chat resets it),
# independent of any single pending-goal's lifecycle.
# Shape: {thread_id: {normalized_label: email}}
_THREAD_KNOWN_CONTACTS: dict = {}

APPROVE_WORDS = {
    "approve", "approved", "yes", "y", "confirm", "approve it", "yes approve",
    "looks good", "looks good approve", "send it", "go ahead", "go ahead and send",
    "do it",
}
CANCEL_WORDS = {"cancel", "no", "n", "discard", "cancel it", "no thanks", "do not send", "don't send"}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def new_thread_id() -> str:
    thread_id = str(uuid.uuid4())
    logger.info("New thread: %s", thread_id)
    return thread_id


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def db_choices() -> list:
    return sql_database.list_available_databases()


def default_db_choice() -> Optional[str]:
    choices = db_choices()
    if config.DEFAULT_DB_NAME in choices:
        return config.DEFAULT_DB_NAME
    return choices[0] if choices else None


def result_to_table(sql_result: dict) -> dict:
    columns = sql_result.get("columns", [])
    rows = sql_result.get("rows", [])
    return {"columns": columns, "rows": [list(row) for row in rows]}


def format_debug_info(retry_history: list, validation_error: str, retry_count: int) -> str:
    if not retry_history:
        return "No retries were needed - the SQL worked on the first try."

    lines = [f"Query needed {retry_count} repair attempt(s):"]
    for entry in retry_history:
        lines.append(f"\nAttempt {entry['attempt']}:")
        lines.append(f"  Failed SQL: {entry['failed_sql']}")
        lines.append(f"  Error: {entry['error']}")
        lines.append(f"  Fixed SQL: {entry['fixed_sql']}")
    return "\n".join(lines)


def format_rag_debug_info(debug: dict) -> str:
    lines = [
        f"Intent: {debug.get('intent', 'qa')}",
        f"Chunks retrieved: {debug.get('retrieval_count', 0)}",
        f"Documents searched: {debug.get('documents_searched', 0)}",
    ]
    if debug.get("top_scores"):
        lines.append("Top scores: " + ", ".join(str(s) for s in debug["top_scores"]))
    latency = debug.get("latency_ms") or {}
    if latency:
        lines.append("Latency: " + ", ".join(f"{stage}={ms}ms" for stage, ms in latency.items()))
    return "\n".join(lines)


def schema_summary_safe(db_name: Optional[str]) -> dict:
    if not db_name:
        return {"db_name": None, "tables": []}
    db_path = sql_database.db_name_to_path(db_name)
    summary = sql_database.get_schema_summary(db_path)
    return {"db_name": db_name, "tables": summary["tables"]}


def sources_for_query(db_name: str, sql_query: str, tables: list) -> list:
    """Which tables in db_name's schema were actually referenced by
    sql_query, derived by matching table names as whole words - real,
    derived provenance rather than a fabricated "sources" list."""
    referenced = []
    for table in tables:
        name = table["name"]
        if re.search(rf"\b{re.escape(name)}\b", sql_query, re.IGNORECASE):
            referenced.append({"table": name, "db_name": db_name, "row_count": table.get("row_count", 0)})
    return referenced


def resolve_effective_mode(mode_label: str, message: str, db_name: Optional[str]) -> str:
    if mode_label != "Auto":
        return mode_label
    decision = orchestrator.decide(
        message, has_selected_database=bool(db_name), has_rag_documents=rag_documents.has_ready_documents(),
    )
    return {
        "build": "Build Database",
        "query": "Query Database",
        "general": "General Chat",
        "mcp": "MCP Tool",
        "rag": "RAG",
    }[decision.kind]


def resolve_auto_decision(mode_label: str, message: str, db_name: Optional[str]) -> Optional[orchestrator.Decision]:
    if mode_label != "Auto":
        return None
    return orchestrator.decide(
        message, has_selected_database=bool(db_name), has_rag_documents=rag_documents.has_ready_documents(),
    )


def find_mcp_tool_action(tool_key: str, action_name: str):
    tool = next((t for t in ALL_TOOLS if t.key == tool_key), None)
    if not tool:
        return None, None
    action = next((a for a in tool.actions if a.name == action_name), None)
    return tool, action


def normalized_control_text(message: str) -> str:
    return re.sub(r"\s+", " ", (message or "").strip().lower()).strip(" .!,")


def is_approval_text(message: str) -> bool:
    return normalized_control_text(message) in APPROVE_WORDS


def is_cancel_text(message: str) -> bool:
    return normalized_control_text(message) in CANCEL_WORDS


def _safe_pending_args(tool_key: str, action_name: str, instruction: str) -> Optional[dict]:
    if tool_key == "gmail" and action_name in {"send_email", "draft_reply"}:
        from tools.mcp.gmail_tool import parse_email_request

        email_request = parse_email_request(instruction)
        if not email_request:
            return None
        return {
            "to_email": email_request.to_email,
            "subject": email_request.subject,
            "body": email_request.body,
        }
    if tool_key == "google_calendar" and action_name == "create_meeting":
        from tools.mcp.google_calendar_tool import parse_calendar_event_request

        event_request = parse_calendar_event_request(instruction)
        if not event_request:
            return None
        return {
            "summary": event_request.summary,
            "start": event_request.start.isoformat(),
            "end": event_request.end.isoformat(),
            "attendee_email": event_request.attendee_email,
        }
    return {}


def _call_tool_with_payload(tool, action, *, confirmed: bool, instruction: str, arguments: Optional[dict]) -> str:
    kwargs = {"instruction": instruction}
    if tool.key == "gmail" and action.name in {"send_email", "draft_reply"} and arguments:
        kwargs["email_request"] = arguments
    if tool.key == "google_calendar" and action.name == "create_meeting" and arguments:
        kwargs["event_request"] = arguments
    return tool.call(action.name, confirmed=confirmed, **kwargs)


def store_pending_write(
    thread_id: str,
    *,
    tool,
    action,
    goal: str,
    instruction: str,
    preview: str,
    mode_path: str,
    observations: Optional[list] = None,
) -> bool:
    arguments = _safe_pending_args(tool.key, action.name, instruction)
    if arguments is None:
        return False
    _PENDING_MCP_ACTIONS[thread_id] = {
        "conversation_id": thread_id,
        "tool_key": tool.key,
        "action_name": action.name,
        "status": "awaiting_approval",
        "idempotency_key": str(uuid.uuid4()),
        "goal": goal,
        "original_message": goal,
        "instruction": instruction,
        "arguments": arguments,
        "preview": preview,
        "mode_path": mode_path,
        "observations": observations or [],
    }
    pending_goal = _THREAD_PENDING_GOAL.get(thread_id)
    if pending_goal:
        pending_goal["pending_write"] = _PENDING_MCP_ACTIONS[thread_id]
        pending_goal["status"] = "awaiting_approval"
    else:
        _THREAD_PENDING_GOAL[thread_id] = {
            "conversation_id": thread_id,
            "status": "awaiting_approval",
            "missing": None,
            "missing_fields": [],
            "field_values": {},
            "observations": observations or [],
            "pending_write": _PENDING_MCP_ACTIONS[thread_id],
            "goal": goal,
            "message": goal,
            "intent": "pending_write",
        }
    return True


def execute_pending_write(thread_id: str) -> tuple[str, str, bool, str]:
    pending = _PENDING_MCP_ACTIONS.get(thread_id)
    if not pending:
        return "There is no pending tool action to approve.", "Agentic Auto", False, "No pending write action."
    if pending.get("status") != "awaiting_approval":
        return "That pending tool action is already being handled.", pending.get("mode_path") or "Agentic Auto", False, "Duplicate approval ignored."

    pending["status"] = "executing"
    tool, action = find_mcp_tool_action(pending["tool_key"], pending["action_name"])
    mode_used = pending.get("mode_path") or "Agentic Auto -> MCP Tool"
    try:
        if not tool or not action:
            return "That pending tool action is no longer available.", mode_used, False, "Pending write action was unavailable."
        tool.connected = oauth.is_tool_connected(tool.key)
        reply = _call_tool_with_payload(
            tool,
            action,
            confirmed=True,
            instruction=pending.get("instruction", pending.get("original_message", "")),
            arguments=pending.get("arguments") or {},
        )
        success = tool_output_success(reply)
        debug = (
            f"Agentic approval executed pending write {pending['tool_key']}.{pending['action_name']} "
            f"idempotency_key={pending['idempotency_key']} status={'success' if success else 'failure'}."
        )
        return reply, mode_used, success, debug
    except Exception as exc:  # noqa: BLE001 - report the real tool/API error
        logger.exception("Pending write execution failed: %s", exc)
        return f"Sorry, the approved action failed: {exc}", mode_used, False, f"Pending write failed: {exc}"
    finally:
        _PENDING_MCP_ACTIONS.pop(thread_id, None)
        pending_goal = _THREAD_PENDING_GOAL.get(thread_id)
        if pending_goal and pending_goal.get("status") in {"awaiting_approval", "executing"}:
            _THREAD_PENDING_GOAL.pop(thread_id, None)


def do_approve(thread_id: str, include_sample_data: bool):
    """Returns (reply_text, success, new_db_name)."""
    try:
        summary = builder.approve_build(thread_id, include_sample_data)
        reply = (
            f"Created **{summary['db_name']}** with {summary['table_count']} table(s). "
            f"{summary['sample_data_note']} It's selected in the Database dropdown — "
            f"ask your question in Auto and I can query it for you."
        )
        return reply, True, summary["db_name"]
    except Exception as exc:  # noqa: BLE001 - surface any failure to the user
        logger.exception("Database creation failed: %s", exc)
        return f"Sorry, I couldn't create the database: {exc}", False, None


def do_cancel(thread_id: str) -> str:
    builder.cancel_build(thread_id)
    return "Build cancelled - the proposed schema was discarded."


def _rag_scope_readiness(document_ids: Optional[List[str]]) -> str:
    """Return one of: ready, missing, indexing, failed."""
    indexing_statuses = {"uploading", "indexing", "processing", "pending"}

    if document_ids is not None:
        if not document_ids:
            return "missing"
        statuses = []
        for doc_id in document_ids:
            record = rag_documents.get_document(doc_id)
            statuses.append(record.status if record else "missing")
        if all(status == "ready" for status in statuses):
            return "ready"
        if any(status in indexing_statuses for status in statuses):
            return "indexing"
        if any(status == "failed" for status in statuses):
            return "failed"
        if any(status == "ready" for status in statuses):
            return "ready"
        return "missing"

    if rag_documents.has_ready_documents():
        return "ready"
    records = rag_documents.list_documents()
    if not records:
        return "missing"
    if any(record.status in indexing_statuses for record in records):
        return "indexing"
    if all(record.status == "failed" for record in records):
        return "failed"
    return "missing"


def _usable_as_email_body_context(text: Optional[str]) -> bool:
    """Prior assistant text can fill an email body only if it is a real result."""
    if not text or not str(text).strip():
        return False
    lowered = str(text).lower()
    blocked = (
        "who should i send",
        "what should the email say",
        "who should i send it to",
        "which database",
        "attach or select",
        "reply **yes**",
        "please confirm before",
        "cancelled the pending",
    )
    return not any(marker in lowered for marker in blocked)


def auto_planning_context(req: "ChatRequest") -> orchestrator.PlanningContext:
    sql_context = sql_memory.conversation_memory.get_context(req.thread_id)
    last_context = _THREAD_AUTO_CONTEXT.get(req.thread_id, {})
    # None => all ready docs are available; [] => explicitly no document scope.
    # Attached ids count as available resources even while still indexing so the
    # planner can choose RAG (and the executor can report indexing honestly).
    if req.document_ids is not None:
        has_docs = len(req.document_ids) > 0
    else:
        has_docs = rag_documents.has_ready_documents()
    last_path = last_context.get("last_path") or []
    last_mode = (last_context.get("last_mode") or "")
    has_rag_context = ("RAG" in last_path) or ("RAG" in last_mode)
    last_answer = last_context.get("last_answer")
    if req.document_ids is not None:
        doc_records = [
            record.to_public_dict()
            for doc_id in req.document_ids
            if (record := rag_documents.get_document(doc_id))
        ]
    else:
        doc_records = [record.to_public_dict() for record in rag_documents.list_documents()]
    return orchestrator.PlanningContext(
        has_selected_database=bool(req.db_name),
        has_rag_documents=has_docs,
        has_sql_context=bool(sql_context.strip()),
        has_last_answer=_usable_as_email_body_context(last_answer),
        has_rag_context=has_rag_context,
        selected_database_name=req.db_name,
        rag_documents=doc_records,
        connected_capabilities={tool.key: oauth.is_tool_connected(tool.key) for tool in ALL_TOOLS},
        last_tool_summary="available" if last_context.get("last_answer") else "",
        use_llm_planner=config.AUTO_PLANNER_USE_LLM,
    )


def _db_name_from_message(message: str) -> Optional[str]:
    text = (message or "").strip().lower()
    if not text:
        return None
    choices = db_choices()
    for name in choices:
        stem = name[:-3] if name.lower().endswith(".db") else name
        candidates = {
            name.lower(),
            stem.lower(),
            f"use {name.lower()}",
            f"use {stem.lower()}",
            f"{stem.lower()}.db",
        }
        if text in candidates or text.rstrip(".") in candidates:
            return name
    return None


def _message_is_short_ack(message: str) -> bool:
    text = (message or "").strip().lower().rstrip(".!")
    return text in {"ok", "okay", "done", "ready", "here", "attached", "uploaded", "this one", "that one"}


def _extract_email_body_from_message(message: str) -> Optional[str]:
    text = (message or "").strip()
    if not text:
        return None
    body_match = re.search(r"\b(?:saying|say|that says|body)\s*:?\s+(.+)$", text, re.IGNORECASE | re.DOTALL)
    if body_match:
        body = body_match.group(1).strip()
        body = re.split(r"\b(?:recipient detail|send to|recipient):?\s+", body, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        return body or None
    # Clarifying replies that are not just an email address count as body text.
    if _extract_email_recipient(text) and len(text.split()) <= 3:
        return None
    if re.fullmatch(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        return None
    if len(text.split()) >= 3:
        return text
    return None


def _field_values_from_message(req: "ChatRequest", message: str, fields: list[str]) -> dict:
    text = (message or "").strip()
    lowered = text.lower()
    values = {}
    if "database" in fields:
        named = _db_name_from_message(message)
        if named:
            req.db_name = named
            values["database"] = named
        elif req.db_name:
            values["database"] = req.db_name
    if "document" in fields and (req.document_ids or rag_documents.has_ready_documents()):
        values["document"] = list(req.document_ids or [])
    if "email_recipient" in fields:
        recipient = _extract_email_recipient(message)
        if recipient or re.search(r"\b(to\s+)?(myself|me)\b", lowered):
            values["email_recipient"] = recipient or "me"
    if "attendee_email" in fields:
        attendee = _extract_email_recipient(message)
        if attendee:
            values["attendee_email"] = attendee
    if "date" in fields and orchestrator._has_date_expression(lowered):
        values["date"] = text
    if "time" in fields and orchestrator._has_time_expression(lowered):
        values["time"] = text
    if "email_body" in fields:
        body = _extract_email_body_from_message(message)
        if body:
            values["email_body"] = body
    elif any(field.startswith("email") for field in fields):
        # Opportunistically capture body when the user answers a recipient ask
        # with a full "to X saying Y" clarification in one turn.
        body = _extract_email_body_from_message(message)
        if body:
            values["email_body"] = body
    if "lookup" in fields and len(text.split()) > 2:
        values["lookup"] = text
    return values


def _infer_missing_fields(plan: orchestrator.AutoPlan) -> list[str]:
    if plan.missing_fields:
        return list(plan.missing_fields)
    text = (plan.missing_info or "").lower()
    fields = []
    if "database" in text:
        fields.append("database")
    if "document" in text or "attach" in text:
        fields.append("document")
    if "recipient" in text:
        fields.append("email_recipient")
    if "attendee" in text:
        fields.append("attendee_email")
    if "date" in text:
        fields.append("date")
    if "time" in text:
        fields.append("time")
    if "body" in text or "say" in text or "message" in text:
        fields.append("email_body")
    return list(dict.fromkeys(fields))


def _combined_pending_goal_message(pending: dict) -> str:
    original = (pending.get("goal") or pending.get("message") or "").strip()
    values = pending.get("field_values") or {}
    details = []
    if values.get("database"):
        details.append(f"Database: {values['database']}")
    if values.get("email_recipient"):
        recipient = values["email_recipient"]
        if recipient in {"me", "myself"}:
            details.append("to myself")
        else:
            details.append(f"to {recipient}")
    if values.get("attendee_email"):
        details.append(f"Attendee: {values['attendee_email']}")
    if values.get("date"):
        details.append(f"Date detail: {values['date']}")
    if values.get("time"):
        details.append(f"Time detail: {values['time']}")
    if values.get("lookup"):
        details.append(f"Lookup detail: {values['lookup']}")
    # Replies we couldn't map to a slot still carry meaning ("14th to 15th of
    # august"). Keep them before the body so they aren't swallowed by the
    # greedy "saying ..." parse.
    for note in pending.get("notes") or []:
        details.append(str(note))
    if values.get("email_body"):
        details.append(f"saying {values['email_body']}")
    return "\n".join([original, *details]).strip() or original


_MISSING_FIELD_PROMPTS = {
    "email_recipient": "the recipient's email address",
    "email_body": "what the email should say",
    "attendee_email": "the attendee's email address",
    "date": "the date",
    "time": "the time",
    "database": "which database to use",
    "document": "which document to use",
    "lookup": "what to look up",
}


def nothing_to_approve_reply(thread_id: str) -> str:
    """Approval with no queued write. Say plainly that nothing was sent, and
    restate whatever is still missing."""
    base = "I haven't prepared anything yet, so there's nothing to approve — nothing has been sent."
    pending = _THREAD_PENDING_GOAL.get(thread_id) or {}
    fields = [field for field in (pending.get("missing_fields") or []) if field in _MISSING_FIELD_PROMPTS]
    if fields:
        needed = ", ".join(_MISSING_FIELD_PROMPTS[field] for field in fields)
        return f"{base} I still need {needed}."
    goal = (pending.get("goal") or "").strip()
    if goal:
        first_line = goal.splitlines()[0].strip()
        return f"{base} Tell me what to do next for: {first_line}"
    return f"{base} Tell me what you'd like me to do."


def _remember_unmatched_reply(pending: dict, message: str, original: str, *, contributed: bool) -> None:
    """Keep a clarification reply that filled no slot, so the planner still sees
    what the user actually typed instead of only the stored goal."""
    text = (message or "").strip()
    if contributed or not text or _message_is_short_ack(text) or is_approval_text(text):
        return
    lowered = text.lower()
    if lowered in original.lower():
        return
    notes = pending.setdefault("notes", [])
    if any(lowered == str(note).lower() for note in notes):
        return
    notes.append(text)


def _is_email_pending_goal(pending: dict) -> bool:
    intent = str(pending.get("intent") or "")
    fields = pending.get("missing_fields") or []
    return intent.startswith("email") or any(str(field).startswith("email") for field in fields)


def resolve_pending_auto_goal(req: "ChatRequest", message: str) -> str:
    """Resume a prior Auto goal once the user supplies missing information."""
    pending = _THREAD_PENDING_GOAL.get(req.thread_id)
    if not pending:
        return message

    if pending.get("status") == "awaiting_approval":
        return message

    fields = list(pending.get("missing_fields") or ([pending["missing"]] if pending.get("missing") else []))
    original = (pending.get("goal") or pending.get("message") or "").strip() or message
    pending.setdefault("field_values", {})
    values_before = dict(pending["field_values"])
    pending["field_values"].update(_field_values_from_message(req, message, fields))
    _remember_unmatched_reply(pending, message, original, contributed=pending["field_values"] != values_before)

    # A recipient was just resolved this turn - if we know what label the
    # user used for them ("dad", "my manager", ...), remember the mapping so
    # later turns in this same conversation don't have to re-ask.
    if "email_recipient" in fields and "email_recipient" in pending["field_values"]:
        resolved_email = pending["field_values"].get("email_recipient")
        label = pending.get("contact_label") or _extract_recipient_label(original) or _extract_recipient_label(message)
        if resolved_email and resolved_email not in {"me", "myself"} and "@" in str(resolved_email):
            _remember_known_contact(req.thread_id, label, resolved_email)
    # Body may arrive opportunistically with a recipient clarification.
    remaining = [
        field
        for field in fields
        if field not in pending["field_values"]
    ]
    pending["missing_fields"] = remaining
    pending["missing"] = remaining[0] if len(remaining) == 1 else ("multiple" if remaining else None)

    if not remaining:
        _THREAD_PENDING_GOAL.pop(req.thread_id, None)
        if _is_email_pending_goal(pending):
            combined = _combined_pending_goal_message({**pending, "goal": original})
            # Prefer the user's clarification when it already contains the full email ask.
            if original.lower() in message.lower() and (
                orchestrator._has_email_recipient(message.lower())
                or orchestrator._has_email_body_expression(message.lower())
            ):
                return message
            return combined
        combined = _combined_pending_goal_message({**pending, "goal": original})
        if original.lower() in message.lower():
            return message
        return combined

    pending["message"] = original
    pending["goal"] = original
    # Still waiting: ask only for remaining fields on the next planner pass.
    # Keep the original goal as the resume message seed via field_values.
    return _combined_pending_goal_message(pending)


def remember_pending_auto_goal(thread_id: str, message: str, plan: orchestrator.AutoPlan) -> None:
    existing = _THREAD_PENDING_GOAL.get(thread_id, {})
    existing_values = dict(existing.get("field_values") or {})
    inferred_missing_fields = _infer_missing_fields(plan)
    if plan.missing_info and inferred_missing_fields:
        carried_values = {
            field: value
            for field, value in existing_values.items()
            if field not in inferred_missing_fields
        }
        # Preserve the original user goal across clarification turns. The
        # resolved planner message is more faithful than an LLM paraphrase.
        original_goal = existing.get("goal") or message or plan.goal
        if existing.get("status") == "waiting_for_fields" and existing.get("goal"):
            original_goal = existing["goal"]
        _THREAD_PENDING_GOAL[thread_id] = {
            "conversation_id": thread_id,
            "status": "waiting_for_fields",
            "missing": inferred_missing_fields[0] if len(inferred_missing_fields) == 1 else "multiple",
            "missing_fields": inferred_missing_fields,
            "field_values": carried_values,
            "notes": list(existing.get("notes") or []),
            "observations": [],
            "pending_write": None,
            "goal": original_goal,
            "message": original_goal,
            "intent": plan.intent,
        }
        return

    if plan.intent == "sql_needs_database":
        _THREAD_PENDING_GOAL[thread_id] = {
            "conversation_id": thread_id,
            "status": "waiting_for_fields",
            "missing": "database",
            "missing_fields": ["database"],
            "field_values": {k: v for k, v in existing_values.items() if k != "database"},
            "notes": list(existing.get("notes") or []),
            "observations": [],
            "pending_write": None,
            "goal": existing.get("goal") or message,
            "message": existing.get("goal") or message,
            "intent": plan.intent,
        }
        return
    if plan.intent == "rag_needs_document":
        _THREAD_PENDING_GOAL[thread_id] = {
            "conversation_id": thread_id,
            "status": "waiting_for_fields",
            "missing": "document",
            "missing_fields": ["document"],
            "field_values": {k: v for k, v in existing_values.items() if k != "document"},
            "notes": list(existing.get("notes") or []),
            "observations": [],
            "pending_write": None,
            "goal": existing.get("goal") or message,
            "message": existing.get("goal") or message,
            "intent": plan.intent,
        }
        return
    if plan.intent == "email_needs_recipient" or (
        plan.missing_info and "recipient" in (plan.missing_info or "").lower()
    ):
        # Capture "dad"/"my manager"/etc. from whichever turn first named a
        # recipient, so once the user supplies the address we know what label
        # to remember it under (see resolve_pending_auto_goal).
        contact_label = existing.get("contact_label") or _extract_recipient_label(
            existing.get("goal") or message
        )
        _THREAD_PENDING_GOAL[thread_id] = {
            "conversation_id": thread_id,
            "status": "waiting_for_fields",
            "missing": "email_recipient",
            "missing_fields": ["email_recipient"],
            "field_values": {k: v for k, v in existing_values.items() if k != "email_recipient"},
            "notes": list(existing.get("notes") or []),
            "observations": [],
            "pending_write": None,
            "goal": existing.get("goal") or message,
            "message": existing.get("goal") or message,
            "intent": plan.intent,
            "contact_label": contact_label,
        }
        return
    if plan.intent == "leave_needs_date" or (
        plan.missing_info and "what date" in (plan.missing_info or "").lower()
    ):
        _THREAD_PENDING_GOAL[thread_id] = {
            "conversation_id": thread_id,
            "status": "waiting_for_fields",
            "missing": "date",
            "missing_fields": ["date"],
            "field_values": {k: v for k, v in existing_values.items() if k != "date"},
            "notes": list(existing.get("notes") or []),
            "observations": [],
            "pending_write": None,
            "goal": existing.get("goal") or message,
            "message": existing.get("goal") or message,
            "intent": plan.intent,
        }
        return
    if plan.intent == "sql_needs_clarification":
        # Keep the original goal when we only need a DB name; otherwise wait for
        # a concrete lookup on the next turn without forcing mode switches.
        if "which database" in (plan.missing_info or "").lower():
            _THREAD_PENDING_GOAL[thread_id] = {
                "conversation_id": thread_id,
                "status": "waiting_for_fields",
                "missing": "database",
                "missing_fields": ["database"],
                "field_values": {k: v for k, v in existing_values.items() if k != "database"},
                "notes": list(existing.get("notes") or []),
                "observations": [],
                "pending_write": None,
                "goal": existing.get("goal") or message,
                "message": existing.get("goal") or message,
                "intent": plan.intent,
            }
        return
    if plan.missing_info and not plan.steps:
        # A free-form clarification ("what's the reason?", "approval required")
        # yields no structured slot, but the goal is still in flight. Dropping it
        # here is what made later turns restart from scratch and re-ask for the
        # recipient and date the user had already given.
        _THREAD_PENDING_GOAL[thread_id] = {
            "conversation_id": thread_id,
            "status": "waiting_for_fields",
            "missing": None,
            "missing_fields": [],
            "field_values": existing_values,
            "notes": list(existing.get("notes") or []),
            "observations": [],
            "pending_write": None,
            "goal": existing.get("goal") or message or plan.goal,
            "message": existing.get("goal") or message or plan.goal,
            "intent": plan.intent,
        }
        return
    # A concrete Auto plan means we are no longer waiting on prior missing info.
    if plan.steps:
        _THREAD_PENDING_GOAL.pop(thread_id, None)


def remember_auto_context(thread_id: str, response: "ChatResponse", plan: Optional[orchestrator.AutoPlan] = None) -> None:
    if not response.reply:
        return
    _THREAD_AUTO_CONTEXT[thread_id] = {
        "last_answer": response.answer or response.reply,
        "last_result": response.result,
        "last_sources": response.sources,
        "last_mode": response.mode_used,
        "last_path": plan.executed_path if plan else [],
    }


def last_context_text(thread_id: str, history: list) -> str:
    cached = _THREAD_AUTO_CONTEXT.get(thread_id, {})
    if cached.get("last_answer"):
        return str(cached["last_answer"])
    for role, content in reversed(history or []):
        if role == "assistant" and content:
            return str(content)
    return ""


def tool_output_success(output: str) -> bool:
    lowered = (output or "").lower()
    failure_markers = [
        "not connected",
        "not implemented",
        "not wired",
        "need a recipient",
        "unknown action",
        "couldn't",
        "could not",
        "failed",
        "no real action was performed",
        "no events found",
    ]
    return not any(marker in lowered for marker in failure_markers)


def mcp_action_is_implemented(tool_key: str, action_name: str) -> bool:
    # Be conservative: only write paths with real API implementations go here.
    return (tool_key, action_name) in {
        ("gmail", "send_email"),
        ("gmail", "draft_reply"),
        ("google_calendar", "create_meeting"),
    }


def build_mcp_instruction(req: "ChatRequest", step: orchestrator.PlanStep, plan: orchestrator.AutoPlan) -> str:
    message = req.message.strip()
    if step.capability == "gmail" and step.action in {"send_email", "draft_reply"}:
        if plan.intent.startswith("leave"):
            leave_instruction = build_leave_email_instruction(message, plan)
            if leave_instruction:
                return leave_instruction
        has_body = re.search(r"\b(?:saying|say|that says|body)\s*:?\s+.+", message, re.IGNORECASE)
        if not has_body:
            # Prefer a successful observation from this same Auto turn (e.g. SQL → Gmail).
            context = ""
            for obs in reversed(plan.observations):
                if obs.success and obs.output and _usable_as_email_body_context(obs.output):
                    context = str(obs.output)
                    break
            if not context:
                context = last_context_text(req.thread_id, req.history)
            if _usable_as_email_body_context(context):
                return f"{message} subject AgentHub Studio result saying {context}"
    return message


def build_leave_email_instruction(message: str, plan: orchestrator.AutoPlan) -> Optional[str]:
    recipient = _extract_email_recipient(message)
    if not recipient:
        return None
    resolved_dates = []
    for obs in plan.observations:
        if obs.capability == "date_time" and obs.success:
            for raw in re.findall(r"\b(\d{4}-\d{2}-\d{2})\b", obs.output):
                value = datetime.fromisoformat(raw).date()
                if value not in resolved_dates:
                    resolved_dates.append(value)
            if resolved_dates:
                break
    if not resolved_dates:
        return None

    # Trust the full structured result the Date/Time tool actually resolved
    # (every ISO date it returned) rather than re-deriving "is this a range"
    # from a narrower, independent regex over the raw message. That second
    # check only recognized literal "X to Y" phrasing and silently dropped
    # the end date for anything phrased as a duration ("for 15 days", "for
    # 3 days starting next Monday", etc.), even though the Date/Time
    # observation had already resolved both endpoints correctly.
    start = min(resolved_dates)
    end = max(resolved_dates) if len(resolved_dates) > 1 else None
    start_label = _long_date_label(start)
    if end:
        subject = f"Leave request for {start_label} to {_long_date_label(end)}"
        when = f"from {start_label} ({start.strftime('%A')}) to {_long_date_label(end)} ({end.strftime('%A')})"
    else:
        subject = f"Leave request for {start_label}"
        when = f"on {start_label} ({start.strftime('%A')})"

    reason = _leave_reason(message)
    reason_sentence = f" {reason}" if reason else ""
    body = (
        "Hi,\n\n"
        f"I would like to request leave {when}.{reason_sentence} "
        "Please let me know if you need any additional information.\n\n"
        "Thank you."
    )
    return f"send an email to {recipient} subject {subject} saying {body}"


def _long_date_label(value) -> str:
    return value.strftime("%B %-d, %Y") if os.name != "nt" else value.strftime("%B %#d, %Y")


def _mentions_date_range(message: str) -> bool:
    text = (message or "").lower()
    if orchestrator.MONTH_FIRST_DATE_RE.search(text) and any(
        match.group(3) for match in orchestrator.MONTH_FIRST_DATE_RE.finditer(text)
    ):
        return True
    return any(
        match.group(2)
        for pattern in orchestrator.DAY_FIRST_DATE_RES
        for match in pattern.finditer(text)
    )


def _leave_reason(message: str) -> Optional[str]:
    """A reason the user volunteered ("because I am sick") belongs in the draft,
    but is never something to block on."""
    match = re.search(r"\b(?:because|since|due to|as)\s+(.+?)(?:[.\n]|$)", message or "", re.IGNORECASE)
    if match:
        reason = match.group(1).strip(" .,")
        if reason:
            return f"Reason: {reason[0].upper() + reason[1:]}."
    if re.search(r"\b(?:i am|i'm|im)\s+sick\b", message or "", re.IGNORECASE):
        return "Reason: I am unwell."
    if re.search(r"\bsick\s+leave\b", message or "", re.IGNORECASE):
        return "Reason: Sick leave."
    return None


def _extract_email_recipient(message: str) -> Optional[str]:
    match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", message or "")
    return match.group(0) if match else None


# Words that can never be a contact label on their own - pronouns, articles,
# and the recipient markers themselves. Kept intentionally short: the goal is
# to filter obvious noise, not to hardcode which relations/names are allowed
# (any other word or short phrase - "dad", "landlord", "my manager Priya",
# "the vendor" - is fair game as a label).
_CONTACT_LABEL_STOPWORDS = {
    "me", "myself", "i", "him", "her", "them", "it", "someone", "somebody",
    "email", "mail", "an", "a", "the",
}


def _normalize_contact_label(label: str) -> str:
    return re.sub(r"\s+", " ", (label or "").strip().lower())


# Words that naturally end a recipient phrase - once one of these shows up
# after "to"/"my"/"our"/"the", the label stops there. Kept general: these are
# connective/structural words and pronouns common to *any* instruction, not
# relation names, so this doesn't hardcode which relations are allowed.
_LABEL_BOUNDARY_WORDS = {
    "saying", "say", "that", "says", "about", "and", "to", "regarding",
    "asking", "telling", "letting", "with", "for", "again", "reminding",
    "him", "her", "them", "it", "the", "a", "an", "he", "she", "they",
    "updated", "new", "his", "hers", "their", "reminder", "note",
}


def _extract_recipient_label(message: str) -> Optional[str]:
    """Pull a general name/relation label for who an email is going to, e.g.
    "email my dad" -> "dad", "send it to my manager Priya" -> "manager priya",
    "email mom about dinner" -> "mom", "email John" -> "john". Returns None
    when the message has no such phrase (e.g. it's a bare email address, or
    doesn't mention a recipient at all). Deliberately not tied to any fixed
    word list like "dad"/"mom"/"boss" - it just captures the short noun
    phrase that follows a recipient-introducing word ("to", "my", "our",
    "the", or directly after "email"/"mail") and stops at the first natural
    boundary word or punctuation.
    """
    text = (message or "").strip()
    if not text:
        return None
    if _extract_email_recipient(text):
        return None  # an explicit address is present - no label to remember

    patterns = [
        # "email/mail dad ..." / "email the landlord ..." - most direct signal,
        # tried first so it wins over an unrelated "to <verb>" elsewhere in
        # the sentence (e.g. "email dad again reminding him to bring dessert").
        r"\b(?:email|mail)\s+(?:my|our|the)?\s*([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,3})",
        r"\b(?:send\s+(?:it|this|a\s+mail|an?\s+email)?\s*(?:to|for))\s+(?:my|our|the)?\s*([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,3})",
        r"\b(?:to|for)\s+(?:my|our|the)\s+([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,3})",
        r"\b(?:my|our)\s+([A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*){0,3})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        words = match.group(1).split()
        kept = []
        for word in words:
            if word.lower().strip(".,'") in _LABEL_BOUNDARY_WORDS:
                break
            kept.append(word)
        if not kept:
            continue
        label = _normalize_contact_label(" ".join(kept))
        if not label or label in _CONTACT_LABEL_STOPWORDS:
            continue
        if "@" in label or len(label.split()) > 4:
            continue
        return label
    return None


def _remember_known_contact(thread_id: str, label: Optional[str], email: Optional[str]) -> None:
    if not label or not email or email.lower() in {"me", "myself"}:
        return
    normalized = _normalize_contact_label(label)
    if not normalized:
        return
    contacts = _THREAD_KNOWN_CONTACTS.setdefault(thread_id, {})
    contacts[normalized] = email


def _known_contact_email(thread_id: str, label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    normalized = _normalize_contact_label(label)
    return _THREAD_KNOWN_CONTACTS.get(thread_id, {}).get(normalized)


def substitute_known_contacts(thread_id: str, message: str) -> str:
    """If this turn refers to a name/relation the user already mapped to an
    email address earlier in the SAME conversation (e.g. "dad" ->
    drmadala99@gmail.com), splice that address into the message before
    planning, so the planner sees a resolvable recipient instead of asking
    "who should I send this to?" again. Only touches turns that actually
    mention a recipient label and don't already contain an email address.
    """
    contacts = _THREAD_KNOWN_CONTACTS.get(thread_id)
    if not contacts:
        return message
    if _extract_email_recipient(message):
        return message  # user already gave/has an explicit address this turn
    label = _extract_recipient_label(message)
    if not label:
        return message
    email = _known_contact_email(thread_id, label)
    if not email:
        return message
    return f"{message} (recipient: {email})"


_MONTH_LOOKUP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_WEEKDAY_LOOKUP = {
    "mon": 0, "monday": 0, "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2, "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4, "sat": 5, "saturday": 5, "sun": 6, "sunday": 6,
}


def _future_or_same_date(now: datetime, month: int, day: int, year_text: Optional[str]) -> datetime.date:
    year = int(year_text) if year_text else now.year
    candidate = datetime(year, month, day).date()
    if not year_text and candidate < now.date():
        candidate = datetime(year + 1, month, day).date()
    return candidate


def _range_days(start_text: str, end_text: Optional[str]) -> list[int]:
    """Days covered by "14" or "14 to 15". Inverted or oversized ranges fall
    back to the start day rather than inventing dates."""
    start = int(start_text)
    if not end_text:
        return [start]
    end = int(end_text)
    if end <= start or end - start > 31:
        return [start]
    return [start, end]


def _next_weekday(now: datetime, weekday: int, force_next: bool) -> datetime.date:
    days = (weekday - now.weekday()) % 7
    if force_next and days == 0:
        days = 7
    if force_next and days < 7:
        days = days or 7
    if not force_next and days == 0:
        days = 0
    return now.date() + timedelta(days=days)


_DATE_CLARIFICATION = (
    "[Date/Time] I couldn't resolve that date. "
    "Try 'August 16', '16 Aug', 'today', or 'tomorrow'."
)


def execute_date_time(message: str) -> tuple[bool, str]:
    now = datetime.now()
    text = message.lower()
    resolved = []
    if any(word in text for word in orchestrator.TOMORROW_WORDS):
        resolved.append(("tomorrow", now.date() + timedelta(days=1)))
    if "today" in text:
        resolved.append(("today", now.date()))
    if "yesterday" in text:
        resolved.append(("yesterday", now.date() - timedelta(days=1)))

    # "august 14", "august 14 to 15" (month, day, optional end day, optional year)
    for match in orchestrator.MONTH_FIRST_DATE_RE.finditer(text):
        month_key = match.group(1)[:3]
        try:
            for day in _range_days(match.group(2), match.group(3)):
                resolved.append((match.group(0), _future_or_same_date(now, _MONTH_LOOKUP[month_key], day, match.group(4))))
        except (ValueError, KeyError):
            return False, _DATE_CLARIFICATION

    # "14th of august", "14th to 15th of august" (day, optional end day, month, year)
    for pattern in orchestrator.DAY_FIRST_DATE_RES:
        for match in pattern.finditer(text):
            month_key = match.group(3)[:3]
            try:
                for day in _range_days(match.group(1), match.group(2)):
                    resolved.append((match.group(0), _future_or_same_date(now, _MONTH_LOOKUP[month_key], day, match.group(4))))
            except (ValueError, KeyError):
                return False, _DATE_CLARIFICATION

    for match in re.finditer(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text):
        try:
            resolved.append((match.group(0), datetime(int(match.group(1)), int(match.group(2)), int(match.group(3))).date()))
        except ValueError:
            return False, _DATE_CLARIFICATION

    for match in re.finditer(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", text):
        month = int(match.group(1))
        day = int(match.group(2))
        year_text = match.group(3)
        if year_text and len(year_text) == 2:
            year_text = f"20{year_text}"
        try:
            resolved.append((match.group(0), _future_or_same_date(now, month, day, year_text)))
        except ValueError:
            return False, _DATE_CLARIFICATION

    for match in re.finditer(
        r"\b(?:(next|this)\s+)?(mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:r|rs|rsday|rday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b",
        text,
    ):
        prefix = match.group(1)
        weekday = _WEEKDAY_LOOKUP.get(match.group(2))
        if weekday is None:
            continue
        resolved.append((match.group(0), _next_weekday(now, weekday, force_next=(prefix == "next"))))

    # Relative offsets: "in 10 days", "10 days from now"/"later", "10 days ago",
    # and the "week(s)" equivalents (a week is treated as 7 days).
    for match in re.finditer(r"\bin\s+(\d+)\s+(day|days|week|weeks)\b", text):
        amount = int(match.group(1))
        days = amount * 7 if match.group(2).startswith("week") else amount
        resolved.append((match.group(0), now.date() + timedelta(days=days)))

    for match in re.finditer(r"\b(\d+)\s+(day|days|week|weeks)\s+(?:from now|from today|later)\b", text):
        amount = int(match.group(1))
        days = amount * 7 if match.group(2).startswith("week") else amount
        resolved.append((match.group(0), now.date() + timedelta(days=days)))

    for match in re.finditer(r"\b(\d+)\s+(day|days|week|weeks)\s+ago\b", text):
        amount = int(match.group(1))
        days = amount * 7 if match.group(2).startswith("week") else amount
        resolved.append((match.group(0), now.date() - timedelta(days=days)))

    # "for 15 days", "for 3 weeks" - a DURATION, not a single offset. Anchor
    # it to whatever start date has already been resolved above (today,
    # tomorrow, an explicit date, a weekday); if nothing else was resolved
    # yet, default the anchor to today so a bare "for 15 days" still gets a
    # start point. Inclusive counting: a 15-day span starting today covers
    # today through today+14, matching how a multi-day leave request is
    # normally counted. Without this, a duration like "from today for 15
    # days" resolved only the single "today" anchor and silently dropped
    # the requested span entirely.
    for match in re.finditer(r"\bfor\s+(\d+)\s+(day|days|week|weeks)\b", text):
        amount = int(match.group(1))
        days = amount * 7 if match.group(2).startswith("week") else amount
        if days < 1:
            continue
        anchor = min((value for _, value in resolved), default=now.date())
        if not resolved:
            resolved.append(("today", anchor))
        resolved.append((match.group(0), anchor + timedelta(days=days - 1)))

    deduped = []
    seen = set()
    for label, value in resolved:
        key = (label, value)
        if key not in seen:
            seen.add(key)
            deduped.append((label, value))
    resolved = deduped

    if not resolved and ("time" in text or "date" in text):
        return True, f"[Date/Time] Current local time is {now.isoformat(timespec='seconds')}."
    if not resolved:
        return False, _DATE_CLARIFICATION

    lines = ["[Date/Time] Resolved date information:"]
    for label, value in resolved:
        lines.append(f"- {label}: {value.isoformat()} ({value.strftime('%A')})")
    if any(word in text for word in ["holiday", "leave", "pto", "vacation", "sick"]):
        lines.append(_holiday_status_line({value for _, value in resolved}))
    return True, "\n".join(lines)


_HOLIDAY_CALENDAR_CACHE: dict[tuple[str, int], Any] = {}


def _holiday_calendar_for_year(year: int):
    """One holidays.country_holidays(...) instance per (country, year),
    reused across calls - the library computes the whole year's calendar up
    front, so there's no reason to rebuild it per request."""
    key = (config.HOLIDAY_CALENDAR_COUNTRY, year)
    calendar_for_year = _HOLIDAY_CALENDAR_CACHE.get(key)
    if calendar_for_year is None:
        calendar_for_year = holidays_lib.country_holidays(config.HOLIDAY_CALENDAR_COUNTRY, years=year)
        _HOLIDAY_CALENDAR_CACHE[key] = calendar_for_year
    return calendar_for_year


def _holiday_status_line(dates) -> str:
    """Real (offline) holiday-calendar verification for every resolved date -
    replaces the old placeholder that always said this wasn't checked."""
    try:
        matches = []
        for value in sorted(dates):
            calendar_for_year = _holiday_calendar_for_year(value.year)
            name = calendar_for_year.get(value)
            if name:
                matches.append(f"{_long_date_label(value)} is a holiday ({name}, {config.HOLIDAY_CALENDAR_COUNTRY})")
        if matches:
            return "Holiday check (" + config.HOLIDAY_CALENDAR_COUNTRY + "): " + "; ".join(matches) + "."
        return f"Holiday check ({config.HOLIDAY_CALENDAR_COUNTRY}): none of these dates are public holidays."
    except Exception as exc:  # noqa: BLE001 - a bad country code or lib issue must not break date resolution
        logger.warning("Holiday calendar lookup failed: %s", exc)
        return "Holiday status could not be checked right now."


_CALC_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def execute_calculator(message: str) -> tuple[bool, str]:
    text = message.lower()
    # "X% of Y" -> "(X/100)*Y" - must run before the word replacements below,
    # since "of" isn't valid Python and a bare "%" -> "/100" swap would leave
    # "18/100 of 245" (still invalid) rather than a multiplication.
    text = re.sub(
        r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)",
        r"(\1/100)*\2",
        text,
    )
    expression = (
        text
        .replace("what's", "")
        .replace("whats", "")
        .replace("how much is", "")
        .replace("calculate", "")
        .replace("what is", "")
        .replace("plus", "+")
        .replace("minus", "-")
        .replace("times", "*")
        .replace("divided by", "/")
        .replace("%", "/100")
        .strip(" ?")
    )
    try:
        value = _eval_numeric_ast(ast.parse(expression, mode="eval").body)
    except Exception as exc:  # noqa: BLE001
        return False, f"[Calculator] I couldn't evaluate that expression: {exc}"
    return True, f"[Calculator] {expression} = {value}"


def _eval_numeric_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPERATORS:
        return _CALC_OPERATORS[type(node.op)](_eval_numeric_ast(node.left), _eval_numeric_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPERATORS:
        return _CALC_OPERATORS[type(node.op)](_eval_numeric_ast(node.operand))
    raise ValueError("Only numeric expressions are supported.")


def handle_auto_message(req: "ChatRequest", message: str, text_lower: str) -> "ChatResponse":
    message = resolve_pending_auto_goal(req, message)
    message = substitute_known_contacts(req.thread_id, message)
    req.message = message
    text_lower = message.lower().rstrip(".!")
    plan = orchestrator.plan_turn(message, auto_planning_context(req))
    remember_pending_auto_goal(req.thread_id, message, plan)
    seen_calls = set()

    sql_update = None
    sql_valid = None
    retry_count_out = None
    execution_ms = None
    result_update = None
    row_count = None
    answer_update = None
    tool_labels = []
    for step in plan.steps:
        if step.capability == "general_chat":
            continue
        cap = orchestrator.CAPABILITY_REGISTRY.get(step.capability)
        tool_labels.append(cap.display_name if cap else step.capability)
    debug_lines = [
        f"Intent: {plan.intent}",
        "Tools: " + (" → ".join(tool_labels) if tool_labels else "none"),
        "Planned steps: " + (" -> ".join(f"{s.capability}.{s.action}" for s in plan.steps) if plan.steps else "(none)"),
    ]
    schema_update = None
    sources_update = None
    retrieved_contexts_update = None
    approve_visible = False
    db_dropdown_choices = None
    db_dropdown_value = None
    replies: list[str] = []

    while True:
        step = orchestrator.next_step(plan)
        if step is None:
            break

        call_sig = (step.capability, step.action, len(plan.observations))
        if call_sig in seen_calls:
            observation = orchestrator.Observation(
                step.capability,
                step.action,
                False,
                f"[Auto] Duplicate tool call blocked for {step.capability}.{step.action}.",
            )
            orchestrator.observe(plan, observation)
            replies.append(observation.output)
            debug_lines.append(observation.output)
            break
        seen_calls.add(call_sig)

        if step.depends_on_previous and not any(obs.success for obs in plan.observations) and not last_context_text(req.thread_id, req.history):
            observation = orchestrator.Observation(
                step.capability,
                step.action,
                False,
                f"[Auto] Skipped {step.capability}: the previous required step did not return a successful real result.",
            )
            orchestrator.observe(plan, observation)
            replies.append(observation.output)
            debug_lines.append(observation.output)
            break

        observation = None

        if step.capability == "database_builder":
            try:
                if builder.has_pending_build(req.thread_id):
                    reply = builder.refine_build(req.thread_id, message)
                else:
                    reply = builder.start_build(req.thread_id, message)
                approve_visible = True
                observation = orchestrator.Observation(step.capability, step.action, True, reply)
            except SchemaGenerationError as exc:
                reply = f"Sorry, I couldn't design that schema: {exc}"
                approve_visible = builder.has_pending_build(req.thread_id)
                observation = orchestrator.Observation(step.capability, step.action, False, reply)

        elif step.capability == "sql_agent":
            if not req.db_name:
                reply = "Which database would you like me to use?"
                observation = orchestrator.Observation(step.capability, step.action, False, reply)
            else:
                db_path = sql_database.db_name_to_path(req.db_name)
                conversation_context = sql_memory.conversation_memory.get_context(req.thread_id)
                started_at = time.perf_counter()
                try:
                    result = compiled_app.invoke(
                        {"question": message, "db_path": db_path, "conversation_context": conversation_context},
                        config={"configurable": {"thread_id": req.thread_id}},
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Auto SQL execution failed: %s", exc)
                    reply = f"Sorry, something went wrong while answering that: {exc}"
                    observation = orchestrator.Observation(step.capability, step.action, False, reply)
                else:
                    execution_ms = int((time.perf_counter() - started_at) * 1000)
                    final_answer = result.get("final_answer", "I couldn't generate an answer.")
                    sql_query = result.get("sql_query", "")
                    sql_result = result.get("sql_result", {"columns": [], "rows": []})
                    retry_history = result.get("retry_history", [])
                    validation_error = result.get("validation_error", "")
                    retry_count = result.get("retry_count", 0)
                    execution_success = result.get("execution_success", False)

                    sql_memory.conversation_memory.add_turn(req.thread_id, message, sql_query, final_answer)
                    reply = final_answer
                    sql_update = sql_query
                    sql_valid = execution_success
                    retry_count_out = retry_count
                    result_update = result_to_table(sql_result)
                    row_count = len(result_update["rows"])
                    answer_update = final_answer
                    schema_update = schema_summary_safe(req.db_name)
                    if execution_success:
                        sources_update = sources_for_query(req.db_name, sql_query, schema_update["tables"])
                    else:
                        sources_update = []
                    debug_lines.append(format_debug_info(retry_history, validation_error, retry_count))
                    observation = orchestrator.Observation(step.capability, step.action, bool(execution_success), reply)

        elif step.capability == "rag_agent":
            scoped_ids = list(req.document_ids) if req.document_ids is not None else None
            readiness = _rag_scope_readiness(scoped_ids)
            if readiness == "missing":
                reply = (
                    "Attach or select the document you'd like me to check, then ask again "
                    "once it shows Ready."
                )
                sources_update = []
                observation = orchestrator.Observation(step.capability, step.action, False, reply)
            elif readiness == "indexing":
                reply = "The document is still being indexed. Give me a moment and try again."
                sources_update = []
                observation = orchestrator.Observation(step.capability, step.action, False, reply)
            elif readiness == "failed":
                reply = "I couldn't index that document successfully, so I can't read it yet."
                sources_update = []
                observation = orchestrator.Observation(step.capability, step.action, False, reply)
            else:
                try:
                    rag_result = execute_rag_query(
                        question=message,
                        document_ids=scoped_ids,
                        conversation_id=req.thread_id,
                    )
                except RAGError as exc:
                    logger.exception("Auto RAG failed: %s", exc)
                    reply = f"Sorry, something went wrong while searching your documents: {exc}"
                    sources_update = []
                    observation = orchestrator.Observation(step.capability, step.action, False, reply)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Auto RAG failed unexpectedly: %s", exc)
                    reply = f"Sorry, something went wrong while searching your documents: {exc}"
                    sources_update = []
                    observation = orchestrator.Observation(step.capability, step.action, False, reply)
                else:
                    reply = rag_result["answer"]
                    answer_update = rag_result["answer"]
                    sources_update = rag_result["sources"]
                    retrieved_contexts_update = rag_result.get("retrieved_contexts")
                    debug_lines.append(format_rag_debug_info(rag_result["debug"]))
                    observation = orchestrator.Observation(step.capability, step.action, True, reply)

        elif step.capability in {"gmail", "google_drive", "google_calendar"}:
            tool, action = find_mcp_tool_action(step.capability, step.action)
            if not tool or not action:
                reply = f"[Auto] I couldn't find {step.capability}.{step.action}."
                observation = orchestrator.Observation(step.capability, step.action, False, reply)
            else:
                tool.connected = oauth.is_tool_connected(tool.key)
                instruction = build_mcp_instruction(req, step, plan)
                if action.write_action and not mcp_action_is_implemented(tool.key, action.name):
                    reply = f"[{tool.display_name}] '{action.description}' is not implemented yet. No external action was performed."
                    observation = orchestrator.Observation(step.capability, step.action, False, reply)
                else:
                    output = _call_tool_with_payload(tool, action, confirmed=False, instruction=instruction, arguments=None)
                    mode_path = "Agentic Auto -> " + " -> ".join([*plan.executed_path, tool.display_name])
                    pending = bool(action.write_action and tool.connected and tool_output_success(output))
                    if pending:
                        pending = store_pending_write(
                            req.thread_id,
                            tool=tool,
                            action=action,
                            goal=plan.goal or message,
                            instruction=instruction,
                            preview=output,
                            mode_path=mode_path,
                            observations=[
                                {
                                    "capability": obs.capability,
                                    "action": obs.action,
                                    "success": obs.success,
                                    "output": obs.output,
                                }
                                for obs in plan.observations
                            ],
                        )
                    if pending:
                        reply = f"{output}\n\nReply **yes** to approve, or **cancel** to stop."
                    else:
                        reply = output
                    observation = orchestrator.Observation(
                        step.capability,
                        step.action,
                        tool_output_success(output),
                        output,
                        pending_approval=pending,
                    )

        elif step.capability == "date_time":
            success, reply = execute_date_time(message)
            observation = orchestrator.Observation(step.capability, step.action, success, reply)

        elif step.capability == "calculator":
            success, reply = execute_calculator(message)
            observation = orchestrator.Observation(step.capability, step.action, success, reply)

        elif step.capability == "weather":
            reply = (
                "Weather isn't available yet — no weather provider is connected. "
                "I can't look up live conditions right now."
            )
            observation = orchestrator.Observation(step.capability, step.action, False, reply)

        else:
            history_for_llm = [tuple(turn) for turn in req.history[-(config.MAX_MEMORY_TURNS * 2):]]
            try:
                reply = general_chat.respond(message, history_for_llm, auto_mode=True)
                observation = orchestrator.Observation("general_chat", "respond", True, reply)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Auto general chat failed: %s", exc)
                reply = f"Sorry, something went wrong: {exc}"
                observation = orchestrator.Observation("general_chat", "respond", False, reply)

        orchestrator.observe(plan, observation)
        replies.append(reply)
        debug_lines.append(
            f"Observation {len(plan.observations)}: {observation.capability}.{observation.action} "
            f"{'succeeded' if observation.success else 'failed'}"
            f"{' and is pending approval' if observation.pending_approval else ''}."
        )

        if observation.pending_approval:
            break
        if not observation.success:
            break

    if plan.missing_info:
        replies.append(plan.missing_info)
        if plan.completion_status == "complete":
            plan.completion_status = "needs_user"

    reply = "\n\n".join(part for part in replies if part) or "I couldn't complete that request."
    debug_lines.append(f"Completion status: {plan.completion_status}")
    debug_lines.append("Capability registry: " + ", ".join(
        f"{cap['display_name']} ({'real' if cap['implemented'] else 'not implemented'})"
        for cap in orchestrator.describe_capabilities()
    ))

    response = ChatResponse(
        reply=reply,
        mode_used=orchestrator.mode_path(plan),
        thread_id=req.thread_id,
        approve_visible=approve_visible,
        timestamp=now_iso(),
        sql=sql_update,
        sql_valid=sql_valid,
        retry_count=retry_count_out,
        execution_ms=execution_ms,
        result=result_update,
        row_count=row_count,
        answer=answer_update or reply,
        debug="\n".join(debug_lines),
        schema=schema_update,
        sources=sources_update if sources_update is not None else [],
        retrieved_contexts=retrieved_contexts_update,
        db_choices=db_dropdown_choices,
        db_value=db_dropdown_value,
    )
    remember_auto_context(req.thread_id, response, plan)
    return response


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    thread_id: str
    mode: str = "Auto"
    db_name: Optional[str] = None
    include_sample_data: bool = True
    # Recent (role, content) turns, oldest first, already bounded by the
    # client - only used to give General Chat mode conversational context.
    history: list = []
    # Optional RAG scope for Auto/RAG. None => every ready document;
    # [] => no documents (empty scope); non-empty => only those ids.
    document_ids: Optional[List[str]] = None


class ChatResponse(BaseModel):
    reply: str
    mode_used: str
    thread_id: str
    approve_visible: bool
    timestamp: str
    sql: Optional[str] = None
    sql_valid: Optional[bool] = None
    retry_count: Optional[int] = None
    execution_ms: Optional[int] = None
    result: Optional[dict] = None
    row_count: Optional[int] = None
    answer: Optional[str] = None
    debug: Optional[str] = None
    schema: Optional[dict] = None
    sources: Optional[list] = None
    retrieved_contexts: Optional[list] = None
    db_choices: Optional[list] = None
    db_value: Optional[str] = None


class ApproveRequest(BaseModel):
    thread_id: str
    include_sample_data: bool = True


class CancelRequest(BaseModel):
    thread_id: str


class NewChatRequest(BaseModel):
    thread_id: str


class DeleteDatabaseRequest(BaseModel):
    db_name: str


class McpRunRequest(BaseModel):
    label: str


class DisconnectRequest(BaseModel):
    provider: str


class RAGQueryRequest(BaseModel):
    question: str
    document_ids: Optional[List[str]] = None
    # Thread/conversation id for follow-up memory (see agents/rag_agent/memory.py).
    # Optional so one-off queries (e.g. from a script) don't need one.
    conversation_id: Optional[str] = None


class RAGQueryResponse(BaseModel):
    answer: str
    sources: list
    retrieved_chunks: list
    debug: dict


# ---------------------------------------------------------------------------
# Core chat dispatch (shared by /api/chat)
# ---------------------------------------------------------------------------


def handle_message(req: ChatRequest) -> ChatResponse:
    message = (req.message or "").strip()
    text_lower = normalized_control_text(message)
    if not message:
        return ChatResponse(
            reply="Please type a message.",
            mode_used=req.mode,
            thread_id=req.thread_id,
            approve_visible=builder.has_pending_build(req.thread_id),
            timestamp=now_iso(),
        )

    pending_mcp = _PENDING_MCP_ACTIONS.get(req.thread_id)
    if pending_mcp and is_approval_text(message):
        reply, mode_used, success, debug = execute_pending_write(req.thread_id)
        response = ChatResponse(
            reply=reply,
            mode_used=mode_used,
            thread_id=req.thread_id,
            approve_visible=builder.has_pending_build(req.thread_id),
            timestamp=now_iso(),
            debug=debug,
            sources=[],
        )
        remember_auto_context(req.thread_id, response)
        return response

    if pending_mcp and is_cancel_text(message):
        _PENDING_MCP_ACTIONS.pop(req.thread_id, None)
        pending_goal = _THREAD_PENDING_GOAL.get(req.thread_id)
        if pending_goal and pending_goal.get("status") == "awaiting_approval":
            _THREAD_PENDING_GOAL.pop(req.thread_id, None)
        return ChatResponse(
            reply="Cancelled the pending tool action.",
            mode_used="Agentic Auto -> MCP Tool",
            thread_id=req.thread_id,
            approve_visible=builder.has_pending_build(req.thread_id),
            timestamp=now_iso(),
            debug="Agentic approval was cancelled before any external write action ran.",
            sources=[],
        )

    if (
        req.mode == "Auto"
        and is_approval_text(message)
        and not pending_mcp
        and not builder.has_pending_build(req.thread_id)
    ):
        # Never hand a bare approval to the chat model: it will happily invent a
        # "your email has been sent" confirmation for an action that never ran.
        return ChatResponse(
            reply=nothing_to_approve_reply(req.thread_id),
            mode_used="Agentic Auto",
            thread_id=req.thread_id,
            approve_visible=False,
            timestamp=now_iso(),
            debug="Approval received with no pending write action; nothing was executed.",
            sources=[],
        )

    if req.mode in UNAVAILABLE_MODES:
        return ChatResponse(
            reply=(
                f"**{req.mode}** isn't available yet in this build. Switch to Auto, Build "
                f"Database, Query Database, General Chat, or RAG to keep going."
            ),
            mode_used=req.mode,
            thread_id=req.thread_id,
            approve_visible=False,
            timestamp=now_iso(),
        )

    if req.mode == "Auto" and not (
        builder.has_pending_build(req.thread_id) and (text_lower in APPROVE_WORDS or text_lower in CANCEL_WORDS)
    ):
        return handle_auto_message(req, message, text_lower)

    if builder.has_pending_build(req.thread_id) and text_lower in APPROVE_WORDS:
        effective_mode = "Build Database"
        auto_decision = orchestrator.Decision("build", "Continuing a pending database approval.")
    elif builder.has_pending_build(req.thread_id) and text_lower in CANCEL_WORDS:
        effective_mode = "Build Database"
        auto_decision = orchestrator.Decision("build", "Cancelling a pending database approval.")
    else:
        auto_decision = resolve_auto_decision(req.mode, message, req.db_name)
        effective_mode = resolve_effective_mode(req.mode, message, req.db_name)

    sql_update = None
    sql_valid = None
    retry_count_out = None
    execution_ms = None
    result_update = None
    row_count = None
    answer_update = None
    debug_update = None
    schema_update = None
    sources_update = None
    retrieved_contexts_update = None
    approve_visible = False
    db_dropdown_choices = None
    db_dropdown_value = None
    reply = ""

    # -------------------------------------------------------------- Build --
    if effective_mode == "Build Database":
        if builder.has_pending_build(req.thread_id) and text_lower in APPROVE_WORDS:
            reply, success, new_db_name = do_approve(req.thread_id, req.include_sample_data)
            if success:
                db_dropdown_choices = db_choices()
                db_dropdown_value = new_db_name
                schema_update = schema_summary_safe(new_db_name)

        elif builder.has_pending_build(req.thread_id) and text_lower in CANCEL_WORDS:
            reply = do_cancel(req.thread_id)

        else:
            try:
                if builder.has_pending_build(req.thread_id):
                    reply = builder.refine_build(req.thread_id, message)
                else:
                    reply = builder.start_build(req.thread_id, message)
                approve_visible = True
            except SchemaGenerationError as exc:
                reply = f"Sorry, I couldn't design that schema: {exc}"
                approve_visible = builder.has_pending_build(req.thread_id)

    # --------------------------------------------------------------- MCP --
    elif effective_mode == "MCP Tool":
        if not auto_decision or not auto_decision.tool_key or not auto_decision.action_name:
            reply = "I picked the tool layer, but couldn't determine the exact action. Try naming the tool or action more directly."
        else:
            tool, action = find_mcp_tool_action(auto_decision.tool_key, auto_decision.action_name)
            if not tool or not action:
                reply = "I couldn't find that tool action."
            else:
                tool.connected = oauth.is_tool_connected(tool.key)
                output = _call_tool_with_payload(tool, action, confirmed=False, instruction=message, arguments=None)
                pending = bool(action.write_action and tool.connected and tool_output_success(output))
                if pending:
                    pending = store_pending_write(
                        req.thread_id,
                        tool=tool,
                        action=action,
                        goal=message,
                        instruction=message,
                        preview=output,
                        mode_path=f"Agentic Auto -> {tool.display_name}",
                    )
                if pending:
                    reply = (
                        f"I chose **{tool.display_name}** for this request.\n\n"
                        f"{output}\n\n"
                        "Reply **yes** to approve, or **cancel** to stop."
                    )
                else:
                    reply = f"I chose **{tool.display_name}** for this request.\n\n{output}"
                debug_update = f"Agentic decision: {orchestrator.describe_decision(auto_decision)}"
                sources_update = []

    # -------------------------------------------------------------- Query --
    elif effective_mode == "Query Database":
        if not req.db_name:
            reply = "Select a database and I can check that for you."
        else:
            db_path = sql_database.db_name_to_path(req.db_name)
            conversation_context = sql_memory.conversation_memory.get_context(req.thread_id)
            started_at = time.perf_counter()
            try:
                initial_state = {
                    "question": message,
                    "db_path": db_path,
                    "conversation_context": conversation_context,
                }
                run_config = {"configurable": {"thread_id": req.thread_id}}
                result = compiled_app.invoke(initial_state, config=run_config)
            except Exception as exc:  # noqa: BLE001 - LLM/API failures must not crash the UI
                logger.exception("Graph execution failed: %s", exc)
                reply = f"Sorry, something went wrong while answering that: {exc}"
            else:
                execution_ms = int((time.perf_counter() - started_at) * 1000)

                final_answer = result.get("final_answer", "I couldn't generate an answer.")
                sql_query = result.get("sql_query", "")
                sql_result = result.get("sql_result", {"columns": [], "rows": []})
                retry_history = result.get("retry_history", [])
                validation_error = result.get("validation_error", "")
                retry_count = result.get("retry_count", 0)
                execution_success = result.get("execution_success", False)

                sql_memory.conversation_memory.add_turn(req.thread_id, message, sql_query, final_answer)
                reply = final_answer

                sql_update = sql_query
                sql_valid = execution_success
                retry_count_out = retry_count
                result_update = result_to_table(sql_result)
                row_count = len(result_update["rows"])
                answer_update = final_answer
                debug_update = format_debug_info(retry_history, validation_error, retry_count)
                schema_update = schema_summary_safe(req.db_name)
                if execution_success:
                    sources_update = sources_for_query(req.db_name, sql_query, schema_update["tables"])
                else:
                    sources_update = []

    # ----------------------------------------------------------------- RAG --
    elif effective_mode == "RAG":
        if not rag_documents.has_ready_documents() and not req.document_ids:
            reply = (
                "There are no indexed documents yet. Attach a document with the paperclip "
                "in chat, wait until it shows Ready, then ask again."
            )
            sources_update = []
        else:
            try:
                rag_result = execute_rag_query(
                    question=message,
                    document_ids=req.document_ids,
                    conversation_id=req.thread_id,
                )
            except RAGError as exc:
                logger.exception("RAG agent failed: %s", exc)
                reply = f"Sorry, something went wrong while searching your documents: {exc}"
                sources_update = []
            except Exception as exc:  # noqa: BLE001 - must not crash the UI
                logger.exception("RAG agent failed unexpectedly: %s", exc)
                reply = f"Sorry, something went wrong while searching your documents: {exc}"
                sources_update = []
            else:
                reply = rag_result["answer"]
                answer_update = rag_result["answer"]
                sources_update = rag_result["sources"]
                retrieved_contexts_update = rag_result.get("retrieved_contexts")
                debug_update = format_rag_debug_info(rag_result["debug"])

    # ------------------------------------------------------------ General --
    else:
        history_for_llm = [tuple(turn) for turn in req.history[-(config.MAX_MEMORY_TURNS * 2):]]
        try:
            reply = general_chat.respond(message, history_for_llm)
        except Exception as exc:  # noqa: BLE001
            logger.exception("General chat failed: %s", exc)
            reply = f"Sorry, something went wrong: {exc}"
        sources_update = []

    if req.mode == "Auto":
        mode_badge = f"Agentic Auto -> {effective_mode}"
    else:
        mode_badge = effective_mode
    if auto_decision and not debug_update:
        debug_update = f"Agentic decision: {orchestrator.describe_decision(auto_decision)}"

    return ChatResponse(
        reply=reply,
        mode_used=mode_badge,
        thread_id=req.thread_id,
        approve_visible=approve_visible,
        timestamp=now_iso(),
        sql=sql_update,
        sql_valid=sql_valid,
        retry_count=retry_count_out,
        execution_ms=execution_ms,
        result=result_update,
        row_count=row_count,
        answer=answer_update,
        debug=debug_update,
        schema=schema_update,
        sources=sources_update,
        retrieved_contexts=retrieved_contexts_update,
        db_choices=db_dropdown_choices,
        db_value=db_dropdown_value,
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="AgentHub Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/bootstrap")
def bootstrap():
    # New sessions start with no database selected. A database in the
    # dropdown is available for SQL turns - it is never auto-attached.
    return {
        "thread_id": new_thread_id(),
        "modes": MODES,
        "unavailable_modes": list(UNAVAILABLE_MODES),
        "db_choices": db_choices(),
        "default_db": None,
        "schema": schema_summary_safe(None),
        "mcp_sidebar_lines": sidebar_lines(),
        "mcp_actions": list(_MCP_ACTION_MAP.keys()),
        "build_examples": BUILD_EXAMPLES,
        "query_examples": QUERY_EXAMPLES,
        "rag_document_count": len(rag_documents.ready_document_ids()),
    }


@app.get("/api/databases")
def get_databases():
    return {"choices": db_choices()}


@app.get("/api/databases/detail")
def get_databases_detail():
    details = []
    for name in db_choices():
        db_path = sql_database.db_name_to_path(name)
        try:
            stat = os.stat(db_path)
            summary = sql_database.get_schema_summary(db_path)
            details.append({
                "name": name,
                "table_count": len(summary["tables"]),
                "row_count": sum(t.get("row_count", 0) for t in summary["tables"]),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            })
        except OSError:
            continue
    return {"databases": details}


@app.post("/api/databases/delete")
def delete_database(req: DeleteDatabaseRequest):
    try:
        deleted_path = db_creator.delete_database(req.db_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(
            status_code=423,
            detail=f"Database '{req.db_name}' could not be deleted because it is locked or unavailable.",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Database '{req.db_name}' could not be deleted right now. Close any active connection and try again.",
        ) from exc

    sql_cache.clear_schema_cache(deleted_path)
    default_db = default_db_choice()
    return {
        "deleted": req.db_name,
        "db_choices": db_choices(),
        "default_db": default_db,
        "schema": schema_summary_safe(default_db),
        "message": f"Deleted {req.db_name}.",
    }


@app.get("/api/schema")
def get_schema(db_name: Optional[str] = None):
    return schema_summary_safe(db_name)


@app.get("/api/settings")
def get_settings():
    return {
        "model_name": config.MODEL_NAME,
        "openai_configured": bool(config.OPENAI_API_KEY),
        "max_retries": config.MAX_RETRIES,
        "max_memory_turns": config.MAX_MEMORY_TURNS,
        "default_db_name": config.DEFAULT_DB_NAME,
    }


@app.post("/api/rag/upload")
async def rag_upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file name was provided.")

    content = await file.read()
    try:
        record, is_duplicate = rag_ingestion.ingest_file(content, file.filename)
    except RAGError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response = record.to_public_dict()
    if is_duplicate:
        response["status"] = "already_indexed"
    return response


@app.get("/api/rag/documents")
def rag_list_documents():
    return {"documents": [r.to_public_dict() for r in rag_documents.list_documents()]}


@app.get("/api/rag/documents/{document_id}")
def rag_get_document(document_id: str):
    record = rag_documents.get_document(document_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found.")
    return record.to_public_dict()


@app.delete("/api/rag/documents/{document_id}")
def rag_delete_document(document_id: str):
    deleted = rag_ingestion.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"status": "deleted", "document_id": document_id}


@app.post("/api/rag/query", response_model=RAGQueryResponse)
def api_rag_query(req: RAGQueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required.")
    if not rag_documents.has_ready_documents() and not req.document_ids:
        raise HTTPException(status_code=400, detail="No indexed documents are available yet.")

    try:
        result = execute_rag_query(
            question=req.question,
            document_ids=req.document_ids,
            conversation_id=req.conversation_id,
        )
    except RAGError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return RAGQueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        retrieved_chunks=result["retrieved_contexts"],
        debug=result["debug"],
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return handle_message(req)


@app.post("/api/approve", response_model=ChatResponse)
def approve(req: ApproveRequest):
    reply, success, new_db_name = do_approve(req.thread_id, req.include_sample_data)
    return ChatResponse(
        reply=reply,
        mode_used="Build Database",
        thread_id=req.thread_id,
        approve_visible=False,
        timestamp=now_iso(),
        schema=schema_summary_safe(new_db_name) if success else None,
        db_choices=db_choices() if success else None,
        db_value=new_db_name if success else None,
    )


@app.post("/api/cancel", response_model=ChatResponse)
def cancel(req: CancelRequest):
    reply = do_cancel(req.thread_id)
    return ChatResponse(
        reply=reply, mode_used="Build Database", thread_id=req.thread_id, approve_visible=False,
        timestamp=now_iso(),
    )


@app.post("/api/new-chat")
def new_chat(req: NewChatRequest):
    # Conversation-scoped reset only. Do not touch OAuth, indexed docs, or DB files.
    sql_memory.conversation_memory.clear(req.thread_id)
    builder.cancel_build(req.thread_id)
    _THREAD_AUTO_CONTEXT.pop(req.thread_id, None)
    _PENDING_MCP_ACTIONS.pop(req.thread_id, None)
    _THREAD_PENDING_GOAL.pop(req.thread_id, None)
    _THREAD_KNOWN_CONTACTS.pop(req.thread_id, None)
    try:
        from agents.rag_agent import memory as rag_memory
        rag_memory.conversation_memory.clear(req.thread_id)
    except Exception:  # noqa: BLE001
        pass
    fresh_id = new_thread_id()
    return {"thread_id": fresh_id, "default_db": None}


@app.post("/api/mcp-run")
def mcp_run(req: McpRunRequest):
    if not req.label or req.label not in _MCP_ACTION_MAP:
        return {"output": "Pick a tool action above, then click Run."}
    tool, action = _MCP_ACTION_MAP[req.label]
    tool.connected = oauth.is_tool_connected(tool.key)
    return {"output": tool.call(action.name, confirmed=False, instruction=req.label)}


@app.get("/api/mcp-tools")
def mcp_tools():
    return {
        "tools": [
            {
                "key": tool.key,
                "display_name": tool.display_name,
                "icon": tool.icon,
                "description": tool.description,
                "connected": oauth.is_tool_connected(tool.key),
                "actions": [
                    {"name": a.name, "description": a.description, "write_action": a.write_action}
                    for a in tool.actions
                ],
            }
            for tool in ALL_TOOLS
        ]
    }


@app.get("/api/integrations/status")
def integrations_status():
    return oauth.integration_status()


@app.post("/api/integrations/disconnect")
def disconnect_integration(req: DisconnectRequest):
    oauth.disconnect(req.provider)
    return oauth.integration_status()


@app.get("/auth/google/login")
def google_login():
    try:
        return RedirectResponse(oauth.google_login_url())
    except oauth.OAuthConfigError as exc:
        return HTMLResponse(
            f"<h1>Google OAuth is not configured</h1><p>{exc}</p>"
            "<p>Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET to .env, then restart the app.</p>",
            status_code=400,
        )


@app.get("/auth/google/callback")
def google_callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    if error:
        return HTMLResponse(f"<h1>Google connection cancelled</h1><p>{error}</p>", status_code=400)
    if not code or not state:
        return HTMLResponse("<h1>Google callback is missing code or state.</h1>", status_code=400)
    try:
        oauth.finish_google_callback(code, state)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Google OAuth callback failed: %s", exc)
        return HTMLResponse(f"<h1>Google connection failed</h1><p>{exc}</p>", status_code=400)
    return RedirectResponse("/?connected=google")


# Static frontend last, so /api/* routes above take precedence.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(STATIC_DIR / "index.html"))
else:
    logger.warning("static/ directory not found - the web UI will not be served.")


if __name__ == "__main__":
    import uvicorn

    # Canonical local entrypoint. Prefer:
    #   .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8010
    uvicorn.run("app:app", host="127.0.0.1", port=8010, reload=False)
