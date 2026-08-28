"""
orchestrator.py
===============
Turn-scoped planner for Agentic Auto mode.

Auto is no longer a one-shot router. Each user turn becomes a fresh goal,
then app.py executes a bounded plan one capability at a time and feeds back
real observations. When enabled, a structured LLM planner proposes the
operational plan from a sanitized resource summary; the deterministic planner
remains the validation fallback.

The legacy router.py classifier remains available only as a database fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

import config
import router
from tools.mcp.registry import ALL_TOOLS

logger = config.get_logger(__name__)

MAX_AUTO_STEPS = 6


@dataclass(frozen=True)
class Capability:
    key: str
    display_name: str
    description: str
    constraints: str
    read_only: bool = True
    implemented: bool = True
    requires_connection: bool = False


@dataclass
class PlanStep:
    capability: str
    action: str
    reason: str
    write_action: bool = False
    depends_on_previous: bool = False


@dataclass
class Observation:
    capability: str
    action: str
    success: bool
    output: str
    pending_approval: bool = False


@dataclass
class PlanningContext:
    has_selected_database: bool
    has_rag_documents: bool
    has_sql_context: bool = False
    has_last_answer: bool = False
    # True when the previous Auto turn in this thread successfully used RAG.
    has_rag_context: bool = False
    selected_database_name: Optional[str] = None
    rag_documents: list[dict[str, Any]] = field(default_factory=list)
    connected_capabilities: dict[str, bool] = field(default_factory=dict)
    last_tool_summary: str = ""
    use_llm_planner: bool = False


@dataclass
class AutoPlan:
    goal: str
    intent: str
    steps: list[PlanStep]
    observations: list[Observation] = field(default_factory=list)
    current_step: int = 0
    completion_status: str = "running"
    missing_info: Optional[str] = None
    missing_fields: list[str] = field(default_factory=list)

    @property
    def executed_path(self) -> list[str]:
        labels: list[str] = []
        for obs in self.observations:
            name = CAPABILITY_REGISTRY.get(obs.capability, Capability(obs.capability, obs.capability, "", "")).display_name
            if name not in labels:
                labels.append(name)
        return labels


TOOL_BY_KEY = {tool.key: tool for tool in ALL_TOOLS}

CAPABILITY_REGISTRY: dict[str, Capability] = {
    "sql_agent": Capability(
        key="sql_agent",
        display_name="SQL",
        description="Answer read-only questions about the selected SQLite database.",
        constraints=(
            "Use only when the user is asking about data in the selected database. "
            "Do not use just because a database is selected or because the previous turn used SQL."
        ),
    ),
    "database_builder": Capability(
        key="database_builder",
        display_name="Build Database",
        description="Design a new local SQLite database schema from plain English.",
        constraints="Requires explicit approval before creating the database file.",
        read_only=False,
    ),
    "general_chat": Capability(
        key="general_chat",
        display_name="General Chat",
        description="Answer conversational requests that need no specialized tool.",
        constraints=(
            "Must not claim facts from databases, Gmail, Calendar, RAG, or other tools. "
            "Must never tell the user to switch modes or manually pick an internal tool."
        ),
    ),
    "rag_agent": Capability(
        key="rag_agent",
        display_name="RAG",
        description=(
            "Answer questions using uploaded/attached documents (retrieval or document overview)."
        ),
        constraints=(
            "Use when the user asks about attached/uploaded document contents, including "
            "summaries, headings/titles, comparisons, and pronoun references like "
            "'this file' / 'the document' / 'it' when documents are available. "
            "Do not use for unrelated chat, SQL, email, calendar, or weather. "
            "If no document is available, ask them to attach one — never tell them to switch modes."
        ),
    ),
    "date_time": Capability(
        key="date_time",
        display_name="Date/Time",
        description="Resolve relative dates, explicit dates, and current local time.",
        constraints="May resolve dates and checks them against an offline holiday calendar (config.HOLIDAY_CALENDAR_COUNTRY).",
    ),
    "calculator": Capability(
        key="calculator",
        display_name="Calculator",
        description="Evaluate arithmetic, percentages, and simple numeric expressions.",
        constraints="Use only for deterministic calculations; do not infer business data.",
    ),
    "weather": Capability(
        key="weather",
        display_name="Weather",
        description="Weather lookup.",
        constraints="No real weather provider is currently implemented.",
        implemented=False,
    ),
}

for _tool in ALL_TOOLS:
    CAPABILITY_REGISTRY[_tool.key] = Capability(
        key=_tool.key,
        display_name=_tool.display_name,
        description=_tool.description,
        constraints=(
            "Read actions may run when connected. Write actions require approval. "
            "If the tool action is a placeholder, report that no real action was performed."
        ),
        read_only=not any(action.write_action for action in _tool.actions),
        requires_connection=True,
    )


@dataclass
class Decision:
    """Legacy shape kept for explicit fallback callers."""

    kind: str  # build, query, general, mcp, rag
    reason: str
    tool_key: Optional[str] = None
    action_name: Optional[str] = None


class LLMPlanStep(BaseModel):
    capability: str = Field(description="One registered capability key.")
    action: str = Field(description="One supported action for that capability.")
    reason: str = Field(description="Short operational reason, no hidden reasoning.")
    depends_on_previous: bool = False


class LLMPlannerOutput(BaseModel):
    intent: str = Field(description="Short intent label such as general, sql, rag, leave_request.")
    goal: str = Field(description="User-facing goal summary.")
    tools_needed: bool = Field(description="False for general questions that need no capability.")
    steps: list[LLMPlanStep] = Field(default_factory=list)
    missing_info: Optional[str] = Field(default=None, description="Ask only for missing required information.")
    requires_approval: bool = Field(default=False, description="True when a planned write/risky step needs approval.")
    next_action: str = Field(default="execute", description="execute, ask_user, or answer_general.")


_llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=0,
    api_key=config.OPENAI_API_KEY,
) if config.OPENAI_API_KEY else None


_RAG_DOC_WORDS = [
    "document", "documents", "pdf", "file", "files", "report", "reports",
    "handbook", "policy", "policies", "upload", "uploaded", "docs",
    "attachment", "attachments", "attached", "paper", "essay", "assignment",
]
_RAG_INTENT_HINTS = [
    "summarize", "summary", "summarise", "compare", "according to",
    "say about", "says about", "said about", "mention", "state that",
    "what does", "tell me about", "explain the", "find",
    "heading", "title", "about", "conclusion", "conclude", "concludes",
    "overview", "key points",
]
# Pronoun / deixis that can refer to the currently attached document(s).
_RAG_ATTACHMENT_REFS = [
    "this file", "that file", "the file", "this document", "that document",
    "the document", "this pdf", "the pdf", "this attachment", "the attachment",
    "attached file", "attached document", "these files", "these documents",
    "both documents", "both files", "the attached",
]
_SELF_QUESTION_RE = re.compile(
    r"\b(?:"
    r"who\s+are\s+(?:you|u)|"
    r"what(?:'s|\s+is)\s+(?:your|ur)\s+name|"
    r"what\s+(?:do|can)\s+(?:you|u)\s+do|"
    r"what\s+is\s+it\s+(?:that\s+)?(?:you|u)\s+do|"
    r"how\s+(?:do|can|could|would)\s+(?:you|u)\s+help(?:\s+me)?|"
    r"tell\s+me\s+about\s+(?:yourself|urself)"
    r")\b",
    re.IGNORECASE,
)
# Date grammar shared with app.execute_date_time so the planner and the
# resolver always agree on what counts as a date.
_MONTH_TOKEN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_RANGE_JOIN = r"\s*(?:to|through|until|till|thru|-|–|—)\s*"
# Misspelling "tomorrow" should not turn a complete request into an interrogation.
TOMORROW_WORDS = ("tomorrow", "tommorow", "tommorrow", "tomorow", "tomorrw", "tmrw")
# "august 14", "aug16th", "august 14 to 15"
MONTH_FIRST_DATE_RE = re.compile(
    rf"\b({_MONTH_TOKEN})\s*(\d{{1,2}})(?!\d)(?:st|nd|rd|th)?"
    rf"(?:{_RANGE_JOIN}(\d{{1,2}})(?!\d)(?:st|nd|rd|th)?)?(?:,?\s*(\d{{4}}))?\b",
    re.IGNORECASE,
)
# Day-before-month forms. An ordinal suffix or an explicit "of" is required so
# that phrases like "3 may be enough" are not read as a date.
DAY_FIRST_DATE_RES = (
    re.compile(
        rf"\b(\d{{1,2}})(?!\d)(?:st|nd|rd|th)"
        rf"(?:{_RANGE_JOIN}(\d{{1,2}})(?!\d)(?:st|nd|rd|th)?)?"
        rf"\s*(?:of\s*)?({_MONTH_TOKEN})\b(?:,?\s*(\d{{4}}))?",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(\d{{1,2}})(?!\d)(?:{_RANGE_JOIN}(\d{{1,2}})(?!\d))?"
        rf"\s*of\s*({_MONTH_TOKEN})\b(?:,?\s*(\d{{4}}))?",
        re.IGNORECASE,
    ),
)

_WEATHER_HINTS = ["weather", "forecast", "temperature", "rain", "humidity"]
_BUILD_HINTS = ["create a database", "build a database", "make a database", "set up a database", "design a schema"]
_EMAIL_HINTS = ["gmail", "email", "e-mail", "mail", "inbox", "send this", "send that", "send it", "email me"]
_CALENDAR_HINTS = ["calendar", "meeting", "meetings", "schedule", "event", "events", "appointment"]
_LEAVE_HINTS = ["leave", "pto", "vacation", "sick day", "sick leave", "time off"]
# Hard blocks for SQL. "email"/"mail" are handled separately so a combined
# "look up X and email me" goal can still plan SQL → Gmail.
_HARD_SQL_BLOCKS = _LEAVE_HINTS + ["calendar", "meeting", "holiday", "weather"]
_NEW_INTENT_BLOCKS_SQL = _HARD_SQL_BLOCKS + ["email", "mail"]


def _supported_actions() -> dict[str, set[str]]:
    actions = {
        "general_chat": {"respond"},
        "sql_agent": {"query"},
        "database_builder": {"start_or_refine"},
        "rag_agent": {"query"},
        "date_time": {"resolve_date", "current_time"},
        "calculator": {"calculate"},
        "weather": {"lookup"},
    }
    for tool in ALL_TOOLS:
        actions[tool.key] = {action.name for action in tool.actions}
    return actions


def _action_is_write(capability: str, action_name: str) -> bool:
    if capability == "database_builder":
        return True
    tool = TOOL_BY_KEY.get(capability)
    if not tool:
        return False
    action = next((item for item in tool.actions if item.name == action_name), None)
    return bool(action and action.write_action)


def _plan_turn_with_llm(message: str, context: PlanningContext) -> Optional[AutoPlan]:
    if not _llm:
        return None

    actions = _supported_actions()
    capability_lines = [
        f"- {cap['key']}: {cap['display_name']} | actions={sorted(actions.get(cap['key'], []))} | "
        f"use={cap['description']} | constraints={cap['constraints']}"
        for cap in describe_capabilities()
    ]
    safe_resources = {
        "database_selected": context.has_selected_database,
        "document_count_available": len(context.rag_documents) if context.rag_documents else int(context.has_rag_documents),
        "document_statuses": sorted({str(doc.get("status", "")) for doc in context.rag_documents if doc.get("status")})[:6],
        "sql_followup_available": context.has_sql_context,
        "rag_followup_available": context.has_rag_context,
        "last_answer_available": context.has_last_answer,
        "connected_capability_keys": sorted(k for k, connected in context.connected_capabilities.items() if connected),
    }
    system = (
        "You are AgentHub Studio's Auto capability planner. Return only structured operational fields; "
        "do not expose chain-of-thought. Treat each user turn as a fresh goal. "
        "Selected databases, attached documents, previous tool results, and connected services are available resources, not forced routes. "
        "General identity/explanation/chat questions should use general_chat.respond only. "
        "Never select SQL merely because a database is selected. Never select RAG merely because a document is attached. "
        "Select one or more steps only when required to complete the goal. "
        "Writes or risky external actions require preview and approval; never plan silent execution. "
        "If required info is missing, ask only for that missing info. "
        "Never invent tool results, connection state, recipients, dates, document content, SQL rows, sources, or holidays."
    )
    user = (
        "Capabilities:\n"
        + "\n".join(capability_lines)
        + "\n\nAvailable resource summary JSON:\n"
        + json.dumps(safe_resources)
        + "\n\nUser message:\n"
        + message
    )
    try:
        raw_plan = _llm.with_structured_output(LLMPlannerOutput).invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return _validate_llm_plan(raw_plan, message, context)
    except Exception as exc:  # noqa: BLE001 - planner fallback must be graceful
        logger.warning("LLM Auto planner failed or returned an invalid plan; using rule fallback: %s", exc)
        return None


def _validate_llm_plan(raw_plan: LLMPlannerOutput, message: str, context: PlanningContext) -> AutoPlan:
    if not isinstance(raw_plan, LLMPlannerOutput):
        raw_plan = LLMPlannerOutput.model_validate(raw_plan)

    if raw_plan.missing_info and raw_plan.next_action == "ask_user":
        return AutoPlan(
            goal=raw_plan.goal or message,
            intent=raw_plan.intent or "needs_user",
            steps=[],
            missing_info=raw_plan.missing_info,
            completion_status="needs_user",
        )

    supported = _supported_actions()
    text = _normalize(message)
    validated_steps: list[PlanStep] = []
    for item in raw_plan.steps[:MAX_AUTO_STEPS]:
        if item.capability not in supported:
            raise ValueError(f"unknown capability {item.capability}")
        if item.action not in supported[item.capability]:
            raise ValueError(f"unknown action {item.capability}.{item.action}")
        # The LLM may propose RAG merely because attachments are available.
        # Apply the same document-intent gate used by the rule planner before
        # allowing Auto mode to enter the unchanged RAG execution path.
        if item.capability == "rag_agent" and not _wants_document_capability(text, context):
            continue
        if item.capability == "sql_agent" and not context.has_selected_database:
            return AutoPlan(message, "sql_needs_database", [], missing_info="Which database would you like me to use?")
        if item.capability == "rag_agent" and not context.has_rag_documents:
            return AutoPlan(
                message,
                "rag_needs_document",
                [],
                missing_info="Attach or select the document you'd like me to check, then ask again once it shows Ready.",
            )
        if item.capability == "gmail" and item.action in {"send_email", "draft_reply"}:
            missing: list[str] = []
            if not (_has_email_recipient(text) or context.has_last_answer):
                missing.append("email_recipient")
            if not (
                _has_email_body_expression(text)
                or _is_context_send(text)
                or context.has_last_answer
            ):
                missing.append("email_body")
            if missing:
                if missing == ["email_recipient", "email_body"]:
                    missing_info = "Who should I send it to, and what should it say?"
                    intent_name = "email_needs_details"
                elif "email_recipient" in missing:
                    missing_info = "Who should I send this to?"
                    intent_name = "email_needs_recipient"
                else:
                    missing_info = "What should the email say?"
                    intent_name = "email_needs_body"
                plan = AutoPlan(message, intent_name, [], missing_info=missing_info)
                plan.missing_fields = missing
                return plan
        validated_steps.append(
            PlanStep(
                item.capability,
                item.action,
                item.reason[:240] or "Selected by the Auto planner.",
                write_action=_action_is_write(item.capability, item.action),
                depends_on_previous=item.depends_on_previous,
            )
        )

    if not validated_steps:
        validated_steps = [PlanStep("general_chat", "respond", "No specialized capability is required for this turn.")]

    if _is_leave_request(text) and _has_date_expression(text) and any(
        step.capability == "gmail" and step.action == "send_email" for step in validated_steps
    ) and not any(step.capability == "date_time" for step in validated_steps):
        validated_steps.insert(0, PlanStep("date_time", "resolve_date", "Resolve the requested leave date before preparing the email."))

    return AutoPlan(raw_plan.goal or message, raw_plan.intent or "planned", validated_steps, missing_info=raw_plan.missing_info)


def plan_turn(message: str, context: PlanningContext) -> AutoPlan:
    if context.use_llm_planner:
        llm_plan = _plan_turn_with_llm(message, context)
        if llm_plan:
            rule_plan = _rule_plan_turn(message, context)
            if _llm_plan_lost_operational_intent(llm_plan, rule_plan):
                return rule_plan
            return llm_plan
    return _rule_plan_turn(message, context)


def _llm_plan_lost_operational_intent(llm_plan: AutoPlan, rule_plan: AutoPlan) -> bool:
    """Never let the LLM downgrade clear tool/workflow requests to chat or to an
    open-ended question. A plan with no operational step (chat-only *or* a bare
    ask_user with no steps at all) must not beat a rule plan that knows how to
    make progress."""
    rule_caps = {step.capability for step in rule_plan.steps}
    llm_caps = {step.capability for step in llm_plan.steps}
    operational_caps = {
        "gmail", "google_calendar", "google_drive", "sql_agent",
        "rag_agent", "database_builder", "date_time", "calculator", "weather",
    }

    if not rule_plan.steps:
        # The rule planner knows exactly which slots are required; prefer its
        # narrow question over an invented one (duration, reason, subject...).
        return bool(rule_plan.missing_info) and not (llm_caps & operational_caps)

    return bool(rule_caps & operational_caps) and not (llm_caps & operational_caps)


def _rule_plan_turn(message: str, context: PlanningContext) -> AutoPlan:
    text = _normalize(message)
    steps: list[PlanStep] = []
    intent = "general"

    if _mentions_any(text, _BUILD_HINTS):
        intent = "database_build"
        steps.append(PlanStep("database_builder", "start_or_refine", "The user is asking to create or change a database schema."))
        return AutoPlan(goal=message, intent=intent, steps=steps)

    if _is_leave_request(text):
        intent = "leave_request"
        if not _has_date_expression(text):
            plan = AutoPlan(goal=message, intent="leave_needs_date", steps=[])
            plan.missing_info = "What date should I request leave for?"
            plan.missing_fields = ["date"]
            return plan
        steps.append(PlanStep("date_time", "resolve_date", "Resolve the requested leave date and check it against the holiday calendar."))
        plan = AutoPlan(goal=message, intent=intent, steps=steps)
        if _mentions_any(text, ["email", "mail", "send", "request"]) and _has_email_recipient(text):
            steps.append(PlanStep("gmail", "send_email", "Send the leave request only after approval.", write_action=True))
            plan.steps = steps
        else:
            plan.missing_info = "I can resolve the date, but I need a recipient email before preparing a leave request."
            plan.missing_fields = ["email_recipient"]
        return plan

    # Weather before RAG so an attached file does not steal unrelated asks.
    if _wants_weather(text):
        intent = "weather"
        steps.append(PlanStep("weather", "lookup", "The user is asking about weather."))
        return AutoPlan(goal=message, intent=intent, steps=steps)

    document_question = _wants_document_capability(text, context)
    if document_question and not context.has_rag_documents:
        plan = AutoPlan(goal=message, intent="rag_needs_document", steps=[])
        plan.missing_info = (
            "Attach or select the document you'd like me to check, then ask again once it shows Ready."
        )
        return plan

    wants_rag = document_question
    wants_email = _wants_email(text)
    wants_calendar = _wants_calendar(text)
    wants_calendar_create = _wants_calendar_create(text)
    sql_cues = _has_structured_data_cues(message, text)
    sql_intent = _looks_like_database_question(message, text, context)

    if wants_rag and wants_email:
        intent = "rag_then_email"
        steps.append(PlanStep("rag_agent", "query", "Find the requested information in uploaded documents."))
        steps.append(PlanStep("gmail", "send_email", "Email the document-grounded answer after approval.", write_action=True, depends_on_previous=True))
        return AutoPlan(goal=message, intent=intent, steps=steps)

    # Data lookup + email in one goal: run SQL first, then prepare Gmail.
    if wants_email and sql_cues and not _is_context_send(text):
        if _is_vague_database_request(text):
            plan = AutoPlan(goal=message, intent="sql_needs_clarification", steps=[])
            plan.missing_info = (
                "Which database would you like me to use, and what should I look up?"
                if not context.has_selected_database
                else "What would you like to know from the database before I email it?"
            )
            plan.missing_fields = ["database", "lookup"] if not context.has_selected_database else ["lookup"]
            return plan
        if not context.has_selected_database:
            plan = AutoPlan(goal=message, intent="sql_needs_database", steps=[])
            plan.missing_info = "Which database would you like me to use?"
            plan.missing_fields = ["database"]
            return plan
        if not _has_email_recipient(text):
            plan = AutoPlan(goal=message, intent="email_needs_recipient", steps=[])
            plan.missing_info = "Who should I send this to?"
            plan.missing_fields = ["email_recipient"]
            return plan
        intent = "sql_then_email"
        steps.append(PlanStep("sql_agent", "query", "Look up the requested data before emailing."))
        steps.append(
            PlanStep(
                "gmail",
                "send_email",
                "Email the SQL result after approval.",
                write_action=True,
                depends_on_previous=True,
            )
        )
        return AutoPlan(goal=message, intent=intent, steps=steps)

    if wants_email and _is_context_send(text):
        intent = "email_followup"
        steps.append(PlanStep("gmail", "send_email", "Send the previous answer/result after approval.", write_action=True, depends_on_previous=True))
        return AutoPlan(goal=message, intent=intent, steps=steps)

    if wants_calendar_create:
        missing = []
        if not _has_date_expression(text):
            missing.append("date")
        if not _has_time_expression(text):
            missing.append("time")
        if not _has_email_recipient(text):
            missing.append("attendee_email")
        if missing:
            plan = AutoPlan(goal=message, intent="calendar_needs_info", steps=[])
            labels = {
                "date": "date",
                "time": "time",
                "attendee_email": "attendee email",
            }
            plan.missing_fields = missing
            plan.missing_info = "Please provide the " + ", ".join(labels[item] for item in missing) + "."
            return plan
        intent = "calendar_create"
        steps.append(PlanStep("date_time", "resolve_date", "Resolve the meeting date before preparing the event."))
        steps.append(PlanStep("google_calendar", "create_meeting", "Create the calendar event only after approval.", write_action=True))
        return AutoPlan(goal=message, intent=intent, steps=steps)

    if wants_calendar:
        intent = "calendar"
        steps.append(PlanStep("google_calendar", "check_schedule", "Read the relevant calendar events before any write or email."))
        if wants_email:
            intent = "calendar_then_email"
            steps.append(PlanStep("gmail", "send_email", "Email attendees only if Calendar returns real event information.", write_action=True, depends_on_previous=True))
        return AutoPlan(goal=message, intent=intent, steps=steps)

    if wants_email:
        intent = "email"
        if _mentions_any(text, ["inbox", "summarize inbox", "summarise inbox"]):
            action = "summarize_inbox"
        elif _mentions_any(text, ["draft", "compose", "write an email", "write a mail"]):
            action = "draft_reply"
        elif _mentions_any(text, ["send", "email me", "mail me", "send an email", "send email", "email"]):
            action = "send_email"
        elif _has_email_recipient(text) and _has_email_body_expression(text):
            # A recipient plus a body is an outgoing message, never a request to
            # read the inbox.
            action = "send_email"
        else:
            action = "summarize_inbox"
        if action in {"send_email", "draft_reply"}:
            missing: list[str] = []
            if not _has_email_recipient(text):
                missing.append("email_recipient")
            has_body = (
                _has_email_body_expression(text)
                or _is_context_send(text)
                or context.has_last_answer
            )
            if not has_body:
                missing.append("email_body")
            if missing:
                if missing == ["email_recipient", "email_body"]:
                    missing_info = "Who should I send it to, and what should it say?"
                    intent_name = "email_needs_details"
                elif missing == ["email_recipient"]:
                    missing_info = "Who should I send this to?"
                    intent_name = "email_needs_recipient"
                else:
                    missing_info = "What should the email say?"
                    intent_name = "email_needs_body"
                plan = AutoPlan(goal=message, intent=intent_name, steps=[])
                plan.missing_info = missing_info
                plan.missing_fields = missing
                return plan
        steps.append(PlanStep("gmail", action, "The user is asking for Gmail.", write_action=(action in {"send_email", "draft_reply"})))
        return AutoPlan(goal=message, intent=intent, steps=steps)

    if wants_rag:
        intent = "rag"
        steps.append(PlanStep("rag_agent", "query", "The user is asking about uploaded/indexed document contents."))
        return AutoPlan(goal=message, intent=intent, steps=steps)

    if _wants_date_time(text):
        intent = "date_time"
        steps.append(PlanStep("date_time", "resolve_date", "Resolve date/time from the local clock."))
        return AutoPlan(goal=message, intent=intent, steps=steps)

    if _wants_calculation(text):
        intent = "calculation"
        steps.append(PlanStep("calculator", "calculate", "The user is asking for a deterministic calculation."))
        return AutoPlan(goal=message, intent=intent, steps=steps)

    if sql_intent:
        if _is_vague_database_request(text):
            plan = AutoPlan(goal=message, intent="sql_needs_clarification", steps=[])
            if not context.has_selected_database:
                plan.missing_info = (
                    "Which database would you like me to use, and what should I look up?"
                )
                plan.missing_fields = ["database", "lookup"]
            else:
                plan.missing_info = "What would you like to know from the database?"
                plan.missing_fields = ["lookup"]
            return plan
        if not context.has_selected_database:
            plan = AutoPlan(goal=message, intent="sql_needs_database", steps=[])
            plan.missing_info = "Which database would you like me to use?"
            plan.missing_fields = ["database"]
            return plan
        intent = "sql"
        steps.append(PlanStep("sql_agent", "query", "The user needs structured data from the selected database."))
        return AutoPlan(goal=message, intent=intent, steps=steps)

    # Default / uncertain: General Chat is a first-class outcome, never SQL.
    steps.append(PlanStep("general_chat", "respond", "No specialized capability is required for this turn."))
    return AutoPlan(goal=message, intent="general_conversation", steps=steps)


def next_step(plan: AutoPlan) -> Optional[PlanStep]:
    if plan.completion_status != "running":
        return None
    if plan.current_step >= len(plan.steps):
        plan.completion_status = "complete"
        return None
    if plan.current_step >= MAX_AUTO_STEPS:
        plan.completion_status = "failed"
        plan.missing_info = "The Auto plan hit its step limit before completing."
        return None
    return plan.steps[plan.current_step]


def observe(plan: AutoPlan, observation: Observation) -> None:
    plan.observations.append(observation)
    if observation.pending_approval:
        plan.completion_status = "pending_approval"
        return
    if not observation.success:
        plan.completion_status = "failed"
        return
    plan.current_step += 1
    if plan.current_step >= len(plan.steps):
        plan.completion_status = "complete"


def describe_capabilities() -> list[dict]:
    return [
        {
            "key": cap.key,
            "display_name": cap.display_name,
            "description": cap.description,
            "constraints": cap.constraints,
            "implemented": cap.implemented,
            "requires_connection": cap.requires_connection,
            "read_only": cap.read_only,
        }
        for cap in CAPABILITY_REGISTRY.values()
    ]


def mode_path(plan: AutoPlan) -> str:
    names = plan.executed_path
    if not names:
        names = [CAPABILITY_REGISTRY.get(step.capability, Capability(step.capability, step.capability, "", "")).display_name for step in plan.steps]
    return "Agentic Auto" + (" -> " + " -> ".join(dict.fromkeys(names)) if names else "")


def decide(message: str, has_selected_database: bool, has_rag_documents: bool = False) -> Decision:
    """Legacy compatibility for non-Auto callers that still expect one mode."""
    context = PlanningContext(has_selected_database=has_selected_database, has_rag_documents=has_rag_documents)
    plan = plan_turn(message, context)
    first = plan.steps[0] if plan.steps else PlanStep("general_chat", "respond", "No specialized tool was needed.")
    return _step_to_decision(first)


def describe_decision(decision: Decision) -> str:
    if decision.kind != "mcp":
        return decision.reason
    tool = TOOL_BY_KEY.get(decision.tool_key or "")
    action = next((a for a in (tool.actions if tool else []) if a.name == decision.action_name), None)
    if not tool or not action:
        return decision.reason
    return f"Selected {tool.display_name} -> {action.description}. {decision.reason}"


def _step_to_decision(step: PlanStep) -> Decision:
    if step.capability == "database_builder":
        return Decision("build", step.reason)
    if step.capability == "sql_agent":
        return Decision("query", step.reason)
    if step.capability == "rag_agent":
        return Decision("rag", step.reason)
    if step.capability in TOOL_BY_KEY:
        return Decision("mcp", step.reason, step.capability, step.action)
    return Decision("general", step.reason)


def _is_document_question(text: str) -> bool:
    """Explicit document-content questions (work with or without attachments)."""
    mentions_docs = _mentions_any(text, _RAG_DOC_WORDS) or _mentions_any(text, _RAG_ATTACHMENT_REFS)
    has_question_hint = (
        _mentions_any(text, _RAG_INTENT_HINTS)
        or text.strip().endswith("?")
        or bool(re.search(r"\b(?:give me|what is|what's|whats|tell me|show me)\b", text))
    )
    return mentions_docs and has_question_hint


def _is_self_question_without_document(text: str) -> bool:
    return bool(_SELF_QUESTION_RE.search(text)) and not (
        _mentions_any(text, _RAG_DOC_WORDS) or _mentions_any(text, _RAG_ATTACHMENT_REFS)
    )


def _refers_to_attached_document(text: str) -> bool:
    if _mentions_any(text, _RAG_ATTACHMENT_REFS):
        return True
    if _mentions_any(text, _RAG_DOC_WORDS):
        return True
    # Short deixis when attachments exist: "what is this?", "summarize it", "compare these".
    if re.search(r"\b(?:this|that|it|these|those)\b", text):
        if _mentions_any(text, _RAG_INTENT_HINTS) or re.search(
            r"\b(?:what is|what's|whats|give me|tell me|show me|summar|compar|about|heading|title|conclusion)\b",
            text,
        ):
            return True
    return False


def _wants_document_capability(text: str, context: PlanningContext) -> bool:
    """Whether this turn should use RAG / ask for a document.

    Attached documents are an available resource, not a forced route. Unrelated
    asks (identity, weather, SQL, calendar, email) must not become RAG just
    because a file is attached.
    """
    # "What is it that you do?" contains document-like deixis ("it"), but is
    # about the assistant. Explicit document language still wins, e.g.
    # "What can you do with this PDF?"
    if _is_self_question_without_document(text):
        return False
    if _mentions_any(text, _WEATHER_HINTS):
        return False
    if _wants_calendar(text):
        return False
    # Pure email/calendar-style asks must not become RAG just because a file
    # is attached — but "find X in my docs and email it" still needs RAG.
    if _wants_email(text) and not (
        _is_document_question(text)
        or _mentions_any(text, _RAG_DOC_WORDS)
        or (context.has_rag_documents and _refers_to_attached_document(text))
    ):
        return False
    if _is_document_question(text):
        return True

    if context.has_rag_documents and _refers_to_attached_document(text):
        return True

    # With attachments, overview/content asks ("main conclusion", "heading")
    # refer to the attached file even without saying "file"/"document".
    if context.has_rag_documents and _mentions_any(
        text,
        [
            "heading", "title", "headline", "conclusion", "conclude", "concludes",
            "overview", "key points", "summarize", "summary", "summarise",
        ],
    ):
        return True

    # Policy / benefits style questions often collide with SQL vocabulary
    # ("how many", "employees") but with an attached doc should use RAG —
    # not invent an answer in General Chat. Pure DB asks without policy
    # vocabulary still fall through to the SQL gate.
    if context.has_rag_documents and router.classify(
        text, has_selected_database=True
    ) == "query" and _mentions_any(
        text,
        _LEAVE_HINTS + ["policy", "policies", "handbook", "benefits", "pto", "vacation day", "sick day"],
    ):
        return True

    # After a RAG turn, short topical follow-ups still need retrieval.
    if context.has_rag_documents and context.has_rag_context and _looks_like_rag_followup(text):
        return True

    return False


def _looks_like_rag_followup(text: str) -> bool:
    if len(text.split()) > 12:
        return False
    if text.startswith("what about ") or text.startswith("how about "):
        return True
    if text.startswith("and ") or text in {"same for benefits", "same for vacation", "that one"}:
        return True
    if _mentions_any(text, ["conclusion", "sick leave", "vacation", "policy", "heading", "title"]):
        return True
    return text.strip().endswith("?") and not _wants_calendar(text) and not _wants_email(text)


def _decide_rag(text: str, has_rag_documents: bool) -> Optional[Decision]:
    context = PlanningContext(has_selected_database=False, has_rag_documents=has_rag_documents)
    if _wants_document_capability(text, context):
        return Decision("rag", "The user is asking about content in uploaded/indexed documents.")
    return None


def _wants_weather(text: str) -> bool:
    return _mentions_any(text, _WEATHER_HINTS)


def _is_vague_database_request(text: str) -> bool:
    """True when the user asks to use/query a DB without a concrete lookup."""
    if not re.search(r"\b(?:query|search|check|ask|use)\b.{0,48}\b(?:database|db)\b", text):
        return False
    # Concrete structured-data cues mean we can proceed (or ask only for a DB name).
    if re.search(
        r"\b(?:how many|average|avg|highest|lowest|top \d+|count|total|sum of|"
        r"salary|salaries|employee|employees|department|departments|"
        r"list|show|find)\b",
        text,
    ):
        return False
    return True


def _has_structured_data_cues(message: str, text: str) -> bool:
    """True when the message itself looks like a structured-data lookup.

    Unlike ``_looks_like_database_question``, this does not treat "email" as a
    hard block, so combined SQL→Gmail goals can be planned in one turn.
    """
    if _mentions_any(text, _HARD_SQL_BLOCKS):
        return False
    return router.classify(message, has_selected_database=True) == "query"


def _looks_like_database_question(message: str, text: str, context: PlanningContext) -> bool:
    """Positive-evidence SQL gate.

    A selected database only makes SQL available. Previous SQL turns only
    unlock short referential follow-ups. Conversational questions never
    fall through into SQL by default.
    """
    if _mentions_any(text, _NEW_INTENT_BLOCKS_SQL):
        return False

    # Fresh structured-data questions - database may or may not be selected.
    if _has_structured_data_cues(message, text):
        return True

    # Sticky SQL is intentionally narrow: only short referential follow-ups
    # after a successful SQL turn in this conversation.
    if context.has_sql_context and _looks_like_sql_followup(text):
        return True

    return False


def _looks_like_sql_followup(text: str) -> bool:
    if len(text.split()) > 8:
        return False
    return (
        text.startswith("what about ")
        or text.startswith("how about ")
        or text.startswith("and for ")
        or text.startswith("same for ")
        or text in {"same for marketing", "same for sales", "that one"}
    )


def _is_leave_request(text: str) -> bool:
    if not (
        _mentions_any(text, _LEAVE_HINTS)
        and _mentions_any(text, ["request", "apply", "ask", "take", "need"])
    ):
        return False
    # An email-send instruction whose body happens to mention leave is still Gmail,
    # not the specialized leave workflow. This needs an explicit send/draft verb:
    # a leave goal that only picked up a recipient and body through slot filling
    # is still a leave request.
    if _has_explicit_send_verb(text) and _has_email_body_expression(text):
        return False
    return True


def _has_explicit_send_verb(text: str) -> bool:
    return bool(
        re.search(r"\b(?:send|sent|draft|compose|write)\b", text)
        or _mentions_any(text, ["email me", "mail me", "shoot an email", "fire off an email"])
    )


def _wants_email(text: str) -> bool:
    return _mentions_any(text, _EMAIL_HINTS)


def _is_context_send(text: str) -> bool:
    return _mentions_any(text, ["send this", "send that", "send it", "email me that", "email me this", "mail me that", "mail me this"])


def _has_email_recipient(text: str) -> bool:
    return bool(re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)) or bool(
        re.search(r"\b(to\s+)?(myself|me)\b", text)
    )


def _has_email_body_expression(text: str) -> bool:
    return bool(re.search(r"\b(?:saying|say|that says|body)\s*:?\s+.+", text, re.DOTALL)) or bool(
        re.search(r"\b(?:about|regarding|re:)\s+.+", text, re.DOTALL)
    )


def _wants_calendar(text: str) -> bool:
    return _mentions_any(text, _CALENDAR_HINTS)


def _wants_calendar_create(text: str) -> bool:
    return _mentions_any(
        text,
        [
            "schedule a meeting", "schedule meeting", "create a meeting",
            "create meeting", "set up a meeting", "book a meeting",
            "add a meeting", "schedule an event", "create an event",
            "add an event",
        ],
    )


def _wants_date_time(text: str) -> bool:
    return _has_date_expression(text) or _mentions_any(
        text,
        [
            "date", "time", "day is", "what day",
            "next monday", "next tuesday", "next wednesday", "next thursday", "next friday",
            "next saturday", "next sunday", "jan ", "january", "feb ", "february",
            "mar ", "march", "apr ", "april", "may ", "jun ", "june", "jul ",
            "july", "aug ", "august", "sep ", "september", "oct ", "october",
            "nov ", "november", "dec ", "december",
        ],
    )


def _has_date_expression(text: str) -> bool:
    if _mentions_any(text, ["today", "yesterday", *TOMORROW_WORDS]):
        return True
    if re.search(r"\b(?:(?:next|this)\s+)?(?:mon|monday|tue|tues|tuesday|wed|wednesday|thu|thur|thurs|thursday|fri|friday|sat|saturday|sun|sunday)\b", text):
        return True
    if MONTH_FIRST_DATE_RE.search(text):
        return True
    if any(pattern.search(text) for pattern in DAY_FIRST_DATE_RES):
        return True
    if re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", text):
        return True
    if re.search(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", text):
        return True
    return False


def _has_time_expression(text: str) -> bool:
    return bool(
        re.search(r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b", text)
        or re.search(r"\b(?:at|from)\s+\d{1,2}(?::\d{2})?\b", text)
        or re.search(r"\b(?:noon|midnight)\b", text)
    )


def _wants_calculation(text: str) -> bool:
    if re.search(r"\b(calculate|calculator|percent|percentage|plus|minus|times|divided by)\b", text):
        return True
    return bool(re.fullmatch(r"[0-9\s+\-*/().,%]+", text))


def _normalize(message: str) -> str:
    return " ".join((message or "").lower().strip().split())


def _mentions_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)
