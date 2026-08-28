"""
gmail_tool.py
=============
Gmail MCP-style tool.

The send action uses the real Gmail API after the app has a connected Google
OAuth token and the user has confirmed the write action.
"""

import base64
import re
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Optional
from urllib.parse import urlencode

from . import oauth
from .base import MCPTool, ToolAction

import config

logger = config.get_logger(__name__)

# Guarded like orchestrator.py's _llm: only constructed when a real key is
# configured, so tests / offline runs never try to hit the network.
_llm = None
if config.OPENAI_API_KEY:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    _llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=config.CHAT_TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
    )

GMAIL_SEND_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
GMAIL_DRAFT_URL = "https://gmail.googleapis.com/gmail/v1/users/me/drafts"
GMAIL_MESSAGES_URL = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
GMAIL_INBOX_SUMMARY_LIMIT = 5


@dataclass
class EmailRequest:
    to_email: str
    subject: str
    body: str


class GmailTool(MCPTool):
    def call(self, action_name: str, confirmed: bool = False, **kwargs) -> str:
        action = next((a for a in self.actions if a.name == action_name), None)
        if action is None:
            return f"[{self.display_name}] Unknown action '{action_name}'."

        if not self.connected:
            return (
                "Gmail isn't connected yet. "
                "Connect it in MCP Connections and I can prepare/send that email."
            )

        if action_name == "send_email":
            return self._send_email(
                confirmed=confirmed,
                instruction=kwargs.get("instruction", ""),
                email_request=kwargs.get("email_request"),
            )

        if action_name == "draft_reply":
            return self._create_draft(
                confirmed=confirmed,
                instruction=kwargs.get("instruction", ""),
                email_request=kwargs.get("email_request"),
            )

        if action_name == "summarize_inbox":
            return self._summarize_inbox()

        return super().call(action_name, confirmed=confirmed, **kwargs)

    def _summarize_inbox(self) -> str:
        access_token = oauth.google_access_token()
        list_params = urlencode({"maxResults": str(GMAIL_INBOX_SUMMARY_LIMIT), "labelIds": "INBOX"})
        try:
            payload = oauth.get_json(f"{GMAIL_MESSAGES_URL}?{list_params}", access_token)
        except RuntimeError as exc:
            return f"[Gmail] Couldn't read your inbox: {exc}"

        message_refs = payload.get("messages", [])
        if not message_refs:
            return "[Gmail] Your inbox is empty."

        lines = [f"[Gmail] Your {len(message_refs)} most recent inbox message(s):"]
        detail_params = urlencode(
            [("format", "metadata"), ("metadataHeaders", "Subject"), ("metadataHeaders", "From")]
        )
        for ref in message_refs:
            try:
                detail = oauth.get_json(f"{GMAIL_MESSAGES_URL}/{ref['id']}?{detail_params}", access_token)
            except RuntimeError as exc:
                lines.append(f"- (couldn't load message {ref.get('id', '?')}: {exc})")
                continue
            headers = {h.get("name"): h.get("value") for h in detail.get("payload", {}).get("headers", [])}
            subject = headers.get("Subject") or "(No subject)"
            sender = headers.get("From") or "(Unknown sender)"
            snippet = (detail.get("snippet") or "").strip()
            lines.append(f"- From {sender} - \"{subject}\": {snippet}")
        return "\n".join(lines)

    def _send_email(self, confirmed: bool, instruction: str, email_request=None) -> str:
        email_request = coerce_email_request(email_request) or parse_email_request(instruction)
        if not email_request:
            return (
                "[Gmail] I need a recipient and message. Try: "
                "`send a mail to myself saying hi` or "
                "`send an email to someone@example.com saying hello`."
            )

        if not confirmed:
            return (
                "Email draft ready:\n\n"
                f"To: {email_request.to_email}\n"
                f"Subject: {email_request.subject}\n"
                f"Body:\n{email_request.body}\n\n"
                "Reply yes to send, or cancel to discard."
            )

        access_token = oauth.google_access_token()
        raw = build_raw_message(email_request)
        response = oauth.post_json(GMAIL_SEND_URL, access_token, {"raw": raw})
        message_id = response.get("id", "unknown")
        return f"Sent email to {email_request.to_email}. Gmail message id: {message_id}"

    def _create_draft(self, confirmed: bool, instruction: str, email_request=None) -> str:
        email_request = coerce_email_request(email_request) or parse_email_request(instruction)
        if not email_request:
            return "[Gmail] I need a recipient and message before I can create that Gmail draft."

        if not confirmed:
            return (
                "Gmail draft ready:\n\n"
                f"To: {email_request.to_email}\n"
                f"Subject: {email_request.subject}\n"
                f"Body:\n{email_request.body}\n\n"
                "Reply yes to create this draft in Gmail, or cancel to discard."
            )

        access_token = oauth.google_access_token()
        raw = build_raw_message(email_request)
        response = oauth.post_json(GMAIL_DRAFT_URL, access_token, {"message": {"raw": raw}})
        draft_id = response.get("id", "unknown")
        message_id = (response.get("message") or {}).get("id", "unknown")
        return f"Created Gmail draft to {email_request.to_email}. Draft id: {draft_id}. Gmail message id: {message_id}"


def coerce_email_request(value) -> Optional[EmailRequest]:
    if isinstance(value, EmailRequest):
        return value
    if isinstance(value, dict):
        to_email = str(value.get("to_email") or "").strip()
        subject = str(value.get("subject") or "").strip()
        body = str(value.get("body") or "").strip()
        if to_email and subject and body:
            return EmailRequest(to_email=to_email, subject=subject, body=body)
    return None


_LITERAL_QUOTE_RE = re.compile(r"[\"'“‘](.+?)[\"'”’]", re.DOTALL)

# Phrases that mean "you decide the wording" rather than "type exactly this."
# General on purpose - matches the intent, not a fixed sentence.
_COMPOSE_FOR_ME_RE = re.compile(
    r"\b(?:write|compose|draft|come up with|make up)\s+(?:the\s+)?"
    r"(?:mail|email|message|body|content|it)\b.{0,40}\b"
    r"(?:yourself|for me|on your own|for the message)\b"
    r"|\byou\s+(?:write|compose|decide|figure out)\b.{0,20}\b(?:mail|email|message|body|content)\b",
    re.IGNORECASE,
)


def _looks_like_intent_only(raw_instruction_tail: str) -> bool:
    """True when the text after 'saying/that/about ...' reads like a
    description of what to convey rather than literal dictated wording.
    Heuristic, not exhaustive - a few weak signals combined:
    - explicitly says "write/compose it yourself"
    - has no quoted literal text AND is phrased as a description (starts with
      a first-person clause like "I built..." reported back in third person,
      or contains words like "tell them/let them know") rather than direct
      address ("Hi ..., ...").
    """
    text = (raw_instruction_tail or "").strip()
    if not text:
        return False
    if _COMPOSE_FOR_ME_RE.search(text):
        return True
    if _LITERAL_QUOTE_RE.search(text):
        return False
    if re.match(r"(?i)^(hi|hello|dear)\b", text):
        return False
    return bool(re.search(r"\b(tell|let)\s+(?:them|him|her)\b|\bi\s+(?:built|made|created|finished|launched|started)\b", text, re.IGNORECASE))


def compose_email_body_with_llm(intent_text: str, full_instruction: str) -> Optional[str]:
    """Generate a natural, well-composed email body from the user's stated
    intent, instead of echoing their raw instruction into a template.
    Returns None (caller falls back to the template) if no LLM is
    configured or generation fails for any reason.
    """
    if _llm is None:
        return None
    try:
        system = SystemMessage(
            content=(
                "You write short, natural, professional email bodies for a user "
                "based on what they want to convey. Output ONLY the email body "
                "text (greeting, a couple of sentences, sign-off if it fits) - "
                "no subject line, no explanations, no markdown, no quotes around "
                "the whole thing. Keep it concise and genuinely composed, not a "
                "restatement of the instruction itself."
            )
        )
        human = HumanMessage(
            content=(
                f"Write an email body conveying: {intent_text}\n\n"
                f"(Full original request for context: {full_instruction})"
            )
        )
        response = _llm.invoke([system, human])
        body = (response.content or "").strip()
        return body or None
    except Exception:  # pragma: no cover - defensive: never break email flow
        logger.warning("compose_email_body_with_llm failed; falling back to template", exc_info=True)
        return None


def parse_email_request(instruction: str) -> Optional[EmailRequest]:
    text = " ".join((instruction or "").strip().split())
    if not text:
        return None

    profile = oauth.google_profile()
    own_email = profile.get("email", "")

    to_email = ""
    email_match = re.search(r"\bto\s+([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", text, re.IGNORECASE)
    if email_match:
        to_email = email_match.group(1)
    else:
        email_match = re.search(r"\b(?:recipient detail|send to|recipient|to):?\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", text, re.IGNORECASE)
        if email_match:
            to_email = email_match.group(1)
        else:
            email_match = re.search(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", text, re.IGNORECASE)
            if email_match:
                to_email = email_match.group(1)
    if not to_email and re.search(r"\b(to\s+)?(myself|me)\b", text, re.IGNORECASE):
        to_email = own_email

    if not to_email:
        return None

    body = ""
    body_match = re.search(r"\b(?:saying|say|that says|body)\s*:?\s+(.+)$", text, re.IGNORECASE | re.DOTALL)
    intent_tail = ""
    if body_match:
        intent_tail = body_match.group(1).strip()
        intent_tail = re.split(r"\b(?:recipient detail|send to|recipient):?\s+", intent_tail, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        body = intent_tail
    else:
        # No explicit "saying/say/body ..." marker - if the whole instruction
        # reads like "write the mail yourself" / "compose it for me" (rather
        # than a literal dictated body), treat the full text as the intent
        # to compose from instead of falling straight to a blank "Hi".
        if _COMPOSE_FOR_ME_RE.search(text):
            intent_tail = text
        body = ""

    composed = None
    if intent_tail and _looks_like_intent_only(intent_tail):
        composed = compose_email_body_with_llm(intent_tail, text)

    if composed:
        body = composed
    else:
        body = body.strip(" .")
        if not body:
            body = "Hi"
        body = polish_email_body(body)

    subject = "Message from AgentHub Studio"
    subject_match = re.search(r"\bsubject\s+['\"]?(.+?)['\"]?(?:\s+(?:saying|body|that says)\b|$)", text, re.IGNORECASE)
    if subject_match:
        subject = subject_match.group(1).strip(" .\"'")
    elif subject == "Message from AgentHub Studio":
        subject = infer_email_subject(body, text)

    return EmailRequest(to_email=to_email, subject=subject, body=body)


def polish_email_body(body: str) -> str:
    """Light formatting for short one-line bodies; leave multi-line drafts alone."""
    text = (body or "").strip()
    if not text:
        return "Hi"
    if "\n" in text or re.match(r"(?i)^hi\b|^hello\b|^dear\b", text):
        return text
    cleaned = text.rstrip(".")
    return f"Hi,\n\n{cleaned}.\n\nThank you."


def infer_email_subject(body: str, full_text: str) -> str:
    """Derive a concise subject from the message content when none was given."""
    source = f"{body} {full_text}"
    leave_match = re.search(
        r"\b(?:leave|pto|time off|vacation|sick(?:\s+day|\s+leave)?)\b.*?"
        r"\b(?:on\s+)?("
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?"
        r"|\d{4}-\d{2}-\d{2}"
        r")\b",
        source,
        re.IGNORECASE | re.DOTALL,
    )
    if leave_match:
        return f"Leave request for {_pretty_date_label(leave_match.group(1))}"

    if re.search(r"\b(?:leave|pto|time off)\b", source, re.IGNORECASE):
        return "Leave request"

    first = re.split(r"[.!\n]", body or "", maxsplit=1)[0].strip()
    first = re.sub(r"^(?:hi|hello|dear)\b[\s,]*", "", first, flags=re.IGNORECASE).strip()
    if 8 <= len(first) <= 72:
        return first[0].upper() + first[1:]
    return "Message from AgentHub Studio"


def _pretty_date_label(raw: str) -> str:
    text = (raw or "").strip()
    iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if iso:
        year, month, day = map(int, iso.groups())
        try:
            from datetime import date

            return f"{date(year, month, day).strftime('%B')} {day}"
        except ValueError:
            return text
    month_match = re.fullmatch(
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
        r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"\s+(\d{1,2})(?:st|nd|rd|th)?",
        text,
        re.IGNORECASE,
    )
    if not month_match:
        return text
    month_key = month_match.group(1)[:3].lower()
    day = int(month_match.group(2))
    months = {
        "jan": "January", "feb": "February", "mar": "March", "apr": "April",
        "may": "May", "jun": "June", "jul": "July", "aug": "August",
        "sep": "September", "oct": "October", "nov": "November", "dec": "December",
    }
    return f"{months.get(month_key, text)} {day}"


def build_raw_message(email_request: EmailRequest) -> str:
    message = EmailMessage()
    profile = oauth.google_profile()
    from_email = profile.get("email")
    if from_email:
        message["From"] = from_email
    message["To"] = email_request.to_email
    message["Subject"] = email_request.subject
    message.set_content(email_request.body)
    return base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")


gmail_tool = GmailTool(
    key="gmail",
    display_name="Gmail",
    icon="📧",
    description="Summarize emails and send approved messages through Gmail.",
    actions=[
        ToolAction("summarize_inbox", "Summarize recent emails", write_action=False),
        ToolAction("draft_reply", "Create a Gmail draft", write_action=True),
        ToolAction("send_email", "Send an email", write_action=True),
    ],
)
