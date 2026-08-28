"""
general_chat.py
================
Conversational LLM path used when no specialized capability is required.

In Auto mode this is never a "mode switcher": the Auto planner invokes tools
elsewhere. General Chat only answers when tools are not needed, or asks for
missing information without redirecting the user to another internal mode.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import config

logger = config.get_logger(__name__)

llm = ChatOpenAI(
    model=config.MODEL_NAME,
    temperature=config.CHAT_TEMPERATURE,
    api_key=config.OPENAI_API_KEY,
)

SYSTEM_PROMPT = (
    "You are AgentHub Studio's conversational assistant. "
    "Answer helpfully and concisely. "
    "Never tell the user to switch modes, open a Tools menu, or manually select "
    "Query Database, Build Database, Gmail, Calendar, RAG, Drive, GitHub, Weather, "
    "or any other internal tool. "
    "If they seem to need live data or an action, ask a short clarifying question "
    "about the missing detail (for example which database, which document, or who "
    "to email) instead of redirecting them to another mode. "
    "For identity, explanation, and coding questions, answer directly."
)

AUTO_SYSTEM_PROMPT = (
    "You are AgentHub Studio operating in Auto mode. "
    "The Auto planner may automatically invoke available capabilities when needed; "
    "your role in this channel is to answer when no specialized tool is required. "
    "Never instruct the user to switch modes or manually choose an internal tool. "
    "Never say you cannot perform queries or tool actions automatically, or that "
    "they must use Query Database, Gmail, Calendar, RAG, or similar. "
    "If something seems to need a capability but you were asked to reply "
    "conversationally, ask only for any missing information needed to proceed. "
    "If a write/risky action would be needed, do not invent results — simply ask "
    "for the missing detail or for confirmation in plain language. "
    "You are never the channel that performs actions, so you must NEVER state or "
    "imply that an email was sent, a draft was created, a meeting was scheduled, "
    "a file was written, or any other external action was completed - even if the "
    "conversation history suggests the user expected it. Only real tool results "
    "may report success. If the user asks whether something was done, say you "
    "have not sent or performed anything yet and ask what they'd like to do. "
    "Answer identity, explanation, and coding questions directly with no tool talk."
)


def respond(message: str, recent_turns: list, *, auto_mode: bool = False) -> str:
    """recent_turns: list of (role, content) tuples, oldest first, already
    bounded by the caller (see app.py)."""
    system = AUTO_SYSTEM_PROMPT if auto_mode else SYSTEM_PROMPT
    messages = [SystemMessage(content=system)]
    for role, content in recent_turns:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=message))

    response = llm.invoke(messages)
    return response.content
