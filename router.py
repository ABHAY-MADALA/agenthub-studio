"""
router.py
=========
Legacy lightweight classifier used by Auto's SQL gate.

IMPORTANT:
  A selected database means a database is *available*, not that every
  message should become a SQL query. Classification therefore requires
  positive evidence of a structured-data question. Uncertainty falls
  through to 'general' - never to SQL.
"""

import re

BUILD_PATTERNS = [
    r"\bcreate a (?:new )?database\b",
    r"\bbuild a (?:new )?database\b",
    r"\bmake a (?:new )?database\b",
    r"\bset up a database\b",
    r"\badd a table\b",
    r"\bdesign a schema\b",
]

# Strong, data-shaped query evidence. Deliberately does NOT include bare
# "who is/are", "what is the", or "ends with ?" - those over-route chat
# into SQL (e.g. "who are you?").
QUERY_PATTERNS = [
    r"\bhow many\b",
    r"\baverage\b",
    r"\bavg\b",
    r"\bhighest\b",
    r"\blowest\b",
    r"\bmaximum\b",
    r"\bminimum\b",
    r"\btop \d+\b",
    r"\bcount\b",
    r"\btotal\b",
    r"\bsum of\b",
    r"\bmedian\b",
    r"\bwho (?:earns?|makes?|has|have|works?|got|receives?)\b",
    r"\bwho (?:is|are) the (?:highest|lowest|top|best|worst)\b",
    r"\bwhich (?:department|employee|employees|city|customer|product|person|team|role|job)\b",
    r"\b(?:list|show|find)\s+(?:all\s+)?(?:employees|departments|customers|orders|products|rows|records)\b",
    r"\bin the (?:database|db|table)\b",
    r"\bfrom the (?:database|db|table)\b",
    r"\bquery (?:the )?(?:database|db)\b",
    r"\b(?:search|check|ask) (?:the )?(?:database|db)\b",
    r"\bsalary\b",
    r"\bsalaries\b",
    r"\bemployee(?:s)?\b",
    r"\bdepartment(?:s)?\b",
]

_BUILD_RE = re.compile("|".join(BUILD_PATTERNS), re.IGNORECASE)
_QUERY_RE = re.compile("|".join(QUERY_PATTERNS), re.IGNORECASE)

def classify(message: str, has_selected_database: bool) -> str:
    """Returns one of: 'build', 'query', 'general'.

    ``has_selected_database`` only unlocks SQL when the message itself
    looks like a structured-data question. A selected database is never
    enough on its own, and uncertainty falls through to ``general``.
    """
    text = (message or "").strip()
    if not text:
        return "general"

    if _BUILD_RE.search(text):
        return "build"

    # Selecting a database makes SQL *available*; it does not make SQL
    # the default for every question mark / conversational utterance.
    if has_selected_database and _QUERY_RE.search(text):
        return "query"

    return "general"
