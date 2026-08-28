"""
config.py
=========
All the "knobs" for this project live here: model name, retry limits,
storage paths, memory/cache settings, and logging setup.

Every other module does:
    import config
    config.MAX_RETRIES
instead of scattering magic numbers everywhere.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = Path(os.getenv("AGENTHUB_STORAGE_DIR", str(BASE_DIR / "storage"))).expanduser().resolve()
DATABASES_DIR = STORAGE_DIR / "databases"
UPLOADS_DIR = STORAGE_DIR / "uploads"
TOKENS_DIR = STORAGE_DIR / "tokens"

DATABASES_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
TOKENS_DIR.mkdir(parents=True, exist_ok=True)

# RAG Documents storage - source files, the persistent vector index, and
# document registry metadata each get their own subfolder so none of them
# accidentally clobber the others. See agents/rag_agent/ for the code that
# reads/writes these.
RAG_DIR = STORAGE_DIR / "rag"
RAG_DOCUMENTS_DIR = RAG_DIR / "documents"
RAG_CHROMA_DIR = RAG_DIR / "chroma"
RAG_METADATA_DIR = RAG_DIR / "metadata"

RAG_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
RAG_CHROMA_DIR.mkdir(parents=True, exist_ok=True)
RAG_METADATA_DIR.mkdir(parents=True, exist_ok=True)

# The demo ships with one ready-made sample database so "Query Database"
# mode has something to query on first launch.
DEFAULT_DB_NAME = "sample_company.db"
DEFAULT_DB_PATH = str(DATABASES_DIR / DEFAULT_DB_NAME)

# ---------------------------------------------------------------------------
# OpenAI / LLM settings
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = 0  # deterministic SQL / schema generation
CHAT_TEMPERATURE = 0  # a little more natural for general chat
AUTO_PLANNER_USE_LLM = os.getenv("AUTO_PLANNER_USE_LLM", "true").strip().lower() in {"1", "true", "yes"} and bool(OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# OAuth integration settings (Google Workspace)
# ---------------------------------------------------------------------------
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://127.0.0.1:8010")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", f"{APP_BASE_URL}/auth/google/callback")
GOOGLE_SCOPES = [
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
    "https://www.googleapis.com/auth/calendar.events.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]

# ---------------------------------------------------------------------------
# Holiday calendar (Date/Time tool - see app.py: execute_date_time)
# ---------------------------------------------------------------------------
# Offline lookup via the `holidays` package - no API key, no network call.
# Uses the "holidays" package's own country codes (ISO 3166-1 alpha-2, e.g.
# "US", "GB", "IN"). Nothing in the app tracks a per-user locale yet, so this
# is a single global default; override with the env var if your team is
# elsewhere.
HOLIDAY_CALENDAR_COUNTRY = os.getenv("HOLIDAY_CALENDAR_COUNTRY", "US")

# ---------------------------------------------------------------------------
# Retry settings (SQL self-repair loop)
# ---------------------------------------------------------------------------
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Conversation memory settings
# ---------------------------------------------------------------------------
# How many previous turns to feed back into prompts for follow-up questions.
# Kept small on purpose - see agents/sql_agent/memory.py.
MAX_MEMORY_TURNS = 4

# ---------------------------------------------------------------------------
# RAG Documents settings
# ---------------------------------------------------------------------------
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")

# Chunk size/overlap are in tokens (counted with tiktoken), not characters -
# see agents/rag_agent/chunking.py.
RAG_CHUNK_SIZE_TOKENS = int(os.getenv("RAG_CHUNK_SIZE_TOKENS", "1000"))
RAG_CHUNK_OVERLAP_TOKENS = int(os.getenv("RAG_CHUNK_OVERLAP_TOKENS", "150"))

# Retrieval defaults - see agents/rag_agent/retrieval.py.
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_USE_MMR = os.getenv("RAG_USE_MMR", "true").strip().lower() in {"1", "true", "yes"}
RAG_MMR_FETCH_MULTIPLIER = int(os.getenv("RAG_MMR_FETCH_MULTIPLIER", "4"))
RAG_MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.5"))  # 1.0 = pure relevance, 0.0 = pure diversity

# Conversation memory (kept separate from SQL agent memory - see
# agents/rag_agent/memory.py).
RAG_MAX_MEMORY_TURNS = int(os.getenv("RAG_MAX_MEMORY_TURNS", "4"))

# Upload limits / allow-list - enforced in agents/rag_agent/ingestion.py.
RAG_MAX_FILE_SIZE_MB = int(os.getenv("RAG_MAX_FILE_SIZE_MB", "25"))
RAG_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".docx", ".csv"}

# Summarization (map-reduce) settings - see agents/rag_agent/summarization.py.
RAG_SUMMARY_MAP_BATCH_TOKENS = int(os.getenv("RAG_SUMMARY_MAP_BATCH_TOKENS", "6000"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
