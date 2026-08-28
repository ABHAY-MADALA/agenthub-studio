# AgentHub Studio (MVP)

A LibreChat-style AI workspace: a chat platform where the main feature is
an **AI database workflow** - describe a database in English and it gets
built, then ask questions about it in English and get SQL, results, and
answers back. LangGraph, LangChain, and an MCP-style tool interface power
this from behind the scenes; the product is the chat workspace, not a
LangChain script.

Built for an internship demo. Optimized for a working local demo over
production scale - see [Current limitations](#current-limitations) and
[docs/architecture.md](docs/architecture.md) for what's deliberately
simplified and how it would grow up.


## What it does

```text
User (Build Database mode):
  "Create a database for a college with students, professors, courses,
   enrollments, and grades."

Platform:
  -> proposes a normalized schema (tables, columns, PK/FK) as a preview
  -> waits for approval
  -> creates college_db.db (and optional sample data) after you approve

User (Query Database mode, college_db.db selected):
  "Which course has the most students?"

Platform:
  -> generates SQL, validates it's read-only, runs it, self-repairs on
     error (up to 3 tries), answers in plain English
  -> shows the SQL, the result table, the schema, and retry/debug details

User (follow-up, same chat):
  "How many students are enrolled in it?"

Platform:
  -> resolves "it" using short conversation memory, answers correctly
```

## Feature overview

- **Chat-style web UI** (FastAPI backend + a static HTML/CSS/JS frontend)
  with a sidebar: mode selector, database selector, MCP tool placeholders,
  New Chat / Clear Memory.
- **Modes**: General Chat, Build Database, Query Database, and Auto
  (a fast keyword-based router picks a mode for you - see
  [router.py](router.py)).
- **Database Builder**: English description -> schema preview (tables,
  columns, PK/FK) -> your approval -> `.db` file created, optionally
  seeded with LLM-generated sample data.
- **SQL Query Agent**: a LangGraph pipeline (schema -> generate SQL ->
  validate -> run -> repair on failure -> answer), read-only by
  construction.
- **Schema Viewer**, **Generated SQL panel**, **Query Result table**,
  **Final Answer panel**, and a **Debug/Retry panel** - all visible at
  once, not buried in a transcript.
- **Bounded conversation memory** (last few turns) per chat thread, with
  a one-click reset.
- **In-process schema cache**, swappable for Redis later without touching
  calling code (see [agents/sql_agent/cache.py](agents/sql_agent/cache.py)).
- **MCP-style Google Workspace tools** for Gmail and Google Calendar, with
  write actions behind preview and approval. Google Drive is currently a
  placeholder surface for future import/summarization tools.

## Installation

```bash
cd agenthub-studio
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment setup

```bash
cp .env.example .env
```

Edit `.env` and set your real key:

```text
OPENAI_API_KEY=your-openai-api-key
```

## Running it

```bash
python3 app.py
```

Open `http://127.0.0.1:7860` in your browser. A sample
`sample_company.db` (departments/employees) is created automatically in
`storage/databases/` on first run, so Query Database mode has something
to query immediately.

For local development with auto-reload, run `uvicorn app:app --reload
--port 7860` instead.

## Try it

1. **Build a database** - switch to *Build Database* mode (or leave it on
   *Auto* and just say "create a database for..."):
   > Create a database for a college with students, professors, courses,
   > enrollments, and grades.

   Review the schema preview, check "include sample data" if you want a
   few demo rows, then click **Approve & Create Database**.

2. **Query it** - switch the Database dropdown to your new `.db`, switch
   to *Query Database* mode, and ask:
   > Which course has the most students?

3. **Follow up** - ask a related question without repeating context:
   > How many students are enrolled in it?

4. **Try General Chat** - switch modes and just talk; no tools involved.

5. **Explore the MCP placeholders** - open "Try a placeholder tool" in the
   sidebar and run a write action (e.g. "Gmail: Send an email") to see
   the approval-required response pattern.

## Project structure

```text
agenthub-studio/
├── app.py                       FastAPI backend - modes, panels, wiring
├── static/                      Frontend (HTML/CSS/JS) served by app.py
├── orchestrator.py              Agentic Auto router across agents and tools
├── router.py                    Basic database/general classifier used by orchestrator
├── config.py                    Model name, paths, retry/memory limits, logging
├── agents/
│   ├── general_chat.py          General Chat mode (plain LLM, no tools)
│   ├── sql_agent/                Query Database mode
│   │   ├── agent.py               LangGraph: State, nodes, conditional edges
│   │   ├── database.py            SQLite connection, schema, validation, execution
│   │   ├── memory.py              Bounded per-thread conversation memory
│   │   ├── cache.py               In-process schema cache (per database)
│   │   └── prompts.py             SQL generation / repair / answer prompts
│   └── database_builder/         Build Database mode
│       ├── builder.py             Orchestration + approval-flow state
│       ├── schema_generator.py    English -> schema JSON (LLM)
│       ├── db_creator.py          Schema JSON -> .db file (plain sqlite3)
│       └── prompts.py             Schema generation / refinement / sample data prompts
├── tools/mcp/                    MCP-style integrations + local OAuth helpers
│   ├── base.py                    Shared MCPTool/ToolAction shape
│   ├── gmail_tool.py, google_drive_tool.py, google_calendar_tool.py
│   ├── oauth.py                   Google OAuth connect + token storage
│   └── registry.py                Lists all tools for the sidebar
├── storage/
│   ├── databases/                 Created .db files live here (selectable in the UI)
│   ├── uploads/                   Reserved for future file-import tools
│   └── tokens/                    Local OAuth tokens, gitignored
└── docs/
    ├── architecture.md            Deeper design notes and diagrams
    └── demo_script.md             Step-by-step internship demo script
```

## State vs Memory vs Cache

| Concept | Lives in | Scope | Example |
|---|---|---|---|
| **State** | `agents/sql_agent/agent.py` (`State`) | One graph run (one question) | `sql_query`, `retry_count` |
| **Memory** | `agents/sql_agent/memory.py` | Across turns, within one chat thread | "the previous question was about Engineering" |
| **Cache** | `agents/sql_agent/cache.py` | Across all threads, per database | Schema text, computed once per `.db` file |

See [docs/architecture.md](docs/architecture.md) for the full explanation
of the LangGraph flow, SQL safety validation, retry/repair loop, caching,
and how MCP tools would be plugged in for real.

## Agentic Auto Mode

Auto mode is the main assistant experience. The user can type naturally
instead of manually choosing a tool first. `orchestrator.py` decides whether
the message should go to:

- Database Builder
- SQL Query Agent
- General Chat
- Gmail
- Google Drive
- Google Calendar

Read-style actions can run directly once connected. Write-style actions
such as sending email, creating Gmail drafts, or creating calendar events
pause and ask the user to reply `yes` before anything external is changed.

## Safety

- **Query Database mode is always read-only.** `validate_sql_query()`
  requires the query to start with `SELECT`/`WITH`, rejects multiple
  statements, and blocks a keyword blocklist (`INSERT`, `UPDATE`,
  `DELETE`, `DROP`, `ALTER`, `CREATE`, ...) before anything ever executes.
- **Build Database mode is the only place `CREATE TABLE` (and optional
  sample `INSERT`s) happen**, and only after you click **Approve**.
- **MCP tools** mark write actions
  (`send_email`, `draft_reply`, `create_meeting`, ...) as requiring confirmation - the
  same pattern a real integration would use before it's allowed to touch
  anything outside the app.

## Connecting Gmail, Google Drive, and Calendar

The MCP Tools page has real OAuth connection buttons. Google Workspace is
one connection that powers Gmail, Google Drive, and Google Calendar.

Add these values to `.env`, then restart the app:

```text
APP_BASE_URL=http://127.0.0.1:8010

GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=http://127.0.0.1:8010/auth/google/callback
```

For Google Cloud:

- Enable Gmail API, Google Drive API, and Google Calendar API.
- Configure the OAuth consent screen.
- Add yourself as a test user.
- Create a Web OAuth client.
- Add this authorized redirect URI:
  `http://127.0.0.1:8010/auth/google/callback`

Tokens are saved locally in `storage/tokens/`, which is ignored by git.
This is meant for a local single-user demo; production should encrypt
tokens and store them per authenticated user.

## Current limitations

- SQLite only; everything runs in one local process.
- Conversation memory and the schema cache are in-process - they reset
  if the app restarts.
- No authentication - anyone with access to the running app can use it.
- MCP tools have real OAuth connect/disconnect status. Gmail send, Gmail
  draft creation, inbox summaries, Calendar schedule checks, and Calendar
  writes use real API paths after approval. Google Drive actions are still
  placeholders.
- Auto mode's router is a keyword heuristic, not an LLM classifier - fast
  and predictable, but simpler than true intent detection.

## Future improvements

- Redis-backed (or otherwise persistent) schema cache and conversation
  memory - both are already written behind small interfaces so the
  swap wouldn't touch calling code.
- Implement additional real API actions for connected Google Workspace
  tools while keeping write actions behind approval.
- CSV/PDF import via Google Drive -> feed the Database Builder directly.
- PostgreSQL/MySQL support alongside SQLite.
- Authentication and per-user databases.
- Persistent LangGraph checkpointing (swap `MemorySaver` for a durable
  backend in `agents/sql_agent/agent.py`).
