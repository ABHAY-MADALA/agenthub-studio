# Architecture

## Positioning

This is a **chat platform** with an AI database workflow as its flagship
feature - not a LangChain script with a UI bolted on. `app.py` is the
product surface; LangGraph, LangChain, and the MCP-style tool interfaces
are implementation details of individual modes/tools, swappable without
changing how the platform looks or feels.

```text
                         ┌─────────────────────────┐
                         │         app.py           │  <- the product
                         │  (Gradio chat platform)   │
                         └────────────┬─────────────┘
                                      │
                 mode resolved (explicit, or Auto via orchestrator.py)
                                      │
        ┌─────────────────┬──────────┴───────────┬─────────────────┐
        │                 │                       │                 │
        v                 v                       v                 v
 General Chat     Database Builder          SQL Query Agent    MCP Tools
 (agents/          (agents/                  (agents/           (tools/mcp/*)
  general_chat.py)  database_builder/)        sql_agent/)        Gmail, Drive,
        │                 │                       │              Calendar
        v                 v                       v
   plain LLM call   schema_generator.py     LangGraph pipeline
                     -> db_creator.py        (get_schema -> generate_sql
                     (approval required      -> validate -> run
                      before any write)       -> repair loop -> answer)
```

## Auto mode planning: orchestrator.py

`orchestrator.py` (not `router.py`) is what actually drives **Auto** mode
today: it plans each turn as one or more steps across the available
capabilities (`sql_agent`, `database_builder`, `rag`, `gmail`,
`google_drive`, `google_calendar`, `date_time`, `calculator`,
`general_chat`), tracks pending multi-step goals (e.g. "find the highest
paid employee and email me their name"), and decides when a write action
needs user confirmation before it runs. `app.py` calls
`orchestrator.plan_turn()` / `orchestrator.next_step()` / `orchestrator.observe()`
to drive this loop - see `tests/test_agentic_auto.py` for the full set of
routing scenarios it covers.

`router.py` is a smaller, older regex classifier (its own docstring calls
it "Legacy") that is now only used for a narrower internal gate rather than
as the primary Auto-mode router - see its module docstring for the reasoning
it was built with.

## The SQL Query Agent (LangGraph)

```text
    START
      |
      v
   get_schema  ---- cache hit? (agents/sql_agent/cache.py, keyed by db_path)
      |                 \
      | miss             '-> return cached schema text
      v
  inspect DB (sqlite_master + PRAGMA), cache it
      |
      v
  generate_sql  (LLM: English question + schema + short memory -> SQL)
      |
      v
  validate_sql  (plain Python - no LLM; see "SQL safety" below)
      |
      +-- invalid, retries left --> fix_sql (LLM) --> validate_sql (loop)
      |
      v (valid)
    run_sql
      |
      +-- error, retries left --> fix_sql (LLM) --> validate_sql (loop)
      |
      v (success, or MAX_RETRIES exhausted)
  generate_answer  (LLM: turns the result table into an English answer)
      |
      v
     END
```

Every node reads/writes a single `State` (a `TypedDict` in
`agents/sql_agent/agent.py`) that includes `db_path` - the graph is
multi-database aware because the platform lets users switch between
several SQLite files, not just one.

`MAX_RETRIES` (default 3, `config.py`) caps the repair loop so a
persistently broken query can never loop forever; the user gets an honest
failure message instead (`prompts.build_failure_message`).

## The Database Builder

Deliberately **not** a LangGraph - it's a two-step wizard (propose, then
approve), not a multi-branch agent loop, so a plain function pipeline is
easier to follow and just as capable:

```text
English description
        |
        v
schema_generator.generate_schema()   <- one LLM call, strict JSON parsing
        |                               (agents/database_builder/schema_generator.py)
        v
builder.format_schema_preview()      <- markdown shown in chat
        |
        v
   [user reviews - types a refinement, or clicks Approve/Cancel]
        |
        v (Approve)
db_creator.create_database()         <- CREATE TABLE, in dependency order
        |                               (agents/database_builder/db_creator.py)
        v (optional: "include sample data" was checked)
schema_generator.generate_sample_data() -> db_creator.insert_sample_data()
        |
        v
new .db file appears in the Database dropdown, ready for Query Database mode
```

**Foreign-key-aware table ordering**: `db_creator.topological_order()`
does a simple dependency sort (Kahn's-algorithm-style DFS) so a child
table (e.g. `enrollments`) is always created - and seeded - after every
table its foreign keys point to, regardless of what order the LLM listed
them in.

**Approval is enforced by design, not just convention**: `CREATE TABLE`
only ever runs inside `db_creator.create_database()`, which is only ever
called from `builder.approve_build()`, which is only ever called after
the user clicks **Approve & Create Database** (or types "approve") in the
UI. There's no code path from "describe a database" straight to a
written file.

## State vs Memory vs Cache

| Concept | Lives in | Scope | Example |
|---|---|---|---|
| **State** | `agents/sql_agent/agent.py` (`State`) | One graph run (one question) | `sql_query`, `sql_result`, `retry_count` |
| **Memory** | `agents/sql_agent/memory.py` (`ConversationMemory`) | Across turns, within one chat thread | "the previous question was about Engineering" |
| **Cache** | `agents/sql_agent/cache.py` (`SimpleCache`) | Across *all* threads, per database file | Schema text, computed once per `.db` |

- **State** is thrown away after each question.
- **Memory** is bounded to the last `config.MAX_MEMORY_TURNS` turns (default
  4) so prompts don't grow forever, and is keyed by `thread_id` so
  separate chats never mix. "New Chat / Clear Memory" wipes it and issues
  a fresh `thread_id`.
- **Cache** has nothing to do with conversations - it just avoids
  re-running `PRAGMA table_info` on every question, since a database's
  schema rarely changes mid-session. It's cleared implicitly whenever a
  new database is created (a fresh `db_path` has never been cached).

### Why in-process instead of Redis for the MVP

Both memory and cache are plain Python dicts, wrapped behind small
interfaces (`ConversationMemory`, `SimpleCache`) so a persistent backend
can be swapped in later **without touching any calling code** - `agent.py`,
`app.py`, and `builder.py` only ever call the public methods on those
classes, never their internals. For a 2-day local demo, adding Redis
would be extra infrastructure (a running server, a new dependency, a new
failure mode) for zero visible benefit - the in-process version is
simpler, faster to run, and behaves identically from the demo's point of
view. Swapping it in later is a contained change to two files.

Query *results* are deliberately never cached (see the note at the bottom
of `agents/sql_agent/cache.py`) - the underlying data can change (e.g.
right after Database Builder seeds a table), and a stale cached result
would be a real correctness bug, not just a staleness inconvenience.

## SQL safety validation

`agents/sql_agent/database.py: validate_sql_query()` is a plain Python
function (no LLM, no way to talk it out of its rules) that runs before
every execution, in Query Database mode only:

1. Rejects empty queries.
2. Requires the query to start with `SELECT` or `WITH`.
3. Rejects multiple statements (`SELECT ...; DROP ...`).
4. Blocks a keyword blocklist anywhere in the query: `INSERT`, `UPDATE`,
   `DELETE`, `DROP`, `ALTER`, `CREATE`, `REPLACE`, `TRUNCATE`, `ATTACH`,
   `DETACH`, `VACUUM`, `REINDEX`, `GRANT`, `REVOKE`. (`PRAGMA` is also
   blocked outright by rule 2, since it can't start with `SELECT`/`WITH`.)
5. Runs `EXPLAIN QUERY PLAN <query>` against SQLite - this parses and
   binds the query (catching syntax errors and invalid table/column
   references) **without executing it or touching any data**.

Only after all five checks pass does `run_sql` actually execute the
query. This is the mechanism that keeps Query Database mode strictly
read-only, independent of anything the LLM was told in its prompt.

Build Database mode is the intentional exception: `CREATE TABLE` and
sample `INSERT`s are allowed there, but only inside `db_creator.py`,
only after explicit user approval (see above) - the two modes never share
a code path.

## Automatic error repair

If `validate_sql` or `run_sql` fails, the graph routes to `fix_sql`,
which sends the LLM the original question, the schema, the exact failed
SQL, and the exact SQLite error message, and asks for a corrected query.
The fixed query always re-enters `validate_sql` before it's trusted -
`fix_sql` never runs anything directly. This repeats up to `MAX_RETRIES`
times (default 3); if it's still failing, the user gets a clear message
instead of an infinite loop. The Debug/Retry panel in the UI shows every
attempt: the failed SQL, the error, and the fix.

## Auto mode routing

`router.py` classifies a message into `build`, `query`, or `general`
using regex keyword matching - not an LLM call. This is a deliberate
tradeoff for a live demo: zero added latency/cost per message, and a
predictable, explainable rule ("mentions 'create a database'" ->
`build`; "looks like a question and a database is selected" -> `query`;
otherwise `general`) beats an LLM classifier that might occasionally pick
an unexpected mode on stage. Explicit mode selection in the sidebar
always overrides Auto - Auto is a convenience default, not the only path.

## MCP-style tool placeholders

`tools/mcp/base.py` defines a small shared shape:

```python
MCPTool(key, display_name, icon, description, actions, connected=False)
ToolAction(name, description, write_action: bool)
```

Every tool (`gmail_tool.py`, `google_drive_tool.py`,
`google_calendar_tool.py`) is just an `MCPTool`
instance listing its actions. Calling `tool.call(action_name,
confirmed=False)`:

- Returns a "not connected yet" message if `connected=False` (true for
  Google Workspace is not connected).
- For a real connection, returns a "needs your confirmation" message for
  any `write_action=True` action (e.g. `send_email`, `draft_reply`, `create_meeting`)
  unless called with `confirmed=True` - the same ask-before-writing
  pattern the platform-wide safety rules require.

**Adding a real integration later** means implementing the body of one
tool's `call()` (or giving `MCPTool` a real client + `connected=True`) -
nothing in `app.py` or `router.py` needs to know the difference, since
they only ever call `tool.call(...)` through `tools/mcp/registry.py`.

Implemented / planned actions:

- **Gmail**: create Gmail drafts and send email after approval.
- **Google Drive**: import CSV/PDF/docs into `storage/uploads/`, summarize
  files (a natural on-ramp into Database Builder: "build a database from
  this CSV").
- **Google Calendar**: create meetings, check schedule.
