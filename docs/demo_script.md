# Internship Demo Script

Target length: ~8-10 minutes. Run `python3 app.py` before the audience
arrives so the sample database is already created and the page is loaded
- don't burn demo time on `pip install`.

## 0. Framing (30 seconds)

> "This is an AI workspace platform, similar in spirit to LibreChat - a
> chat app where you can talk to different tools and agents. The
> flagship feature is an AI database workflow: describe a database in
> English and it gets built; ask questions about it in English and get
> SQL, results, and an answer back. LangGraph and LangChain power that
> workflow behind the scenes, but the product is the chat workspace, not
> a script."

Point at the sidebar: mode selector, database selector, MCP tool
placeholders. Point at the right-hand panel: Schema / SQL / Result /
Answer / Debug - all visible at once, not buried in a transcript.

## 1. Build a database from English (2 minutes)

Switch mode to **Build Database** (or leave it on **Auto** and let the
router pick it - worth calling out either way).

Type:

> Create a database for a college with students, professors, courses,
> enrollments, and grades.

- While it's generating: "This is one LLM call that returns a structured
  schema as JSON - tables, columns, primary keys, foreign keys - which
  we then render as this preview."
- Point out the preview: table names, PK markers, FK arrows.
- Check **"Include sample data when building"**.
- Click **Approve & Create Database**.

> "Nothing gets written to disk until I click Approve - that's enforced
> in code, not just UI convention: `CREATE TABLE` only ever runs inside
> one function, and that function is only ever called after approval."

The new `.db` file appears in the Database dropdown automatically.

## 2. Query the new database in English (2 minutes)

Switch the **Database** dropdown to the new database. Switch mode to
**Query Database**.

Type:

> Which course has the most students?

- Point at the **Generated SQL** tab: the actual `SELECT`/`JOIN`/`GROUP
  BY` query.
- Point at the **Query Result** tab: the raw table.
- Point at the **Final Answer** tab (and the chat bubble): the plain-
  English answer.

## 3. Follow-up using memory (1 minute)

Without re-explaining context:

> How many students are enrolled in it?

> "The agent resolves 'it' using a short rolling memory of the last few
> turns - kept bounded on purpose so prompts don't grow forever. It's
> scoped per chat thread, so two tabs never leak into each other."

## 4. SQL safety validation (1-2 minutes)

Switch to the original sample database (`sample_company.db`) or stay on
the new one. Ask something that sounds like a write:

> Delete all the students.

- The agent will either refuse to generate a destructive query, or (to
  show the safety net explicitly) you can open the **Debug/Retry** panel
  and explain: "Every generated query passes through a plain Python
  validator before it's ever run - it must start with SELECT/WITH, can't
  contain a second statement, and is blocked outright if it contains
  INSERT/UPDATE/DELETE/DROP/ALTER/CREATE, before SQLite even sees it."
- Optionally ask a question you know will need repair (e.g. reference a
  column that doesn't quite exist) to show the **Debug/Retry** panel
  populate with a failed attempt, the error, and the automatic fix.

## 5. Generated SQL + result table recap (30 seconds)

Ask one more clean question to leave the panels looking good for
questions afterward, e.g.:

> Who are the top 3 highest paid employees?

## 6. MCP tool placeholders (1 minute)

Open the sidebar's **"Try a placeholder tool"** accordion. Pick a
write action, e.g. **"Gmail: Send an email"**, click **Run**.

> "This shows the pattern for external tools: read actions execute
> directly once connected, but write actions - sending an email, creating
> a Gmail draft, or creating a calendar event - always require confirmation
> first, the same rule that applies to database writes."

## 7. Wrap-up: what's next for production (1 minute)

> "Everything here runs in one local process by design - that was the
> right tradeoff for a 2-day MVP demo. Memory and the schema cache are
> both written behind small interfaces specifically so they can be
> swapped for Redis or a persistent store later without touching the
> agent or the UI code. Same story for the MCP tools - the shape is
> there, the real OAuth connections aren't, on purpose."

Close by pointing back at the sidebar/mode selector:

> "The bigger idea is that this is a workspace, not a single tool -
> Build Database, Query Database, General Chat, and future MCP tools all
> live in the same chat surface, which is what makes it feel like a
> product rather than a demo script."

## Fallback / recovery tips

- If schema generation returns malformed JSON (rare, but LLMs
  occasionally do this): just retry the same description - the button
  can be resubmitted, nothing was written to disk yet.
- If a query needs a repair attempt live: that's a **feature to
  highlight**, not a bug to hide - point at the Debug/Retry panel.
- Keep `sample_company.db` selected as a safe fallback database if a
  freshly-built one behaves unexpectedly during Q&A.
