# Screenshots and Demo Notes

This repo is ready for public screenshots, but no screenshots are committed yet.

Recommended screenshots:

1. Main AgentHub Studio chat workspace in Auto mode.
2. Build Database mode showing a schema preview before approval.
3. Query Database mode showing generated SQL, result table, schema, and debug panel.
4. RAG mode after uploading a document, showing retrieved sources.
5. MCP Connections panel showing Google connection status without exposing account details.

Do not include screenshots that show:

- API keys or OAuth client secrets.
- Personal emails, calendar events, or contact information.
- Uploaded private documents.
- Local token files or browser profile details.

Suggested demo flow:

1. Start the app with a fresh `.env` created from `.env.example`.
2. Ask AgentHub to create a small database, review the schema, then approve it.
3. Ask a SQL question and show the generated query plus answer.
4. Upload a non-private sample document in RAG mode and ask one grounded question.
5. Mention Google Workspace support as a local OAuth demo with guarded write actions.
