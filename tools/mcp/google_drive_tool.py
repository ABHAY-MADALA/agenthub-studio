"""
google_drive_tool.py
=====================
Placeholder Google Drive MCP tool. Future actions: import CSV/PDF/docs,
summarize files. Imported files would land in storage/uploads/ and could
feed the Database Builder (e.g. "build a database from this CSV").
"""

from .base import MCPTool, ToolAction

google_drive_tool = MCPTool(
    key="google_drive",
    display_name="Google Drive",
    icon="📁",
    description="Import CSV/PDF/docs and summarize files.",
    actions=[
        ToolAction("import_file", "Import a file into storage/uploads/", write_action=False),
        ToolAction("summarize_file", "Summarize an imported file", write_action=False),
    ],
)
