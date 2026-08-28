"""
registry.py
===========
One place that lists every MCP-style tool so the sidebar (and, later, a
real tool-routing agent) can enumerate them without importing each module
by hand. Adding a new integration is: write tool.py exporting an MCPTool
instance (see gmail_tool.py for the shape), then add it to ALL_TOOLS here.
"""

from .gmail_tool import gmail_tool
from .google_calendar_tool import google_calendar_tool
from .google_drive_tool import google_drive_tool
from .oauth import is_tool_connected

ALL_TOOLS = [gmail_tool, google_drive_tool, google_calendar_tool]


def sidebar_lines() -> list:
    """Human-readable lines for the sidebar, e.g. '📧 Gmail (not connected)'."""
    return [
        f"{tool.icon} {tool.display_name} — {'connected' if is_tool_connected(tool.key) else 'not connected'}"
        for tool in ALL_TOOLS
    ]
