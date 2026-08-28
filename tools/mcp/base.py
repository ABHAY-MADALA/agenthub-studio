"""
base.py
=======
Shared shape for MCP-style tool integrations (Gmail, Google Drive, Google
Calendar, ...). OAuth connect/disconnect status is handled by
tools/mcp/oauth.py. Individual API actions can be filled in here one by
one while keeping the same read-vs-write safety model.

Each tool is a list of ToolAction entries. Read-only actions ("summarize",
"list", "analyze") can execute immediately once connected. Actions marked
write_action=True ("send", "draft", "create issue") must go through
call(..., confirmed=True) - calling without confirmation returns a
"needs your approval" message instead of making external changes.

To add a real action: implement MCPTool.call() for that one tool (e.g. swap
the generic body in gmail_tool.py for real Gmail API calls) - app.py only
needs to call tool.call(action_name, confirmed=...).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class ToolAction:
    name: str
    description: str
    write_action: bool = False


@dataclass
class MCPTool:
    key: str  # stable id, e.g. "gmail"
    display_name: str
    icon: str
    description: str
    actions: List[ToolAction]
    connected: bool = False
    # Optional per-action placeholder response overrides.
    _custom_responses: Dict[str, str] = field(default_factory=dict)

    def call(self, action_name: str, confirmed: bool = False, **kwargs) -> str:
        action = next((a for a in self.actions if a.name == action_name), None)
        if action is None:
            return f"[{self.display_name}] Unknown action '{action_name}'."

        if not self.connected:
            return (
                f"{self.display_name} isn't connected. "
                "Connect it in MCP Connections and I can continue."
            )

        if action.write_action and not confirmed:
            return (
                f"[{self.display_name}] This action ({action.description}) would modify "
                f"external data. Please confirm before it runs."
            )

        custom = self._custom_responses.get(action_name)
        if custom:
            return custom
        return (
            f"[{self.display_name}] '{action.description}' is not implemented yet. "
            "No external action was performed."
        )
