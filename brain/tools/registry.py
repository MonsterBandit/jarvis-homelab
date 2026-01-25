from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Set


class ToolFamily(str, Enum):
    WEB = "web"
    UTILITY = "utility"
    LOCAL_READ = "local_read"


@dataclass(frozen=True)
class ToolDef:
    name: str
    family: ToolFamily
    description: str


# v4-A baseline tool set (silent, retrieval-only)
TOOL_DEFS: Dict[str, ToolDef] = {
    # Web retrieval (placeholders; actual implementation later)
    "web.search": ToolDef("web.search", ToolFamily.WEB, "Search the web for sources"),
    "web.open": ToolDef("web.open", ToolFamily.WEB, "Open a web page by ref/URL"),
    "web.find": ToolDef("web.find", ToolFamily.WEB, "Find text pattern in an opened page"),
    "web.click": ToolDef("web.click", ToolFamily.WEB, "Follow a link by id from an opened page"),
    "web.screenshot_pdf": ToolDef("web.screenshot_pdf", ToolFamily.WEB, "Screenshot a PDF page for reading"),

    # Utility retrieval
    "util.time": ToolDef("util.time", ToolFamily.UTILITY, "Get current time for an offset"),
    "util.weather": ToolDef("util.weather", ToolFamily.UTILITY, "Get weather forecast for a location"),
    "util.calc": ToolDef("util.calc", ToolFamily.UTILITY, "Evaluate a basic expression"),

    # Local read-only
    "local.read_file": ToolDef("local.read_file", ToolFamily.LOCAL_READ, "Read a file (allowlisted)"),
    "local.read_snippet": ToolDef("local.read_snippet", ToolFamily.LOCAL_READ, "Read a snippet (allowlisted)"),
}


# Capability gating by phase. For v4-A, this is the allowlist.
V4A_ENABLED_TOOLS: Set[str] = set(TOOL_DEFS.keys())


def is_known_tool(tool_name: str) -> bool:
    return tool_name in TOOL_DEFS


def is_tool_allowed(tool_name: str, enabled_tools: Optional[Set[str]] = None) -> bool:
    """
    Single authoritative gate for tool availability.
    enabled_tools defaults to v4-A baseline. Later phases may override.
    """
    enabled = enabled_tools if enabled_tools is not None else V4A_ENABLED_TOOLS
    return tool_name in enabled
