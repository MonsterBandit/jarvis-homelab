from __future__ import annotations

from ..types import ToolRequestSpec


def request_read_file(path: str, reason: str = "") -> ToolRequestSpec:
    return ToolRequestSpec(
        tool_name="local.read_file",
        args={"path": path},
        reason=reason or "Read file content for accurate reasoning.",
        surface="toolbelt_local_read",
    )


def request_read_snippet(path: str, start_line: int, end_line: int, reason: str = "") -> ToolRequestSpec:
    return ToolRequestSpec(
        tool_name="local.read_snippet",
        args={"path": path, "start_line": int(start_line), "end_line": int(end_line)},
        reason=reason or "Read a bounded snippet for precise edits.",
        surface="toolbelt_local_read",
    )
