from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional, Set

from tools.registry import TOOL_DEFS, is_known_tool, is_tool_allowed
from tools.types import ToolFailureClass, ToolRequest, ToolResult
from tools.web_tools import run_web_tool
from tools.utility_tools import run_utility_tool
from tools.local_read_tools import run_local_read_tool


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def run_tool(req: ToolRequest, enabled_tools: Optional[Set[str]] = None, governance_ctx: Optional[dict] = None) -> ToolResult:
    """
    Single, canonical tool execution entry point.

    v4-A contract:
    - No stubs for enabled tools.
    - Fail closed with governance-aligned failure classes.
    - Tool-family logic is delegated to family modules (e.g. tools/web_tools.py).
    """
    t0 = time.monotonic()
    started = _iso_now()

    # Bundle 4 Phase 1: Sandbox boundary (mechanical guardrail)
    # If a caller supplies governance_ctx={"sandbox": True}, fail-closed with structured boundary signal.
    if governance_ctx and bool(governance_ctx.get("sandbox")):
        return ToolResult(
            ok=False,
            tool_name=req.tool_name,
            failure_class=ToolFailureClass.SANDBOX_BOUNDARY,
            failure_message="Sandbox mode forbids tool execution.",
            started_at=started,
            ended_at=_iso_now(),
            latency_ms=int((time.monotonic() - t0) * 1000),
        )


    if not is_known_tool(req.tool_name):
        return ToolResult(
            ok=False,
            tool_name=req.tool_name,
            failure_class=ToolFailureClass.TOOL_NOT_ALLOWED,
            failure_message=f"Unknown tool: {req.tool_name}",
            started_at=started,
            ended_at=_iso_now(),
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

    if not is_tool_allowed(req.tool_name, enabled_tools=enabled_tools):
        return ToolResult(
            ok=False,
            tool_name=req.tool_name,
            failure_class=ToolFailureClass.TOOL_NOT_ALLOWED,
            failure_message=f"Tool not enabled: {req.tool_name}",
            started_at=started,
            ended_at=_iso_now(),
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

    tdef = TOOL_DEFS[req.tool_name]

    if tdef.family.value == "web":
        result = await run_web_tool(req)
        if not result.started_at:
            result.started_at = started
        result.ended_at = result.ended_at or _iso_now()
        if result.latency_ms is None:
            result.latency_ms = int((time.monotonic() - t0) * 1000)
        return result

    if tdef.family.value == "utility":
        result = await run_utility_tool(req)
        if not result.started_at:
            result.started_at = started
        result.ended_at = result.ended_at or _iso_now()
        if result.latency_ms is None:
            result.latency_ms = int((time.monotonic() - t0) * 1000)
        return result

    if tdef.family.value == "local_read":
        result = await run_local_read_tool(req)
        if not result.started_at:
            result.started_at = started
        result.ended_at = result.ended_at or _iso_now()
        if result.latency_ms is None:
            result.latency_ms = int((time.monotonic() - t0) * 1000)
        return result

    return ToolResult(
        ok=False,
        tool_name=req.tool_name,
        failure_class=ToolFailureClass.TOOL_INTERNAL_ERROR,
        failure_message=f"Tool family not implemented: {tdef.family.value}",
        started_at=started,
        ended_at=_iso_now(),
        latency_ms=int((time.monotonic() - t0) * 1000),
    )