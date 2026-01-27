from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ToolFailureClass(str, Enum):
    # Governance-aligned failure taxonomy (v4-4 compatible)
    TOOL_NOT_ALLOWED = "TOOL_NOT_ALLOWED"
    SANDBOX_BOUNDARY = "SANDBOX_BOUNDARY"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    TOOL_UPSTREAM_ERROR = "TOOL_UPSTREAM_ERROR"
    TOOL_BAD_INPUT = "TOOL_BAD_INPUT"
    TOOL_PARSE_ERROR = "TOOL_PARSE_ERROR"
    TOOL_INTERNAL_ERROR = "TOOL_INTERNAL_ERROR"


@dataclass
class ToolProvenance:
    """
    Human-meaningful provenance.
    Never expose raw internal IDs unless explicitly requested.
    """
    sources: List[str]          # e.g. URLs, filenames, identifiers
    retrieved_at: str           # ISO timestamp
    notes: Optional[str] = None


@dataclass
class ToolRequest:
    """
    Canonical request envelope for all tools.
    Alice never constructs this directly; ISAC does.
    """
    tool_name: str
    args: Dict[str, Any]
    purpose: str                # Short human sentence: why this tool is used
    task_id: int
    step_id: Optional[int]
    user_id: str
    chat_id: str


@dataclass
class ToolResult:
    """
    Canonical result envelope.
    This is what ISAC hands back to Alice (sanitized).
    """
    ok: bool
    tool_name: str
    primary: Optional[Any] = None          # Human-safe payload
    provenance: Optional[ToolProvenance] = None

    failure_class: Optional[ToolFailureClass] = None
    failure_message: Optional[str] = None  # Human-safe explanation

    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    latency_ms: Optional[int] = None
