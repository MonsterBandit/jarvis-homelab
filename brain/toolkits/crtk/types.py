from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class CRTKMode(str, Enum):
    """
    Operational mode for CRTK behavior shaping.

    LAP_INERT is the only permitted mode during LAP.
    """
    LAP_INERT = "LAP_INERT"


ExecutionSurface = Literal[
    "internal",              # CRTK-only reasoning (no calls)
    "toolbelt_local_read",   # Callable LOCAL_READ tools (requested, not executed here)
    "toolbelt_web_okd",      # Callable WEB tools under OKD (requested, not executed here)
    "toolbelt_utility",      # Callable UTILITY tools (requested, not executed here)
    "runner",                # Callable runner/execution surfaces (requested, not executed here)
    "future",                # Placeholder for later additions (not available in LAP)
]


@dataclass(frozen=True)
class ToolSpec:
    """
    Declarative description of a specialist coding tool capability.

    IMPORTANT:
    - This is an inventory/contract record, not an executable function.
    """
    name: str
    purpose: str
    surface: ExecutionSurface
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    guardrails: List[str] = field(default_factory=list)
    available_modes: List[str] = field(default_factory=lambda: [CRTKMode.LAP_INERT.value])


@dataclass(frozen=True)
class ToolRequestSpec:
    """
    A non-executing request specification for a callable tool.

    CRTK may emit these specs to communicate what it *would* call later,
    but CRTK never executes tools.
    """
    tool_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    surface: ExecutionSurface = "future"


@dataclass(frozen=True)
class CRTKRequest:
    """
    Pure-input request for CRTK.

    IMPORTANT:
    - No filesystem paths here that will be auto-read.
    - If file paths are mentioned, they are treated as user-provided strings only.
    """
    prompt: str
    context: Dict[str, Any] = field(default_factory=dict)
    mode: CRTKMode = CRTKMode.LAP_INERT


@dataclass(frozen=True)
class CRTKResponse:
    """
    Pure-output response from CRTK.

    IMPORTANT:
    - Data only. No execution. No I/O.
    - Any tool usage is expressed as ToolRequestSpec entries (requests only).
    """
    mode: str
    intent: str
    assumptions: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    proposed_requests: List[ToolRequestSpec] = field(default_factory=list)
    verification: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
