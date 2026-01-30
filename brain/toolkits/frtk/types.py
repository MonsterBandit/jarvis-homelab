from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal


class FRTKMode(str, Enum):
    LAP_INERT = "LAP_INERT"


ExecutionSurface = Literal[
    "internal",          # FRTK-only reasoning (no calls)
    "ui_supervised",     # Human executes in Firefly UI
    "api_governed",      # ISAC executes via governed API (future)
    "future",
]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    purpose: str
    surface: ExecutionSurface
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    guardrails: List[str] = field(default_factory=list)
    available_modes: List[str] = field(default_factory=lambda: [FRTKMode.LAP_INERT.value])


@dataclass(frozen=True)
class ToolRequestSpec:
    tool_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    surface: ExecutionSurface = "future"


@dataclass(frozen=True)
class FRTKRequest:
    prompt: str
    context: Dict[str, Any] = field(default_factory=dict)
    mode: FRTKMode = FRTKMode.LAP_INERT


@dataclass(frozen=True)
class FRTKResponse:
    mode: str
    intent: str
    phase: str
    assumptions: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    proposed_requests: List[ToolRequestSpec] = field(default_factory=list)
    verification: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
