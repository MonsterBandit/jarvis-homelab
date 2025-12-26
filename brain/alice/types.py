"""
alice/types.py

Domain-neutral message and response types for Alice v1.

Design goals:
- No side effects
- No domain coupling (finance explicitly excluded by non-reference)
- Explicit uncertainty and boundary signaling
- Structured, auditable mediation outputs

This module is safe to import anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class IsacSignalKind(str, Enum):
    """
    High-level classification of what ISAC is sending to Alice.
    Alice must not change severity; she only translates and mediates.

    These map to the Phase 8 design:
    - INFORMATIONAL: facts/system state, no detected tension
    - UNCERTAIN: ambiguity/low confidence/incomplete data
    - TENSION: inconsistency or conflict in observations/structure
    - RISK: boundary or policy is being approached
    """

    INFORMATIONAL = "informational"
    UNCERTAIN = "uncertain"
    TENSION = "tension"
    RISK = "risk"


class BoundaryType(str, Enum):
    """
    Boundary classifications for how Alice communicates limits.

    - HARD: non-negotiable (gates/authority/locks)
    - SOFT: contextual (insufficient clarity, low confidence, timing)
    - VISIBILITY: out-of-scope by design (domain excluded, tool not onboarded)
    """

    HARD = "hard"
    SOFT = "soft"
    VISIBILITY = "visibility"


class AliceMove(str, Enum):
    """
    Finite set of allowed high-level moves Alice can take.
    Alice may not recommend, decide, or prioritize.
    """

    EXPLAIN = "explain"
    ASK = "ask"
    REFLECT = "reflect"
    PAUSE = "pause"
    DEFER = "defer"
    OPTIONS = "options"
    ACK_DISMISSAL = "ack_dismissal"


class ConfidenceLevel(str, Enum):
    """
    Coarse confidence buckets. Keep simple and readable for humans.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Confidence:
    """
    Confidence descriptor. `score` is optional and only used if ISAC provides it.
    """

    level: ConfidenceLevel = ConfidenceLevel.UNKNOWN
    score: Optional[float] = None  # 0.0 - 1.0 if present


@dataclass(frozen=True)
class BoundaryNotice:
    """
    A boundary that Alice must communicate, including source and what remains possible.
    """

    boundary_type: BoundaryType
    source: str  # e.g., "Global Execution Rules", "Gate D lock", "Admin constraint"
    message: str  # short, plain-language description
    allowed_alternatives: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class IsacSignal:
    """
    Canonical message from ISAC to Alice.
    This is not conversational; it's structured analysis.

    IMPORTANT:
    - `kind` guides mediation mode
    - `confidence` must be surfaced if not HIGH
    - `boundaries` must be surfaced if present
    - `data` is optional, but must be safe-to-display and non-secret
    """

    kind: IsacSignalKind
    title: str
    summary: str
    confidence: Confidence = field(default_factory=Confidence)
    boundaries: List[BoundaryNotice] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None  # for correlating logs later (Gate D evidence)


@dataclass(frozen=True)
class AliceUtterance:
    """
    A single piece of text Alice will present to the human.
    Keep it calm, reversible, and uncertainty-visible.
    """

    text: str


@dataclass(frozen=True)
class AliceResponse:
    """
    Alice's structured output, suitable for a UI layer.

    - `move` declares what Alice is doing
    - `utterances` are the human-facing messages in order
    - `questions` are explicit prompts (0 or 1 per your global rule)
    - `options` are unranked (Alice must not imply preference)
    - `boundary_echo` repeats boundary info for auditability
    """

    move: AliceMove
    utterances: List[AliceUtterance] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    options: List[str] = field(default_factory=list)
    boundary_echo: List[BoundaryNotice] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
