from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Protocol, runtime_checkable


class ValidatorStatus(str, Enum):
    """
    Canonical validator outcome states.
    """
    OK = "ok"
    WARNING = "warning"
    INVALID = "invalid"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ValidatorContext:
    """
    Immutable factual context passed into validators.

    This must contain facts only — no intent, no instructions.
    """
    domain: str                 # e.g. "grocy", "homeassistant", "finance"
    source: str                 # e.g. "grocy_api", "sqlite", "firefly_api"
    now_iso: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidatorResult:
    """
    Output from a validator.

    Explainable, serializable, and side-effect free.
    """
    validator: str              # stable identifier, e.g. "presence.non_empty"
    status: ValidatorStatus
    reason: str                 # short, human-readable explanation
    evidence: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BaseValidator(Protocol):
    """
    Base interface for all validators.
    """

    @property
    def name(self) -> str:
        """
        Stable, namespaced identifier.
        Example: "presence.non_empty"
        """
        ...

    def validate(self, data: Any, ctx: ValidatorContext) -> ValidatorResult:
        """
        Perform validation against provided data.
        Must not mutate state or perform I/O.
        """
        ...
