from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, timezone

from .base import ValidatorResult


@dataclass(frozen=True)
class ValidationSnapshot:
    """
    Immutable snapshot of validation results for a given domain at a point in time.

    This is a passive data structure:
    - No behavior
    - No persistence
    - No decisions
    """

    domain: str
    created_at_iso: str
    results: List[ValidatorResult] = field(default_factory=list)
    source: Optional[str] = None

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
