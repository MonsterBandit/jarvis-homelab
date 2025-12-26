"""
alice package

Domain-neutral Alice v1.

Exports only the public, safe entry points.
No runtime wiring is performed here.
"""

from .mediator import mediate
from .types import (
    IsacSignal,
    IsacSignalKind,
    AliceResponse,
    AliceMove,
    BoundaryNotice,
    BoundaryType,
    Confidence,
    ConfidenceLevel,
)

__all__ = [
    "mediate",
    "IsacSignal",
    "IsacSignalKind",
    "AliceResponse",
    "AliceMove",
    "BoundaryNotice",
    "BoundaryType",
    "Confidence",
    "ConfidenceLevel",
]
