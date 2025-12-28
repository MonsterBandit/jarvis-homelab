"""
Debug-only health inspection for ISAC.

This module exists to allow a human (admin) or developer
to explicitly inspect ISAC's internal health state.

It is:
- Explicit
- Read-only
- In-memory only
- Never auto-invoked
"""

from brain.health_capture import capture_internal_health
from brain.validators.health import HealthAggregate


def inspect_internal_health() -> HealthAggregate:
    """
    Capture and return ISAC internal health for inspection.

    This function must be called explicitly and is intended
    for debugging and development only.
    """
    return capture_internal_health()
