"""
Internal, explicit health capture for ISAC.

This module:
- Creates an in-memory HealthAggregate
- Runs selected validations explicitly
- Returns the aggregate
- Does not persist, log, or act on results
"""

from brain.validators.health import HealthAggregate
from brain.validators.base import ValidatorContext
from brain.services.validation import run_domain_validation


def capture_internal_health() -> HealthAggregate:
    """
    Capture a snapshot of internal ISAC health.

    This is:
    - Explicit
    - Read-only
    - In-memory only
    - Safe under Gate D
    """

    health = HealthAggregate()

    ctx = ValidatorContext(
        domain="isac_internal",
        source="internal_capture",
    )

    # Example minimal capture (expand later, deliberately)
    run_domain_validation(
        domain="isac_internal",
        data={},
        ctx=ctx,
        health=health,
    )

    return health
