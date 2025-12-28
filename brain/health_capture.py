"""
Internal, explicit health capture for ISAC.
"""

from validators.health import HealthAggregate
from validators.base import ValidatorContext
from services.validation import run_domain_validation
from validators.sanity import non_empty_string


def capture_internal_health() -> HealthAggregate:
    """
    Capture a snapshot of internal ISAC health.

    Explicit, read-only, in-memory only.
    """

    health = HealthAggregate()

    ctx = ValidatorContext(
        domain="isac_internal",
        source="internal_capture",
    )

    # A1: Domain identity sanity (explicit, deliberate)
    domain_result = non_empty_string(ctx.domain, ctx)

    run_domain_validation(
        domain="isac_internal",
        data={},
        ctx=ctx,
        health=health,
        extra_results=[domain_result],
    )

    return health
