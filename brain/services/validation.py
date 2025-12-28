from typing import Any, Optional

from brain.validators.base import ValidatorContext
from brain.validators.runner import run_validators
from brain.validators.snapshot import ValidationSnapshot
from brain.validators.health import HealthAggregate


def run_domain_validation(
    *,
    domain: str,
    data: Any,
    ctx: ValidatorContext,
    health: Optional[HealthAggregate] = None,
) -> ValidationSnapshot:
    """
    Explicitly run validators for a given domain against provided data
    and return an immutable validation snapshot.

    If a HealthAggregate is provided, the snapshot is appended to it.

    This function is:
    - Read-only
    - Side-effect free (except optional in-memory aggregation)
    - Explicitly invoked
    - Not wired to any execution path by default
    """

    results = run_validators(
        domain=domain,
        data=data,
        ctx=ctx,
    )

    snapshot = ValidationSnapshot(
        domain=domain,
        created_at_iso=ValidationSnapshot.now_iso(),
        results=results,
        source=ctx.source,
    )

    if health is not None:
        health.add(snapshot)

    return snapshot
