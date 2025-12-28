from typing import Any, Optional, List

from brain.validators.base import ValidatorContext, ValidatorResult
from brain.validators.runner import run_validators
from brain.validators.snapshot import ValidationSnapshot
from brain.validators.health import HealthAggregate


def run_domain_validation(
    *,
    domain: str,
    data: Any,
    ctx: ValidatorContext,
    health: Optional[HealthAggregate] = None,
    extra_results: Optional[List[ValidatorResult]] = None,
) -> ValidationSnapshot:
    """
    Explicitly run validators for a given domain against provided data
    and return an immutable validation snapshot.

    Optional extra_results allow deliberate, one-off checks to be
    included without registry coupling.

    This function is:
    - Read-only
    - Explicit
    - Side-effect free (except optional in-memory aggregation)
    """

    results = run_validators(
        domain=domain,
        data=data,
        ctx=ctx,
    )

    if extra_results:
        results = list(results) + list(extra_results)

    snapshot = ValidationSnapshot(
        domain=domain,
        created_at_iso=ValidationSnapshot.now_iso(),
        results=results,
        source=ctx.source,
    )

    if health is not None:
        health.add(snapshot)

    return snapshot
