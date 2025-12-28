from typing import Any

from brain.validators.base import ValidatorContext
from brain.validators.runner import run_validators
from brain.validators.snapshot import ValidationSnapshot


def run_domain_validation(
    *,
    domain: str,
    data: Any,
    ctx: ValidatorContext,
) -> ValidationSnapshot:
    """
    Explicitly run validators for a given domain against provided data
    and return an immutable validation snapshot.

    This function is:
    - Read-only
    - Side-effect free
    - Explicitly invoked
    - Not wired to any execution path yet
    """

    results = run_validators(
        domain=domain,
        data=data,
        ctx=ctx,
    )

    return ValidationSnapshot(
        domain=domain,
        created_at_iso=ValidationSnapshot.now_iso(),
        results=results,
        source=ctx.source,
    )
