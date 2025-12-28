from __future__ import annotations

from typing import Any, List, Optional

from brain.validators.base import ValidatorContext, ValidatorResult
from brain.validators.registry import ValidatorRegistry


def run_validators(
    *,
    domain: str,
    data: Any,
    ctx: ValidatorContext,
    registry: Optional[ValidatorRegistry] = None,
) -> List[ValidatorResult]:
    """
    Run the registered validators for a domain.

    - No side effects
    - No I/O
    - Deterministic given (data, ctx, registry)

    Returns a raw list of ValidatorResult for maximum flexibility.
    """
    reg = registry or ValidatorRegistry()
    validators = reg.get_validators(domain)

    results: List[ValidatorResult] = []
    for v in validators:
        results.append(v.validate(data, ctx))

    return results
