from __future__ import annotations

from typing import Any, Optional

from validators.base import (
    BaseValidator,
    ValidatorContext,
    ValidatorResult,
    ValidatorStatus,
)


class NonEmptyValidator:
    """
    Generic presence validator.

    Rules:
    - None            -> INVALID
    - Empty container -> WARNING
    - Otherwise       -> OK
    """

    @property
    def name(self) -> str:
        return "presence.non_empty"

    def validate(self, data: Any, ctx: ValidatorContext) -> ValidatorResult:
        # Explicit null
        if data is None:
            return ValidatorResult(
                validator=self.name,
                status=ValidatorStatus.INVALID,
                reason=f"{ctx.domain} returned null data.",
                evidence={
                    "domain": ctx.domain,
                    "source": ctx.source,
                    "value": None,
                },
            )

        # Try to detect emptiness (lists, dicts, strings, etc.)
        try:
            length: Optional[int] = len(data)  # type: ignore[arg-type]
        except Exception:
            length = None

        if length == 0:
            return ValidatorResult(
                validator=self.name,
                status=ValidatorStatus.WARNING,
                reason=f"{ctx.domain} dataset is empty.",
                evidence={
                    "domain": ctx.domain,
                    "source": ctx.source,
                    "length": 0,
                },
            )

        return ValidatorResult(
            validator=self.name,
            status=ValidatorStatus.OK,
            reason=f"{ctx.domain} dataset is present.",
            evidence={
                "domain": ctx.domain,
                "source": ctx.source,
                "length": length,
            },
        )
