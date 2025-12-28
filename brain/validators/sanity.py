from typing import Any

from .base import (
    ValidatorContext,
    ValidatorResult,
    ValidatorStatus,
)


# ---------------------------------------------------------------------------
# sanity.non_negative_number
# ---------------------------------------------------------------------------

NON_NEGATIVE_NUMBER_VALIDATOR = "sanity.non_negative_number"


def non_negative_number(
    value: Any,
    ctx: ValidatorContext,
) -> ValidatorResult:
    """
    Validate that a value is a non-negative number.

    Strict semantics:
    - OK: value is int or float and >= 0
    - INVALID: value is numeric but < 0, OR value is a numeric-looking string
    - UNKNOWN: value is None or not numeric
    """

    if value is None:
        return ValidatorResult(
            validator=NON_NEGATIVE_NUMBER_VALIDATOR,
            status=ValidatorStatus.UNKNOWN,
            reason="Value missing; cannot validate numeric sanity.",
            evidence={},
        )

    # Reject numeric-looking strings explicitly (strict mode)
    if isinstance(value, str):
        return ValidatorResult(
            validator=NON_NEGATIVE_NUMBER_VALIDATOR,
            status=ValidatorStatus.INVALID,
            reason="Numeric strings are not accepted; coercion is not allowed.",
            evidence={
                "received_type": "str",
                "received_value": value,
            },
        )

    if isinstance(value, (int, float)):
        if value < 0:
            return ValidatorResult(
                validator=NON_NEGATIVE_NUMBER_VALIDATOR,
                status=ValidatorStatus.INVALID,
                reason="Numeric value is negative.",
                evidence={
                    "value": value,
                },
            )

        return ValidatorResult(
            validator=NON_NEGATIVE_NUMBER_VALIDATOR,
            status=ValidatorStatus.OK,
            reason="Numeric value is non-negative.",
            evidence={
                "value": value,
            },
        )

    return ValidatorResult(
        validator=NON_NEGATIVE_NUMBER_VALIDATOR,
        status=ValidatorStatus.UNKNOWN,
        reason="Value is not numeric; cannot validate numeric sanity.",
        evidence={
            "received_type": type(value).__name__,
        },
    )


# ---------------------------------------------------------------------------
# sanity.non_empty_string
# ---------------------------------------------------------------------------

NON_EMPTY_STRING_VALIDATOR = "sanity.non_empty_string"


def non_empty_string(
    value: Any,
    ctx: ValidatorContext,
) -> ValidatorResult:
    """
    Validate that a value is a non-empty string.

    Strict semantics:
    - OK: value is str and value.strip() is non-empty
    - INVALID: value is str but empty or whitespace-only
    - UNKNOWN: value is None or not a string
    """

    if value is None:
        return ValidatorResult(
            validator=NON_EMPTY_STRING_VALIDATOR,
            status=ValidatorStatus.UNKNOWN,
            reason="Value missing; cannot validate string content.",
            evidence={},
        )

    if not isinstance(value, str):
        return ValidatorResult(
            validator=NON_EMPTY_STRING_VALIDATOR,
            status=ValidatorStatus.UNKNOWN,
            reason="Value is not a string; cannot validate string content.",
            evidence={
                "received_type": type(value).__name__,
            },
        )

    if value.strip() == "":
        return ValidatorResult(
            validator=NON_EMPTY_STRING_VALIDATOR,
            status=ValidatorStatus.INVALID,
            reason="String is empty or whitespace-only.",
            evidence={
                "value": value,
            },
        )

    return ValidatorResult(
        validator=NON_EMPTY_STRING_VALIDATOR,
        status=ValidatorStatus.OK,
        reason="String is non-empty.",
        evidence={
            "value": value,
        },
    )


# ---------------------------------------------------------------------------
# sanity.boolean_strict
# ---------------------------------------------------------------------------

BOOLEAN_STRICT_VALIDATOR = "sanity.boolean_strict"


def boolean_strict(
    value: Any,
    ctx: ValidatorContext,
) -> ValidatorResult:
    """
    Validate that a value is a strict boolean.

    Strict semantics:
    - OK: value is exactly True or False
    - INVALID: value is a boolean-like (e.g., 1, 0, "true", "false")
    - UNKNOWN: value is None or not boolean-like
    """

    if value is None:
        return ValidatorResult(
            validator=BOOLEAN_STRICT_VALIDATOR,
            status=ValidatorStatus.UNKNOWN,
            reason="Value missing; cannot validate boolean.",
            evidence={},
        )

    # Exact booleans only
    if value is True or value is False:
        return ValidatorResult(
            validator=BOOLEAN_STRICT_VALIDATOR,
            status=ValidatorStatus.OK,
            reason="Value is a strict boolean.",
            evidence={
                "value": value,
            },
        )

    # Boolean-like values are explicitly rejected
    if isinstance(value, (int, str)):
        return ValidatorResult(
            validator=BOOLEAN_STRICT_VALIDATOR,
            status=ValidatorStatus.INVALID,
            reason="Boolean-like value is not accepted; coercion is not allowed.",
            evidence={
                "received_type": type(value).__name__,
                "received_value": value,
            },
        )

    return ValidatorResult(
        validator=BOOLEAN_STRICT_VALIDATOR,
        status=ValidatorStatus.UNKNOWN,
        reason="Value is not boolean; cannot validate boolean.",
        evidence={
            "received_type": type(value).__name__,
        },
    )


# ---------------------------------------------------------------------------
# sanity.enum_allowed_values
# ---------------------------------------------------------------------------

ENUM_ALLOWED_VALUES_VALIDATOR = "sanity.enum_allowed_values"


def enum_allowed_values(
    value: Any,
    ctx: ValidatorContext,
    *,
    allowed_values: list[Any],
) -> ValidatorResult:
    """
    Validate that a value is one of an explicit allowed set.

    Strict semantics:
    - OK: value is exactly one of allowed_values
    - INVALID: value is present but not in allowed_values
    - UNKNOWN: value is None
    """

    if value is None:
        return ValidatorResult(
            validator=ENUM_ALLOWED_VALUES_VALIDATOR,
            status=ValidatorStatus.UNKNOWN,
            reason="Value missing; cannot validate enum membership.",
            evidence={
                "allowed_values": allowed_values,
            },
        )

    if value not in allowed_values:
        return ValidatorResult(
            validator=ENUM_ALLOWED_VALUES_VALIDATOR,
            status=ValidatorStatus.INVALID,
            reason="Value is not in allowed enum set.",
            evidence={
                "value": value,
                "allowed_values": allowed_values,
            },
        )

    return ValidatorResult(
        validator=ENUM_ALLOWED_VALUES_VALIDATOR,
        status=ValidatorStatus.OK,
        reason="Value is in allowed enum set.",
        evidence={
            "value": value,
        },
    )
