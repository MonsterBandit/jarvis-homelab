from typing import Any, Dict, List

from .base import (
    ValidatorContext,
    ValidatorResult,
    ValidatorStatus,
)


# ---------------------------------------------------------------------------
# structure.required_keys
# ---------------------------------------------------------------------------

REQUIRED_KEYS_VALIDATOR = "structure.required_keys"


def required_keys(
    data: Any,
    ctx: ValidatorContext,
    *,
    required_keys: List[str],
) -> ValidatorResult:
    """
    Validate that required keys are present in a dict-like structure.

    Semantics:
    - OK: all required keys exist and are not None
    - INVALID: one or more required keys are missing or None
    - UNKNOWN: data is missing or not a dict
    """

    if data is None or not isinstance(data, dict):
        return ValidatorResult(
            validator=REQUIRED_KEYS_VALIDATOR,
            status=ValidatorStatus.UNKNOWN,
            reason="Data missing or not a dict; cannot validate required keys.",
            evidence={
                "expected_type": "dict",
                "received_type": type(data).__name__ if data is not None else "None",
            },
        )

    missing = [
        key for key in required_keys
        if key not in data or data.get(key) is None
    ]

    if missing:
        return ValidatorResult(
            validator=REQUIRED_KEYS_VALIDATOR,
            status=ValidatorStatus.INVALID,
            reason="Required keys missing or null.",
            evidence={
                "missing_keys": missing,
                "required_keys": required_keys,
            },
        )

    return ValidatorResult(
        validator=REQUIRED_KEYS_VALIDATOR,
        status=ValidatorStatus.OK,
        reason="All required keys present.",
        evidence={
            "required_keys": required_keys,
        },
    )


# ---------------------------------------------------------------------------
# structure.nested_path_exists
# ---------------------------------------------------------------------------

NESTED_PATH_VALIDATOR = "structure.nested_path_exists"


def nested_path_exists(
    data: Any,
    ctx: ValidatorContext,
    *,
    path: str,
) -> ValidatorResult:
    """
    Validate that a dotted path exists within a nested dict structure.

    Example path:
        "attributes.quantity.unit"

    Semantics:
    - OK: full path exists and resolves to a non-None value
    - INVALID: path traversal fails or resolves to None
    - UNKNOWN: data is missing or not a dict
    """

    if data is None or not isinstance(data, dict):
        return ValidatorResult(
            validator=NESTED_PATH_VALIDATOR,
            status=ValidatorStatus.UNKNOWN,
            reason="Data missing or not a dict; cannot traverse nested path.",
            evidence={
                "expected_type": "dict",
                "received_type": type(data).__name__ if data is not None else "None",
                "path": path,
            },
        )

    current: Any = data
    segments = path.split(".")

    for segment in segments:
        if not isinstance(current, dict) or segment not in current:
            return ValidatorResult(
                validator=NESTED_PATH_VALIDATOR,
                status=ValidatorStatus.INVALID,
                reason="Nested path does not exist.",
                evidence={
                    "path": path,
                    "failed_at": segment,
                },
            )

        current = current.get(segment)

        if current is None:
            return ValidatorResult(
                validator=NESTED_PATH_VALIDATOR,
                status=ValidatorStatus.INVALID,
                reason="Nested path resolves to null.",
                evidence={
                    "path": path,
                    "failed_at": segment,
                },
            )

    return ValidatorResult(
        validator=NESTED_PATH_VALIDATOR,
        status=ValidatorStatus.OK,
        reason="Nested path exists.",
        evidence={
            "path": path,
        },
    )
