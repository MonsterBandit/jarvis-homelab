from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from brain.validators.base import (
    ValidatorContext,
    ValidatorResult,
    ValidatorStatus,
)


class AgeStalenessValidator:
    """
    Staleness validator based on ISO 8601 timestamps.

    Inputs:
    - data: any object (unused; timestamp comes from ctx.meta or explicit arg)
    - ctx.meta["timestamp_iso"]: ISO 8601 timestamp string

    Rules:
    - Missing timestamp         -> UNKNOWN
    - Unparseable timestamp     -> INVALID
    - Age <= warn_after_seconds -> OK
    - Age > warn_after_seconds  -> WARNING
    - Age > invalid_after_secs  -> INVALID
    """

    def __init__(
        self,
        *,
        warn_after_seconds: int,
        invalid_after_seconds: Optional[int] = None,
        timestamp_meta_key: str = "timestamp_iso",
    ) -> None:
        self._warn_after = warn_after_seconds
        self._invalid_after = invalid_after_seconds
        self._meta_key = timestamp_meta_key

    @property
    def name(self) -> str:
        return "staleness.age"

    def _parse_iso(self, ts: str) -> datetime:
        # Allow Z suffix and naive UTC
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def validate(self, data: Any, ctx: ValidatorContext) -> ValidatorResult:
        ts_value = ctx.meta.get(self._meta_key)

        # No timestamp provided
        if not ts_value:
            return ValidatorResult(
                validator=self.name,
                status=ValidatorStatus.UNKNOWN,
                reason="No timestamp available to assess freshness.",
                evidence={
                    "domain": ctx.domain,
                    "source": ctx.source,
                    "meta_key": self._meta_key,
                },
            )

        # Parse timestamp
        try:
            ts_dt = self._parse_iso(str(ts_value))
        except Exception as exc:
            return ValidatorResult(
                validator=self.name,
                status=ValidatorStatus.INVALID,
                reason="Timestamp is not a valid ISO 8601 value.",
                evidence={
                    "domain": ctx.domain,
                    "source": ctx.source,
                    "timestamp": ts_value,
                    "error": str(exc),
                },
            )

        # Determine 'now'
        if ctx.now_iso:
            try:
                now_dt = self._parse_iso(ctx.now_iso)
            except Exception:
                now_dt = datetime.now(timezone.utc)
        else:
            now_dt = datetime.now(timezone.utc)

        age_seconds = int((now_dt - ts_dt).total_seconds())

        # Negative age (clock skew)
        if age_seconds < 0:
            return ValidatorResult(
                validator=self.name,
                status=ValidatorStatus.WARNING,
                reason="Timestamp is in the future relative to now.",
                evidence={
                    "domain": ctx.domain,
                    "source": ctx.source,
                    "timestamp": ts_value,
                    "age_seconds": age_seconds,
                },
            )

        # Invalid threshold
        if self._invalid_after is not None and age_seconds > self._invalid_after:
            return ValidatorResult(
                validator=self.name,
                status=ValidatorStatus.INVALID,
                reason="Data is too old to be considered valid.",
                evidence={
                    "domain": ctx.domain,
                    "source": ctx.source,
                    "timestamp": ts_value,
                    "age_seconds": age_seconds,
                    "invalid_after_seconds": self._invalid_after,
                },
            )

        # Warning threshold
        if age_seconds > self._warn_after:
            return ValidatorResult(
                validator=self.name,
                status=ValidatorStatus.WARNING,
                reason="Data may be stale.",
                evidence={
                    "domain": ctx.domain,
                    "source": ctx.source,
                    "timestamp": ts_value,
                    "age_seconds": age_seconds,
                    "warn_after_seconds": self._warn_after,
                },
            )

        return ValidatorResult(
            validator=self.name,
            status=ValidatorStatus.OK,
            reason="Data is fresh.",
            evidence={
                "domain": ctx.domain,
                "source": ctx.source,
                "timestamp": ts_value,
                "age_seconds": age_seconds,
            },
        )
