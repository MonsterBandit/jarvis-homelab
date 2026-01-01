from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

# Reuse Stage 3 helpers (internal-only, no HTTP self-calls)
from services.irr_ups import (
    DEFAULT_MOUNT_DIR,
    DEFAULT_LOG_PATH,
    _latest_snapshot_file,
    _parse_poll_line,
    _read_last_line,
    _snapshot_metadata,
)

router = APIRouter(prefix="/irr", tags=["irr"])


def _interpret(status: Optional[str], runtime_s: Optional[int], batt_pct: Optional[int]) -> str:
    if status is None:
        return "UPS status is unavailable; observability is degraded."
    if status == "OL":
        return "Utility power is present; system is operating normally."
    if status in {"OB", "OB LB"}:
        if runtime_s is not None and runtime_s < 600:
            return "Running on battery with limited runtime remaining."
        return "Running on battery; monitor runtime."
    return "UPS state observed; no action implied."


@router.get("/ups/summary")
async def ups_summary() -> Dict[str, Any]:
    """
    IRR Stage 4A: Point-in-time incident summary (read-only).
    Consumes latest poll + latest snapshot. Light interpretation only.
    """
    warnings: List[str] = []

    # Inputs
    log_line = _read_last_line(DEFAULT_LOG_PATH)
    snap_path = _latest_snapshot_file(DEFAULT_MOUNT_DIR)

    # Mirror Stage 3 behavior: 503 only if nothing is readable
    if log_line is None and snap_path is None:
        raise HTTPException(
            status_code=503,
            detail="IRR summary unavailable: missing log and snapshot",
        )

    parsed = None
    fields: Dict[str, Any] = {}
    ts_raw: Optional[str] = None

    if log_line is not None:
        pr = _parse_poll_line(log_line)
        parsed = pr.parsed
        warnings.extend(pr.warnings)
        ts_raw = parsed.get("timestamp")
        fields = parsed.get("fields", {})
    else:
        warnings.append("log_missing")

    snapshot_meta = None
    if snap_path is not None:
        try:
            snapshot_meta = _snapshot_metadata(snap_path)
        except Exception as exc:
            warnings.append(f"snapshot_metadata_failed:{exc}")
    else:
        warnings.append("snapshot_missing")

    status = fields.get("status")
    runtime_s = fields.get("runtime_s")
    batt_pct = fields.get("batt_pct")

    interpretation = _interpret(status, runtime_s, batt_pct)

    return {
        "status": "ok" if log_line is not None else "degraded",
        "timestamp": ts_raw,
        "operational": {
            "status": status,
            "runtime_s": runtime_s,
            "batt_pct": batt_pct,
            "load": fields.get("load"),
            "in_v": fields.get("in_v"),
            "out_v": fields.get("out_v"),
            "hz": fields.get("hz"),
        },
        "snapshot": snapshot_meta,
        "interpretation": interpretation,
        "warnings": warnings,
        "notes": [
            "IRR Stage 4A: incident summary (read-only)",
            "Light interpretation only; no recommendations or actions",
        ],
    }


@router.get("/ups/narrative")
async def ups_narrative_stub() -> Dict[str, Any]:
    """
    Reserved for windowed incident reconstruction.
    Stub only in Stage 4A.
    """
    return {
        "status": "not_implemented",
        "notes": [
            "IRR Stage 4A stub",
            "Windowed narrative will be implemented later",
        ],
    }
