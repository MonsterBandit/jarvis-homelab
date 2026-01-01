from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/irr", tags=["irr"])

DEFAULT_MOUNT_DIR = Path("/irr")
DEFAULT_LOG_PATH = DEFAULT_MOUNT_DIR / "ups_poll.log"


@dataclass
class ParseResult:
    parsed: Dict[str, Any]
    warnings: List[str]


def _read_text_limited(path: Path, max_bytes: int = 80_000) -> str:
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
        return data.decode("utf-8", errors="replace") + "\n...[truncated]..."
    return data.decode("utf-8", errors="replace")


def _read_last_line(path: Path, max_bytes: int = 16_384) -> Optional[str]:
    if not path.exists() or path.stat().st_size == 0:
        return None

    with path.open("rb") as f:
        size = f.seek(0, 2)
        seek_back = min(size, max_bytes)
        f.seek(-seek_back, 2)
        chunk = f.read(seek_back)

    lines = chunk.splitlines()
    if not lines:
        return None
    return lines[-1].decode("utf-8", errors="replace").strip()


def _parse_iso_best_effort(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return None


def _parse_poll_line(line: str) -> ParseResult:
    warnings: List[str] = []
    parsed: Dict[str, Any] = {"timestamp": None, "fields": {}}

    if not line:
        return ParseResult(parsed=parsed, warnings=["empty_line"])

    parts = [p.strip() for p in line.split("|")]
    if not parts:
        return ParseResult(parsed=parsed, warnings=["split_failed"])

    ts_raw = parts[0]
    parsed["timestamp"] = ts_raw
    if _parse_iso_best_effort(ts_raw) is None:
        warnings.append("timestamp_parse_failed")

    fields: Dict[str, Any] = {}
    for seg in parts[1:]:
        if not seg:
            continue
        if "=" not in seg:
            warnings.append(f"unparsed_segment:{seg}")
            continue
        k, v = seg.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue

        if v == "":
            fields[k] = None
            continue

        try:
            if "." in v:
                fields[k] = float(v)
            else:
                fields[k] = int(v)
        except Exception:
            fields[k] = v

    parsed["fields"] = fields

    # Hybrid strictness: require these keys
    required = ["status", "load", "runtime_s", "batt_pct"]
    for rk in required:
        if rk not in fields:
            warnings.append(f"missing_required:{rk}")

    return ParseResult(parsed=parsed, warnings=warnings)


def _latest_snapshot_file(mount_dir: Path) -> Optional[Path]:
    if not mount_dir.exists():
        return None
    snaps = sorted(
        mount_dir.glob("ups_snapshot_*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return snaps[0] if snaps else None


def _snapshot_metadata(path: Path) -> Dict[str, Any]:
    captured_at = None
    name = path.name
    try:
        stem = name.replace("ups_snapshot_", "").replace(".txt", "")
        dt = datetime.strptime(stem, "%Y%m%d_%H%M%S")
        captured_at = dt.isoformat()
    except Exception:
        captured_at = None

    st = path.stat()
    return {
        "file": str(path),
        "name": name,
        "size_bytes": st.st_size,
        "mtime_epoch": st.st_mtime,
        "captured_at_guess": captured_at,
    }


@router.get("/ups/latest")
async def ups_latest() -> Dict[str, Any]:
    mount_dir = DEFAULT_MOUNT_DIR
    log_path = DEFAULT_LOG_PATH

    log_line = _read_last_line(log_path)
    snap_path = _latest_snapshot_file(mount_dir)

    # M3: 200 if anything readable; otherwise 503
    if log_line is None and snap_path is None:
        raise HTTPException(
            status_code=503,
            detail="IRR data unavailable: missing /irr/ups_poll.log and no ups_snapshot_*.txt",
        )

    parsed_line = None
    parse_warnings: List[str] = []
    if log_line is not None:
        pr = _parse_poll_line(log_line)
        parsed_line = pr.parsed
        parse_warnings = pr.warnings

    snapshot_meta = None
    snapshot_raw = None
    snapshot_warnings: List[str] = []
    if snap_path is not None:
        try:
            snapshot_meta = _snapshot_metadata(snap_path)
            snapshot_raw = _read_text_limited(snap_path, max_bytes=80_000)
        except Exception as exc:
            snapshot_warnings.append(f"snapshot_read_failed:{exc}")

    status = "ok"
    warnings: List[str] = []
    if log_line is None:
        status = "degraded"
        warnings.append("log_missing")
    if snap_path is None:
        status = "degraded"
        warnings.append("snapshot_missing")

    warnings.extend(parse_warnings)
    warnings.extend(snapshot_warnings)

    return {
        "status": status,
        "mount_dir": str(mount_dir),
        "log": {
            "path": str(log_path),
            "latest_line_raw": log_line,
            "latest_line_parsed": parsed_line,
        },
        "snapshot": {
            "latest_meta": snapshot_meta,
            "latest_raw": snapshot_raw,
        },
        "warnings": warnings,
        "notes": [
            "IRR Stage 3: read-only observability",
            "No command execution, no writes, no automation",
            "Hybrid strictness parse warnings included",
        ],
    }


@router.get("/ups/log")
async def ups_log(
    since: Optional[str] = Query(default=None, description="ISO timestamp; return lines strictly after this time"),
    tail: Optional[int] = Query(default=None, ge=1, le=1000, description="Return last N lines (default 60). Max 1000."),
) -> Dict[str, Any]:
    log_path = DEFAULT_LOG_PATH
    if not log_path.exists():
        # M3: log endpoint strict
        raise HTTPException(status_code=503, detail="IRR log missing: /irr/ups_poll.log")

    if since and tail is not None:
        raise HTTPException(status_code=400, detail="Provide either 'since' or 'tail', not both")

    # K1 default
    if not since and tail is None:
        tail = 60

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

    if since:
        since_dt = _parse_iso_best_effort(since)
        if since_dt is None:
            raise HTTPException(status_code=400, detail="Invalid 'since' timestamp (expected ISO format)")

        out: List[str] = []
        filtered_count = 0
        for ln in lines:
            ts_part = ln.split(" | ", 1)[0].strip()
            ts_dt = _parse_iso_best_effort(ts_part)
            if ts_dt is None:
                filtered_count += 1
                continue
            if ts_dt > since_dt:
                out.append(ln)
            else:
                filtered_count += 1

        return {
            "status": "ok",
            "mode": "since",
            "since": since,
            "count": len(out),
            "lines": out,
            "notes": [
                "Timestamps parsed internally for filtering only; output preserves original log lines",
                f"Filtered out {filtered_count} line(s) at/before since or with unparsable timestamps",
            ],
        }

    n = int(tail or 60)
    out_lines = lines[-n:] if len(lines) > n else lines

    return {
        "status": "ok",
        "mode": "tail",
        "tail": n,
        "count": len(out_lines),
        "lines": out_lines,
        "notes": [
            "Output preserves original log lines exactly",
            "Default tail=60 (~1 hour at 60s polling)",
        ],
    }
