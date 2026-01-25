from __future__ import annotations

import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from tools.types import ToolFailureClass, ToolProvenance, ToolRequest, ToolResult


# ---------------------------------------------------------
# Local Read Guardrails (v4-A)
# - Read-only
# - Allowlisted paths only
# - Conservative caps (size + lines)
# - No binary reads
# ---------------------------------------------------------

MAX_FILE_BYTES = int(os.getenv("ISAC_LOCAL_READ_MAX_BYTES", "65536"))          # 64 KB default
MAX_SNIPPET_LINES = int(os.getenv("ISAC_LOCAL_SNIPPET_MAX_LINES", "200"))      # 200 lines default
MAX_SNIPPET_BYTES = int(os.getenv("ISAC_LOCAL_SNIPPET_MAX_BYTES", "32768"))     # 32 KB default


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fail(req: ToolRequest, cls: ToolFailureClass, msg: str, started_at: str, t0: float) -> ToolResult:
    return ToolResult(
        ok=False,
        tool_name=req.tool_name,
        failure_class=cls,
        failure_message=msg,
        started_at=started_at,
        ended_at=_iso_now(),
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


def _is_path_allowed(path: str) -> bool:
    """Read allowlist (LOCKED to match write allowlist + OKD):
    - /opt/jarvis/brain/**
    - /opt/jarvis/data/index.html
    - /opt/jarvis/governance/operational_knowledge/**
    """
    if not path:
        return False
    norm = os.path.abspath(path)

    if norm.startswith("/opt/jarvis/brain/"):
        return True

    if norm == "/opt/jarvis/data/index.html":
        return True

    if norm.startswith("/opt/jarvis/governance/operational_knowledge/"):
        return True

    return False


def resolve_target_path(user_path: str, must_exist: bool = True) -> Optional[str]:
    """Resolve a user-facing path (host-style) to an on-container filesystem path.

    Mirrors the existing conservative mapping used elsewhere:
    - /opt/jarvis/brain/** -> /app/**
    - /opt/jarvis/brain/main.py -> /app/main.py
    """
    p = (user_path or "").strip()
    if not p:
        return None

    candidates = [p]

    # Known mount: /opt/jarvis/brain -> /app
    if p == "/opt/jarvis/brain/main.py":
        candidates.append("/app/main.py")
    if p.startswith("/opt/jarvis/brain/"):
        rel = p[len("/opt/jarvis/brain/"):]
        candidates.append("/app/" + rel)
    if p == "/opt/jarvis/brain":
        candidates.append("/app")

    # As a last resort, if user passed a relative-ish path, try under /app
    if not p.startswith("/") and p:
        candidates.append("/app/" + p)

    for c in candidates:
        try:
            if must_exist:
                if Path(c).exists():
                    return c
            else:
                parent = Path(c).parent
                if parent.exists():
                    return c
        except Exception:
            continue
    return None


def _read_bytes_checked(path: str) -> Tuple[bytes, int]:
    p = Path(path)
    size = int(p.stat().st_size)
    if size > MAX_FILE_BYTES:
        raise ValueError(f"file exceeds cap ({size} bytes > {MAX_FILE_BYTES} bytes)")
    data = p.read_bytes()
    if b"\x00" in data:
        raise ValueError("binary file (NUL byte) not allowed")
    return data, size


def _decode_utf8_strict(data: bytes) -> str:
    return data.decode("utf-8")  # strict by default


def _util_read_file(req: ToolRequest, started_at: str, t0: float) -> ToolResult:
    args = req.args or {}
    user_path = str(args.get("path") or "").strip()
    if not user_path:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "local.read_file requires path", started_at, t0)

    if not _is_path_allowed(user_path):
        return _fail(req, ToolFailureClass.TOOL_NOT_ALLOWED, "Path not allowlisted", started_at, t0)

    resolved = resolve_target_path(user_path, must_exist=True)
    if not resolved:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "File not found (or not resolvable)", started_at, t0)

    try:
        data, size = _read_bytes_checked(resolved)
        text = _decode_utf8_strict(data)
    except ValueError as e:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, str(e), started_at, t0)
    except UnicodeDecodeError:
        return _fail(req, ToolFailureClass.TOOL_PARSE_ERROR, "File is not UTF-8 text", started_at, t0)
    except Exception as e:
        return _fail(req, ToolFailureClass.TOOL_INTERNAL_ERROR, f"Read failed: {e}", started_at, t0)

    sha = hashlib.sha256(data).hexdigest()
    prov = ToolProvenance(
        sources=[user_path],
        retrieved_at=_iso_now(),
        notes=f"local.read_file resolved={resolved} bytes={size} sha256={sha}",
    )

    return ToolResult(
        ok=True,
        tool_name=req.tool_name,
        primary={
            "path": user_path,
            "resolved_path": resolved,
            "bytes": size,
            "sha256": sha,
            "content": text,
        },
        provenance=prov,
        started_at=started_at,
        ended_at=_iso_now(),
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


def _parse_line_range(args: Dict[str, Any]) -> Tuple[int, int]:
    # 1-based, inclusive line numbers
    start_line = args.get("start_line", 1)
    end_line = args.get("end_line")
    try:
        s = int(start_line)
    except Exception:
        raise ValueError("start_line must be an int")
    if end_line is None:
        raise ValueError("end_line is required")
    try:
        e = int(end_line)
    except Exception:
        raise ValueError("end_line must be an int")
    if s < 1 or e < 1:
        raise ValueError("start_line and end_line must be >= 1")
    if e < s:
        raise ValueError("end_line must be >= start_line")
    if (e - s + 1) > MAX_SNIPPET_LINES:
        raise ValueError(f"snippet too large (max {MAX_SNIPPET_LINES} lines)")
    return s, e


def _util_read_snippet(req: ToolRequest, started_at: str, t0: float) -> ToolResult:
    args = req.args or {}
    user_path = str(args.get("path") or "").strip()
    if not user_path:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "local.read_snippet requires path", started_at, t0)

    try:
        start_line, end_line = _parse_line_range(args)
    except ValueError as e:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, str(e), started_at, t0)

    if not _is_path_allowed(user_path):
        return _fail(req, ToolFailureClass.TOOL_NOT_ALLOWED, "Path not allowlisted", started_at, t0)

    resolved = resolve_target_path(user_path, must_exist=True)
    if not resolved:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "File not found (or not resolvable)", started_at, t0)

    # Streaming read: do NOT load entire file; enforce snippet byte cap.
    snippet_lines: List[str] = []
    total_lines: int = 0
    snippet_bytes: int = 0

    try:
        sha = hashlib.sha256()
        with open(resolved, "rb") as fb:
            for raw in fb:
                total_lines += 1
                sha.update(raw)

                if total_lines < start_line:
                    continue
                if total_lines > end_line:
                    break

                if b"\x00" in raw:
                    return _fail(req, ToolFailureClass.TOOL_PARSE_ERROR, "Binary file (NUL byte) not allowed", started_at, t0)

                snippet_bytes += len(raw)
                if snippet_bytes > MAX_SNIPPET_BYTES:
                    return _fail(
                        req,
                        ToolFailureClass.TOOL_BAD_INPUT,
                        f"snippet exceeds cap ({snippet_bytes} bytes > {MAX_SNIPPET_BYTES} bytes)",
                        started_at,
                        t0,
                    )

                try:
                    snippet_lines.append(raw.decode("utf-8").rstrip("\n"))
                except UnicodeDecodeError:
                    return _fail(req, ToolFailureClass.TOOL_PARSE_ERROR, "File is not UTF-8 text", started_at, t0)

        if start_line > total_lines:
            return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, f"start_line out of range (file has {total_lines} lines)", started_at, t0)
        if end_line > total_lines:
            return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, f"end_line out of range (file has {total_lines} lines)", started_at, t0)

        file_sha256 = sha.hexdigest()

    except FileNotFoundError:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "File not found (or not resolvable)", started_at, t0)
    except Exception as e:
        return _fail(req, ToolFailureClass.TOOL_INTERNAL_ERROR, f"Read failed: {e}", started_at, t0)

    prov = ToolProvenance(
        sources=[user_path],
        retrieved_at=_iso_now(),
        notes=f"local.read_snippet resolved={resolved} lines={start_line}-{end_line} total_lines={total_lines} snippet_bytes={snippet_bytes} sha256={file_sha256}",
    )

    return ToolResult(
        ok=True,
        tool_name=req.tool_name,
        primary={
            "path": user_path,
            "resolved_path": resolved,
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": total_lines,
            "snippet_bytes": snippet_bytes,
            "sha256": file_sha256,
            "lines": snippet_lines,
        },
        provenance=prov,
        started_at=started_at,
        ended_at=_iso_now(),
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


async def run_local_read_tool(req: ToolRequest) -> ToolResult:
    """Dispatch for LOCAL_READ tools."""
    t0 = time.monotonic()
    started_at = _iso_now()

    try:
        if req.tool_name == "local.read_file":
            return _util_read_file(req, started_at, t0)
        if req.tool_name == "local.read_snippet":
            return _util_read_snippet(req, started_at, t0)

        return _fail(req, ToolFailureClass.TOOL_NOT_ALLOWED, f"Unsupported local_read tool: {req.tool_name}", started_at, t0)
    except Exception as e:
        return _fail(req, ToolFailureClass.TOOL_INTERNAL_ERROR, f"Internal error: {e}", started_at, t0)
