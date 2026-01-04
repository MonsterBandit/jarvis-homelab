from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException

router = APIRouter(prefix="/alice/memory", tags=["alice-memory"])


def _db_path() -> str:
    """
    In-container default is /app/data/jarvis_brain.db (volume-mounted from /opt/jarvis/brain-data).
    Allow override via JARVIS_BRAIN_DB_PATH for flexibility.
    """
    return os.getenv("JARVIS_BRAIN_DB_PATH", "/app/data/jarvis_brain.db")


def _connect() -> sqlite3.Connection:
    path = _db_path()
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail=f"DB not found at {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _require_reader(
    x_isac_admin_key: Optional[str],
    x_isac_readonly_key: Optional[str],
) -> Dict[str, str]:
    """
    Reader authorization:
      - Admin: X-ISAC-ADMIN-KEY must match ISAC_ADMIN_API_KEY
      - Read-only (secondary): X-ISAC-READONLY-KEY must match ISAC_READONLY_API_KEY

    Notes:
      - Read-only key is separate by design to preserve authority boundaries.
      - If ISAC_READONLY_API_KEY is unset, read-only access is effectively disabled (admin still works).
    """
    admin_expected = os.getenv("ISAC_ADMIN_API_KEY", "").strip()
    ro_expected = os.getenv("ISAC_READONLY_API_KEY", "").strip()

    if admin_expected and x_isac_admin_key and x_isac_admin_key.strip() == admin_expected:
        return {"role": "admin"}

    if ro_expected and x_isac_readonly_key and x_isac_readonly_key.strip() == ro_expected:
        return {"role": "read_only"}

    raise HTTPException(status_code=401, detail="Unauthorized")


def _rows_to_dicts(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


@router.get("/concepts")
def get_memory_concepts(
    x_isac_admin_key: Optional[str] = Header(default=None),
    x_isac_readonly_key: Optional[str] = Header(default=None),
):
    _require_reader(x_isac_admin_key, x_isac_readonly_key)

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
              id, user_id, concept_key, concept_value, confidence, source, created_at, updated_at
            FROM memory_concepts
            ORDER BY updated_at DESC, created_at DESC
            """
        ).fetchall()
    return {"items": _rows_to_dicts(rows)}


@router.get("/aliases")
def get_memory_aliases(
    x_isac_admin_key: Optional[str] = Header(default=None),
    x_isac_readonly_key: Optional[str] = Header(default=None),
):
    _require_reader(x_isac_admin_key, x_isac_readonly_key)

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
              id, user_id, canonical, alias, scope, confidence, source, created_at, updated_at
            FROM memory_aliases
            ORDER BY updated_at DESC, created_at DESC
            """
        ).fetchall()
    return {"items": _rows_to_dicts(rows)}


@router.get("/preferences")
def get_memory_preferences(
    x_isac_admin_key: Optional[str] = Header(default=None),
    x_isac_readonly_key: Optional[str] = Header(default=None),
):
    _require_reader(x_isac_admin_key, x_isac_readonly_key)

    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
              id, user_id, pref_key, pref_value, confidence, source, created_at, updated_at
            FROM memory_preferences
            ORDER BY updated_at DESC, created_at DESC
            """
        ).fetchall()
    return {"items": _rows_to_dicts(rows)}


@router.get("/_debug/headers")
def debug_headers(
    x_isac_admin_key: Optional[str] = Header(default=None),
    x_isac_readonly_key: Optional[str] = Header(default=None),
):
    return {
        "x_isac_admin_key": x_isac_admin_key,
        "x_isac_readonly_key": x_isac_readonly_key,
    }
