"""
/opt/jarvis/brain/alice/memory_api.py

Alice Memory API (read + admin writes) for the SQLite-backed memory system.

Authoritative facts (do not change here):
- SQLite DB: /app/data/jarvis_brain.db (volume-mapped from /opt/jarvis/brain-data/jarvis_brain.db)
- memory_concepts primary key: concept_id
- memory_aliases primary key: alias_id
- memory_aliases includes: concept_id, user_id, preferred_name, pattern_notes, confidence, locked, created_at, updated_at
- memory_aliases has UNIQUE(concept_id, user_id)

Design goals:
- No dependency on alice.boundaries
- Uses ISAC_ADMIN_API_KEY / ISAC_READONLY_API_KEY for auth gating
- No schema drift: introspect table columns; never assume "id"
- Read endpoints:
    GET  /alice/memory/health
    GET  /alice/memory/concepts
    GET  /alice/memory/aliases
    GET  /alice/memory/aliases/by-concept/{concept_id}
    GET  /alice/memory/aliases/by-user/{user_id}
- Admin write endpoints:
    POST /alice/memory/concepts
    POST /alice/memory/aliases

Notes:
- Write endpoints accept arbitrary JSON and insert only columns that exist.
- Unknown keys are ignored (never invent columns). If you want strict mode later, we can add it.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fastapi import APIRouter, Body, Header, HTTPException

router = APIRouter(prefix="/alice/memory", tags=["Alice Memory"])


# ---------------------------------------------------------------------
# DB / schema helpers
# ---------------------------------------------------------------------

def _db_path() -> str:
    # Container path (compose maps /opt/jarvis/brain-data -> /app/data)
    return (
        os.getenv("JARVIS_BRAIN_DB_PATH")
        or os.getenv("JARVIS_DB_PATH")
        or "/app/data/jarvis_brain.db"
    )


@contextmanager
def _connect():
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _env_key(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    return v if v else None


def _pragma_table_info(conn: sqlite3.Connection, table: str) -> List[sqlite3.Row]:
    return conn.execute(f"PRAGMA table_info({table})").fetchall()


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    info = _pragma_table_info(conn, table)
    return [r["name"] for r in info] if info else []


def _primary_key_column(conn: sqlite3.Connection, table: str) -> Optional[str]:
    """
    Returns the single PK column name if present (pk==1).
    If composite PK exists, this returns the first pk column.
    """
    info = _pragma_table_info(conn, table)
    for r in info:
        if int(r["pk"] or 0) == 1:
            return str(r["name"])
    return None


def _best_order_by(cols: Sequence[str]) -> str:
    # Favor updated_at then created_at if present; else rowid.
    if "updated_at" in cols and "created_at" in cols:
        return " ORDER BY updated_at DESC, created_at DESC"
    if "updated_at" in cols:
        return " ORDER BY updated_at DESC"
    if "created_at" in cols:
        return " ORDER BY created_at DESC"
    return " ORDER BY rowid DESC"


def _select_rows(
    conn: sqlite3.Connection,
    table: str,
    preferred_cols: Sequence[str],
    limit: int,
) -> List[Dict[str, Any]]:
    cols = _table_columns(conn, table)
    if not cols:
        return []

    selected = [c for c in preferred_cols if c in cols]
    if not selected:
        selected = cols

    sql = f"SELECT {', '.join(selected)} FROM {table}{_best_order_by(cols)} LIMIT ?"
    rows = conn.execute(sql, (limit,)).fetchall()
    return [dict(r) for r in rows]


def _fetch_one_by_pk(
    conn: sqlite3.Connection,
    table: str,
    pk_col: str,
    pk_val: Any,
) -> Optional[Dict[str, Any]]:
    cols = _table_columns(conn, table)
    if not cols:
        return None
    sql = f"SELECT {', '.join(cols)} FROM {table} WHERE {pk_col} = ? LIMIT 1"
    row = conn.execute(sql, (pk_val,)).fetchone()
    return dict(row) if row else None


def _insert_schema_safe(
    conn: sqlite3.Connection,
    table: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Insert a row using only keys that are real columns in the table.
    Unknown keys are ignored (never invent columns).
    Returns the inserted row by PK if possible, else a minimal ack.
    """
    cols = _table_columns(conn, table)
    if not cols:
        raise HTTPException(status_code=500, detail=f"Table not found or has no columns: {table}")

    insert_cols = [k for k in payload.keys() if k in cols]
    if not insert_cols:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "No insertable fields in payload for table",
                "table": table,
                "known_columns": cols,
            },
        )

    placeholders = ", ".join(["?"] * len(insert_cols))
    col_list = ", ".join(insert_cols)
    values = [payload[c] for c in insert_cols]

    cur = conn.execute(f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})", values)
    conn.commit()

    pk_col = _primary_key_column(conn, table)
    if pk_col:
        # Normal case: INTEGER PRIMARY KEY AUTOINCREMENT => lastrowid
        pk_val = cur.lastrowid
        # If the caller provided the PK explicitly and it was used, prefer that.
        if pk_col in payload and payload.get(pk_col) is not None and pk_col in insert_cols:
            pk_val = payload[pk_col]
        row = _fetch_one_by_pk(conn, table, pk_col, pk_val)
        if row is not None:
            return row

    return {
        "inserted": True,
        "table": table,
        "lastrowid": cur.lastrowid,
        "accepted_columns": insert_cols,
    }


def _select_aliases_filtered(
    conn: sqlite3.Connection,
    *,
    concept_id: Optional[int] = None,
    user_id: Optional[str] = None,
    limit: int = 200,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Read-only helper for memory_aliases with optional filters.
    Uses only real columns and a sane ORDER BY.
    Returns (items, count).
    """
    table = "memory_aliases"
    cols = _table_columns(conn, table)
    if not cols:
        return ([], 0)

    preferred = [
        "alias_id",
        "concept_id",
        "user_id",
        "preferred_name",
        "pattern_notes",
        "confidence",
        "locked",
        "created_at",
        "updated_at",
    ]
    selected = [c for c in preferred if c in cols] or cols

    where = []
    params: List[Any] = []

    if concept_id is not None and "concept_id" in cols:
        where.append("concept_id = ?")
        params.append(concept_id)

    if user_id is not None and "user_id" in cols:
        where.append("user_id = ?")
        params.append(user_id)

    where_sql = f" WHERE {' AND '.join(where)}" if where else ""
    sql = f"SELECT {', '.join(selected)} FROM {table}{where_sql}{_best_order_by(cols)} LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, tuple(params)).fetchall()
    items = [dict(r) for r in rows]
    return (items, len(items))


# ---------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------

def _require_reader(x_isac_admin_key: Optional[str], x_isac_readonly_key: Optional[str]) -> None:
    admin = _env_key("ISAC_ADMIN_API_KEY")
    readonly = _env_key("ISAC_READONLY_API_KEY")

    provided_admin = (x_isac_admin_key or "").strip() or None
    provided_ro = (x_isac_readonly_key or "").strip() or None

    ok = False
    if admin and provided_admin == admin:
        ok = True
    if readonly and provided_ro == readonly:
        ok = True
    # Convenience: allow admin key in readonly header too
    if admin and provided_ro == admin:
        ok = True
    if readonly and provided_admin == readonly:
        ok = True

    if not ok:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _require_admin(x_isac_admin_key: Optional[str], x_isac_readonly_key: Optional[str]) -> None:
    admin = _env_key("ISAC_ADMIN_API_KEY")
    if not admin:
        raise HTTPException(status_code=500, detail="ISAC_ADMIN_API_KEY not configured")

    provided_admin = (x_isac_admin_key or "").strip() or None
    provided_ro = (x_isac_readonly_key or "").strip() or None

    if provided_admin == admin or provided_ro == admin:
        return

    raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@router.get("/health")
def health():
    """
    Minimal health check: can we open DB and see our memory tables.
    """
    with _connect() as conn:
        tables = [
            r["name"]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        ]
        return {
            "ok": True,
            "db_path": _db_path(),
            "tables_present": {
                "memory_concepts": "memory_concepts" in tables,
                "memory_aliases": "memory_aliases" in tables,
            },
        }


@router.get("/concepts")
def get_memory_concepts(
    x_isac_admin_key: Optional[str] = Header(default=None),
    x_isac_readonly_key: Optional[str] = Header(default=None),
    limit: int = 200,
):
    _require_reader(x_isac_admin_key, x_isac_readonly_key)

    preferred = [
        "concept_id",
        "user_id",
        "concept_key",
        "category",
        "default_unit",
        "notes",
        "confidence",
        "locked",
        "created_at",
        "updated_at",
    ]

    with _connect() as conn:
        items = _select_rows(conn, "memory_concepts", preferred, limit=limit)
        return {"items": items, "meta": {"limit": limit, "count": len(items)}}


@router.post("/concepts")
def create_memory_concept(
    payload: Dict[str, Any] = Body(...),
    x_isac_admin_key: Optional[str] = Header(default=None),
    x_isac_readonly_key: Optional[str] = Header(default=None),
):
    """
    Admin-only concept creation.
    Inserts only real DB columns; ignores unknown keys.
    """
    _require_admin(x_isac_admin_key, x_isac_readonly_key)

    with _connect() as conn:
        row = _insert_schema_safe(conn, "memory_concepts", payload)
        return {"item": row}


@router.get("/aliases")
def get_memory_aliases(
    x_isac_admin_key: Optional[str] = Header(default=None),
    x_isac_readonly_key: Optional[str] = Header(default=None),
    limit: int = 200,
):
    _require_reader(x_isac_admin_key, x_isac_readonly_key)

    preferred = [
        "alias_id",
        "concept_id",
        "user_id",
        "preferred_name",
        "pattern_notes",
        "confidence",
        "locked",
        "created_at",
        "updated_at",
    ]

    with _connect() as conn:
        items = _select_rows(conn, "memory_aliases", preferred, limit=limit)
        return {"items": items, "meta": {"limit": limit, "count": len(items)}}


@router.get("/aliases/by-concept/{concept_id}")
def get_aliases_by_concept(
    concept_id: int,
    x_isac_admin_key: Optional[str] = Header(default=None),
    x_isac_readonly_key: Optional[str] = Header(default=None),
    limit: int = 200,
):
    _require_reader(x_isac_admin_key, x_isac_readonly_key)

    with _connect() as conn:
        items, count = _select_aliases_filtered(conn, concept_id=concept_id, user_id=None, limit=limit)
        return {"items": items, "meta": {"limit": limit, "count": count, "concept_id": concept_id}}


@router.get("/aliases/by-user/{user_id}")
def get_aliases_by_user(
    user_id: str,
    x_isac_admin_key: Optional[str] = Header(default=None),
    x_isac_readonly_key: Optional[str] = Header(default=None),
    limit: int = 200,
):
    _require_reader(x_isac_admin_key, x_isac_readonly_key)

    with _connect() as conn:
        items, count = _select_aliases_filtered(conn, concept_id=None, user_id=user_id, limit=limit)
        return {"items": items, "meta": {"limit": limit, "count": count, "user_id": user_id}}


@router.post("/aliases")
def create_memory_alias(
    payload: Dict[str, Any] = Body(...),
    x_isac_admin_key: Optional[str] = Header(default=None),
    x_isac_readonly_key: Optional[str] = Header(default=None),
):
    """
    Admin-only alias creation.
    Inserts only real DB columns; ignores unknown keys.
    """
    _require_admin(x_isac_admin_key, x_isac_readonly_key)

    with _connect() as conn:
        row = _insert_schema_safe(conn, "memory_aliases", payload)
        return {"item": row}
