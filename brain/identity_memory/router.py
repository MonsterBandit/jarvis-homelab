from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from . import db as idb


# Public routers (read-gated by main.py include_router dependencies)
identity_router = APIRouter(prefix="/identity", tags=["identity"])
memory_router = APIRouter(prefix="/memory", tags=["memory"])

# Admin routers (admin-gated by main.py include_router dependencies)
admin_identity_router = APIRouter(prefix="/identity", tags=["identity-admin"])
admin_memory_router = APIRouter(prefix="/memory", tags=["memory-admin"])
admin_inbox_router = APIRouter(prefix="/admin", tags=["admin-inbox"])



# ---------------------------------------------------------
# Bundle 4 (Sandbox & Exploratory Reasoning) — Phase 1
# Mechanical guardrails: explicit sandbox trigger + hard blocks
# Trigger: header 'X-ISAC-SANDBOX: true|1|yes'
# ---------------------------------------------------------

def _is_sandbox_request(request: Request) -> bool:
    try:
        v = (request.headers.get("X-ISAC-SANDBOX") or request.headers.get("x-isac-sandbox") or "").strip()
        return v.lower() in {"1", "true", "yes"}
    except Exception:
        return False

def _sandbox_boundary_http(surface: str) -> None:
    raise HTTPException(
        status_code=403,
        detail={
            "ok": False,
            "error": "SANDBOX_BOUNDARY",
            "blocked": True,
            "blocked_surface": surface,
            "message": "Sandbox mode forbids tools, observation, execution, and memory access.",
            "next_allowed": ["exit_sandbox", "discard_sandbox", "summarize_sandbox"],
        },
    )

class UserResolveResponse(BaseModel):
    user_id: str
    status: str
    created: bool
    expires_at: Optional[str] = None
    last_seen_at: str


class MemoryWriteRequest(BaseModel):
    memory_key: str = Field(..., min_length=1, max_length=200)
    memory_value: str = Field(..., min_length=1, max_length=4000)


class MemoryWriteResponse(BaseModel):
    id: int
    action: str
    memory_key: str
    memory_value: str
    tier: int


class MemoryDeleteRequest(BaseModel):
    memory_key: str = Field(..., min_length=1, max_length=200)


def _clean_user_id(v: Optional[str]) -> str:
    u = (v or "").strip()
    if not u:
        raise HTTPException(status_code=400, detail="X-ISAC-USER-ID required")
    if len(u) > 80:
        raise HTTPException(status_code=400, detail="user_id too long")
    return u


def _db_path_from_env(request: Request) -> str:
    # main.py uses DB_PATH env var; we read the same from app state when set by main.py
    p = getattr(request.app.state, "db_path", None)
    if not p:
        # Fallback: if state not set, use the same default main.py uses.
        p = "/app/data/jarvis_brain.db"
    return str(p)


def _with_conn(request: Request):
    db_path = _db_path_from_env(request)
    conn = idb.connect(db_path)
    return conn


@identity_router.post("/users/resolve", response_model=UserResolveResponse)
def resolve_user(
    request: Request,
    x_isac_user_id: Optional[str] = Header(default=None, alias="X-ISAC-USER-ID"),
) -> UserResolveResponse:
    user_id = _clean_user_id(x_isac_user_id)
    conn = _with_conn(request)
    try:
        idb.opportunistic_sweep(conn)
        u = idb.ensure_user(conn, user_id)
        idb.reinforce_user(conn, user_id)  # reinforcement on activity (extends expiry)
        conn.commit()
        return UserResolveResponse(
            user_id=u["user_id"],
            status=u["status"],
            created=bool(u.get("created")),
            expires_at=u.get("expires_at"),
            last_seen_at=u["last_seen_at"],
        )
    finally:
        conn.close()


@identity_router.get("/users/me")
def me(
    request: Request,
    x_isac_user_id: Optional[str] = Header(default=None, alias="X-ISAC-USER-ID"),
) -> Dict[str, Any]:
    user_id = _clean_user_id(x_isac_user_id)
    conn = _with_conn(request)
    try:
        idb.opportunistic_sweep(conn)
        idb.ensure_user(conn, user_id)
        idb.reinforce_user(conn, user_id)
        u = idb.get_user(conn, user_id)
        conn.commit()
        return {"status": "ok", "user": u}
    finally:
        conn.close()


@memory_router.get("/show")
def show_memory(
    request: Request,
    tier: Optional[int] = Query(default=None, description="Optional: 1 or 2"),
    x_isac_user_id: Optional[str] = Header(default=None, alias="X-ISAC-USER-ID"),
) -> Dict[str, Any]:
    if _is_sandbox_request(request):
        _sandbox_boundary_http("memory.show")

    user_id = _clean_user_id(x_isac_user_id)
    conn = _with_conn(request)
    try:
        idb.opportunistic_sweep(conn)
        idb.ensure_user(conn, user_id)
        idb.reinforce_user(conn, user_id)
        items = idb.list_memory(conn, user_id, tier=tier)
        conn.commit()
        return {"status": "ok", "user_id": user_id, "count": len(items), "items": items}
    finally:
        conn.close()


@memory_router.post("/write", response_model=MemoryWriteResponse)
def write_memory_tier1(
    request: Request,
    body: MemoryWriteRequest,
    x_isac_user_id: Optional[str] = Header(default=None, alias="X-ISAC-USER-ID"),
) -> MemoryWriteResponse:
    if _is_sandbox_request(request):
        _sandbox_boundary_http("memory.write")

    """Tier 1 write: safe self-profile, explicit confirmation is UI-mediated.
    Admin is NOT required for Tier 1.
    """
    user_id = _clean_user_id(x_isac_user_id)
    key = (body.memory_key or "").strip()
    val = (body.memory_value or "").strip()
    if not key or not val:
        raise HTTPException(status_code=400, detail="memory_key and memory_value required")

    conn = _with_conn(request)
    try:
        idb.opportunistic_sweep(conn)
        idb.ensure_user(conn, user_id)
        idb.reinforce_user(conn, user_id)
        r = idb.upsert_memory(conn, user_id, key, val, tier=1, actor="user")
        conn.commit()
        return MemoryWriteResponse(**r)
    finally:
        conn.close()


@memory_router.post("/forget")
def forget_memory_tier1(
    request: Request,
    body: MemoryDeleteRequest,
    x_isac_user_id: Optional[str] = Header(default=None, alias="X-ISAC-USER-ID"),
) -> Dict[str, Any]:
    if _is_sandbox_request(request):
        _sandbox_boundary_http("memory.forget")

    user_id = _clean_user_id(x_isac_user_id)
    key = (body.memory_key or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="memory_key required")

    conn = _with_conn(request)
    try:
        idb.opportunistic_sweep(conn)
        idb.ensure_user(conn, user_id)
        idb.reinforce_user(conn, user_id)
        r = idb.delete_memory(conn, user_id, key, actor="user")
        conn.commit()
        return {"status": "ok", **r}
    finally:
        conn.close()


# ---------------- Admin endpoints ----------------

@admin_identity_router.post("/users/{user_id}/promote")
def promote_user_admin(request: Request, user_id: str) -> Dict[str, Any]:
    uid = _clean_user_id(user_id)
    conn = _with_conn(request)
    try:
        idb.opportunistic_sweep(conn)
        idb.ensure_user(conn, uid)
        idb.promote_user(conn, uid)
        conn.commit()
        return {"status": "ok", "user_id": uid, "promoted": True}
    finally:
        conn.close()


class AdminMemoryWriteRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=80)
    memory_key: str = Field(..., min_length=1, max_length=200)
    memory_value: str = Field(..., min_length=1, max_length=4000)
    memory_tier: int = Field(..., ge=2, le=2)


@admin_memory_router.post("/write", response_model=MemoryWriteResponse)
def write_memory_admin(request: Request, body: AdminMemoryWriteRequest) -> MemoryWriteResponse:
    if _is_sandbox_request(request):
        _sandbox_boundary_http("memory.write_admin")

    uid = _clean_user_id(body.user_id)
    key = (body.memory_key or "").strip()
    val = (body.memory_value or "").strip()
    if not key or not val:
        raise HTTPException(status_code=400, detail="memory_key and memory_value required")

    conn = _with_conn(request)
    try:
        idb.opportunistic_sweep(conn)
        idb.ensure_user(conn, uid)
        r = idb.upsert_memory(conn, uid, key, val, tier=2, actor="admin")
        conn.commit()
        return MemoryWriteResponse(**r)
    finally:
        conn.close()


@admin_inbox_router.get("/inbox")
def inbox(request: Request, limit: int = Query(50, ge=1, le=200), include_ack: bool = Query(False)) -> Dict[str, Any]:
    conn = _with_conn(request)
    try:
        idb.opportunistic_sweep(conn)
        items = idb.admin_inbox_list(conn, limit=limit, include_ack=bool(include_ack))
        conn.commit()
        return {"status": "ok", "count": len(items), "items": items}
    finally:
        conn.close()


@admin_inbox_router.post("/inbox/{inbox_id}/ack")
def inbox_ack(request: Request, inbox_id: int) -> Dict[str, Any]:
    conn = _with_conn(request)
    try:
        ok = idb.admin_inbox_ack(conn, inbox_id)
        conn.commit()
        return {"status": "ok", "acknowledged": bool(ok), "id": int(inbox_id)}
    finally:
        conn.close()


@admin_inbox_router.post("/sweep/run")
def sweep_run(request: Request) -> Dict[str, Any]:
    conn = _with_conn(request)
    try:
        res = idb.opportunistic_sweep(conn, min_interval_seconds=0)
        conn.commit()
        return {"status": "ok", "result": res}
    finally:
        conn.close()
