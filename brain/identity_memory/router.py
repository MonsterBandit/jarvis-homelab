from __future__ import annotations

"""
identity_memory/router.py

Identity & Memory v1 — minimal API surface (LOCK-READY).

LAP defaults:
- Identity & Memory is INACTIVE unless explicitly activated later.
- All endpoints below are present for structure + guard verification, but fail-closed in LAP.
- No background jobs are scheduled.
"""

import os
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, Request

from . import db as idb


def _db_path_from_request(request: Request) -> str:
    # main.py sets app.state.db_path = DB_PATH; fall back to env or default.
    p = getattr(getattr(request, "app", None), "state", None)
    candidate = getattr(p, "db_path", None) if p else None
    return str(candidate or os.getenv("JARVIS_DB_PATH") or os.getenv("JARVIS_BRAIN_DB_PATH") or "/app/data/jarvis_brain.db")


# ----------------------------
# LAP guard (fail-closed)
# ----------------------------

def _is_truthy(v: Optional[str]) -> bool:
    return bool(v and str(v).strip().lower() in {"1", "true", "yes", "on"})

def _require_identity_memory_active() -> None:
    # Default: inactive unless explicitly enabled later.
    if not _is_truthy(os.getenv("ISAC_IDENTITY_MEMORY_ACTIVE")):
        raise HTTPException(
            status_code=403,
            detail={
                "ok": False,
                "error": "IDENTITY_MEMORY_INACTIVE",
                "blocked": True,
                "phase": "LAP",
                "message": "Identity & Memory is not active during the Last Administrative Phase (LAP).",
                "next_allowed": ["continue_lap_implementation", "activate_identity_memory_later_with_admin"],
            },
        )


# Routers (names used by main.py includes)
identity_router = APIRouter(prefix="/identity", tags=["identity"])
admin_identity_router = APIRouter(prefix="/identity", tags=["identity-admin"])

memory_router = APIRouter(prefix="/memory", tags=["memory"])
admin_memory_router = APIRouter(prefix="/memory", tags=["memory-admin"])

admin_inbox_router = APIRouter(prefix="/admin", tags=["admin-inbox"])
audit_router = APIRouter(prefix="/audit", tags=["audit"])


# ----------------------------
# Identity APIs
# ----------------------------

@identity_router.get("/me")
def identity_me(
    request: Request,
    x_alice_session_id: Optional[str] = Header(default=None, alias="X-ALICE-SESSION-ID"),
) -> Dict[str, Any]:
    _require_identity_memory_active()

    sid = (x_alice_session_id or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="X-ALICE-SESSION-ID header is required")

    db_path = _db_path_from_request(request)
    conn = idb.connect(db_path)
    try:
        user = idb.resolve_or_create_user_for_session(conn, session_id=sid)
        conn.commit()
        return {
            "user_id": user["user_id"],
            "lifecycle_state": user["lifecycle_state"],
            "expiry_at": user.get("expiry_at"),
            "reinforced_at": user.get("reinforced_at"),
            "created": bool(user.get("created")),
        }
    finally:
        conn.close()


@admin_identity_router.get("/users/{user_id}")
def get_user_admin(
    request: Request,
    user_id: int,
) -> Dict[str, Any]:
    _require_identity_memory_active()

    conn = idb.connect(_db_path_from_request(request))
    try:
        u = idb.get_user_admin(conn, int(user_id))
        if not u:
            raise HTTPException(status_code=404, detail="User not found")
        return u
    finally:
        conn.close()


@admin_identity_router.post("/users/{user_id}/promote")
def promote_user(
    request: Request,
    user_id: int,
) -> Dict[str, Any]:
    _require_identity_memory_active()

    conn = idb.connect(_db_path_from_request(request))
    try:
        idb.promote_user_to_permanent(conn, int(user_id))
        conn.commit()
        return {"ok": True, "user_id": int(user_id), "lifecycle_state": "permanent"}
    finally:
        conn.close()


# ----------------------------
# Memory APIs
# ----------------------------

@memory_router.get("")
def list_memory(
    request: Request,
    tier: Optional[int] = Query(default=None),
    # user_id is admin-only in the locked API. Enforced by routing: only admin router accepts it.
) -> Dict[str, Any]:
    _require_identity_memory_active()

    # Normal user view: derive user from session id.
    sid = (request.headers.get("X-ALICE-SESSION-ID") or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="X-ALICE-SESSION-ID header is required")

    conn = idb.connect(_db_path_from_request(request))
    try:
        u = idb.resolve_or_create_user_for_session(conn, session_id=sid)
        items = idb.list_memory_entries(conn, user_id=int(u["user_id"]), tier=(int(tier) if tier is not None else None))
        return {"items": items, "meta": {"count": len(items), "tier": tier}}
    finally:
        conn.close()


@admin_memory_router.get("")
def list_memory_admin(
    request: Request,
    tier: Optional[int] = Query(default=None),
    user_id: Optional[int] = Query(default=None),
) -> Dict[str, Any]:
    _require_identity_memory_active()

    if user_id is None:
        raise HTTPException(status_code=400, detail="user_id is required for admin memory listing")

    conn = idb.connect(_db_path_from_request(request))
    try:
        items = idb.list_memory_entries(conn, user_id=int(user_id), tier=(int(tier) if tier is not None else None))
        return {"items": items, "meta": {"count": len(items), "tier": tier, "user_id": int(user_id)}}
    finally:
        conn.close()


@memory_router.post("/proposals")
def propose_memory_write(
    request: Request,
    body: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    _require_identity_memory_active()

    # Payload per locked API: user_id, tier, type, value, reason
    tier = int(body.get("tier") or 0)
    if tier not in (1, 2):
        raise HTTPException(status_code=400, detail="tier must be 1 or 2")

    type_ = str(body.get("type") or "").strip()
    value = body.get("value")
    reason = str(body.get("reason") or "").strip()

    if not type_:
        raise HTTPException(status_code=400, detail="type is required")
    if value is None:
        raise HTTPException(status_code=400, detail="value is required")
    if not reason:
        raise HTTPException(status_code=400, detail="reason is required")

    # Determine target user:
    # - If caller provided user_id, use it only on admin router (not here). For user router, derive from session.
    sid = (request.headers.get("X-ALICE-SESSION-ID") or "").strip()
    if not sid:
        raise HTTPException(status_code=400, detail="X-ALICE-SESSION-ID header is required")

    conn = idb.connect(_db_path_from_request(request))
    try:
        u = idb.resolve_or_create_user_for_session(conn, session_id=sid)
        requires_admin = (tier == 2)
        value_json = value if isinstance(value, dict) else {"value": value}
        value_text = str(value) if not isinstance(value, str) else value
        proposal = idb.create_memory_proposal(
            conn,
            user_id=int(u["user_id"]),
            tier=tier,
            type_=type_,
            value_json=value_json,
            value_text=value_text,
            reason=reason,
            requires_admin=requires_admin,
        )
        conn.commit()
        return proposal
    finally:
        conn.close()


@memory_router.post("/proposals/{proposal_id}/confirm")
def confirm_proposal(
    request: Request,
    proposal_id: str,
    body: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    _require_identity_memory_active()

    if body.get("confirm") is not True:
        raise HTTPException(status_code=400, detail="confirm=true is required")

    # Determine actor type:
    # - Tier 1 confirmations are user
    # - Tier 2 confirmations must be admin, enforced by mounting this endpoint also under admin router in main.py if desired.
    conn = idb.connect(_db_path_from_request(request))
    try:
        prop = idb.get_proposal(conn, proposal_id)
        if not prop:
            raise HTTPException(status_code=404, detail="Proposal not found")

        # Expiry check
        try:
            exp = prop["expires_at"]
            if exp and exp <= idb.utc_now_iso():
                idb.mark_proposal_status(conn, proposal_id, status="expired")
                conn.commit()
                raise HTTPException(status_code=409, detail="Proposal expired")
        except HTTPException:
            raise
        except Exception:
            pass

        if prop["status"] != "pending":
            raise HTTPException(status_code=409, detail=f"Proposal is not pending (status={prop['status']})")

        tier = int(prop["tier"])
        created_by = "user" if tier == 1 else "admin"

        # If Tier 2, require admin router in practice. Here we hard-enforce by checking for X-ISAC-ADMIN-TOKEN presence,
        # but do not validate it here (validation happens in main.py require_admin_if_configured dependency).
        if tier == 2:
            admin_token = (request.headers.get("X-ISAC-ADMIN-TOKEN") or "").strip()
            admin_key = (request.headers.get("X-ISAC-ADMIN-KEY") or "").strip()
            if not (admin_token or admin_key):
                raise HTTPException(status_code=401, detail="Admin confirmation required for Tier 2")

        mid = idb.create_memory_entry_from_proposal(conn, proposal=prop, created_by=created_by)
        idb.mark_proposal_status(conn, proposal_id, status="confirmed", ts_field="confirmed_at")
        conn.commit()
        return {"ok": True, "memory_id": mid, "proposal_id": proposal_id}
    finally:
        conn.close()


@memory_router.post("/proposals/{proposal_id}/reject")
def reject_proposal(
    request: Request,
    proposal_id: str,
) -> Dict[str, Any]:
    _require_identity_memory_active()

    conn = idb.connect(_db_path_from_request(request))
    try:
        prop = idb.get_proposal(conn, proposal_id)
        if not prop:
            raise HTTPException(status_code=404, detail="Proposal not found")
        if prop["status"] != "pending":
            raise HTTPException(status_code=409, detail=f"Proposal is not pending (status={prop['status']})")
        idb.mark_proposal_status(conn, proposal_id, status="rejected", ts_field="rejected_at")
        conn.commit()
        return {"ok": True, "proposal_id": proposal_id, "status": "rejected"}
    finally:
        conn.close()


@memory_router.delete("/{memory_id}")
def forget_memory(
    request: Request,
    memory_id: int,
) -> Dict[str, Any]:
    _require_identity_memory_active()

    # Tier enforcement:
    # - Tier 1: user can delete
    # - Tier 2: admin required
    conn = idb.connect(_db_path_from_request(request))
    try:
        row = conn.execute(
            "SELECT tier FROM memory_entries WHERE id = ? AND is_deleted = 0",
            (int(memory_id),),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Memory not found")

        tier = int(row["tier"])
        if tier == 2:
            admin_token = (request.headers.get("X-ISAC-ADMIN-TOKEN") or "").strip()
            admin_key = (request.headers.get("X-ISAC-ADMIN-KEY") or "").strip()
            if not (admin_token or admin_key):
                raise HTTPException(status_code=401, detail="Admin required to delete Tier 2 memory")

        ok = idb.delete_memory_entry(conn, memory_id=int(memory_id), actor=("admin" if tier == 2 else "user"), actor_user_id=None)
        conn.commit()
        return {"ok": True, "deleted": bool(ok), "memory_id": int(memory_id)}
    finally:
        conn.close()


# ----------------------------
# Admin Inbox
# ----------------------------

@admin_inbox_router.get("/inbox")
def list_inbox(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    include_acked: bool = Query(default=False),
) -> Dict[str, Any]:
    _require_identity_memory_active()

    conn = idb.connect(_db_path_from_request(request))
    try:
        items = idb.list_admin_inbox(conn, limit=int(limit), include_acked=bool(include_acked))
        return {"items": items, "meta": {"count": len(items), "limit": int(limit), "include_acked": bool(include_acked)}}
    finally:
        conn.close()


@admin_inbox_router.post("/inbox/{event_id}/ack")
def ack_inbox(
    request: Request,
    event_id: str,
) -> Dict[str, Any]:
    _require_identity_memory_active()

    conn = idb.connect(_db_path_from_request(request))
    try:
        ok = idb.ack_admin_inbox_event(conn, event_id=event_id, acked_by_user_id=None)
        conn.commit()
        return {"ok": True, "acked": bool(ok), "event_id": event_id}
    finally:
        conn.close()


# ----------------------------
# Audit (read-only, admin-gated by main.py dependencies)
# ----------------------------

@audit_router.get("/identity")
def audit_identity(request: Request, limit: int = Query(default=200, ge=1, le=500)) -> Dict[str, Any]:
    _require_identity_memory_active()
    conn = idb.connect(_db_path_from_request(request))
    try:
        items = idb.list_audit(conn, domain="identity", limit=int(limit))
        return {"items": items, "meta": {"count": len(items), "limit": int(limit)}}
    finally:
        conn.close()

@audit_router.get("/memory")
def audit_memory(request: Request, limit: int = Query(default=200, ge=1, le=500)) -> Dict[str, Any]:
    _require_identity_memory_active()
    conn = idb.connect(_db_path_from_request(request))
    try:
        items = idb.list_audit(conn, domain="memory", limit=int(limit))
        return {"items": items, "meta": {"count": len(items), "limit": int(limit)}}
    finally:
        conn.close()