from __future__ import annotations

"""
identity_memory/db.py

Identity & Memory v1 — DB substrate (LAP implementation: STRUCTURE ONLY, INERT).

LOCKED governance inputs:
- identity_memory_design_v1.md
- identity_memory_api_surface_v1.md
- identity_memory_ui_flows_v1.md
- identity_memory_implementation_plan_v1.md

LAP semantics:
- This module may create tables (non-destructive, CREATE TABLE IF NOT EXISTS).
- This module MUST NOT schedule background jobs.
- No request-path sweepers. No opportunistic deletion.
- Activation/permission is enforced in routers (fail-closed by default).

Storage:
- SQLite at /app/data/jarvis_brain.db (volume-mapped to /opt/jarvis/brain-data/jarvis_brain.db)
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Time helpers
# ----------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _int_env(name: str, default: int) -> int:
    try:
        raw = (os.getenv(name) or "").strip()
        return int(raw) if raw else int(default)
    except Exception:
        return int(default)


# Safe defaults (implementation plan v1)
ALICE_PROVISIONAL_TTL_HOURS = _int_env("ALICE_PROVISIONAL_TTL_HOURS", 168)
ALICE_REINFORCEMENT_EXTEND_HOURS = _int_env("ALICE_REINFORCEMENT_EXTEND_HOURS", 168)
ALICE_PROPOSAL_TTL_MINUTES = _int_env("ALICE_PROPOSAL_TTL_MINUTES", 60)


# ----------------------------
# Connection + schema
# ----------------------------

def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create required Identity & Memory v1 tables if missing (non-destructive)."""
    cur = conn.cursor()

    # Users (internal canonical users)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            handle TEXT NULL,
            lifecycle_state TEXT NOT NULL CHECK (lifecycle_state IN ('provisional','reinforced','permanent')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            reinforced_at TEXT NULL,
            expiry_at TEXT NULL,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            deleted_at TEXT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_users_lifecycle_state ON users (lifecycle_state)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_users_expiry_at ON users (expiry_at)")

    # Session identity mapping (browser session -> internal user)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS session_identities (
            session_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
        """
    )

    # Memory entries (persisted only after proposal confirm)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            tier INTEGER NOT NULL CHECK (tier IN (0,1,2)),
            type TEXT NOT NULL,
            value_json TEXT NOT NULL,
            value_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            created_by TEXT NOT NULL CHECK (created_by IN ('user','admin')),
            is_deleted INTEGER NOT NULL DEFAULT 0,
            deleted_at TEXT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_user_tier ON memory_entries (user_id, tier)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries (type)")

    # Memory proposals (non-persistent until confirmed)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_proposals (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            tier INTEGER NOT NULL CHECK (tier IN (1,2)),
            type TEXT NOT NULL,
            value_json TEXT NOT NULL,
            value_text TEXT NOT NULL,
            reason TEXT NOT NULL,
            requires_admin INTEGER NOT NULL CHECK (requires_admin IN (0,1)),
            status TEXT NOT NULL CHECK (status IN ('pending','confirmed','rejected','expired')),
            created_at TEXT NOT NULL,
            confirmed_at TEXT NULL,
            rejected_at TEXT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_proposals_user_status ON memory_proposals (user_id, status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_proposals_expires_at ON memory_proposals (expires_at)")

    # Admin inbox events
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_inbox_events (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL CHECK (event_type IN ('provisional_user_created','user_reinforced','user_expiring','user_deleted')),
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            acked_at TEXT NULL,
            acked_by_user_id INTEGER NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_inbox_acked ON admin_inbox_events (acked_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_inbox_created_at ON admin_inbox_events (created_at)")

    # Audit log (immutable-ish append-only)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id TEXT PRIMARY KEY,
            domain TEXT NOT NULL CHECK (domain IN ('identity','memory','admin_inbox')),
            action TEXT NOT NULL,
            actor TEXT NOT NULL CHECK (actor IN ('system','user','admin')),
            actor_user_id INTEGER NULL,
            subject_user_id INTEGER NULL,
            target_id TEXT NULL,
            created_at TEXT NOT NULL,
            summary TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_domain_created_at ON audit_log (domain, created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_subject_created_at ON audit_log (subject_user_id, created_at)")

    conn.commit()


# ----------------------------
# Audit helpers
# ----------------------------

def audit(
    conn: sqlite3.Connection,
    *,
    domain: str,
    action: str,
    actor: str,
    actor_user_id: Optional[int],
    subject_user_id: Optional[int],
    target_id: Optional[str],
    summary: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    aid = str(uuid.uuid4())
    now = utc_now_iso()
    conn.execute(
        """
        INSERT INTO audit_log
          (id, domain, action, actor, actor_user_id, subject_user_id, target_id, created_at, summary, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            aid,
            domain,
            action,
            actor,
            actor_user_id,
            subject_user_id,
            target_id,
            now,
            summary,
            json.dumps(metadata or {}, ensure_ascii=False),
        ),
    )
    return aid


# ----------------------------
# Identity helpers (session -> user)
# ----------------------------

def resolve_or_create_user_for_session(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    ttl_hours: int = ALICE_PROVISIONAL_TTL_HOURS,
) -> Dict[str, Any]:
    """
    Resolve the internal user for a session_id. If missing, create provisional user.
    Side effects (design-allowed when activated): provisional creation + admin inbox + audit.

    NOTE: Routers enforce LAP inactive by default; this function is inert unless called.
    """
    now = utc_now_iso()

    row = conn.execute(
        """
        SELECT si.session_id, si.user_id, u.lifecycle_state, u.expiry_at, u.reinforced_at
        FROM session_identities si
        JOIN users u ON u.id = si.user_id
        WHERE si.session_id = ? AND u.is_deleted = 0
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()

    if row:
        user_id = int(row["user_id"])
        conn.execute("UPDATE session_identities SET last_seen_at = ? WHERE session_id = ?", (now, session_id))
        conn.execute("UPDATE users SET updated_at = ? WHERE id = ?", (now, user_id))
        return {
            "user_id": user_id,
            "lifecycle_state": row["lifecycle_state"],
            "expiry_at": row["expiry_at"],
            "reinforced_at": row["reinforced_at"],
            "created": False,
        }

    # Create user
    expiry = (datetime.now(timezone.utc) + timedelta(hours=int(ttl_hours))).isoformat()
    cur = conn.execute(
        """
        INSERT INTO users (handle, lifecycle_state, created_at, updated_at, reinforced_at, expiry_at, is_deleted, deleted_at)
        VALUES (NULL, 'provisional', ?, ?, NULL, ?, 0, NULL)
        """,
        (now, now, expiry),
    )
    new_user_id = int(cur.lastrowid)

    conn.execute(
        """
        INSERT INTO session_identities (session_id, user_id, created_at, last_seen_at)
        VALUES (?, ?, ?, ?)
        """,
        (session_id, new_user_id, now, now),
    )

    # Admin inbox + audit (design requires inbox notification)
    inbox_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO admin_inbox_events (id, event_type, user_id, created_at, metadata_json, acked_at, acked_by_user_id)
        VALUES (?, 'provisional_user_created', ?, ?, ?, NULL, NULL)
        """,
        (inbox_id, new_user_id, now, json.dumps({"ttl_hours": int(ttl_hours)}, ensure_ascii=False)),
    )
    audit(
        conn,
        domain="identity",
        action="user_created_provisional",
        actor="system",
        actor_user_id=None,
        subject_user_id=new_user_id,
        target_id=inbox_id,
        summary="Provisional user auto-created for new session identity.",
        metadata={"ttl_hours": int(ttl_hours)},
    )

    return {
        "user_id": new_user_id,
        "lifecycle_state": "provisional",
        "expiry_at": expiry,
        "reinforced_at": None,
        "created": True,
    }


def get_user_admin(conn: sqlite3.Connection, user_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT id, handle, lifecycle_state, created_at, updated_at, reinforced_at, expiry_at, is_deleted, deleted_at
        FROM users
        WHERE id = ?
        LIMIT 1
        """,
        (int(user_id),),
    ).fetchone()
    if not row:
        return None
    return dict(row)


def promote_user_to_permanent(conn: sqlite3.Connection, user_id: int, *, actor_user_id: Optional[int] = None) -> None:
    now = utc_now_iso()
    conn.execute(
        """
        UPDATE users
        SET lifecycle_state = 'permanent',
            expiry_at = NULL,
            updated_at = ?
        WHERE id = ? AND is_deleted = 0
        """,
        (now, int(user_id)),
    )
    audit(
        conn,
        domain="identity",
        action="user_promoted_permanent",
        actor="admin",
        actor_user_id=actor_user_id,
        subject_user_id=int(user_id),
        target_id=None,
        summary="User promoted to permanent.",
        metadata={},
    )


# ----------------------------
# Memory proposal helpers
# ----------------------------

def create_memory_proposal(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    tier: int,
    type_: str,
    value_json: Dict[str, Any],
    value_text: str,
    reason: str,
    requires_admin: bool,
) -> Dict[str, Any]:
    now = utc_now_iso()
    expires = (datetime.now(timezone.utc) + timedelta(minutes=int(ALICE_PROPOSAL_TTL_MINUTES))).isoformat()
    pid = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO memory_proposals
          (id, user_id, tier, type, value_json, value_text, reason, requires_admin, status, created_at, confirmed_at, rejected_at, expires_at)
        VALUES
          (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, NULL, NULL, ?)
        """,
        (pid, int(user_id), int(tier), type_, json.dumps(value_json, ensure_ascii=False), value_text, reason, 1 if requires_admin else 0, now, expires),
    )
    audit(
        conn,
        domain="memory",
        action="proposal_created",
        actor="system",
        actor_user_id=None,
        subject_user_id=int(user_id),
        target_id=pid,
        summary="Memory proposal created.",
        metadata={"tier": int(tier), "type": type_},
    )
    return {"proposal_id": pid, "expires_at": expires, "requires_admin": bool(requires_admin)}


def get_proposal(conn: sqlite3.Connection, proposal_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT *
        FROM memory_proposals
        WHERE id = ?
        LIMIT 1
        """,
        (proposal_id,),
    ).fetchone()


def mark_proposal_status(
    conn: sqlite3.Connection,
    proposal_id: str,
    *,
    status: str,
    ts_field: Optional[str] = None,
) -> None:
    now = utc_now_iso()
    if ts_field in {"confirmed_at", "rejected_at"}:
        conn.execute(
            f"UPDATE memory_proposals SET status = ?, {ts_field} = ? WHERE id = ?",
            (status, now, proposal_id),
        )
    else:
        conn.execute(
            "UPDATE memory_proposals SET status = ? WHERE id = ?",
            (status, proposal_id),
        )


def create_memory_entry_from_proposal(
    conn: sqlite3.Connection,
    *,
    proposal: sqlite3.Row,
    created_by: str,
) -> int:
    now = utc_now_iso()
    cur = conn.execute(
        """
        INSERT INTO memory_entries (user_id, tier, type, value_json, value_text, created_at, created_by, is_deleted, deleted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL)
        """,
        (
            int(proposal["user_id"]),
            int(proposal["tier"]),
            str(proposal["type"]),
            str(proposal["value_json"]),
            str(proposal["value_text"]),
            now,
            created_by,
        ),
    )
    mid = int(cur.lastrowid)
    audit(
        conn,
        domain="memory",
        action="memory_created",
        actor=("admin" if created_by == "admin" else "user"),
        actor_user_id=None,
        subject_user_id=int(proposal["user_id"]),
        target_id=str(mid),
        summary="Memory entry created from proposal confirmation.",
        metadata={"proposal_id": str(proposal["id"]), "tier": int(proposal["tier"]), "type": str(proposal["type"])},
    )
    return mid


def list_memory_entries(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    tier: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if tier is None:
        rows = conn.execute(
            """
            SELECT id, tier, type, value_text, created_at, created_by
            FROM memory_entries
            WHERE user_id = ? AND is_deleted = 0
            ORDER BY id DESC
            """,
            (int(user_id),),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, tier, type, value_text, created_at, created_by
            FROM memory_entries
            WHERE user_id = ? AND tier = ? AND is_deleted = 0
            ORDER BY id DESC
            """,
            (int(user_id), int(tier)),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_memory_entry(
    conn: sqlite3.Connection,
    *,
    memory_id: int,
    actor: str,
    actor_user_id: Optional[int],
) -> bool:
    now = utc_now_iso()
    cur = conn.execute(
        """
        UPDATE memory_entries
        SET is_deleted = 1, deleted_at = ?
        WHERE id = ? AND is_deleted = 0
        """,
        (now, int(memory_id)),
    )
    if cur.rowcount and cur.rowcount > 0:
        audit(
            conn,
            domain="memory",
            action="memory_deleted",
            actor=actor,
            actor_user_id=actor_user_id,
            subject_user_id=None,
            target_id=str(int(memory_id)),
            summary="Memory entry deleted (soft-delete).",
            metadata={},
        )
        return True
    return False


# ----------------------------
# Admin inbox
# ----------------------------

def list_admin_inbox(
    conn: sqlite3.Connection,
    *,
    limit: int = 50,
    include_acked: bool = False,
) -> List[Dict[str, Any]]:
    if include_acked:
        rows = conn.execute(
            """
            SELECT id, event_type, user_id, created_at, metadata_json, acked_at, acked_by_user_id
            FROM admin_inbox_events
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, event_type, user_id, created_at, metadata_json, acked_at, acked_by_user_id
            FROM admin_inbox_events
            WHERE acked_at IS NULL
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        item = dict(r)
        try:
            item["metadata"] = json.loads(item.get("metadata_json") or "{}")
        except Exception:
            item["metadata"] = item.get("metadata_json")
        out.append(item)
    return out


def ack_admin_inbox_event(
    conn: sqlite3.Connection,
    *,
    event_id: str,
    acked_by_user_id: Optional[int],
) -> bool:
    now = utc_now_iso()
    cur = conn.execute(
        """
        UPDATE admin_inbox_events
        SET acked_at = ?, acked_by_user_id = ?
        WHERE id = ? AND acked_at IS NULL
        """,
        (now, acked_by_user_id, event_id),
    )
    if cur.rowcount and cur.rowcount > 0:
        audit(
            conn,
            domain="admin_inbox",
            action="inbox_acked",
            actor="admin",
            actor_user_id=acked_by_user_id,
            subject_user_id=None,
            target_id=event_id,
            summary="Admin inbox event acknowledged.",
            metadata={},
        )
        return True
    return False


# ----------------------------
# Audit queries (read-only)
# ----------------------------

def list_audit(conn: sqlite3.Connection, *, domain: str, limit: int = 200) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, domain, action, actor, actor_user_id, subject_user_id, target_id, created_at, summary, metadata_json
        FROM audit_log
        WHERE domain = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (domain, int(limit)),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        item = dict(r)
        try:
            item["metadata"] = json.loads(item.get("metadata_json") or "{}")
        except Exception:
            item["metadata"] = item.get("metadata_json")
        out.append(item)
    return out