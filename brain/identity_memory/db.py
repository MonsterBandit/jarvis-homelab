from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def ensure_user(conn: sqlite3.Connection, user_id: str, *, ttl_days: int = 30) -> Dict[str, Any]:
    now = utc_now_iso()
    ttl = timedelta(days=int(ttl_days))
    expires = (datetime.now(timezone.utc) + ttl).isoformat()

    row = conn.execute(
        "SELECT user_id, display_name, status, created_at, expires_at, last_seen_at FROM users WHERE user_id = ?",
        (user_id,),
    ).fetchone()

    if row:
        conn.execute("UPDATE users SET last_seen_at = ? WHERE user_id = ?", (now, user_id))
        return {
            "user_id": row[0],
            "display_name": row[1],
            "status": row[2],
            "created_at": row[3],
            "expires_at": row[4],
            "last_seen_at": now,
            "created": False,
        }

    conn.execute(
        "INSERT INTO users (user_id, display_name, status, created_at, expires_at, last_seen_at) VALUES (?, ?, 'provisional', ?, ?, ?)",
        (user_id, None, now, expires, now),
    )
    conn.execute(
        "INSERT INTO user_lifecycle_events (user_id, event_type, details, created_at) VALUES (?, 'auto_created', ?, ?)",
        (user_id, json.dumps({"ttl_days": ttl_days}), now),
    )
    conn.execute(
        "INSERT INTO admin_inbox (event_type, user_id, payload, created_at, acknowledged) VALUES ('provisional_user_created', ?, ?, ?, 0)",
        (user_id, json.dumps({"ttl_days": ttl_days}), now),
    )
    return {
        "user_id": user_id,
        "display_name": None,
        "status": "provisional",
        "created_at": now,
        "expires_at": expires,
        "last_seen_at": now,
        "created": True,
    }

def get_user(conn: sqlite3.Connection, user_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT user_id, display_name, status, created_at, expires_at, last_seen_at FROM users WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    if not row:
        return None
    return {
        "user_id": row[0],
        "display_name": row[1],
        "status": row[2],
        "created_at": row[3],
        "expires_at": row[4],
        "last_seen_at": row[5],
    }

def promote_user(conn: sqlite3.Connection, user_id: str) -> None:
    now = utc_now_iso()
    conn.execute(
        "UPDATE users SET status='permanent', expires_at=NULL, last_seen_at=? WHERE user_id=?",
        (now, user_id),
    )
    conn.execute(
        "INSERT INTO user_lifecycle_events (user_id, event_type, details, created_at) VALUES (?, 'promoted_to_permanent', NULL, ?)",
        (user_id, now),
    )
    conn.execute(
        "INSERT INTO admin_inbox (event_type, user_id, payload, created_at, acknowledged) VALUES ('user_promoted', ?, NULL, ?, 0)",
        (user_id, now),
    )

def reinforce_user(conn: sqlite3.Connection, user_id: str, *, ttl_days: int = 30) -> None:
    # Extend expiry for provisional/reinforced users; permanent ignores.
    now = utc_now_iso()
    row = conn.execute("SELECT status FROM users WHERE user_id=?", (user_id,)).fetchone()
    if not row:
        return
    status = str(row[0] or "")
    if status == "permanent":
        conn.execute("UPDATE users SET last_seen_at=? WHERE user_id=?", (now, user_id))
        return
    expires = (datetime.now(timezone.utc) + timedelta(days=int(ttl_days))).isoformat()
    conn.execute(
        "UPDATE users SET status='reinforced', expires_at=?, last_seen_at=? WHERE user_id=?",
        (expires, now, user_id),
    )
    conn.execute(
        "INSERT INTO user_lifecycle_events (user_id, event_type, details, created_at) VALUES (?, 'reinforced', ?, ?)",
        (user_id, json.dumps({"ttl_days": ttl_days}), now),
    )
    conn.execute(
        "INSERT INTO admin_inbox (event_type, user_id, payload, created_at, acknowledged) VALUES ('user_reinforced', ?, ?, ?, 0)",
        (user_id, json.dumps({"ttl_days": ttl_days}), now),
    )

def upsert_memory(conn: sqlite3.Connection, user_id: str, key: str, value: str, tier: int, actor: str) -> Dict[str, Any]:
    now = utc_now_iso()
    # Upsert by (user_id, memory_key)
    row = conn.execute(
        "SELECT id, memory_value, memory_tier FROM memory_items WHERE user_id=? AND memory_key=?",
        (user_id, key),
    ).fetchone()

    if row:
        mem_id = int(row[0])
        conn.execute(
            "UPDATE memory_items SET memory_value=?, memory_tier=?, updated_at=? WHERE id=?",
            (value, int(tier), now, mem_id),
        )
        conn.execute(
            "INSERT INTO memory_events (memory_id, user_id, action, actor, created_at) VALUES (?, ?, 'updated', ?, ?)",
            (mem_id, user_id, actor, now),
        )
        return {"id": mem_id, "action": "updated", "memory_key": key, "memory_value": value, "tier": int(tier)}
    else:
        cur = conn.execute(
            "INSERT INTO memory_items (user_id, memory_key, memory_value, memory_tier, confidence, created_at, updated_at) VALUES (?, ?, ?, ?, 1.0, ?, ?)",
            (user_id, key, value, int(tier), now, now),
        )
        mem_id = int(cur.lastrowid)
        conn.execute(
            "INSERT INTO memory_events (memory_id, user_id, action, actor, created_at) VALUES (?, ?, 'created', ?, ?)",
            (mem_id, user_id, actor, now),
        )
        return {"id": mem_id, "action": "created", "memory_key": key, "memory_value": value, "tier": int(tier)}

def list_memory(conn: sqlite3.Connection, user_id: str, *, tier: Optional[int] = None) -> List[Dict[str, Any]]:
    if tier in (1, 2):
        rows = conn.execute(
            "SELECT id, memory_key, memory_value, memory_tier, confidence, created_at, updated_at FROM memory_items WHERE user_id=? AND memory_tier=? ORDER BY memory_key ASC",
            (user_id, int(tier)),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, memory_key, memory_value, memory_tier, confidence, created_at, updated_at FROM memory_items WHERE user_id=? ORDER BY memory_key ASC",
            (user_id,),
        ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "id": int(r[0]),
            "memory_key": r[1],
            "memory_value": r[2],
            "tier": int(r[3]),
            "confidence": float(r[4]) if r[4] is not None else None,
            "created_at": r[5],
            "updated_at": r[6],
        })
    return out

def delete_memory(conn: sqlite3.Connection, user_id: str, key: str, actor: str) -> Dict[str, Any]:
    now = utc_now_iso()
    row = conn.execute(
        "SELECT id FROM memory_items WHERE user_id=? AND memory_key=?",
        (user_id, key),
    ).fetchone()
    if not row:
        return {"ok": True, "deleted": False, "reason": "not_found"}
    mem_id = int(row[0])
    conn.execute("DELETE FROM memory_items WHERE id=?", (mem_id,))
    conn.execute(
        "INSERT INTO memory_events (memory_id, user_id, action, actor, created_at) VALUES (?, ?, 'deleted', ?, ?)",
        (mem_id, user_id, actor, now),
    )
    return {"ok": True, "deleted": True, "id": mem_id}

def admin_inbox_list(conn: sqlite3.Connection, *, limit: int = 50, include_ack: bool = False) -> List[Dict[str, Any]]:
    if include_ack:
        rows = conn.execute(
            "SELECT id, event_type, user_id, payload, created_at, acknowledged FROM admin_inbox ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, event_type, user_id, payload, created_at, acknowledged FROM admin_inbox WHERE acknowledged=0 ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    out = []
    for r in rows:
        payload = None
        if r[3]:
            try:
                payload = json.loads(r[3])
            except Exception:
                payload = r[3]
        out.append({
            "id": int(r[0]),
            "event_type": r[1],
            "user_id": r[2],
            "payload": payload,
            "created_at": r[4],
            "acknowledged": bool(r[5]),
        })
    return out

def admin_inbox_ack(conn: sqlite3.Connection, inbox_id: int) -> bool:
    cur = conn.execute("UPDATE admin_inbox SET acknowledged=1 WHERE id=?", (int(inbox_id),))
    return cur.rowcount > 0

# Opportunistic sweeper (best-effort): deletes expired provisional/reinforced users and logs inbox.
_LAST_SWEEP_AT = 0.0

def opportunistic_sweep(conn: sqlite3.Connection, *, min_interval_seconds: int = 60) -> Dict[str, Any]:
    global _LAST_SWEEP_AT
    now_epoch = time.time()
    if now_epoch - _LAST_SWEEP_AT < float(min_interval_seconds):
        return {"ran": False, "reason": "rate_limited"}

    _LAST_SWEEP_AT = now_epoch
    now = utc_now_iso()

    # Delete expired users (non-permanent) and cascade their memory via FK.
    rows = conn.execute(
        "SELECT user_id FROM users WHERE status != 'permanent' AND expires_at IS NOT NULL AND expires_at <= ?",
        (now,),
    ).fetchall()

    deleted = 0
    for (uid,) in rows:
        conn.execute(
            "INSERT INTO admin_inbox (event_type, user_id, payload, created_at, acknowledged) VALUES ('user_auto_deleted', ?, NULL, ?, 0)",
            (uid, now),
        )
        conn.execute(
            "INSERT INTO user_lifecycle_events (user_id, event_type, details, created_at) VALUES (?, 'expired_deleted', NULL, ?)",
            (uid, now),
        )
        conn.execute("DELETE FROM users WHERE user_id=?", (uid,))
        deleted += 1

    return {"ran": True, "deleted_users": deleted}
