from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from tools.task_ledger import add_artifact


DB_PATH = os.getenv("JARVIS_DB_PATH", "/app/data/jarvis_brain.db")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def store_plan(task_id: int, plan: Dict[str, Any], step_id: Optional[int] = None) -> None:
    add_artifact(
        task_id=task_id,
        step_id=step_id,
        artifact_type="okd_observation_plan",
        metadata_json=json.dumps({"stored_at": _iso_now(), "plan": plan}),
    )


def store_plan_approval(task_id: int, approved_by: str, step_id: Optional[int] = None) -> None:
    add_artifact(
        task_id=task_id,
        step_id=step_id,
        artifact_type="okd_observation_plan_approval",
        metadata_json=json.dumps({"stored_at": _iso_now(), "approved": True, "approved_by": approved_by}),
    )


def append_expansion_event(task_id: int, event: Dict[str, Any], step_id: Optional[int] = None) -> None:
    add_artifact(
        task_id=task_id,
        step_id=step_id,
        artifact_type="okd_expansion_event",
        metadata_json=json.dumps({"stored_at": _iso_now(), "event": event}),
    )


# ---- Compatibility wrappers required by governed_observation.py ----

def okd_store_plan(task_id: int, plan: Dict[str, Any], step_id: Optional[int] = None) -> None:
    store_plan(task_id, plan, step_id=step_id)


def okd_store_plan_approval(task_id: int, approved_by: str, step_id: Optional[int] = None) -> None:
    store_plan_approval(task_id, approved_by, step_id=step_id)


def okd_append_expansion_event(task_id: int, event: Dict[str, Any], step_id: Optional[int] = None) -> None:
    append_expansion_event(task_id, event, step_id=step_id)


def okd_store_expansion_log(task_id: int, events: List[Dict[str, Any]], step_id: Optional[int] = None) -> None:
    store_expansion_log(task_id, events, step_id=step_id)


def okd_store_update_brief(task_id: int, brief: Dict[str, Any], step_id: Optional[int] = None) -> None:
    store_update_brief(task_id, brief, step_id=step_id)


def okd_load_latest_plan(task_id: int) -> Optional[Dict[str, Any]]:
    return load_latest_plan(task_id)


def okd_is_plan_approved(task_id: int) -> bool:
    return is_plan_approved(task_id)

# ------------------------------------------------------------------

def store_expansion_log(task_id: int, events: List[Dict[str, Any]], step_id: Optional[int] = None) -> None:
    add_artifact(
        task_id=task_id,
        step_id=step_id,
        artifact_type="okd_expansion_log",
        metadata_json=json.dumps({"stored_at": _iso_now(), "events": events}),
    )


def store_update_brief(task_id: int, brief: Dict[str, Any], step_id: Optional[int] = None) -> None:
    add_artifact(
        task_id=task_id,
        step_id=step_id,
        artifact_type="okd_update_brief",
        metadata_json=json.dumps({"stored_at": _iso_now(), "brief": brief}),
    )


def _fetch_artifacts(task_id: int) -> List[Tuple[int, str, str]]:
    """
    Return list of (id, artifact_type, metadata_json) ordered by id asc.
    """
    conn = _db()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, artifact_type, metadata_json FROM task_artifacts WHERE task_id = ? ORDER BY id ASC",
            (task_id,),
        )
        return cur.fetchall() or []
    finally:
        conn.close()


def load_latest_plan(task_id: int) -> Optional[Dict[str, Any]]:
    rows = _fetch_artifacts(task_id)
    plan: Optional[Dict[str, Any]] = None
    for _id, atype, meta in rows:
        if atype != "okd_observation_plan":
            continue
        try:
            payload = json.loads(meta or "{}")
            plan = payload.get("plan")
        except Exception:
            plan = None
    return plan


def is_plan_approved(task_id: int) -> bool:
    rows = _fetch_artifacts(task_id)
    for _id, atype, meta in reversed(rows):
        if atype != "okd_observation_plan_approval":
            continue
        try:
            payload = json.loads(meta or "{}")
            return bool(payload.get("approved") is True)
        except Exception:
            return False
    return False


def load_expansion_events(task_id: int) -> List[Dict[str, Any]]:
    rows = _fetch_artifacts(task_id)
    out: List[Dict[str, Any]] = []
    for _id, atype, meta in rows:
        if atype != "okd_expansion_event":
            continue
        try:
            payload = json.loads(meta or "{}")
            ev = payload.get("event")
            if isinstance(ev, dict):
                out.append(ev)
        except Exception:
            continue
    return out
