import sqlite3
from typing import Optional, Dict, Any

DB_PATH = "/app/data/jarvis_brain.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_task(title: str, resume_hint: Optional[str] = None) -> int:
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO tasks (title, status, resume_hint)
            VALUES (?, 'pending', ?)
            """,
            (title, resume_hint),
        )
        return int(cur.lastrowid)


def add_step(task_id: int, step_index: int, description: str) -> int:
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO task_steps (task_id, step_index, description, status)
            VALUES (?, ?, ?, 'pending')
            """,
            (task_id, step_index, description),
        )
        return int(cur.lastrowid)


def add_artifact(
    task_id: int,
    artifact_type: str,
    path: Optional[str] = None,
    metadata_json: Optional[str] = None,
    step_id: Optional[int] = None,
) -> int:
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO task_artifacts (task_id, step_id, artifact_type, path, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (task_id, step_id, artifact_type, path, metadata_json),
        )
        return int(cur.lastrowid)
