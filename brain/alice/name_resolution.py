"""
/opt/jarvis/brain/alice/name_resolution.py

Read-only name resolution for Alice.

Purpose:
- Resolve a user phrase to a known concept via memory_aliases (preferred_name) and memory_concepts.
- No writes. No learning. No confidence updates. No schema changes.

Authoritative schema expectations (do not assume beyond this):
- memory_aliases: alias_id (PK), concept_id, user_id, preferred_name, confidence, locked, created_at, updated_at, pattern_notes
- memory_concepts: concept_id (PK), concept_key, ...

Resolution rules (v1):
1) Exact match on preferred_name after normalization (strip + casefold) for the given user_id
2) If multiple rows somehow match, pick highest confidence then newest updated_at/created_at

Returns:
- dict with concept_id, concept_key, preferred_name, confidence, alias_id, source="alias"
- or None if no match

NOTE:
- DB path uses same env fallbacks as memory_api.py
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Dict, Optional


def _db_path() -> str:
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


def _norm(s: str) -> str:
    # v1 normalization: trim + casefold (no stemming, no fuzzy)
    return (s or "").strip().casefold()


def resolve_alias_to_concept(*, text: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Resolve an input phrase to a concept via memory_aliases for a specific user.

    Args:
        text: user phrase (e.g. "test concept")
        user_id: canonical user id (e.g. "admin")

    Returns:
        dict with keys:
          - concept_id
          - concept_key
          - preferred_name
          - confidence
          - alias_id
          - source = "alias"
        or None if not found.
    """
    q = _norm(text)
    if not q or not user_id:
        return None

    # We do a case-insensitive match by normalizing in Python.
    # SQLite doesn't have a built-in casefold, so we keep this simple:
    # fetch candidate rows for user_id and compare normalized preferred_name in Python.
    with _connect() as conn:
        # Pull minimal columns first.
        rows = conn.execute(
            """
            SELECT
                a.alias_id,
                a.concept_id,
                a.user_id,
                a.preferred_name,
                a.confidence,
                a.created_at,
                a.updated_at,
                c.concept_key
            FROM memory_aliases a
            JOIN memory_concepts c ON c.concept_id = a.concept_id
            WHERE a.user_id = ?
            """,
            (user_id,),
        ).fetchall()

        matches = []
        for r in rows:
            if _norm(r["preferred_name"]) == q:
                matches.append(r)

        if not matches:
            return None

        # Sort: highest confidence, then updated_at, then created_at, then alias_id desc
        def sort_key(r):
            conf = float(r["confidence"] or 0.0)
            upd = r["updated_at"] or ""
            cre = r["created_at"] or ""
            aid = int(r["alias_id"] or 0)
            return (conf, upd, cre, aid)

        best = sorted(matches, key=sort_key, reverse=True)[0]

        return {
            "source": "alias",
            "alias_id": best["alias_id"],
            "concept_id": best["concept_id"],
            "concept_key": best["concept_key"],
            "preferred_name": best["preferred_name"],
            "confidence": best["confidence"],
        }
