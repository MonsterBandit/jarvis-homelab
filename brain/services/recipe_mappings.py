from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field


# Keep self-contained (no importing main.py)
DB_PATH = os.getenv("JARVIS_DB_PATH", "/app/data/jarvis_brain.db")

Household = Literal["home_a", "home_b"]

router = APIRouter(prefix="/recipes/mappings", tags=["Recipes - Mappings"])


# ----------------------------
# Helpers
# ----------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _ensure_table_exists() -> None:
    """
    Safety net. main.py init_db should create this, but we also guard here.
    """
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recipe_ingredient_mapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                household TEXT, -- NULL = global mapping
                ingredient_norm TEXT NOT NULL,
                ingredient_display TEXT NOT NULL,
                product_id INTEGER NOT NULL,
                product_name TEXT,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_recipe_ing_map_household_norm
            ON recipe_ingredient_mapping (household, ingredient_norm)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_recipe_ing_map_norm
            ON recipe_ingredient_mapping (ingredient_norm)
            """
        )
        conn.commit()
    finally:
        conn.close()


def _household_or_none(h: Optional[str]) -> Optional[str]:
    if h is None:
        return None
    hh = h.strip().lower()
    if hh == "":
        return None
    if hh not in {"home_a", "home_b"}:
        raise HTTPException(status_code=400, detail="household must be 'home_a', 'home_b', or omitted for global")
    return hh


# ----------------------------
# Models
# ----------------------------

class MappingSetRequest(BaseModel):
    household: Optional[Household] = Field(
        default=None,
        description="Optional. If omitted, mapping is global (applies to both households).",
    )
    ingredient_name: str = Field(..., min_length=1, description="Ingredient name to map, e.g. 'milk'")
    product_id: int = Field(..., ge=1, description="Grocy product_id")
    product_name: Optional[str] = Field(default=None, description="Optional: cached product name for UX")
    notes: Optional[str] = Field(default=None, description="Optional notes about this mapping")


class MappingRow(BaseModel):
    id: int
    household: Optional[str]
    ingredient_norm: str
    ingredient_display: str
    product_id: int
    product_name: Optional[str] = None
    notes: Optional[str] = None
    created_at: str
    updated_at: str


class MappingListResponse(BaseModel):
    status: Literal["ok"] = "ok"
    count: int
    items: List[MappingRow]


# ----------------------------
# Endpoints (explicit writes)
# ----------------------------

@router.get("", response_model=MappingListResponse, summary="List ingredient mappings (global + optional household).")
def list_mappings(
    household: Optional[str] = Query(default=None, description="Filter by household. Omit to see all."),
    q: Optional[str] = Query(default=None, description="Search by ingredient (normalized)."),
) -> MappingListResponse:
    _ensure_table_exists()

    hh = _household_or_none(household)
    qn = _normalize(q) if q else None

    conn = _connect()
    try:
        cur = conn.cursor()

        where: List[str] = []
        params: List[Any] = []

        if hh is not None:
            where.append("household = ?")
            params.append(hh)

        if qn:
            where.append("ingredient_norm LIKE ?")
            params.append(f"%{qn}%")

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        cur.execute(
            f"""
            SELECT id, household, ingredient_norm, ingredient_display, product_id, product_name, notes, created_at, updated_at
            FROM recipe_ingredient_mapping
            {where_sql}
            ORDER BY household IS NOT NULL, household, ingredient_norm
            """,
            tuple(params),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    items: List[MappingRow] = []
    for r in rows:
        items.append(
            MappingRow(
                id=int(r[0]),
                household=r[1],
                ingredient_norm=r[2],
                ingredient_display=r[3],
                product_id=int(r[4]),
                product_name=r[5],
                notes=r[6],
                created_at=r[7],
                updated_at=r[8],
            )
        )

    return MappingListResponse(count=len(items), items=items)


@router.post("/set", response_model=Dict[str, Any], summary="Create or update a mapping (explicit).")
def set_mapping(body: MappingSetRequest) -> Dict[str, Any]:
    """
    Explicit write: stores mapping in SQLite only (NOT Grocy).
    """
    _ensure_table_exists()

    hh = _household_or_none(body.household)
    ing_display = body.ingredient_name.strip()
    ing_norm = _normalize(ing_display)

    if not ing_norm:
        raise HTTPException(status_code=400, detail="ingredient_name cannot be empty after normalization")

    conn = _connect()
    try:
        cur = conn.cursor()

        now = _utc_now()

        # UPSERT by unique (household, ingredient_norm)
        # SQLite upsert needs the same unique index we created.
        cur.execute(
            """
            INSERT INTO recipe_ingredient_mapping
                (household, ingredient_norm, ingredient_display, product_id, product_name, notes, created_at, updated_at)
            VALUES
                (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(household, ingredient_norm) DO UPDATE SET
                ingredient_display=excluded.ingredient_display,
                product_id=excluded.product_id,
                product_name=excluded.product_name,
                notes=excluded.notes,
                updated_at=excluded.updated_at
            """,
            (
                hh,
                ing_norm,
                ing_display,
                int(body.product_id),
                body.product_name,
                body.notes,
                now,
                now,
            ),
        )
        conn.commit()

        # Return the stored row
        cur.execute(
            """
            SELECT id, household, ingredient_norm, ingredient_display, product_id, product_name, notes, created_at, updated_at
            FROM recipe_ingredient_mapping
            WHERE household IS ? AND ingredient_norm = ?
            """,
            (hh, ing_norm),
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=500, detail="Failed to read back mapping after upsert")

    return {
        "status": "ok",
        "mapping": {
            "id": int(row[0]),
            "household": row[1],
            "ingredient_norm": row[2],
            "ingredient_display": row[3],
            "product_id": int(row[4]),
            "product_name": row[5],
            "notes": row[6],
            "created_at": row[7],
            "updated_at": row[8],
        },
        "notes": [
            "Stored in ISAC SQLite only",
            "No Grocy writes performed",
        ],
    }


@router.delete("/delete", response_model=Dict[str, Any], summary="Delete a mapping by household + ingredient_name (explicit).")
def delete_mapping(
    ingredient_name: str = Query(..., min_length=1),
    household: Optional[str] = Query(default=None, description="Omit to delete global mapping for that ingredient."),
) -> Dict[str, Any]:
    _ensure_table_exists()

    hh = _household_or_none(household)
    ing_norm = _normalize(ingredient_name)

    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            DELETE FROM recipe_ingredient_mapping
            WHERE household IS ? AND ingredient_norm = ?
            """,
            (hh, ing_norm),
        )
        deleted = cur.rowcount
        conn.commit()
    finally:
        conn.close()

    return {"status": "ok", "deleted": int(deleted), "household": hh, "ingredient_norm": ing_norm}
