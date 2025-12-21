"""
Meal Planning Service (Phase 6.76)
- ISAC-owned meal plans referencing finalized recipe snapshots
- Household-aware
- Preview-only shopping aggregation
- Hard guarantees:
  - NO Grocy writes
  - NO inventory changes
  - NO automation / scheduling
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

router = APIRouter()

DEFAULT_DB_PATH = "/app/data/jarvis_brain.db"
DB_PATH_ENV = "JARVIS_DB_PATH"


# -------------------------
# Utilities
# -------------------------

def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _get_db_path() -> str:
    return os.getenv(DB_PATH_ENV, DEFAULT_DB_PATH)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema() -> None:
    """Idempotent schema creation."""
    conn = _connect()
    try:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meal_plans (
                id TEXT PRIMARY KEY,
                household TEXT NOT NULL,
                title TEXT NOT NULL,
                week_start TEXT NOT NULL,  -- YYYY-MM-DD (Sunday start)
                created_at TEXT NOT NULL,
                notes TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_meal_plans_household_week
            ON meal_plans(household, week_start);
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meal_plan_items (
                id TEXT PRIMARY KEY,
                meal_plan_id TEXT NOT NULL,
                day INTEGER NOT NULL,       -- 0=Sun ... 6=Sat
                slot TEXT NOT NULL,         -- breakfast/lunch/dinner/snack/other
                recipe_finalized_id TEXT NOT NULL,
                servings REAL,
                notes TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(meal_plan_id) REFERENCES meal_plans(id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_meal_plan_items_plan
            ON meal_plan_items(meal_plan_id);
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_meal_plan_items_recipe
            ON meal_plan_items(recipe_finalized_id);
            """
        )

        conn.commit()
    finally:
        conn.close()


def _table_names(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    return [r["name"] for r in cur.fetchall()]


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    return [r["name"] for r in cur.fetchall()]


def _find_recipes_table(conn: sqlite3.Connection) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """
    Try to find the canonical recipes table and key columns for:
      - id
      - status OR something that indicates finalized
      - finalized_payload
    Returns: (table_name, column_map) where column_map has keys:
      id_col, status_col (optional), finalized_payload_col (optional)
    """
    tables = _table_names(conn)

    candidates = [
        "recipes",
        "recipe",
        "recipe_snapshots",
        "recipes_snapshots",
        "recipe_drafts",
        "recipes_drafts",
    ]

    for t in candidates:
        if t not in tables:
            continue
        cols = set(_table_columns(conn, t))

        # Try common column names we expect from 6.75
        id_col = "id" if "id" in cols else None
        status_col = "status" if "status" in cols else None
        finalized_payload_col = "finalized_payload" if "finalized_payload" in cols else None

        # Some schemas might store a finalized snapshot payload under different name
        if finalized_payload_col is None:
            for alt in ("final_payload", "snapshot_payload", "payload"):
                if alt in cols:
                    finalized_payload_col = alt
                    break

        if id_col:
            return t, {
                "id_col": id_col,
                "status_col": status_col or "",
                "finalized_payload_col": finalized_payload_col or "",
            }

    return None, None


def _require_finalized_recipe(conn: sqlite3.Connection, recipe_id: str) -> sqlite3.Row:
    """
    Enforces: the recipe reference must exist AND must be finalized.
    This is strict by design.
    """
    table, cmap = _find_recipes_table(conn)
    if not table or not cmap:
        raise HTTPException(
            status_code=500,
            detail="Recipes table not found in ISAC DB. Ensure Phase 6.75 is present and using canonical SQLite store.",
        )

    id_col = cmap["id_col"]
    status_col = cmap.get("status_col") or ""
    finalized_payload_col = cmap.get("finalized_payload_col") or ""

    cur = conn.cursor()

    if status_col:
        cur.execute(
            f"SELECT * FROM {table} WHERE {id_col} = ? AND {status_col} = 'finalized' LIMIT 1;",
            (recipe_id,),
        )
    else:
        # If no status column exists, we still require a finalized payload column to exist.
        if not finalized_payload_col:
            raise HTTPException(
                status_code=500,
                detail="Recipes table found but does not expose status or finalized payload columns. Cannot enforce finalized-only constraint.",
            )
        cur.execute(
            f"SELECT * FROM {table} WHERE {id_col} = ? AND {finalized_payload_col} IS NOT NULL LIMIT 1;",
            (recipe_id,),
        )

    row = cur.fetchone()
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Finalized recipe not found for id={recipe_id}. Use a finalized snapshot id from Phase 6.75.",
        )
    return row


def _parse_finalized_payload(row: sqlite3.Row) -> Dict[str, Any]:
    """
    Pulls and parses finalized payload JSON from whatever column exists.
    Returns {} if missing/unparseable.
    """
    # common: finalized_payload
    for key in ("finalized_payload", "final_payload", "snapshot_payload", "payload"):
        if key in row.keys():
            raw = row[key]
            if raw is None:
                return {}
            if isinstance(raw, (dict, list)):
                return {"_raw": raw}
            if isinstance(raw, (bytes, bytearray)):
                try:
                    raw = raw.decode("utf-8")
                except Exception:
                    return {}
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        return parsed
                    return {"_raw": parsed}
                except Exception:
                    return {}
    return {}


def _extract_ingredients(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Resilient ingredient extraction for multiple finalized payload schemas.

    Supported shapes (observed in Phase 6.75 finalized_payload):
      - payload["resolved_ingredients"] : list[dict]
      - payload["parsed_ingredients"]   : list[dict]
      - payload["raw_recipe"]["ingredient_lines"] : list[str]
      - payload["ingredients"] / other legacy keys (fallback)

    Output ingredient objects in a normalized-ish shape:
      { name, quantity, unit, note, raw }
    """
    if not payload:
        return []

    candidates: List[Any] = []

    # 1) Preferred: resolved_ingredients
    v = payload.get("resolved_ingredients")
    if isinstance(v, list) and v:
        candidates = v
    else:
        # 2) Next: parsed_ingredients
        v = payload.get("parsed_ingredients")
        if isinstance(v, list) and v:
            candidates = v
        else:
            # 3) Fallback: raw_recipe.ingredient_lines
            raw_recipe = payload.get("raw_recipe") if isinstance(payload.get("raw_recipe"), dict) else {}
            v = raw_recipe.get("ingredient_lines") if isinstance(raw_recipe, dict) else None
            if isinstance(v, list) and v:
                candidates = v
            else:
                # 4) Legacy/fallback keys
                for k in ("ingredients", "ingredient_lines", "items", "line_items"):
                    v = payload.get(k)
                    if isinstance(v, list) and v:
                        candidates = v
                        break

    def qty_from_obj(q: Any) -> Any:
        """
        Convert qty object like {"min": 1.0, "max": 2.0, "raw": "1-2"} into
        a usable quantity representation.
        - If min == max: returns float(min)
        - Else: returns raw if present, else "min-max"
        """
        if q is None:
            return None
        if isinstance(q, (int, float, str)):
            return q
        if isinstance(q, dict):
            qmin = q.get("min")
            qmax = q.get("max")
            raw = q.get("raw")
            if qmin is not None and qmax is not None:
                try:
                    fmin = float(qmin)
                    fmax = float(qmax)
                    if fmin == fmax:
                        return fmin
                    if isinstance(raw, str) and raw.strip():
                        return raw.strip()
                    return f"{fmin}-{fmax}"
                except Exception:
                    pass
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        return q

    normalized: List[Dict[str, Any]] = []

    for item in candidates:
        if isinstance(item, str):
            s = item.strip()
            if s:
                normalized.append({"name": s, "quantity": None, "unit": None, "note": None, "raw": item})
            continue

        if isinstance(item, dict):
            name = (
                item.get("name")
                or item.get("ingredient")
                or item.get("title")
                or item.get("text")
                or item.get("original")
                or item.get("raw")
                or ""
            )

            qty = item.get("quantity") if "quantity" in item else item.get("qty")
            unit = item.get("unit") if "unit" in item else item.get("uom")
            note = item.get("note") or item.get("comment") or item.get("preparation")

            normalized.append(
                {
                    "name": str(name).strip() if name is not None else "",
                    "quantity": qty_from_obj(qty),
                    "unit": unit,
                    "note": note,
                    "raw": item,
                }
            )
            continue

        normalized.append({"name": "", "quantity": None, "unit": None, "note": None, "raw": item})

    return [n for n in normalized if n.get("name")]


def _coerce_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _scaled_qty(qty: Any, scale: float) -> Any:
    f = _coerce_float(qty)
    if f is None:
        return qty
    return round(f * scale, 4)


# -------------------------
# Models
# -------------------------

class MealPlanCreate(BaseModel):
    household: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    week_start: str = Field(..., description="YYYY-MM-DD (Sunday start)")
    notes: Optional[str] = None

    @validator("week_start")
    def validate_week_start(cls, v: str) -> str:
        try:
            d = date.fromisoformat(v)
        except Exception:
            raise ValueError("week_start must be YYYY-MM-DD")

        # Enforce Sunday start: Python weekday Mon=0..Sun=6, so Sunday == 6
        if d.weekday() != 6:
            raise ValueError("week_start must be a Sunday (Sunday-start week)")
        return v


class MealPlanOut(BaseModel):
    id: str
    household: str
    title: str
    week_start: str
    created_at: str
    notes: Optional[str] = None


class MealPlanItemCreate(BaseModel):
    day: int = Field(..., ge=0, le=6, description="0=Sun ... 6=Sat")
    slot: str = Field(..., min_length=1, description="breakfast/lunch/dinner/snack/other")
    recipe_finalized_id: str = Field(..., min_length=1)
    servings: Optional[float] = Field(None, gt=0)
    notes: Optional[str] = None


class MealPlanItemOut(BaseModel):
    id: str
    meal_plan_id: str
    day: int
    slot: str
    recipe_finalized_id: str
    servings: Optional[float] = None
    notes: Optional[str] = None
    created_at: str


class MealPlanDetail(BaseModel):
    plan: MealPlanOut
    items: List[MealPlanItemOut]


class ShoppingPreviewLine(BaseModel):
    name: str
    quantity: Any = None
    unit: Any = None
    note: Any = None
    sources: List[Dict[str, Any]]


class ShoppingPreviewOut(BaseModel):
    meal_plan_id: str
    household: str
    week_start: str
    lines: List[ShoppingPreviewLine]
    warnings: List[str] = []


# -------------------------
# Routes
# -------------------------

@router.on_event("startup")
def _startup() -> None:
    _ensure_schema()


@router.post("", response_model=MealPlanOut)
def create_meal_plan(payload: MealPlanCreate) -> MealPlanOut:
    _ensure_schema()

    plan_id = str(uuid.uuid4())
    created_at = _utc_now_iso()

    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO meal_plans (id, household, title, week_start, created_at, notes)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (plan_id, payload.household, payload.title, payload.week_start, created_at, payload.notes),
        )
        conn.commit()

        return MealPlanOut(
            id=plan_id,
            household=payload.household,
            title=payload.title,
            week_start=payload.week_start,
            created_at=created_at,
            notes=payload.notes,
        )
    finally:
        conn.close()


@router.get("", response_model=List[MealPlanOut])
def list_meal_plans(household: Optional[str] = None, week_start: Optional[str] = None) -> List[MealPlanOut]:
    _ensure_schema()

    conn = _connect()
    try:
        cur = conn.cursor()

        where = []
        params: List[Any] = []

        if household:
            where.append("household = ?")
            params.append(household)
        if week_start:
            where.append("week_start = ?")
            params.append(week_start)

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        cur.execute(
            f"""
            SELECT id, household, title, week_start, created_at, notes
            FROM meal_plans
            {where_sql}
            ORDER BY week_start DESC, created_at DESC;
            """,
            params,
        )
        rows = cur.fetchall()

        return [
            MealPlanOut(
                id=r["id"],
                household=r["household"],
                title=r["title"],
                week_start=r["week_start"],
                created_at=r["created_at"],
                notes=r["notes"],
            )
            for r in rows
        ]
    finally:
        conn.close()


@router.get("/{meal_plan_id}", response_model=MealPlanDetail)
def get_meal_plan(meal_plan_id: str) -> MealPlanDetail:
    _ensure_schema()

    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM meal_plans WHERE id = ? LIMIT 1;", (meal_plan_id,))
        plan = cur.fetchone()
        if not plan:
            raise HTTPException(status_code=404, detail="Meal plan not found")

        cur.execute(
            """
            SELECT * FROM meal_plan_items
            WHERE meal_plan_id = ?
            ORDER BY day ASC, slot ASC, created_at ASC;
            """,
            (meal_plan_id,),
        )
        items = cur.fetchall()

        plan_out = MealPlanOut(
            id=plan["id"],
            household=plan["household"],
            title=plan["title"],
            week_start=plan["week_start"],
            created_at=plan["created_at"],
            notes=plan["notes"],
        )

        items_out = [
            MealPlanItemOut(
                id=i["id"],
                meal_plan_id=i["meal_plan_id"],
                day=i["day"],
                slot=i["slot"],
                recipe_finalized_id=i["recipe_finalized_id"],
                servings=i["servings"],
                notes=i["notes"],
                created_at=i["created_at"],
            )
            for i in items
        ]

        return MealPlanDetail(plan=plan_out, items=items_out)
    finally:
        conn.close()


@router.post("/{meal_plan_id}/items", response_model=MealPlanItemOut)
def add_meal_plan_item(meal_plan_id: str, payload: MealPlanItemCreate) -> MealPlanItemOut:
    _ensure_schema()

    item_id = str(uuid.uuid4())
    created_at = _utc_now_iso()

    conn = _connect()
    try:
        cur = conn.cursor()

        # Ensure meal plan exists
        cur.execute("SELECT id FROM meal_plans WHERE id = ? LIMIT 1;", (meal_plan_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Meal plan not found")

        # Enforce: recipe id must refer to a finalized snapshot
        _require_finalized_recipe(conn, payload.recipe_finalized_id)

        cur.execute(
            """
            INSERT INTO meal_plan_items
              (id, meal_plan_id, day, slot, recipe_finalized_id, servings, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                item_id,
                meal_plan_id,
                payload.day,
                payload.slot.strip(),
                payload.recipe_finalized_id,
                payload.servings,
                payload.notes,
                created_at,
            ),
        )
        conn.commit()

        return MealPlanItemOut(
            id=item_id,
            meal_plan_id=meal_plan_id,
            day=payload.day,
            slot=payload.slot.strip(),
            recipe_finalized_id=payload.recipe_finalized_id,
            servings=payload.servings,
            notes=payload.notes,
            created_at=created_at,
        )
    finally:
        conn.close()


@router.delete("/{meal_plan_id}/items/{item_id}")
def delete_meal_plan_item(meal_plan_id: str, item_id: str) -> Dict[str, Any]:
    _ensure_schema()

    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM meal_plan_items WHERE id = ? AND meal_plan_id = ?;",
            (item_id, meal_plan_id),
        )
        conn.commit()
        return {"ok": True, "deleted": cur.rowcount}
    finally:
        conn.close()


@router.get("/{meal_plan_id}/shopping/preview", response_model=ShoppingPreviewOut)
def shopping_preview(meal_plan_id: str) -> ShoppingPreviewOut:
    """
    Preview-only aggregation of ingredients from finalized recipe snapshots referenced by a meal plan.
    NO writes anywhere (ISAC-only read).
    """
    _ensure_schema()

    conn = _connect()
    warnings: List[str] = []
    try:
        cur = conn.cursor()

        cur.execute("SELECT * FROM meal_plans WHERE id = ? LIMIT 1;", (meal_plan_id,))
        plan = cur.fetchone()
        if not plan:
            raise HTTPException(status_code=404, detail="Meal plan not found")

        cur.execute(
            """
            SELECT * FROM meal_plan_items
            WHERE meal_plan_id = ?
            ORDER BY day ASC, slot ASC, created_at ASC;
            """,
            (meal_plan_id,),
        )
        items = cur.fetchall()

        aggregated: Dict[str, Dict[str, Any]] = {}

        for it in items:
            recipe_id = it["recipe_finalized_id"]
            servings = it["servings"]

            recipe_row = _require_finalized_recipe(conn, recipe_id)
            payload = _parse_finalized_payload(recipe_row)
            ingredients = _extract_ingredients(payload)

            if not ingredients:
                warnings.append(f"Recipe {recipe_id} has no parseable ingredients in finalized payload.")
                continue

            # If servings is provided, treat it as a multiplier only if payload exposes base_servings.
            # Otherwise, we will not guess scaling; we’ll attach a warning.
            scale = 1.0
            base_servings = None
            for k in ("servings", "base_servings", "yield_servings"):
                if k in payload:
                    base_servings = _coerce_float(payload.get(k))
                    break

            if servings is not None:
                if base_servings and base_servings > 0:
                    scale = float(servings) / float(base_servings)
                else:
                    warnings.append(
                        f"Item references servings={servings} for recipe {recipe_id}, but base servings not found; quantities not scaled."
                    )

            for ing in ingredients:
                name = (ing.get("name") or "").strip()
                if not name:
                    continue

                # Key by lowercase name for grouping
                key = name.lower()

                qty = ing.get("quantity")
                unit = ing.get("unit")
                note = ing.get("note")

                scaled_qty = _scaled_qty(qty, scale) if scale != 1.0 else qty

                if key not in aggregated:
                    aggregated[key] = {
                        "name": name,
                        "quantity": scaled_qty,
                        "unit": unit,
                        "note": note,
                        "sources": [],
                    }
                else:
                    # If both are numeric and units match, sum. Otherwise, keep as multiple sources.
                    prev_qty = aggregated[key].get("quantity")
                    prev_unit = aggregated[key].get("unit")

                    a = _coerce_float(prev_qty)
                    b = _coerce_float(scaled_qty)
                    if a is not None and b is not None and (prev_unit == unit):
                        aggregated[key]["quantity"] = round(a + b, 4)
                    else:
                        # Preserve first quantity but add a warning if conflict
                        if (prev_unit != unit) or (a is None) or (b is None):
                            warnings.append(
                                f"Could not sum quantities for '{name}' due to unit/format mismatch; see sources."
                            )

                aggregated[key]["sources"].append(
                    {
                        "recipe_finalized_id": recipe_id,
                        "day": it["day"],
                        "slot": it["slot"],
                        "quantity": scaled_qty,
                        "unit": unit,
                        "note": note,
                    }
                )

        lines = [
            ShoppingPreviewLine(
                name=v["name"],
                quantity=v.get("quantity"),
                unit=v.get("unit"),
                note=v.get("note"),
                sources=v.get("sources", []),
            )
            for _, v in sorted(aggregated.items(), key=lambda kv: kv[0])
        ]

        return ShoppingPreviewOut(
            meal_plan_id=meal_plan_id,
            household=plan["household"],
            week_start=plan["week_start"],
            lines=lines,
            warnings=warnings,
        )
    finally:
        conn.close()
