from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Meal Plans Service
# -----------------------------------------------------------------------------
# Goals:
# - Store meal plans and plan items in SQLite (ISAC-owned)
# - Reference recipes by "finalized snapshot id" (recipe_drafts.id where status='finalized')
# - Provide shopping preview aggregation using the finalized payload snapshot
#
# Guardrails:
# - No Grocy writes
# - No inventory changes
# - Preview-only for shopping list aggregation (stored in ISAC DB only)
#
# Phase 6.8x (Step 1.2 - Option C):
# - Enrich shopping preview with:
#   1) recipe titles per source (read-only)
#   2) mapping hints per aggregated line (read-only)
# - Strictly no writes (Grocy or SQLite)
#
# POLISH:
# - Deduplicate warnings (stable order), so repeated "servings not scaled" doesn't spam output.
# -----------------------------------------------------------------------------


router = APIRouter(prefix="/mealplans", tags=["Meal Planner"])


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class MealPlanCreateRequest(BaseModel):
    household: str = Field(..., description="Household key, e.g., home_a")
    week_start: str = Field(..., description="Week start date in YYYY-MM-DD (Sunday-start)")


class MealPlanResponse(BaseModel):
    id: str
    household: str
    week_start: str
    created_at: str


class MealPlanItemCreateRequest(BaseModel):
    day: int = Field(..., ge=0, le=6, description="0=Sunday, 6=Saturday")
    slot: str = Field(..., description="Meal slot e.g., breakfast/lunch/dinner/snack")
    recipe_finalized_id: str = Field(..., description="Finalized snapshot id (recipe_drafts.id with status=finalized)")
    servings: Optional[float] = Field(None, description="Desired servings for this plan item (optional)")
    notes: Optional[str] = Field(None, description="Notes about this item (optional)")


class MealPlanItemResponse(BaseModel):
    id: str
    meal_plan_id: str
    day: int
    slot: str
    recipe_finalized_id: str
    servings: Optional[float]
    notes: Optional[str]
    created_at: str


class ShoppingPreviewSource(BaseModel):
    recipe_finalized_id: str
    recipe_title: Optional[str] = None
    day: int
    slot: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    note: Optional[str] = None


class ShoppingPreviewLine(BaseModel):
    name: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    note: Optional[str] = None

    # Read-only mapping hints (preview only)
    mapped_product_id: Optional[int] = None
    mapped_product_name: Optional[str] = None
    mapped_qu_id: Optional[int] = None
    mapping_source: Optional[str] = None  # e.g., "recipe_ingredient_mappings" or "recipe_ingredient_mapping"
    mapping_note: Optional[str] = None

    sources: List[ShoppingPreviewSource]


class ShoppingPreviewResponse(BaseModel):
    meal_plan_id: str
    household: str
    week_start: str
    lines: List[ShoppingPreviewLine]
    warnings: List[str]


# -----------------------------------------------------------------------------
# DB Helpers
# -----------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # Meal plans table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS meal_plans (
            id TEXT PRIMARY KEY,
            household TEXT NOT NULL,
            week_start TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_meal_plans_household_week ON meal_plans(household, week_start);")

    # Meal plan items table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS meal_plan_items (
            id TEXT PRIMARY KEY,
            meal_plan_id TEXT NOT NULL,
            day INTEGER NOT NULL,
            slot TEXT NOT NULL,
            recipe_finalized_id TEXT NOT NULL,
            servings REAL,
            notes TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(meal_plan_id) REFERENCES meal_plans(id)
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_meal_plan_items_plan ON meal_plan_items(meal_plan_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_meal_plan_items_recipe ON meal_plan_items(recipe_finalized_id);")

    conn.commit()


def _coerce_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        s = str(val).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# SQLite Introspection helpers (read-only)
# -----------------------------------------------------------------------------


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;", (table,))
    return cur.fetchone() is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    try:
        rows = cur.execute(f"PRAGMA table_info({table});").fetchall()
        return [r["name"] for r in rows if isinstance(r, sqlite3.Row)]
    except Exception:
        return []


# -----------------------------------------------------------------------------
# Recipe Finalized Payload Helpers
# -----------------------------------------------------------------------------


def _require_finalized_recipe(conn: sqlite3.Connection, finalized_id: str) -> sqlite3.Row:
    cur = conn.cursor()
    cur.execute("SELECT * FROM recipe_drafts WHERE id = ? LIMIT 1;", (finalized_id,))
    row = cur.fetchone()
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Finalized recipe not found for id={finalized_id}. Use a finalized snapshot id from Phase 6.75.",
        )
    if "status" in row.keys() and row["status"] != "finalized":
        raise HTTPException(status_code=400, detail=f"Recipe id={finalized_id} is not finalized (status={row['status']}).")
    return row


def _parse_finalized_payload(recipe_row: sqlite3.Row) -> Dict[str, Any]:
    # DB column is finalized_payload (TEXT), expected to be JSON
    if "finalized_payload" not in recipe_row.keys():
        return {}
    raw = recipe_row["finalized_payload"]
    if raw is None:
        return {}
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _unwrap_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Some callers wrap the "real" payload under keys like snapshot/finalized_payload.
    We'll peel layers to find where the ingredient lists actually live.
    """
    if not isinstance(payload, dict):
        return {}

    for k in ("snapshot", "finalized_payload", "payload", "data"):
        v = payload.get(k)
        if isinstance(v, dict) and v:
            payload = v
    return payload


def _get_recipe_title(payload: Dict[str, Any], recipe_row: sqlite3.Row) -> Optional[str]:
    """
    Read-only convenience. Prefer snapshot raw_recipe.title, fall back to recipe_drafts.title column.
    """
    try:
        p = _unwrap_payload(payload) if isinstance(payload, dict) else {}
        raw_recipe = p.get("raw_recipe")
        if isinstance(raw_recipe, dict):
            t = raw_recipe.get("title")
            if isinstance(t, str) and t.strip():
                return t.strip()

        t2 = p.get("title")
        if isinstance(t2, str) and t2.strip():
            return t2.strip()

        if "title" in recipe_row.keys():
            t3 = recipe_row["title"]
            if isinstance(t3, str) and t3.strip():
                return t3.strip()
    except Exception:
        return None

    return None


def _find_base_servings(payload: Dict[str, Any]) -> Optional[float]:
    if not isinstance(payload, dict):
        return None

    payload = _unwrap_payload(payload)

    for k in ("base_servings", "yield_servings", "servings"):
        if k in payload:
            v = _coerce_float(payload.get(k))
            if v is not None:
                return v

    raw_recipe = payload.get("raw_recipe")
    if isinstance(raw_recipe, dict) and "servings" in raw_recipe:
        v = _coerce_float(raw_recipe.get("servings"))
        if v is not None:
            return v

    draft = payload.get("draft")
    if isinstance(draft, dict):
        rr = draft.get("raw_recipe")
        if isinstance(rr, dict) and "servings" in rr:
            v = _coerce_float(rr.get("servings"))
            if v is not None:
                return v

    return None


def _extract_ingredients(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    payload = _unwrap_payload(payload)

    structured_keys = (
        "resolved_ingredients",
        "parsed_ingredients",
        "ingredients",
        "items",
        "line_items",
    )

    for k in structured_keys:
        v = payload.get(k)
        if isinstance(v, list) and v:
            out: List[Dict[str, Any]] = []
            for it in v:
                if isinstance(it, str):
                    out.append({"name": it.strip(), "quantity": None, "unit": None, "note": None})
                    continue
                if not isinstance(it, dict):
                    continue

                name = (it.get("name") or it.get("ingredient") or it.get("title") or "").strip()
                unit = it.get("unit")
                note = it.get("note")

                qty_val: Optional[float] = None
                qty_obj = it.get("qty")
                if isinstance(qty_obj, dict):
                    qmin = _coerce_float(qty_obj.get("min"))
                    qmax = _coerce_float(qty_obj.get("max"))
                    if qmin is not None and qmax is not None and abs(qmin - qmax) < 1e-9:
                        qty_val = qmin
                    else:
                        qty_val = _coerce_float(qty_obj.get("raw"))

                if qty_val is None:
                    qty_val = _coerce_float(it.get("quantity"))

                if not name:
                    name = (it.get("original") or "").strip()

                if not name:
                    continue

                out.append(
                    {
                        "name": name,
                        "quantity": qty_val,
                        "unit": unit,
                        "note": note,
                    }
                )
            return out

    for k in ("ingredient_lines", "raw_ingredient_lines", "raw_lines"):
        v = payload.get(k)
        if isinstance(v, list) and v:
            return [{"name": str(line).strip(), "quantity": None, "unit": None, "note": None} for line in v if str(line).strip()]

    return []


def _scaled_qty(qty: Any, scale: float) -> Any:
    if qty is None:
        return None
    x = _coerce_float(qty)
    if x is None:
        return qty
    return round(x * scale, 4)


# -----------------------------------------------------------------------------
# Mapping helpers (read-only hints)
# -----------------------------------------------------------------------------


def _normalize_unit(u: Optional[str]) -> str:
    return (u or "").strip().lower()


def _norm_ing(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _lookup_mapping_hint(
    conn: sqlite3.Connection,
    *,
    household: str,
    ingredient_name: str,
    unit: Optional[str],
) -> Tuple[Optional[int], Optional[str], Optional[int], Optional[str], Optional[str]]:
    hh = (household or "").strip().lower()
    ing = (ingredient_name or "").strip()
    nunit = _normalize_unit(unit)

    if _table_exists(conn, "recipe_ingredient_mappings"):
        cols = set(_table_columns(conn, "recipe_ingredient_mappings"))
        needed = {"household", "ingredient_name", "normalized_unit", "grocy_product_id"}
        if needed.issubset(cols):
            cur = conn.cursor()
            cur.execute(
                """
                SELECT grocy_product_id, grocy_qu_id, note
                FROM recipe_ingredient_mappings
                WHERE lower(household) = lower(?)
                  AND lower(ingredient_name) = lower(?)
                  AND normalized_unit = ?
                LIMIT 1;
                """,
                (hh, ing, nunit),
            )
            row = cur.fetchone()
            if row:
                pid = row["grocy_product_id"] if "grocy_product_id" in row.keys() else None
                qu_id = row["grocy_qu_id"] if "grocy_qu_id" in row.keys() else None
                note = row["note"] if "note" in row.keys() else None
                return (
                    int(pid) if pid is not None else None,
                    None,
                    int(qu_id) if qu_id is not None else None,
                    "recipe_ingredient_mappings",
                    str(note) if isinstance(note, str) and note.strip() else None,
                )

    if _table_exists(conn, "recipe_ingredient_mapping"):
        cols = set(_table_columns(conn, "recipe_ingredient_mapping"))
        needed2 = {"ingredient_norm", "product_id"}
        if needed2.issubset(cols):
            ing_norm = _norm_ing(ing)
            if ing_norm:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT household, product_id, product_name
                    FROM recipe_ingredient_mapping
                    WHERE ingredient_norm = ?
                      AND (lower(household) = lower(?) OR household IS NULL)
                    ORDER BY household IS NOT NULL DESC
                    LIMIT 1;
                    """,
                    (ing_norm, hh),
                )
                row = cur.fetchone()
                if row:
                    pid = row["product_id"] if "product_id" in row.keys() else None
                    pname = row["product_name"] if "product_name" in row.keys() else None
                    return (
                        int(pid) if pid is not None else None,
                        str(pname) if isinstance(pname, str) and pname.strip() else None,
                        None,
                        "recipe_ingredient_mapping",
                        "mapping is normalized; unit not considered",
                    )

    return (None, None, None, None, None)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@router.post("", response_model=MealPlanResponse)
def create_meal_plan(req: MealPlanCreateRequest) -> MealPlanResponse:
    from main import DB_PATH  # local import to avoid circulars

    conn = _connect(DB_PATH)
    try:
        _ensure_schema(conn)

        plan_id = str(uuid.uuid4())
        created_at = _utc_now_iso()

        conn.execute(
            "INSERT INTO meal_plans (id, household, week_start, created_at) VALUES (?, ?, ?, ?);",
            (plan_id, req.household, req.week_start, created_at),
        )
        conn.commit()

        return MealPlanResponse(id=plan_id, household=req.household, week_start=req.week_start, created_at=created_at)
    finally:
        conn.close()


@router.get("", response_model=List[MealPlanResponse])
def list_meal_plans(household: Optional[str] = None, limit: int = 50) -> List[MealPlanResponse]:
    from main import DB_PATH

    conn = _connect(DB_PATH)
    try:
        _ensure_schema(conn)

        cur = conn.cursor()
        if household:
            cur.execute(
                "SELECT * FROM meal_plans WHERE household = ? ORDER BY week_start DESC, created_at DESC LIMIT ?;",
                (household, limit),
            )
        else:
            cur.execute("SELECT * FROM meal_plans ORDER BY week_start DESC, created_at DESC LIMIT ?;", (limit,))
        rows = cur.fetchall()

        return [MealPlanResponse(**dict(r)) for r in rows]
    finally:
        conn.close()


@router.get("/{meal_plan_id}", response_model=MealPlanResponse)
def get_meal_plan(meal_plan_id: str) -> MealPlanResponse:
    from main import DB_PATH

    conn = _connect(DB_PATH)
    try:
        _ensure_schema(conn)

        cur = conn.cursor()
        cur.execute("SELECT * FROM meal_plans WHERE id = ? LIMIT 1;", (meal_plan_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Meal plan not found")
        return MealPlanResponse(**dict(row))
    finally:
        conn.close()


@router.post("/{meal_plan_id}/items", response_model=MealPlanItemResponse)
def add_meal_plan_item(meal_plan_id: str, req: MealPlanItemCreateRequest) -> MealPlanItemResponse:
    from main import DB_PATH

    conn = _connect(DB_PATH)
    try:
        _ensure_schema(conn)

        cur = conn.cursor()
        cur.execute("SELECT id FROM meal_plans WHERE id = ? LIMIT 1;", (meal_plan_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Meal plan not found")

        _require_finalized_recipe(conn, req.recipe_finalized_id)

        item_id = str(uuid.uuid4())
        created_at = _utc_now_iso()

        conn.execute(
            """
            INSERT INTO meal_plan_items
            (id, meal_plan_id, day, slot, recipe_finalized_id, servings, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                item_id,
                meal_plan_id,
                req.day,
                req.slot,
                req.recipe_finalized_id,
                req.servings,
                req.notes,
                created_at,
            ),
        )
        conn.commit()

        return MealPlanItemResponse(
            id=item_id,
            meal_plan_id=meal_plan_id,
            day=req.day,
            slot=req.slot,
            recipe_finalized_id=req.recipe_finalized_id,
            servings=req.servings,
            notes=req.notes,
            created_at=created_at,
        )
    finally:
        conn.close()


@router.get("/{meal_plan_id}/items", response_model=List[MealPlanItemResponse])
def list_meal_plan_items(meal_plan_id: str) -> List[MealPlanItemResponse]:
    from main import DB_PATH

    conn = _connect(DB_PATH)
    try:
        _ensure_schema(conn)

        cur = conn.cursor()
        cur.execute("SELECT id FROM meal_plans WHERE id = ? LIMIT 1;", (meal_plan_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Meal plan not found")

        cur.execute(
            """
            SELECT * FROM meal_plan_items
            WHERE meal_plan_id = ?
            ORDER BY day ASC, slot ASC, created_at ASC;
            """,
            (meal_plan_id,),
        )
        rows = cur.fetchall()

        return [MealPlanItemResponse(**dict(r)) for r in rows]
    finally:
        conn.close()


@router.delete("/{meal_plan_id}/items/{item_id}")
def delete_meal_plan_item(meal_plan_id: str, item_id: str) -> Dict[str, Any]:
    from main import DB_PATH

    conn = _connect(DB_PATH)
    try:
        _ensure_schema(conn)

        cur = conn.cursor()
        cur.execute("SELECT id FROM meal_plans WHERE id = ? LIMIT 1;", (meal_plan_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Meal plan not found")

        cur.execute(
            "DELETE FROM meal_plan_items WHERE id = ? AND meal_plan_id = ?;",
            (item_id, meal_plan_id),
        )
        conn.commit()

        return {"ok": True, "deleted": cur.rowcount}
    finally:
        conn.close()


def _dedupe_warnings_keep_order(warnings: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


@router.get("/{meal_plan_id}/shopping/preview", response_model=ShoppingPreviewResponse)
def shopping_preview(meal_plan_id: str) -> ShoppingPreviewResponse:
    from main import DB_PATH

    conn = _connect(DB_PATH)
    try:
        _ensure_schema(conn)

        warnings: List[str] = []
        cur = conn.cursor()

        cur.execute("SELECT * FROM meal_plans WHERE id = ? LIMIT 1;", (meal_plan_id,))
        plan = cur.fetchone()
        if not plan:
            raise HTTPException(status_code=404, detail="Meal plan not found")

        household = (plan["household"] or "").strip().lower()

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
            recipe_title = _get_recipe_title(payload, recipe_row)

            ingredients = _extract_ingredients(payload)

            if not ingredients:
                warnings.append(f"Recipe {recipe_id} has no parseable ingredients in finalized payload.")
                continue

            scale = 1.0
            base_servings = _find_base_servings(payload)

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

                qty = ing.get("quantity")
                unit = (ing.get("unit") or "").strip() or None
                note = ing.get("note")

                scaled_qty = _scaled_qty(qty, scale) if scale != 1.0 else qty

                key = f"{name.lower()}|{unit or ''}"

                if key not in aggregated:
                    pid, pname, qu_id, msource, mnote = _lookup_mapping_hint(
                        conn,
                        household=household,
                        ingredient_name=name,
                        unit=unit,
                    )

                    aggregated[key] = {
                        "name": name,
                        "quantity": _coerce_float(scaled_qty) if _coerce_float(scaled_qty) is not None else None,
                        "unit": unit,
                        "note": note,
                        "sources": [],
                        "mapped_product_id": pid,
                        "mapped_product_name": pname,
                        "mapped_qu_id": qu_id,
                        "mapping_source": msource,
                        "mapping_note": mnote,
                    }
                else:
                    prev_qty = aggregated[key].get("quantity")
                    a = _coerce_float(prev_qty)
                    b = _coerce_float(scaled_qty)
                    if a is not None and b is not None:
                        aggregated[key]["quantity"] = round(a + b, 4)

                aggregated[key]["sources"].append(
                    {
                        "recipe_finalized_id": recipe_id,
                        "recipe_title": recipe_title,
                        "day": it["day"],
                        "slot": it["slot"],
                        "quantity": _coerce_float(scaled_qty) if _coerce_float(scaled_qty) is not None else None,
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
                mapped_product_id=v.get("mapped_product_id"),
                mapped_product_name=v.get("mapped_product_name"),
                mapped_qu_id=v.get("mapped_qu_id"),
                mapping_source=v.get("mapping_source"),
                mapping_note=v.get("mapping_note"),
                sources=[ShoppingPreviewSource(**s) for s in v.get("sources", [])],
            )
            for _, v in sorted(aggregated.items(), key=lambda kv: kv[1]["name"].lower())
        ]

        if not _table_exists(conn, "recipe_ingredient_mappings") and not _table_exists(conn, "recipe_ingredient_mapping"):
            warnings.append("No recipe mapping tables found; preview returned without mapping hints.")

        warnings = _dedupe_warnings_keep_order(warnings)

        return ShoppingPreviewResponse(
            meal_plan_id=meal_plan_id,
            household=plan["household"],
            week_start=plan["week_start"],
            lines=lines,
            warnings=warnings,
        )
    finally:
        conn.close()
