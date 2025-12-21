"""
/opt/jarvis/brain/services/recipes.py

Phase 6.75/6.8x — Recipes Service Fix (Option A), Full-File Replacement

Guardrails:
- No Grocy writes.
- No inventory changes.
- Recipe finalization persists a frozen snapshot to ISAC SQLite only.
- Ingredient mappings are ISAC-owned metadata (SQLite).

Option A Fix:
- Fix SQLite upsert by introducing normalized_unit (real column) + UNIQUE index
- Safe, idempotent migrations for existing installs
- Schema-adaptive draft inserts to survive historical DB drift:
    - raw_ingredient_lines (legacy, NOT NULL in some DBs)
    - parsed_ingredients (legacy, NOT NULL in some DBs)
    - raw_recipe_json (newer)
    - parsed_ingredients_json (newer)

Stability Fix:
- SQLite lock resilience:
    - busy_timeout
    - WAL mode
    - retry wrapper
    - avoid migrations at import time to reduce lock contention
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(tags=["recipes"])

# -----------------------------------------------------------------------------
# DB config + helpers
# -----------------------------------------------------------------------------

DEFAULT_DB_PATHS = [
    os.getenv("JARVIS_BRAIN_DB_PATH", "").strip(),
    "/app/data/jarvis_brain.db",               # container default
    "/opt/jarvis/brain-data/jarvis_brain.db",  # host fallback (dev)
]

_SQLITE_BUSY_TIMEOUT_MS = int(os.getenv("JARVIS_SQLITE_BUSY_TIMEOUT_MS", "5000"))
_SQLITE_RETRIES = int(os.getenv("JARVIS_SQLITE_RETRIES", "10"))
_SQLITE_RETRY_SLEEP_SEC = float(os.getenv("JARVIS_SQLITE_RETRY_SLEEP_SEC", "0.25"))


def _db_path() -> str:
    for p in DEFAULT_DB_PATHS:
        if p and os.path.exists(p):
            return p
    return os.getenv("JARVIS_BRAIN_DB_PATH", "").strip() or "/app/data/jarvis_brain.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=max(1.0, _SQLITE_BUSY_TIMEOUT_MS / 1000.0))
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _with_retry(fn, *, what: str) -> Any:
    last: Optional[Exception] = None
    for _ in range(_SQLITE_RETRIES):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            last = e
            msg = str(e).lower()
            if "database is locked" in msg or "database is busy" in msg:
                time.sleep(_SQLITE_RETRY_SLEEP_SEC)
                continue
            raise
    raise sqlite3.OperationalError(f"{what}: database remained locked after retries: {last}")


def _rv(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    """
    Safe value fetch for sqlite3.Row (does NOT support .get()).
    """
    try:
        return row[key]
    except Exception:
        return default


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [r["name"] for r in rows]


def _table_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return column in _table_columns(conn, table)


def _index_exists(conn: sqlite3.Connection, index_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
        (index_name,),
    ).fetchone()
    return bool(row)


def _normalize_unit(unit: Optional[str]) -> str:
    return (unit or "").strip().lower()


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, ddl_fragment: str) -> None:
    if _table_has_column(conn, table, col):
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl_fragment}")


# -----------------------------------------------------------------------------
# Migrations (safe + idempotent)
# -----------------------------------------------------------------------------

def _ensure_tables_exist(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS recipe_drafts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL DEFAULT 'draft',
            title TEXT NOT NULL DEFAULT '',
            raw_recipe_json TEXT NOT NULL DEFAULT '{}',
            raw_ingredient_lines TEXT NOT NULL DEFAULT '[]',
            parsed_ingredients TEXT NOT NULL DEFAULT '[]',
            parsed_ingredients_json TEXT NOT NULL DEFAULT '[]',
            resolved_ingredients_json TEXT NOT NULL DEFAULT '[]',
            resolution_status TEXT NOT NULL DEFAULT 'unresolved',
            finalized_household TEXT,
            finalized_at TEXT,
            created_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT ''
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS recipe_ingredient_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            household TEXT NOT NULL,
            ingredient_name TEXT NOT NULL,
            ingredient_unit TEXT,
            normalized_unit TEXT NOT NULL DEFAULT '',
            grocy_product_id INTEGER,
            grocy_qu_id INTEGER,
            note TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(household, ingredient_name, COALESCE(ingredient_unit, ''))
        );
        """
    )

    conn.commit()


def _migrate_recipe_drafts(conn: sqlite3.Connection) -> None:
    def _do():
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='recipe_drafts'"
        ).fetchone()
        if not exists:
            return

        _ensure_column(conn, "recipe_drafts", "status", "TEXT NOT NULL DEFAULT 'draft'")
        _ensure_column(conn, "recipe_drafts", "title", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(conn, "recipe_drafts", "raw_recipe_json", "TEXT NOT NULL DEFAULT '{}'")
        _ensure_column(conn, "recipe_drafts", "raw_ingredient_lines", "TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(conn, "recipe_drafts", "parsed_ingredients", "TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(conn, "recipe_drafts", "parsed_ingredients_json", "TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(conn, "recipe_drafts", "resolved_ingredients_json", "TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(conn, "recipe_drafts", "resolution_status", "TEXT NOT NULL DEFAULT 'unresolved'")
        _ensure_column(conn, "recipe_drafts", "finalized_household", "TEXT")
        _ensure_column(conn, "recipe_drafts", "finalized_at", "TEXT")
        _ensure_column(conn, "recipe_drafts", "created_at", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(conn, "recipe_drafts", "updated_at", "TEXT NOT NULL DEFAULT ''")

        conn.commit()

    _with_retry(_do, what="migrate recipe_drafts")


def _migrate_recipe_ingredient_mappings(conn: sqlite3.Connection) -> None:
    def _do():
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='recipe_ingredient_mappings'"
        ).fetchone()
        if not exists:
            return

        _ensure_column(conn, "recipe_ingredient_mappings", "normalized_unit", "TEXT NOT NULL DEFAULT ''")

        conn.execute(
            """
            UPDATE recipe_ingredient_mappings
            SET normalized_unit = lower(trim(COALESCE(ingredient_unit, '')))
            WHERE normalized_unit IS NULL OR normalized_unit = ''
            """
        )

        idx_name = "uq_recipe_ingredient_mappings_hh_name_nunit"
        if not _index_exists(conn, idx_name):
            conn.execute(
                f"""
                CREATE UNIQUE INDEX {idx_name}
                ON recipe_ingredient_mappings (household, ingredient_name, normalized_unit)
                """
            )

        conn.commit()

    _with_retry(_do, what="migrate recipe_ingredient_mappings")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    _with_retry(lambda: _ensure_tables_exist(conn), what="ensure tables exist")
    _migrate_recipe_drafts(conn)
    _migrate_recipe_ingredient_mappings(conn)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class RecipeDraftIn(BaseModel):
    title: str
    ingredient_lines: List[str] = Field(default_factory=list)
    instructions: Optional[str] = None
    servings: Optional[str] = None
    source_url: Optional[str] = None


class CreateDraftRequest(BaseModel):
    title: str
    ingredient_lines: List[str] = Field(default_factory=list)
    instructions: Optional[str] = None
    servings: Optional[str] = None
    source_url: Optional[str] = None


class CreateDraftResponse(BaseModel):
    draft_id: int
    recipe: RecipeDraftIn
    parsed_ingredients: List[Dict[str, Any]]


class FinalizeRequest(BaseModel):
    household: str = Field(..., description="Household key, e.g. home_a")


class FinalizedSnapshot(BaseModel):
    draft_id: int
    household: str
    raw_recipe: Dict[str, Any]
    parsed_ingredients: List[Dict[str, Any]]
    resolved_ingredients: List[Dict[str, Any]]
    resolution_status: str
    metadata: Dict[str, Any]


class FinalizeResponse(BaseModel):
    draft_id: int
    status: str
    finalized_at: Optional[str]
    finalized_household: Optional[str]
    snapshot: FinalizedSnapshot
    notes: List[str] = Field(default_factory=list)


class AnalyzeIngredientsRequest(BaseModel):
    ingredient_lines: List[str] = Field(default_factory=list)


class AnalyzeUrlRequest(BaseModel):
    url: str


class IngredientMapping(BaseModel):
    household: str
    ingredient_name: str
    ingredient_unit: Optional[str] = None
    grocy_product_id: Optional[int] = None
    grocy_qu_id: Optional[int] = None
    note: Optional[str] = None


class MappingSetRequest(IngredientMapping):
    pass


class MappingDeleteRequest(BaseModel):
    household: str
    ingredient_name: str
    ingredient_unit: Optional[str] = None


class MatchProductsRequest(BaseModel):
    household: Optional[str] = None
    ingredient_names: List[str] = Field(default_factory=list)
    max_results: int = 10


# -----------------------------------------------------------------------------
# Ingredient parsing (simple deterministic)
# -----------------------------------------------------------------------------

_UNIT_ALIASES = {
    "tbsp": "tablespoon",
    "tbsps": "tablespoon",
    "tablespoons": "tablespoon",
    "tablespoon": "tablespoon",
    "tsp": "teaspoon",
    "tsps": "teaspoon",
    "teaspoons": "teaspoon",
    "teaspoon": "teaspoon",
    "cup": "cup",
    "cups": "cup",
    "oz": "ounce",
    "ounce": "ounce",
    "ounces": "ounce",
    "lb": "pound",
    "lbs": "pound",
    "pound": "pound",
    "pounds": "pound",
    "g": "gram",
    "gram": "gram",
    "grams": "gram",
    "kg": "kilogram",
    "kilogram": "kilogram",
    "kilograms": "kilogram",
    "ml": "milliliter",
    "milliliter": "milliliter",
    "milliliters": "milliliter",
    "l": "liter",
    "liter": "liter",
    "liters": "liter",
    "pinch": "pinch",
    "pinches": "pinch",
    "clove": "clove",
    "cloves": "clove",
}

_QTY_RE = re.compile(
    r"""
    ^\s*
    (?P<qty>
        (?:\d+(?:\.\d+)?)
        (?:\s+\d+/\d+)?
        |
        (?:\d+/\d+)
    )
    (?:\s*[-–]\s*
        (?P<qty2>
            (?:\d+(?:\.\d+)?)
            (?:\s+\d+/\d+)?
            |
            (?:\d+/\d+)
        )
    )?
    \s*
    (?P<rest>.*)
    $
    """,
    re.VERBOSE,
)


def _parse_number_token(tok: str) -> Optional[float]:
    tok = tok.strip()
    if not tok:
        return None

    if " " in tok:
        parts = tok.split()
        if len(parts) == 2:
            whole = _parse_number_token(parts[0])
            frac = _parse_number_token(parts[1])
            if whole is not None and frac is not None:
                return whole + frac

    if "/" in tok:
        try:
            num, den = tok.split("/", 1)
            return float(num) / float(den)
        except Exception:
            return None

    try:
        return float(tok)
    except Exception:
        return None


def parse_ingredient_line(line: str) -> Dict[str, Any]:
    original = (line or "").strip()
    if not original:
        return {
            "original": line,
            "qty": None,
            "unit": None,
            "name": "",
            "preparation": None,
            "notes": [],
            "confidence": "low",
        }

    qty_min = qty_max = None
    rest = original
    raw_qty = None

    m = _QTY_RE.match(original)
    if m:
        raw_qty = m.group("qty")
        qty_min = _parse_number_token(m.group("qty")) if m.group("qty") else None
        qty_max = _parse_number_token(m.group("qty2")) if m.group("qty2") else None
        if qty_min is not None and qty_max is None:
            qty_max = qty_min
        rest = (m.group("rest") or "").strip()

    unit = None
    name_part = rest

    tokens = rest.split()
    if tokens:
        first = re.sub(r"[^\w/]+", "", tokens[0].lower())
        if first in _UNIT_ALIASES:
            unit = _UNIT_ALIASES[first]
            name_part = " ".join(tokens[1:]).strip()

    name_part = name_part.strip(" ,")
    confidence = "high" if name_part else "low"
    if qty_min is None and unit is None:
        confidence = "medium" if name_part else "low"

    return {
        "original": original,
        "qty": None if qty_min is None else {"min": qty_min, "max": qty_max, "raw": raw_qty},
        "unit": unit,
        "name": name_part or original,
        "preparation": None,
        "notes": [],
        "confidence": confidence,
    }


# -----------------------------------------------------------------------------
# Grocy helpers (read-only)
# -----------------------------------------------------------------------------

@dataclass
class GrocyConfig:
    base_url: str
    api_key: str


def _grocy_config_for_household(household: Optional[str]) -> Optional[GrocyConfig]:
    household_norm = (household or "").strip().upper()
    if household_norm:
        base = os.getenv(f"GROCY_BASE_URL_{household_norm}")
        key = os.getenv(f"GROCY_API_KEY_{household_norm}") or os.getenv(f"GROCY_TOKEN_{household_norm}")
        if base and key:
            return GrocyConfig(base_url=base.rstrip("/"), api_key=key)

    base = os.getenv("GROCY_BASE_URL")
    key = os.getenv("GROCY_API_KEY") or os.getenv("GROCY_TOKEN")
    if base and key:
        return GrocyConfig(base_url=base.rstrip("/"), api_key=key)

    return None


def _grocy_get(cfg: GrocyConfig, path: str) -> Any:
    url = f"{cfg.base_url.rstrip('/')}{path}"
    headers = {"GROCY-API-KEY": cfg.api_key}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


# -----------------------------------------------------------------------------
# Row JSON helper
# -----------------------------------------------------------------------------

def _row_json(row: sqlite3.Row, keys: List[str], default: Any) -> Any:
    for k in keys:
        val = _rv(row, k, None)
        if val is None:
            continue
        if isinstance(val, (bytes, bytearray)):
            val = val.decode("utf-8", "ignore")
        if isinstance(val, str) and val.strip() == "":
            continue
        try:
            return json.loads(val) if isinstance(val, str) else val
        except Exception:
            continue
    return default


def _build_snapshot_from_row(row: sqlite3.Row, household: str) -> "FinalizedSnapshot":
    raw_recipe = _row_json(row, ["raw_recipe_json"], default={})
    parsed = _row_json(row, ["parsed_ingredients_json", "parsed_ingredients"], default=[])
    resolved = _row_json(row, ["resolved_ingredients_json"], default=[]) or parsed
    resolution_status = _rv(row, "resolution_status", "unresolved")

    metadata = {
        "finalized_by": "isac",
        "finalized_at": _rv(row, "finalized_at", None),
        "notes": [
            "Frozen recipe snapshot for meal planning",
            "No ingredient mappings applied",
            "No Grocy writes",
            "No inventory changes",
            "ISAC-owned data only",
        ],
    }

    return FinalizedSnapshot(
        draft_id=int(_rv(row, "id", 0)),
        household=household,
        raw_recipe=raw_recipe,
        parsed_ingredients=parsed,
        resolved_ingredients=resolved,
        resolution_status=resolution_status or "unresolved",
        metadata=metadata,
    )


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@router.post("/recipes/draft", response_model=CreateDraftResponse)
def create_recipe_draft(req: CreateDraftRequest) -> CreateDraftResponse:
    now = _utc_iso()

    recipe = RecipeDraftIn(
        title=req.title,
        ingredient_lines=req.ingredient_lines,
        instructions=req.instructions,
        servings=req.servings,
        source_url=req.source_url,
    )

    parsed = [parse_ingredient_line(x) for x in (req.ingredient_lines or [])]

    raw_recipe_payload = {
        "title": req.title,
        "ingredient_lines": req.ingredient_lines or [],
        "instructions": req.instructions,
        "servings": req.servings,
        "source_url": req.source_url,
    }
    raw_json = json.dumps(raw_recipe_payload)
    raw_lines_json = json.dumps(req.ingredient_lines or [])
    parsed_json = json.dumps(parsed)

    conn = _conn()
    try:
        _ensure_schema(conn)
        cur = conn.cursor()

        cols = _table_columns(conn, "recipe_drafts")

        columns: List[str] = ["status", "title"]
        values: List[Any] = ["draft", req.title]

        if "raw_recipe_json" in cols:
            columns.append("raw_recipe_json")
            values.append(raw_json)

        if "raw_ingredient_lines" in cols:
            columns.append("raw_ingredient_lines")
            values.append(raw_lines_json)

        if "parsed_ingredients" in cols:
            columns.append("parsed_ingredients")
            values.append(parsed_json)

        if "parsed_ingredients_json" in cols:
            columns.append("parsed_ingredients_json")
            values.append(parsed_json)

        if "created_at" in cols:
            columns.append("created_at")
            values.append(now)
        if "updated_at" in cols:
            columns.append("updated_at")
            values.append(now)

        placeholders = ", ".join(["?"] * len(values))
        sql = f"INSERT INTO recipe_drafts ({', '.join(columns)}) VALUES ({placeholders})"

        def _do_insert():
            cur.execute(sql, tuple(values))

        _with_retry(_do_insert, what="insert recipe_draft")
        draft_id = int(cur.lastrowid)
        conn.commit()

        return CreateDraftResponse(draft_id=draft_id, recipe=recipe, parsed_ingredients=parsed)

    finally:
        conn.close()


@router.post("/recipes/draft/{draft_id}/finalize", response_model=FinalizeResponse)
def finalize_recipe_draft(draft_id: int, req: FinalizeRequest) -> FinalizeResponse:
    conn = _conn()
    try:
        _ensure_schema(conn)
        cur = conn.cursor()

        row = cur.execute("SELECT * FROM recipe_drafts WHERE id = ?", (draft_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Recipe draft not found for id={draft_id}")

        if _rv(row, "status") == "finalized":
            household = _rv(row, "finalized_household", req.household) or req.household
            snapshot = _build_snapshot_from_row(row, household=household)
            return FinalizeResponse(
                draft_id=draft_id,
                status="finalized",
                finalized_at=_rv(row, "finalized_at", None),
                finalized_household=_rv(row, "finalized_household", None),
                snapshot=snapshot,
                notes=["Read-only finalized snapshot", "No Grocy writes", "No inventory changes"],
            )

        parsed = _row_json(row, ["parsed_ingredients_json", "parsed_ingredients"], default=[])
        resolved = parsed
        resolution_status = "unresolved"
        finalized_at = _utc_iso()

        cols = _table_columns(conn, "recipe_drafts")

        set_parts = []
        params: List[Any] = []

        if "status" in cols:
            set_parts.append("status = ?")
            params.append("finalized")
        if "resolved_ingredients_json" in cols:
            set_parts.append("resolved_ingredients_json = ?")
            params.append(json.dumps(resolved))
        if "resolution_status" in cols:
            set_parts.append("resolution_status = ?")
            params.append(resolution_status)
        if "finalized_household" in cols:
            set_parts.append("finalized_household = ?")
            params.append(req.household)
        if "finalized_at" in cols:
            set_parts.append("finalized_at = ?")
            params.append(finalized_at)
        if "updated_at" in cols:
            set_parts.append("updated_at = ?")
            params.append(finalized_at)

        if not set_parts:
            raise HTTPException(status_code=500, detail="recipe_drafts schema missing expected columns for finalize")

        params.append(draft_id)

        sql = f"UPDATE recipe_drafts SET {', '.join(set_parts)} WHERE id = ?"

        def _do_update():
            cur.execute(sql, tuple(params))

        _with_retry(_do_update, what="finalize recipe_draft")
        conn.commit()

        row2 = cur.execute("SELECT * FROM recipe_drafts WHERE id = ?", (draft_id,)).fetchone()
        if not row2:
            raise HTTPException(status_code=500, detail="Finalize failed unexpectedly (draft disappeared).")

        snapshot = _build_snapshot_from_row(row2, household=req.household)

        return FinalizeResponse(
            draft_id=draft_id,
            status="finalized",
            finalized_at=finalized_at,
            finalized_household=req.household,
            snapshot=snapshot,
            notes=[
                "Recipe draft finalized",
                "Frozen snapshot persisted to ISAC SQLite",
                "No Grocy writes",
                "No inventory changes",
            ],
        )
    finally:
        conn.close()


@router.get("/recipes/draft/{draft_id}/finalized", response_model=FinalizeResponse)
def get_finalized_snapshot(draft_id: int) -> FinalizeResponse:
    conn = _conn()
    try:
        _ensure_schema(conn)
        cur = conn.cursor()
        row = cur.execute("SELECT * FROM recipe_drafts WHERE id = ?", (draft_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Recipe draft not found for id={draft_id}")
        if _rv(row, "status") != "finalized":
            raise HTTPException(status_code=400, detail=f"Recipe draft id={draft_id} is not finalized yet")

        household = _rv(row, "finalized_household", "unknown") or "unknown"
        snapshot = _build_snapshot_from_row(row, household=household)

        return FinalizeResponse(
            draft_id=draft_id,
            status="finalized",
            finalized_at=_rv(row, "finalized_at", None),
            finalized_household=_rv(row, "finalized_household", None),
            snapshot=snapshot,
            notes=["Read-only finalized snapshot", "No Grocy writes", "No inventory changes"],
        )
    finally:
        conn.close()


@router.post("/recipes/analyze/ingredients")
def analyze_ingredients(req: AnalyzeIngredientsRequest) -> Dict[str, Any]:
    parsed = [parse_ingredient_line(x) for x in (req.ingredient_lines or [])]
    return {"parsed_ingredients": parsed}


@router.post("/recipes/analyze/url")
def analyze_url(req: AnalyzeUrlRequest) -> Dict[str, Any]:
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    try:
        r = requests.get(url, timeout=25, headers={"User-Agent": "ISAC/recipes"})
        r.raise_for_status()
        html = r.text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    title = None
    m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()

    text = re.sub(r"<[^>]+>", "\n", html)
    text = re.sub(r"\r", "\n", text)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]

    ingredient_lines: List[str] = []
    for ln in lines:
        if _QTY_RE.match(ln):
            ingredient_lines.append(ln)
        elif ln.startswith(("-", "•")) and len(ln) < 200:
            ingredient_lines.append(ln.lstrip("-• ").strip())
        if len(ingredient_lines) >= 30:
            break

    parsed = [parse_ingredient_line(x) for x in ingredient_lines]

    return {
        "url": url,
        "title": title,
        "ingredient_lines": ingredient_lines,
        "parsed_ingredients": parsed,
        "notes": ["Heuristic extraction (best-effort)", "No Grocy writes", "No inventory changes"],
    }


@router.get("/recipes/mappings")
def list_mappings(household: str = Query(..., description="Household key, e.g. home_a")) -> Dict[str, Any]:
    conn = _conn()
    try:
        _ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT household, ingredient_name, ingredient_unit, normalized_unit,
                   grocy_product_id, grocy_qu_id, note, created_at, updated_at
            FROM recipe_ingredient_mappings
            WHERE household = ?
            ORDER BY ingredient_name ASC, normalized_unit ASC
            """,
            (household,),
        ).fetchall()
        return {"household": household, "mappings": [dict(r) for r in rows]}
    finally:
        conn.close()


@router.post("/recipes/mappings/set")
def set_mapping(req: MappingSetRequest) -> Dict[str, Any]:
    now = _utc_iso()
    household = req.household.strip()
    name = req.ingredient_name.strip()
    unit_raw = (req.ingredient_unit or "").strip()
    unit = unit_raw if unit_raw else None
    normalized_unit = _normalize_unit(unit_raw)

    if not household:
        raise HTTPException(status_code=400, detail="household is required")
    if not name:
        raise HTTPException(status_code=400, detail="ingredient_name is required")

    conn = _conn()
    try:
        _ensure_schema(conn)
        cur = conn.cursor()

        def _do():
            cur.execute(
                """
                INSERT INTO recipe_ingredient_mappings
                    (household, ingredient_name, ingredient_unit, normalized_unit,
                     grocy_product_id, grocy_qu_id, note, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(household, ingredient_name, normalized_unit)
                DO UPDATE SET
                    ingredient_unit  = excluded.ingredient_unit,
                    grocy_product_id = excluded.grocy_product_id,
                    grocy_qu_id      = excluded.grocy_qu_id,
                    note             = excluded.note,
                    updated_at       = excluded.updated_at
                """,
                (household, name, unit, normalized_unit, req.grocy_product_id, req.grocy_qu_id, req.note, now, now),
            )

        _with_retry(_do, what="upsert recipe mapping")
        conn.commit()

        return {"ok": True, "mapping": req.dict(), "notes": ["Mapping stored in ISAC SQLite (no Grocy writes)"]}
    finally:
        conn.close()


@router.post("/recipes/mappings/delete")
def delete_mapping(req: MappingDeleteRequest) -> Dict[str, Any]:
    household = req.household.strip()
    name = req.ingredient_name.strip()
    unit_raw = (req.ingredient_unit or "").strip()
    normalized_unit = _normalize_unit(unit_raw)

    if not household:
        raise HTTPException(status_code=400, detail="household is required")
    if not name:
        raise HTTPException(status_code=400, detail="ingredient_name is required")

    conn = _conn()
    try:
        _ensure_schema(conn)
        cur = conn.cursor()

        def _do():
            cur.execute(
                """
                DELETE FROM recipe_ingredient_mappings
                WHERE household = ?
                  AND ingredient_name = ?
                  AND normalized_unit = ?
                """,
                (household, name, normalized_unit),
            )

        _with_retry(_do, what="delete recipe mapping")
        deleted = cur.rowcount
        conn.commit()
        return {"ok": True, "deleted": deleted}
    finally:
        conn.close()


@router.post("/recipes/match-products")
def match_products(req: MatchProductsRequest) -> Dict[str, Any]:
    cfg = _grocy_config_for_household(req.household)
    if not cfg:
        raise HTTPException(
            status_code=400,
            detail="Grocy config not found in env (expected GROCY_BASE_URL + GROCY_API_KEY, optionally per-household)",
        )

    try:
        products = _grocy_get(cfg, "/api/objects/products")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch Grocy products: {e}")

    normalized = []
    for p in products or []:
        pn = (p.get("name") or "").strip()
        if pn:
            normalized.append({"id": p.get("id"), "name": pn})

    results: Dict[str, List[Dict[str, Any]]] = {}
    for ing in req.ingredient_names or []:
        key = (ing or "").strip()
        if not key:
            continue
        needle = key.lower()
        hits = [p for p in normalized if needle in p["name"].lower()]
        if not hits:
            first = needle.split()[0]
            hits = [p for p in normalized if first and first in p["name"].lower()]
        results[key] = hits[: max(1, min(req.max_results, 50))]

    return {
        "household": req.household,
        "matches": results,
        "notes": [
            "Read-only Grocy product lookup",
            "Contains-based matching (simple heuristic)",
            "No Grocy writes",
            "No inventory changes",
        ],
    }
