from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, HttpUrl

router = APIRouter(prefix="/recipes", tags=["Recipes"])

# ============================================================
# Database helpers (Phase 6.75 — Recipe Draft + Finalization)
# ============================================================

DB_PATH = os.getenv("JARVIS_DB_PATH", "/app/data/jarvis_brain.db")


def _get_db_conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = {row["name"] for row in cur.fetchall()}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def _init_recipe_drafts_table() -> None:
    conn = _get_db_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recipe_drafts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                servings TEXT,
                raw_ingredient_lines TEXT NOT NULL,
                parsed_ingredients TEXT NOT NULL,
                instructions TEXT,
                source_url TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        # Phase 6.75 — Step 3.4 (idempotent finalization columns)
        _ensure_column(conn, "recipe_drafts", "status", "status TEXT")
        _ensure_column(conn, "recipe_drafts", "finalized_at", "finalized_at TEXT")
        _ensure_column(conn, "recipe_drafts", "finalized_household", "finalized_household TEXT")
        _ensure_column(conn, "recipe_drafts", "finalized_payload", "finalized_payload TEXT")

        conn.commit()
    finally:
        conn.close()


_init_recipe_drafts_table()

# ============================================================
# Models
# ============================================================


class ImportRecipeUrlRequest(BaseModel):
    url: HttpUrl


class RawRecipe(BaseModel):
    title: Optional[str] = None
    servings: Optional[str] = None
    ingredient_lines: List[str] = Field(default_factory=list)
    instructions: Optional[str] = None
    source_url: Optional[str] = None


class ImportRecipeUrlResponse(BaseModel):
    status: str
    warnings: List[str] = Field(default_factory=list)
    recipe: Optional[RawRecipe] = None


class RecipeDraftCreateRequest(BaseModel):
    title: Optional[str] = None
    servings: Optional[str] = None
    ingredient_lines: List[str]
    instructions: Optional[str] = None
    source_url: Optional[str] = None


class RecipeDraftResponse(BaseModel):
    draft_id: int
    recipe: RawRecipe
    parsed_ingredients: List[Dict[str, Any]]


Household = Literal["home_a", "home_b"]


class RecipeDraftAnalyzeRequest(BaseModel):
    household: Household
    top_k: int = Field(5, ge=1, le=10)
    min_score: float = Field(0.0, ge=0.0, le=1.0)


class RecipeDraftAnalyzeResponse(BaseModel):
    draft_id: int
    recipe: RawRecipe
    parsed_ingredients: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    notes: List[str] = Field(default_factory=list)


class MappingConfirmItem(BaseModel):
    ingredient_name: str
    product_id: int
    product_name: Optional[str] = None
    notes: Optional[str] = None


class RecipeDraftConfirmMappingsRequest(BaseModel):
    household: Optional[Household] = None
    items: List[MappingConfirmItem]
    commit: bool = False


class RecipeDraftConfirmMappingsResponse(BaseModel):
    status: Literal["ok"] = "ok"
    draft_id: int
    commit: bool
    attempted: int
    succeeded: int
    failed: int
    results: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class RecipeDraftFinalizeRequest(BaseModel):
    household: Household
    commit: bool = False


class RecipeDraftFinalizedResponse(BaseModel):
    draft_id: int
    status: str
    finalized_at: Optional[str]
    finalized_household: Optional[str]
    snapshot: Dict[str, Any]
    notes: List[str] = Field(default_factory=list)

# ============================================================
# Utility helpers
# ============================================================


def _strip_html(s: str) -> str:
    s = re.sub(r"<\s*br\s*/?\s*>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"</p\s*>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_title_from_html(html: str) -> Optional[str]:
    m = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        return _strip_html(m.group(1))
    return None


async def _fetch_html(url: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            r = await client.get(url, headers={"User-Agent": "ISAC/RecipeImporter"})
            r.raise_for_status()
            return r.text, None
    except Exception as e:
        return None, str(e)


async def _post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"http://127.0.0.1:8000{path}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload)
        try:
            data = resp.json()
        except Exception:
            data = {"detail": resp.text}

    if resp.status_code >= 400:
        return {"error": True, "detail": data}

    return data

# ============================================================
# Draft persistence helpers
# ============================================================


def _store_recipe_draft(raw: RawRecipe, parsed: List[Dict[str, Any]]) -> int:
    conn = _get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO recipe_drafts (
                title,
                servings,
                raw_ingredient_lines,
                parsed_ingredients,
                instructions,
                source_url,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw.title,
                raw.servings,
                json.dumps(raw.ingredient_lines),
                json.dumps(parsed),
                raw.instructions,
                raw.source_url,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _load_recipe_draft_row(draft_id: int) -> sqlite3.Row:
    conn = _get_db_conn()
    try:
        cur = conn.execute("SELECT * FROM recipe_drafts WHERE id = ?", (draft_id,))
        row = cur.fetchone()
        if not row:
            raise KeyError("draft_not_found")
        return row
    finally:
        conn.close()


def _load_recipe_draft(draft_id: int) -> Tuple[RawRecipe, List[Dict[str, Any]]]:
    row = _load_recipe_draft_row(draft_id)
    raw = RawRecipe(
        title=row["title"],
        servings=row["servings"],
        ingredient_lines=json.loads(row["raw_ingredient_lines"]),
        instructions=row["instructions"],
        source_url=row["source_url"],
    )
    parsed = json.loads(row["parsed_ingredients"])
    return raw, parsed

# ============================================================
# Finalization helpers (Phase 6.75 — Step 3.4)
# ============================================================


async def _resolve_ingredients(
    household: Household, parsed: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Attempt to resolve ingredient mappings.
    If resolution is unavailable or fails, gracefully fall back
    to parsed ingredients with explicit metadata.
    """
    resp = await _post_json(
        "/recipes/mappings/resolve",
        {"household": household, "parsed_ingredients": parsed},
    )

    if resp.get("error"):
        return {
            "resolution_status": "unresolved",
            "ingredients": parsed,
            "notes": ["No ingredient mappings applied"],
        }

    return {
        "resolution_status": "resolved",
        "ingredients": resp.get("resolved", resp),
        "notes": ["Ingredient mappings applied with household precedence"],
    }


def _build_final_snapshot(
    draft_id: int,
    household: Household,
    raw: RawRecipe,
    parsed: List[Dict[str, Any]],
    resolved_block: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "draft_id": draft_id,
        "household": household,
        "raw_recipe": raw.model_dump(),
        "parsed_ingredients": parsed,
        "resolved_ingredients": resolved_block["ingredients"],
        "resolution_status": resolved_block["resolution_status"],
        "metadata": {
            "finalized_by": "isac",
            "finalized_at": datetime.now(timezone.utc).isoformat(),
            "notes": [
                "Frozen recipe snapshot for meal planning",
                *resolved_block.get("notes", []),
                "No Grocy writes",
                "No inventory changes",
                "ISAC-owned data only",
            ],
        },
    }


def _persist_finalization(
    draft_id: int, household: Household, snapshot: Dict[str, Any]
) -> Tuple[str, str]:
    conn = _get_db_conn()
    try:
        ts = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            UPDATE recipe_drafts
            SET status = ?,
                finalized_at = ?,
                finalized_household = ?,
                finalized_payload = ?
            WHERE id = ?
            """,
            ("finalized", ts, household, json.dumps(snapshot), draft_id),
        )
        conn.commit()
        return "finalized", ts
    finally:
        conn.close()

# ============================================================
# Endpoints
# ============================================================


@router.post("/draft", response_model=RecipeDraftResponse)
async def create_recipe_draft(payload: RecipeDraftCreateRequest) -> RecipeDraftResponse:
    from services.ingredient_parser import parse_ingredient_lines

    raw = RawRecipe(
        title=payload.title,
        servings=payload.servings,
        ingredient_lines=payload.ingredient_lines,
        instructions=payload.instructions,
        source_url=payload.source_url,
    )

    parsed = parse_ingredient_lines(payload.ingredient_lines)
    draft_id = _store_recipe_draft(raw, parsed)

    return RecipeDraftResponse(draft_id=draft_id, recipe=raw, parsed_ingredients=parsed)


@router.post(
    "/draft/{draft_id}/finalize",
    response_model=RecipeDraftFinalizedResponse,
)
async def finalize_recipe_draft(
    draft_id: int, body: RecipeDraftFinalizeRequest
) -> RecipeDraftFinalizedResponse:
    row = _load_recipe_draft_row(draft_id)
    raw, parsed = _load_recipe_draft(draft_id)

    resolved_block = await _resolve_ingredients(body.household, parsed)
    snapshot = _build_final_snapshot(
        draft_id, body.household, raw, parsed, resolved_block
    )

    if not body.commit:
        return RecipeDraftFinalizedResponse(
            draft_id=draft_id,
            status=row["status"] or "draft",
            finalized_at=None,
            finalized_household=None,
            snapshot=snapshot,
            notes=[
                "Preview only",
                "No database writes",
                "No Grocy writes",
            ],
        )

    status, ts = _persist_finalization(draft_id, body.household, snapshot)

    return RecipeDraftFinalizedResponse(
        draft_id=draft_id,
        status=status,
        finalized_at=ts,
        finalized_household=body.household,
        snapshot=snapshot,
        notes=[
            "Recipe draft finalized",
            "Frozen snapshot persisted to ISAC SQLite",
            "No Grocy writes",
            "No inventory changes",
        ],
    )


@router.get(
    "/draft/{draft_id}/finalized",
    response_model=RecipeDraftFinalizedResponse,
)
async def get_finalized_recipe_draft(draft_id: int) -> RecipeDraftFinalizedResponse:
    row = _load_recipe_draft_row(draft_id)

    if not row["finalized_payload"]:
        raise HTTPException(status_code=404, detail="Draft not finalized")

    snapshot = json.loads(row["finalized_payload"])

    return RecipeDraftFinalizedResponse(
        draft_id=draft_id,
        status=row["status"],
        finalized_at=row["finalized_at"],
        finalized_household=row["finalized_household"],
        snapshot=snapshot,
        notes=[
            "Read-only finalized snapshot",
            "No Grocy writes",
            "No inventory changes",
        ],
    )
