from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter
from pydantic import BaseModel, Field, HttpUrl

router = APIRouter(prefix="/recipes", tags=["Recipes"])

# ============================================================
# Database helpers (Phase 6.75 — Recipe Draft Persistence)
# ============================================================

DB_PATH = os.getenv("JARVIS_DB_PATH", "/app/data/jarvis_brain.db")


def _get_db_conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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
        conn.commit()
    finally:
        conn.close()


_init_recipe_drafts_table()

# ============================================================
# Models
# ============================================================


class ImportRecipeUrlRequest(BaseModel):
    url: HttpUrl = Field(..., description="Recipe page URL to import")


class RawRecipe(BaseModel):
    title: Optional[str] = None
    servings: Optional[str] = None
    ingredient_lines: List[str] = Field(default_factory=list)
    instructions: Optional[str] = None
    source_url: Optional[str] = None


class ImportRecipeUrlResponse(BaseModel):
    status: str = Field(..., description="complete | partial | error")
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

# ============================================================
# Utility helpers
# ============================================================


def _coerce_to_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


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


def _find_jsonld_blocks(html: str) -> List[str]:
    blocks = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return [b.strip() for b in blocks if b.strip()]


def _find_recipe_in_jsonld(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        t = obj.get("@type") or obj.get("type")
        if isinstance(t, list) and any(str(x).lower() == "recipe" for x in t):
            return obj
        if isinstance(t, str) and t.lower() == "recipe":
            return obj

        graph = obj.get("@graph")
        if isinstance(graph, list):
            for node in graph:
                r = _find_recipe_in_jsonld(node)
                if r:
                    return r

        for v in obj.values():
            r = _find_recipe_in_jsonld(v)
            if r:
                return r

    if isinstance(obj, list):
        for item in obj:
            r = _find_recipe_in_jsonld(item)
            if r:
                return r

    return None


def _normalize_instructions(recipe_node: Dict[str, Any]) -> Optional[str]:
    ins = recipe_node.get("recipeInstructions")
    if ins is None:
        return None

    parts: List[str] = []

    for item in _coerce_to_list(ins):
        if isinstance(item, str):
            parts.append(_strip_html(item))
        elif isinstance(item, dict):
            text = item.get("text") or item.get("name")
            if isinstance(text, str):
                parts.append(_strip_html(text))

    if not parts:
        return None

    return "\n".join(f"{i+1}. {p}" for i, p in enumerate(parts))


def _normalize_ingredients(recipe_node: Dict[str, Any]) -> List[str]:
    ings = recipe_node.get("recipeIngredient") or recipe_node.get("ingredients")
    out: List[str] = []
    for item in _coerce_to_list(ings):
        if isinstance(item, str):
            s = _strip_html(item)
            if s:
                out.append(s)
    return out


def _extract_servings(recipe_node: Dict[str, Any]) -> Optional[str]:
    y = recipe_node.get("recipeYield") or recipe_node.get("yield")
    if isinstance(y, (int, float)):
        return str(y)
    if isinstance(y, str):
        return y.strip()
    if isinstance(y, list) and y:
        return str(y[0])
    return None


async def _fetch_html(url: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            r = await client.get(
                url, headers={"User-Agent": "ISAC/RecipeImporter"}
            )
            r.raise_for_status()
            return r.text, None
    except Exception as e:
        return None, str(e)

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


def _load_recipe_draft(draft_id: int) -> Tuple[RawRecipe, List[Dict[str, Any]]]:
    conn = _get_db_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM recipe_drafts WHERE id = ?", (draft_id,))
        row = cur.fetchone()
        if not row:
            raise KeyError("draft_not_found")

        raw = RawRecipe(
            title=row["title"],
            servings=row["servings"],
            ingredient_lines=json.loads(row["raw_ingredient_lines"]),
            instructions=row["instructions"],
            source_url=row["source_url"],
        )

        parsed = json.loads(row["parsed_ingredients"])
        return raw, parsed
    finally:
        conn.close()

# ============================================================
# Endpoints
# ============================================================


@router.post("/import/url", response_model=ImportRecipeUrlResponse)
async def import_recipe_from_url(
    payload: ImportRecipeUrlRequest,
) -> ImportRecipeUrlResponse:
    html, err = await _fetch_html(str(payload.url))
    if err or not html:
        return ImportRecipeUrlResponse(status="error", warnings=[err or "fetch failed"])

    recipe_node: Optional[Dict[str, Any]] = None
    for block in _find_jsonld_blocks(html):
        try:
            data = json.loads(block)
        except Exception:
            continue
        recipe_node = _find_recipe_in_jsonld(data)
        if recipe_node:
            break

    title = None
    servings = None
    ingredients: List[str] = []
    instructions = None
    warnings: List[str] = []

    if recipe_node:
        title = recipe_node.get("name")
        servings = _extract_servings(recipe_node)
        ingredients = _normalize_ingredients(recipe_node)
        instructions = _normalize_instructions(recipe_node)
    else:
        warnings.append("No JSON-LD recipe found")

    if not title:
        title = _extract_title_from_html(html)

    recipe = RawRecipe(
        title=title,
        servings=servings,
        ingredient_lines=ingredients,
        instructions=instructions,
        source_url=str(payload.url),
    )

    return ImportRecipeUrlResponse(
        status="complete" if not warnings else "partial",
        warnings=warnings,
        recipe=recipe,
    )


@router.post("/draft", response_model=RecipeDraftResponse)
async def create_recipe_draft(
    payload: RecipeDraftCreateRequest,
) -> RecipeDraftResponse:
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

    return RecipeDraftResponse(
        draft_id=draft_id,
        recipe=raw,
        parsed_ingredients=parsed,
    )


@router.get("/draft/{draft_id}", response_model=RecipeDraftResponse)
async def get_recipe_draft(draft_id: int) -> RecipeDraftResponse:
    try:
        raw, parsed = _load_recipe_draft(draft_id)
    except KeyError:
        raise ValueError(f"Recipe draft {draft_id} not found")

    return RecipeDraftResponse(
        draft_id=draft_id,
        recipe=raw,
        parsed_ingredients=parsed,
    )
