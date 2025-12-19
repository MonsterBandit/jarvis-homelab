from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import APIRouter
from pydantic import BaseModel, Field, HttpUrl


router = APIRouter(prefix="/recipes", tags=["Recipes"])

# FUTURE: RecipeBuddyAdapter (Phase 6.75.5)
# Optional ingestion adapter. Must map into RawRecipe.
# No authority. No writes. No normalization.


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


def _coerce_to_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _strip_html(s: str) -> str:
    # minimal tag stripper, avoids extra deps
    s = re.sub(r"<\s*br\s*/?\s*>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"</p\s*>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_title_from_html(html: str) -> Optional[str]:
    # og:title
    m = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # <title>
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if m:
        return _strip_html(m.group(1))
    return None


def _find_jsonld_blocks(html: str) -> List[str]:
    # capture script blocks containing ld+json
    blocks = re.findall(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return [b.strip() for b in blocks if b.strip()]


def _find_recipe_in_jsonld(obj: Any) -> Optional[Dict[str, Any]]:
    """
    JSON-LD can be:
      - dict with @type Recipe
      - list of nodes
      - dict with @graph list
      - stringified weirdness
    We look for the first Recipe node.
    """
    if isinstance(obj, dict):
        # direct recipe
        t = obj.get("@type") or obj.get("type")
        if isinstance(t, list):
            if any(str(x).lower() == "recipe" for x in t):
                return obj
        if isinstance(t, str) and t.lower() == "recipe":
            return obj

        # graph
        graph = obj.get("@graph")
        if isinstance(graph, list):
            for node in graph:
                r = _find_recipe_in_jsonld(node)
                if r:
                    return r

        # sometimes nested mainEntity
        for k in ("mainEntity", "mainEntityOfPage"):
            if k in obj:
                r = _find_recipe_in_jsonld(obj[k])
                if r:
                    return r

        # otherwise scan values
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

    # string
    if isinstance(ins, str):
        return _strip_html(ins)

    parts: List[str] = []

    # list of HowToStep / HowToSection / strings
    for item in _coerce_to_list(ins):
        if isinstance(item, str):
            s = _strip_html(item)
            if s:
                parts.append(s)
            continue

        if isinstance(item, dict):
            # HowToSection can contain itemListElement
            if "itemListElement" in item:
                for sub in _coerce_to_list(item.get("itemListElement")):
                    if isinstance(sub, str):
                        s = _strip_html(sub)
                        if s:
                            parts.append(s)
                    elif isinstance(sub, dict):
                        text = sub.get("text") or sub.get("name")
                        if isinstance(text, str):
                            s = _strip_html(text)
                            if s:
                                parts.append(s)
                continue

            text = item.get("text") or item.get("name")
            if isinstance(text, str):
                s = _strip_html(text)
                if s:
                    parts.append(s)

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
    if y is None:
        return None
    if isinstance(y, (int, float)):
        return str(y)
    if isinstance(y, str):
        return y.strip()
    if isinstance(y, list) and y:
        # take first textual yield
        for it in y:
            if isinstance(it, str) and it.strip():
                return it.strip()
            if isinstance(it, (int, float)):
                return str(it)
    return None


async def _fetch_html(url: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (html, error_message)
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            r = await client.get(url, headers={"User-Agent": "ISAC/RecipeImporter (Phase 6.75.1)"})
            r.raise_for_status()
            return r.text, None
    except Exception as e:
        return None, f"Fetch failed: {type(e).__name__}: {e}"


@router.post("/import/url", response_model=ImportRecipeUrlResponse)
async def import_recipe_from_url(payload: ImportRecipeUrlRequest) -> ImportRecipeUrlResponse:
    url = str(payload.url)
    warnings: List[str] = []

    html, err = await _fetch_html(url)
    if err or not html:
        return ImportRecipeUrlResponse(status="error", warnings=[err or "Unknown fetch error"], recipe=None)

    # Try JSON-LD first (most reliable)
    recipe_node: Optional[Dict[str, Any]] = None
    for block in _find_jsonld_blocks(html):
        try:
            data = json.loads(block)
        except Exception:
            # sometimes multiple JSON objects concatenated; ignore
            continue
        recipe_node = _find_recipe_in_jsonld(data)
        if recipe_node:
            break

    title: Optional[str] = None
    servings: Optional[str] = None
    ingredient_lines: List[str] = []
    instructions: Optional[str] = None

    if recipe_node:
        title = recipe_node.get("name")
        if isinstance(title, str):
            title = title.strip()
        else:
            title = None

        servings = _extract_servings(recipe_node)
        ingredient_lines = _normalize_ingredients(recipe_node)
        instructions = _normalize_instructions(recipe_node)
    else:
        warnings.append("No JSON-LD Recipe schema found; returning partial data when possible.")

    # If missing title, fallback to HTML title/meta
    if not title:
        t = _extract_title_from_html(html)
        if t:
            title = t
        else:
            warnings.append("Title not found.")

    if not ingredient_lines:
        warnings.append("Ingredients not found.")

    if not instructions:
        warnings.append("Instructions not found.")

    recipe = RawRecipe(
        title=title,
        servings=servings,
        ingredient_lines=ingredient_lines,
        instructions=instructions,
        source_url=url,
    )

    status = "complete"
    if warnings:
        status = "partial"

    return ImportRecipeUrlResponse(status=status, warnings=warnings, recipe=recipe)
