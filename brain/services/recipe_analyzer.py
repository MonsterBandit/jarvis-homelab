from __future__ import annotations

import os
import re
import sqlite3
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, HttpUrl


router = APIRouter(prefix="/recipes", tags=["Recipes - Analyze"])


Household = Literal["home_a", "home_b"]

# SQLite path inside the container
DB_PATH = os.getenv("JARVIS_DB_PATH", "/app/data/jarvis_brain.db")


# ----------------------------
# Models
# ----------------------------

class AnalyzeRecipeUrlRequest(BaseModel):
    household: Household = Field(..., description="Target household (home_a | home_b)")
    url: HttpUrl = Field(..., description="Recipe page URL to import and analyze")
    top_k: int = Field(5, ge=1, le=10, description="Top match candidates per ingredient")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Score floor for returned candidates")


class AnalyzeRecipeIngredientsRequest(BaseModel):
    household: Household = Field(..., description="Target household (home_a | home_b)")
    parsed_ingredients: List[Dict[str, Any]] = Field(
        ...,
        description="Parsed ingredient objects (use /ingredients/parse output .parsed).",
    )
    top_k: int = Field(5, ge=1, le=10, description="Top match candidates per ingredient")
    min_score: float = Field(0.0, ge=0.0, le=1.0, description="Score floor for returned candidates")


class GapItem(BaseModel):
    ingredient_original: str
    ingredient_name: Optional[str] = None
    status: Literal["matched", "ambiguous", "no_match"]
    best_product_name: Optional[str] = None
    best_product_id: Optional[int] = None
    confidence_score: Optional[float] = None
    labels: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class AnalyzeRecipeResponse(BaseModel):
    status: Literal["ok"] = "ok"
    household: Household

    recipe: Optional[Dict[str, Any]] = None
    parsed: Optional[Dict[str, Any]] = None
    match: Optional[Dict[str, Any]] = None

    gaps: List[GapItem] = Field(default_factory=list)
    proposals: List[Dict[str, Any]] = Field(default_factory=list)

    notes: List[str] = Field(default_factory=list)


# ----------------------------
# Internal call helpers
# ----------------------------

async def _post_json(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls our own API endpoints through localhost.
    This avoids tight coupling to internal implementation details of other routers.
    """
    url = f"http://127.0.0.1:8000{path}"
    timeout = httpx.Timeout(30.0, connect=5.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            data = {"detail": resp.text}

    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=data)

    if not isinstance(data, dict):
        raise HTTPException(
            status_code=500,
            detail={"error": "Expected JSON object", "path": path, "got": type(data).__name__},
        )

    return data


def _extract_ingredient_lines(import_result: Dict[str, Any]) -> List[str]:
    """
    Given /recipes/import/url output, attempt to find the list of ingredient lines.
    We keep this flexible because recipe import shapes vary between sources/versions.
    """
    recipe = import_result.get("recipe")
    if isinstance(recipe, dict):
        lines = recipe.get("ingredient_lines")
        if isinstance(lines, list):
            return [str(x) for x in lines if str(x).strip()]

    # Fallback: some implementations may return top-level ingredient_lines
    lines2 = import_result.get("ingredient_lines")
    if isinstance(lines2, list):
        return [str(x) for x in lines2 if str(x).strip()]

    return []


# ----------------------------
# Phase 6.75.5 Mapping Memory helpers
# ----------------------------

def _norm(s: str) -> str:
    s = str(s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_mappings(household: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns dict: ingredient_norm -> {product_id, product_name, household}
    Prefers household-specific mapping, falls back to global (household NULL).
    """
    try:
        conn = sqlite3.connect(DB_PATH)
    except Exception:  # noqa: BLE001
        return {}

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT household, ingredient_norm, product_id, product_name
            FROM recipe_ingredient_mapping
            WHERE household = ? OR household IS NULL
            """,
            (household,),
        )
        rows = cur.fetchall()
    except Exception:  # noqa: BLE001
        rows = []
    finally:
        conn.close()

    # Global goes in first; household overwrites later naturally.
    out: Dict[str, Dict[str, Any]] = {}
    for hh, ing_norm, pid, pname in rows:
        key = str(ing_norm)
        row = {"household": hh, "product_id": int(pid), "product_name": pname}
        if hh is None and key not in out:
            out[key] = row
        if hh is not None:
            out[key] = row

    return out


def _apply_saved_mappings(
    household: str, parsed_list: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits parsed ingredients into:
      - mapped_results: synthetic match-results with status=matched (score=1.0)
      - remaining_for_match: parsed ingredient dicts to be fuzzy-matched
    """
    mapping_index = _load_mappings(household)

    mapped_results: List[Dict[str, Any]] = []
    remaining_for_match: List[Dict[str, Any]] = []

    for ing in parsed_list:
        if not isinstance(ing, dict):
            continue

        ing_name = str(ing.get("name") or "").strip()
        key = _norm(ing_name)
        m = mapping_index.get(key)

        if m and ing_name:
            mapped_results.append(
                {
                    "ingredient_original": ing.get("original") or ing_name,
                    "ingredient_name": ing_name,
                    "ingredient_unit": ing.get("unit"),
                    "ingredient_qty_raw": (ing.get("qty") or {}).get("raw") if isinstance(ing.get("qty"), dict) else None,
                    "status": "matched",
                    "best_match": {
                        "product_id": m["product_id"],
                        "product_name": m.get("product_name"),
                        "score": 1.0,
                        "reasons": ["explicit_saved_mapping"],
                    },
                    "candidates": [],
                    "notes": ["matched via saved mapping (SQLite)"],
                }
            )
        else:
            remaining_for_match.append(ing)

    return mapped_results, remaining_for_match


# ----------------------------
# Gap / proposal generation
# ----------------------------

def _build_gaps_and_proposals(match_result: Dict[str, Any]) -> Tuple[List[GapItem], List[Dict[str, Any]]]:
    """
    Locked labeling strategy:
    - "needs_mapping" for ambiguous/no_match
    - "missing_product" only when no_match (and explicitly marked as uncertain)
    """
    results = match_result.get("results")
    if not isinstance(results, list):
        return [], []

    gaps: List[GapItem] = []
    proposals: List[Dict[str, Any]] = []

    for r in results:
        if not isinstance(r, dict):
            continue

        status = r.get("status")
        if status not in ("matched", "ambiguous", "no_match"):
            continue

        original = str(r.get("ingredient_original") or "")
        name = r.get("ingredient_name")
        best = r.get("best_match") if isinstance(r.get("best_match"), dict) else None

        best_name = best.get("product_name") if best else None
        best_id = best.get("product_id") if best else None
        best_score = best.get("score") if best else None

        labels: List[str] = []
        notes: List[str] = []

        if status == "matched":
            # Not a gap
            continue

        # Always: needs_mapping when not matched
        labels.append("needs_mapping")
        notes.append("No confident automatic mapping. User confirmation required.")

        if status == "ambiguous":
            labels.append("ambiguous_match")
            notes.append("Top two candidates are close. Choose the correct product mapping.")
            proposals.append(
                {
                    "type": "choose_mapping",
                    "ingredient_original": original,
                    "ingredient_name": name,
                    "reason": "ambiguous",
                    "action_required": "user_select_candidate",
                }
            )

        if status == "no_match":
            labels.append("missing_product")
            notes.append("Likely missing OR named differently in Grocy. Needs human confirmation.")
            proposals.append(
                {
                    "type": "create_or_map_product",
                    "ingredient_original": original,
                    "ingredient_name": name,
                    "reason": "no_match",
                    "action_required": "user_confirm_create_or_map",
                }
            )

        gaps.append(
            GapItem(
                ingredient_original=original,
                ingredient_name=str(name) if name is not None else None,
                status=status,  # type: ignore[arg-type]
                best_product_name=str(best_name) if best_name is not None else None,
                best_product_id=int(best_id) if isinstance(best_id, int) else (int(best_id) if best_id is not None and str(best_id).isdigit() else None),
                confidence_score=float(best_score) if isinstance(best_score, (int, float)) else None,
                labels=labels,
                notes=notes,
            )
        )

    return gaps, proposals


# ----------------------------
# Endpoints
# ----------------------------

@router.post(
    "/analyze/url",
    response_model=AnalyzeRecipeResponse,
    summary="Import recipe URL -> parse ingredients -> match to Grocy products -> return gaps/proposals (read-only).",
)
async def analyze_recipe_url(body: AnalyzeRecipeUrlRequest) -> AnalyzeRecipeResponse:
    """
    Phase 6.75.4 + 6.75.5 (LOCKED):
    - Read-only analysis only
    - No Grocy writes
    - No stock checks
    - No shopping list changes
    - Apply saved mappings (SQLite) before fuzzy matching
    """
    household = body.household.strip().lower()
    if household not in {"home_a", "home_b"}:
        raise HTTPException(status_code=400, detail="household must be 'home_a' or 'home_b'")

    # 1) Import recipe from URL (Phase 6.75.1)
    imported = await _post_json("/recipes/import/url", {"url": str(body.url)})

    ingredient_lines = _extract_ingredient_lines(imported)
    if not ingredient_lines:
        return AnalyzeRecipeResponse(
            household=household,  # type: ignore[arg-type]
            recipe=imported.get("recipe") if isinstance(imported.get("recipe"), dict) else imported,
            parsed=None,
            match=None,
            gaps=[],
            proposals=[],
            notes=[
                "Recipe imported, but no ingredient_lines were found to parse.",
                "This may be a site-specific import limitation.",
                "Phase 6.75.4 remains read-only.",
            ],
        )

    # 2) Parse ingredient lines (Phase 6.75.2)
    parsed = await _post_json("/ingredients/parse", {"lines": ingredient_lines})

    parsed_list = parsed.get("parsed")
    if not isinstance(parsed_list, list):
        raise HTTPException(status_code=500, detail={"error": "Expected parsed.parsed list", "parsed": parsed})

    # 3) Phase 6.75.5: apply saved mappings before fuzzy match (read-only)
    mapped_results, remaining_for_match = _apply_saved_mappings(household, parsed_list)

    # 4) Match remaining parsed ingredients to Grocy products (Phase 6.75.3)
    match_payload = {
        "household": household,
        "ingredients": remaining_for_match,
        "top_k": body.top_k,
    }
    match = await _post_json(f"/recipes/match-products?min_score={body.min_score}", match_payload)

    # Merge mapped results into match results (mapped first for visibility)
    if mapped_results:
        if isinstance(match.get("results"), list):
            match["results"] = mapped_results + match["results"]
        else:
            match["results"] = mapped_results

    # 5) Build gaps + proposals
    gaps, proposals = _build_gaps_and_proposals(match)

    return AnalyzeRecipeResponse(
        household=household,  # type: ignore[arg-type]
        recipe=imported.get("recipe") if isinstance(imported.get("recipe"), dict) else imported,
        parsed=parsed,
        match=match,
        gaps=gaps,
        proposals=proposals,
        notes=[
            "Phase 6.75.4: analyze pipeline (import -> parse -> match -> gaps) is read-only",
            "Phase 6.75.5: saved mappings (SQLite) are applied before fuzzy matching",
            "No writes, no stock checks, no shopping list changes",
            "Universal recipe pool; household affects matching only",
        ],
    )


@router.post(
    "/analyze/ingredients",
    response_model=AnalyzeRecipeResponse,
    summary="Given parsed ingredients -> apply saved mappings -> match remaining -> return gaps/proposals (read-only).",
)
async def analyze_recipe_ingredients(body: AnalyzeRecipeIngredientsRequest) -> AnalyzeRecipeResponse:
    """
    Debug/lab endpoint (B):
    Lets you bypass URL import and parsing when iterating on matching behavior.
    Applies saved mappings before fuzzy matching.
    """
    household = body.household.strip().lower()
    if household not in {"home_a", "home_b"}:
        raise HTTPException(status_code=400, detail="household must be 'home_a' or 'home_b'")

    if not body.parsed_ingredients:
        raise HTTPException(status_code=400, detail="parsed_ingredients must not be empty")

    mapped_results, remaining_for_match = _apply_saved_mappings(household, body.parsed_ingredients)

    match_payload = {
        "household": household,
        "ingredients": remaining_for_match,
        "top_k": body.top_k,
    }
    match = await _post_json(f"/recipes/match-products?min_score={body.min_score}", match_payload)

    if mapped_results:
        if isinstance(match.get("results"), list):
            match["results"] = mapped_results + match["results"]
        else:
            match["results"] = mapped_results

    gaps, proposals = _build_gaps_and_proposals(match)

    return AnalyzeRecipeResponse(
        household=household,  # type: ignore[arg-type]
        recipe=None,
        parsed={"parsed": body.parsed_ingredients},
        match=match,
        gaps=gaps,
        proposals=proposals,
        notes=[
            "Phase 6.75.4: analyze ingredients endpoint (debug path) is read-only",
            "Phase 6.75.5: saved mappings (SQLite) are applied before fuzzy matching",
            "No writes, no stock checks, no shopping list changes",
        ],
    )
