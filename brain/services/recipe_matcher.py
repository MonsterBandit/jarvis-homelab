from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from services.grocy import GrocyClient, GrocyError, create_grocy_client


router = APIRouter(prefix="/recipes", tags=["Recipes - Matching"])


# ----------------------------
# Models
# ----------------------------

Household = Literal["home_a", "home_b"]


class MatchCandidate(BaseModel):
    product_id: Optional[int] = None
    product_name: str
    score: float
    reasons: List[str] = Field(default_factory=list)


class IngredientMatchResult(BaseModel):
    ingredient_original: str
    ingredient_name: Optional[str] = None
    ingredient_unit: Optional[str] = None
    ingredient_qty_raw: Optional[str] = None

    best_match: Optional[MatchCandidate] = None
    candidates: List[MatchCandidate] = Field(default_factory=list)

    status: Literal["matched", "ambiguous", "no_match"] = "no_match"
    notes: List[str] = Field(default_factory=list)


class GrocyMatchIngredientsRequest(BaseModel):
    household: Household = Field(..., description="Target household (home_a | home_b)")
    # We intentionally accept the parsed ingredient objects from /ingredients/parse
    ingredients: List[Dict[str, Any]] = Field(
        ...,
        description="List of parsed ingredient objects (as returned by /ingredients/parse).",
    )
    top_k: int = Field(5, ge=1, le=10, description="How many match candidates to return per ingredient")


class GrocyMatchIngredientsResponse(BaseModel):
    status: Literal["ok"] = "ok"
    household: Household
    count: int
    results: List[IngredientMatchResult]
    notes: List[str] = Field(default_factory=list)


# ----------------------------
# Grocy dependency
# ----------------------------

async def get_grocy_client() -> Optional[GrocyClient]:
    """
    Local dependency: returns configured GrocyClient or None if missing config.
    Mirrors main.py behavior but keeps this module standalone.
    """
    try:
        client = await create_grocy_client()
    except GrocyError:
        return None
    return client


# ----------------------------
# Matching helpers
# ----------------------------

_STOPWORDS = {
    "fresh", "large", "small", "medium",
    "ground", "grated", "shredded",
    "diced", "minced", "chopped", "sliced",
    "optional", "to", "taste", "and", "or",
    "package", "packages", "pkg", "pkgs",
    "can", "cans", "jar", "jars",
}

def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    # keep letters/numbers/spaces
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokens(s: str) -> List[str]:
    n = _normalize(s)
    toks = [t for t in n.split() if t and t not in _STOPWORDS]
    return toks

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0

def _seq_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def _extract_product_id(p: Dict[str, Any]) -> Optional[int]:
    for key in ("id", "product_id", "created_object_id"):
        if key in p and p[key] is not None:
            try:
                return int(p[key])
            except (TypeError, ValueError):
                return None
    return None

def _product_display_name(p: Dict[str, Any]) -> str:
    for key in ("name", "product", "product_name", "title"):
        val = p.get(key)
        if val:
            return str(val)
    # fallback to something stable
    return str(p.get("id") or "unknown_product")

def _score_match(ingredient_name: str, product_name: str) -> Tuple[float, List[str]]:
    """
    Weighted scoring (C): token overlap + fuzzy similarity + containment.
    Returns score in [0, 1] and reasons.
    """
    reasons: List[str] = []

    ing_norm = _normalize(ingredient_name)
    prod_norm = _normalize(product_name)

    if not ing_norm or not prod_norm:
        return 0.0, ["missing_name_for_scoring"]

    # Exact normalized match
    if ing_norm == prod_norm:
        return 1.0, ["exact_normalized_match"]

    ing_toks = _tokens(ingredient_name)
    prod_toks = _tokens(product_name)

    jac = _jaccard(ing_toks, prod_toks)
    ratio = _seq_ratio(ing_norm, prod_norm)

    # containment boosts (use normalized string)
    contains_boost = 0.0
    if ing_norm in prod_norm:
        contains_boost = 0.12
        reasons.append("ingredient_contained_in_product_name")
    elif prod_norm in ing_norm:
        contains_boost = 0.08
        reasons.append("product_name_contained_in_ingredient")

    # Token overlap: if all ingredient tokens exist in product tokens
    if ing_toks and set(ing_toks).issubset(set(prod_toks)):
        reasons.append("all_ingredient_tokens_present")

    # Weighted sum (kept simple + explainable)
    score = (0.55 * jac) + (0.45 * ratio) + contains_boost
    score = max(0.0, min(1.0, score))

    reasons.append(f"jaccard={jac:.3f}")
    reasons.append(f"ratio={ratio:.3f}")

    return score, reasons


def _ingredient_to_fields(ing: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Pull the fields we care about from the parsed ingredient object.
    We intentionally accept flexible shapes.
    """
    original = str(ing.get("original") or "")
    name = ing.get("name")
    unit = ing.get("unit")
    qty_raw = None
    qty = ing.get("qty")
    if isinstance(qty, dict):
        qty_raw = qty.get("raw")
    return original, (str(name) if name else None), (str(unit) if unit else None), (str(qty_raw) if qty_raw else None)


def _classify_candidates(candidates: List[MatchCandidate]) -> Tuple[Literal["matched","ambiguous","no_match"], List[str]]:
    """
    Decide status based on top scores.
    """
    notes: List[str] = []
    if not candidates:
        return "no_match", ["no_candidates"]

    # candidates are expected sorted by score desc
    top = candidates[0]
    if top.score < 0.55:
        notes.append("top_score_below_threshold")
        return "no_match", notes

    # Ambiguity: second is close to first
    if len(candidates) > 1:
        second = candidates[1]
        if top.score - second.score < 0.08 and second.score >= 0.55:
            notes.append("top_two_too_close")
            return "ambiguous", notes

    return "matched", notes


# ----------------------------
# API
# ----------------------------

@router.post(
    "/match-products",
    response_model=GrocyMatchIngredientsResponse,
    summary="Match parsed ingredients to Grocy products (read-only). Household required.",
)
async def match_products(
    body: GrocyMatchIngredientsRequest,
    client: Optional[GrocyClient] = Depends(get_grocy_client),
    min_score: float = Query(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Optional score floor for returned candidates (still computed for all).",
    ),
) -> GrocyMatchIngredientsResponse:
    """
    Phase 6.75.3 (LOCKED):
    - Read-only matching only
    - No Grocy writes
    - No stock changes
    - No shopping list changes
    """
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Grocy client not configured; matching unavailable.",
        )

    hh = (body.household or "").strip().lower()
    if hh not in {"home_a", "home_b"}:
        raise HTTPException(status_code=400, detail="household must be 'home_a' or 'home_b'")

    try:
        products = await client.get_products(household=hh)
    except GrocyError as exc:
        raise HTTPException(status_code=502, detail=f"Error reading Grocy products: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Unexpected error reading products: {exc}") from exc

    # Normalize products list shape defensively
    if not isinstance(products, list):
        raise HTTPException(status_code=500, detail={"error": "Expected list of products", "got": type(products).__name__})

    results: List[IngredientMatchResult] = []

    for ing in body.ingredients:
        if not isinstance(ing, dict):
            # keep predictable output; no hard crash on malformed element
            results.append(
                IngredientMatchResult(
                    ingredient_original=str(ing),
                    status="no_match",
                    notes=["ingredient_not_a_dict"],
                )
            )
            continue

        original, name, unit, qty_raw = _ingredient_to_fields(ing)

        if not name:
            results.append(
                IngredientMatchResult(
                    ingredient_original=original,
                    ingredient_name=None,
                    ingredient_unit=unit,
                    ingredient_qty_raw=qty_raw,
                    status="no_match",
                    notes=["missing_ingredient_name"],
                )
            )
            continue

        scored: List[MatchCandidate] = []
        for p in products:
            if not isinstance(p, dict):
                continue
            p_name = _product_display_name(p)
            score, reasons = _score_match(name, p_name)
            if score >= min_score:
                scored.append(
                    MatchCandidate(
                        product_id=_extract_product_id(p),
                        product_name=p_name,
                        score=round(score, 4),
                        reasons=reasons,
                    )
                )

        scored.sort(key=lambda c: c.score, reverse=True)
        candidates = scored[: body.top_k]

        status, status_notes = _classify_candidates(candidates)
        best = candidates[0] if candidates else None

        out_notes = list(status_notes)
        if status == "matched":
            out_notes.append("read_only_match_ok")
        elif status == "ambiguous":
            out_notes.append("needs_user_confirmation_for_mapping")
        else:
            out_notes.append("no_confident_match")

        results.append(
            IngredientMatchResult(
                ingredient_original=original,
                ingredient_name=name,
                ingredient_unit=unit,
                ingredient_qty_raw=qty_raw,
                best_match=best,
                candidates=candidates,
                status=status,
                notes=out_notes,
            )
        )

    return GrocyMatchIngredientsResponse(
        household=hh,  # type: ignore[arg-type]
        count=len(results),
        results=results,
        notes=[
            "Phase 6.75.3: read-only Grocy product matching",
            "Household is required; no inference",
            "No writes, no stock checks, no shopping list changes",
        ],
    )
