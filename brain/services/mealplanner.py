from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

# Source of truth for meal plan storage + shopping preview logic
from services import mealplans as mealplans_service


# ----------------------------
# Admin auth (canonical)
# ----------------------------

_CANONICAL_ADMIN_ENV = "ISAC_ADMIN_API_KEY"
_CANONICAL_ADMIN_HEADER = "X-ISAC-Admin-Key"


def _get_expected_admin_key() -> Optional[str]:
    """
    Canonical admin key env var:
      - ISAC_ADMIN_API_KEY

    Legacy fallbacks (supported to smooth migration):
      - ISAC_API_KEY
      - JARVIS_API_KEY
    """
    return (
        os.getenv(_CANONICAL_ADMIN_ENV)
        or os.getenv("ISAC_API_KEY")
        or os.getenv("JARVIS_API_KEY")
    )


def require_admin_key(
    x_isac_admin_key: Optional[str] = Header(None, alias=_CANONICAL_ADMIN_HEADER),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> None:
    """
    Require an admin API key for all endpoints in this router.

    Canonical header: X-ISAC-Admin-Key
    Legacy header supported: X-API-Key
    """
    expected = _get_expected_admin_key()
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=f"Admin key not configured (missing {_CANONICAL_ADMIN_ENV}).",
        )

    provided = (x_isac_admin_key or x_api_key or "").strip()
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


router = APIRouter(
    prefix="/mealplanner",
    tags=["mealplanner"],
    dependencies=[Depends(require_admin_key)],
)


# ----------------------------
# Phase 6 semantic locks (constants)
# ----------------------------

INGESTION_CLASSIFICATION_QUESTION = (
    "Is this a product that is typically consumed as a whole container, "
    "or one that is usually used in parts (slices, pieces, servings)?"
)


# ----------------------------
# Models (Phase 6.76)
# ----------------------------

class PlanContextResponse(BaseModel):
    meal_plan_id: str = Field(..., description="Meal plan UUID/id")
    generated_at: str = Field(..., description="UTC timestamp when this context was generated")
    meal_plan: Dict[str, Any] = Field(
        ...,
        description="Meal plan record (source of truth: mealplans service)",
    )
    shopping_preview: Dict[str, Any] = Field(
        ...,
        description="Shopping preview bundle (computed by mealplans service)",
    )
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings for UI")
    notes: List[str] = Field(default_factory=list, description="Operational notes for UI/debug")


# ----------------------------
# Models (Phase 6.77 suggest-only intents)
# ----------------------------

class ShoppingIntentSource(BaseModel):
    recipe_finalized_id: str
    recipe_title: Optional[str] = None
    day: int
    slot: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    note: Optional[str] = None


class PurchaseUnitSuggestion(BaseModel):
    """
    Phase 6.77: Suggest-only.
    We intentionally do NOT select a specific UPC/variant or enforce purchase-unit math yet.
    This object is a placeholder for the future concept↔variant reconciliation layer.
    """
    purchase_unit: Optional[str] = Field(
        None,
        description="Human purchase unit e.g., jar/bottle/box. None until a preferred variant exists.",
    )
    count: Optional[float] = Field(
        None,
        description="Suggested number of purchase units to buy. None until variant sizing + policy logic exists.",
    )
    variant_hint: Optional[str] = Field(
        None,
        description="Optional hint such as preferred variant size or UPC-level choice (not selected in 6.77).",
    )
    reason: str = Field(
        ...,
        description="Explain why purchase-unit suggestion is missing or how it was derived.",
    )


class ShoppingIntentLine(BaseModel):
    concept_name: str = Field(..., description="Ingredient concept name (recipe-facing identity)")
    concept_quantity: Optional[float] = Field(
        None,
        description="Aggregated quantity at the concept level (planning quantity, not inventory)",
    )
    concept_unit: Optional[str] = Field(
        None,
        description="Unit associated with concept_quantity (planning unit, not necessarily purchase unit)",
    )
    note: Optional[str] = None

    # Read-only mapping hints from preview (if present)
    mapped_product_id: Optional[int] = None
    mapped_product_name: Optional[str] = None
    mapped_qu_id: Optional[int] = None
    mapping_source: Optional[str] = None
    mapping_note: Optional[str] = None

    # Phase 6 semantic lock: classification + policies (suggest-only / unknown until ingestion knows)
    ingestion_classification_status: str = Field(
        ...,
        description="One of: unknown, whole_consumption, partial_consumption.",
    )
    classification_question: str = Field(
        ...,
        description="The exact classification question required at product ingestion.",
    )
    consume_whole_on_open_policy_hint: Optional[str] = Field(
        None,
        description="Optional hint: KEEP_REMAINDER or CONSUME_WHOLE_ON_OPEN (not enforced in 6.77).",
    )

    purchase_suggestion: PurchaseUnitSuggestion
    sources: List[ShoppingIntentSource] = Field(default_factory=list, description="Attribution for explainability")


class ShoppingIntentsResponse(BaseModel):
    meal_plan_id: str
    household: str
    week_start: str
    generated_at: str

    # Grouped-by-household shape to be UI-ready (even if single household today)
    intents_by_household: Dict[str, List[ShoppingIntentLine]]

    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings for UI")
    notes: List[str] = Field(default_factory=list, description="Operational notes for UI/debug")


# ----------------------------
# Helpers
# ----------------------------

def _utc_iso_z() -> str:
    # Example: 2025-12-21T17:33:03Z
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _dedupe_strs(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for x in items:
        s = str(x)
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # type: ignore[no-any-return]
    if isinstance(obj, dict):
        return obj
    return dict(obj)  # type: ignore[arg-type]


def _to_list(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj
    return list(obj)  # type: ignore[arg-type]


def _build_intents_from_preview(preview: Dict[str, Any]) -> List[ShoppingIntentLine]:
    """
    Translate mealplans.shopping_preview output into suggest-only shopping intents.

    Phase 6.77 posture:
      - No purchase-unit math
      - No variant selection
      - No inventory checks
      - No writes / side effects
      - Emphasize explainability and source attribution
    """
    lines_raw = preview.get("lines", [])
    out: List[ShoppingIntentLine] = []

    for ln in _to_list(lines_raw):
        d = _to_dict(ln)

        concept_name = str(d.get("name") or "").strip()
        if not concept_name:
            continue

        sources: List[ShoppingIntentSource] = []
        for s in _to_list(d.get("sources", []) or []):
            sd = _to_dict(s)
            sources.append(
                ShoppingIntentSource(
                    recipe_finalized_id=str(sd.get("recipe_finalized_id") or "").strip(),
                    recipe_title=(str(sd.get("recipe_title")).strip() if isinstance(sd.get("recipe_title"), str) else None),
                    day=int(sd.get("day")) if sd.get("day") is not None else 0,
                    slot=str(sd.get("slot") or "").strip(),
                    quantity=sd.get("quantity") if isinstance(sd.get("quantity"), (int, float)) else None,
                    unit=(str(sd.get("unit")).strip() if isinstance(sd.get("unit"), str) and sd.get("unit").strip() else None),
                    note=(str(sd.get("note")).strip() if isinstance(sd.get("note"), str) and sd.get("note").strip() else None),
                )
            )

        purchase_suggestion = PurchaseUnitSuggestion(
            purchase_unit=None,
            count=None,
            variant_hint=None,
            reason=(
                "Suggest-only (Phase 6.77). Purchase-unit counts and variant selection are deferred to "
                "concept↔variant reconciliation (preferred variant, container size, and policies like consume-whole-on-open)."
            ),
        )

        out.append(
            ShoppingIntentLine(
                concept_name=concept_name,
                concept_quantity=(d.get("quantity") if isinstance(d.get("quantity"), (int, float)) else None),
                concept_unit=(str(d.get("unit")).strip() if isinstance(d.get("unit"), str) and d.get("unit").strip() else None),
                note=(str(d.get("note")).strip() if isinstance(d.get("note"), str) and d.get("note").strip() else None),
                mapped_product_id=(int(d["mapped_product_id"]) if d.get("mapped_product_id") is not None else None),
                mapped_product_name=(str(d.get("mapped_product_name")).strip() if isinstance(d.get("mapped_product_name"), str) and d.get("mapped_product_name").strip() else None),
                mapped_qu_id=(int(d["mapped_qu_id"]) if d.get("mapped_qu_id") is not None else None),
                mapping_source=(str(d.get("mapping_source")).strip() if isinstance(d.get("mapping_source"), str) and d.get("mapping_source").strip() else None),
                mapping_note=(str(d.get("mapping_note")).strip() if isinstance(d.get("mapping_note"), str) and d.get("mapping_note").strip() else None),
                ingestion_classification_status="unknown",
                classification_question=INGESTION_CLASSIFICATION_QUESTION,
                consume_whole_on_open_policy_hint=None,
                purchase_suggestion=purchase_suggestion,
                sources=sources,
            )
        )

    return out


# ----------------------------
# Endpoints (thin wrappers)
# ----------------------------

@router.post("/plan")
def plan(payload: Dict[str, Any]) -> Any:
    """
    Thin wrapper around planner logic (if present) in mealplans service.
    """
    if hasattr(mealplans_service, "plan"):
        return mealplans_service.plan(payload)  # type: ignore[attr-defined]
    raise HTTPException(status_code=501, detail="Planner endpoint not implemented in mealplans service")


@router.post("/plan-multi")
def plan_multi(payload: Dict[str, Any]) -> Any:
    """
    Thin wrapper around multi-household planner logic (if present) in mealplans service.
    """
    if hasattr(mealplans_service, "plan_multi"):
        return mealplans_service.plan_multi(payload)  # type: ignore[attr-defined]
    raise HTTPException(status_code=501, detail="Multi-planner endpoint not implemented in mealplans service")


@router.get("/plan-context", response_model=PlanContextResponse)
def plan_context(
    meal_plan_id: str = Query(..., description="Meal plan UUID/id to build planning context for"),
) -> PlanContextResponse:
    """
    Phase 6.76 — Read-only planning context bundle (no side effects).
    """
    meal_plan_id = (meal_plan_id or "").strip()
    if not meal_plan_id:
        raise HTTPException(status_code=400, detail="meal_plan_id is required")

    warnings: List[str] = []

    if not hasattr(mealplans_service, "get_meal_plan"):
        raise HTTPException(status_code=501, detail="get_meal_plan not implemented in mealplans service")

    try:
        mp = mealplans_service.get_meal_plan(meal_plan_id)  # type: ignore[attr-defined]
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(status_code=404, detail="Meal plan not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to load meal plan: {e}")

    if not hasattr(mealplans_service, "shopping_preview"):
        raise HTTPException(status_code=501, detail="shopping_preview not implemented in mealplans service")

    try:
        sp = mealplans_service.shopping_preview(meal_plan_id)  # type: ignore[attr-defined]
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(status_code=404, detail="Meal plan not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to build shopping preview: {e}")

    meal_plan_dict = mp.model_dump() if hasattr(mp, "model_dump") else dict(mp)  # type: ignore[arg-type]
    shopping_preview_dict = sp.model_dump() if hasattr(sp, "model_dump") else dict(sp)  # type: ignore[arg-type]

    # Bubble up preview warnings (deduped)
    try:
        if isinstance(shopping_preview_dict, dict):
            w = shopping_preview_dict.get("warnings")
            if isinstance(w, list):
                warnings.extend([str(x) for x in w if x is not None])
    except Exception:
        pass

    notes: List[str] = [
        "Read-only: no DB writes, no Grocy writes.",
        "Source of truth: mealplans service.",
        f"Admin protected via {_CANONICAL_ADMIN_HEADER}.",
    ]

    return PlanContextResponse(
        meal_plan_id=meal_plan_id,
        generated_at=_utc_iso_z(),
        meal_plan=meal_plan_dict,
        shopping_preview=shopping_preview_dict,
        warnings=_dedupe_strs(warnings),
        notes=notes,
    )


@router.get("/shopping-intents", response_model=ShoppingIntentsResponse)
def shopping_intents(
    meal_plan_id: str = Query(..., description="Meal plan UUID/id to build suggest-only shopping intents for"),
) -> ShoppingIntentsResponse:
    """
    Phase 6.77 — Suggest-only shopping intents (no side effects).

    Establishes a stable, UI-ready intent envelope:
      - concept-level lines
      - household grouping
      - source attribution (recipe/day/slot)
      - explicit ingestion classification prompt
    """
    meal_plan_id = (meal_plan_id or "").strip()
    if not meal_plan_id:
        raise HTTPException(status_code=400, detail="meal_plan_id is required")

    warnings: List[str] = []

    if not hasattr(mealplans_service, "get_meal_plan"):
        raise HTTPException(status_code=501, detail="get_meal_plan not implemented in mealplans service")

    try:
        mp = mealplans_service.get_meal_plan(meal_plan_id)  # type: ignore[attr-defined]
        mp_dict = _to_dict(mp)
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(status_code=404, detail="Meal plan not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to load meal plan: {e}")

    household = str(mp_dict.get("household") or "").strip()
    week_start = str(mp_dict.get("week_start") or "").strip()

    if not household:
        warnings.append("Meal plan household missing; returning intents under 'unknown'.")
        household_key = "unknown"
    else:
        household_key = household

    if not hasattr(mealplans_service, "shopping_preview"):
        raise HTTPException(status_code=501, detail="shopping_preview not implemented in mealplans service")

    try:
        sp = mealplans_service.shopping_preview(meal_plan_id)  # type: ignore[attr-defined]
        sp_dict = _to_dict(sp)
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(status_code=404, detail="Meal plan not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to build shopping preview: {e}")

    try:
        w = sp_dict.get("warnings")
        if isinstance(w, list):
            warnings.extend([str(x) for x in w if x is not None])
    except Exception:
        pass

    intents = _build_intents_from_preview(sp_dict)

    notes: List[str] = [
        "Suggest-only: no DB writes, no Grocy writes, no side effects.",
        "Concept-level intents with explainable sources.",
        "Purchase-unit counts and variant selection are deferred to concept↔variant reconciliation.",
        "Ingestion classification (whole vs partial consumption) must be confirmed during product ingestion.",
        f"Admin protected via {_CANONICAL_ADMIN_HEADER}.",
    ]

    return ShoppingIntentsResponse(
        meal_plan_id=meal_plan_id,
        household=household or "unknown",
        week_start=week_start,
        generated_at=_utc_iso_z(),
        intents_by_household={household_key: intents},
        warnings=_dedupe_strs(warnings),
        notes=notes,
    )
