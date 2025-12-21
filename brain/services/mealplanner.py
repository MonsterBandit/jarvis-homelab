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
# Models
# ----------------------------

class PlanContextResponse(BaseModel):
    meal_plan_id: str = Field(..., description="Meal plan UUID/id")
    generated_at: str = Field(..., description="ISO timestamp (UTC) when this context was generated")
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
# Endpoints (thin wrappers)
# ----------------------------

@router.post("/plan")
def plan(payload: Dict[str, Any]) -> Any:
    """
    Thin wrapper around the planner logic (if present) in mealplans service.

    Contract preserved:
      - Accept arbitrary JSON payload
      - Delegate to mealplans_service.plan(payload)
    """
    if hasattr(mealplans_service, "plan"):
        return mealplans_service.plan(payload)  # type: ignore[attr-defined]
    raise HTTPException(status_code=501, detail="Planner endpoint not implemented in mealplans service")


@router.post("/plan-multi")
def plan_multi(payload: Dict[str, Any]) -> Any:
    """
    Thin wrapper around the multi-household planner logic (if present) in mealplans service.

    Contract preserved:
      - Accept arbitrary JSON payload
      - Delegate to mealplans_service.plan_multi(payload)
    """
    if hasattr(mealplans_service, "plan_multi"):
        return mealplans_service.plan_multi(payload)  # type: ignore[attr-defined]
    raise HTTPException(status_code=501, detail="Multi-planner endpoint not implemented in mealplans service")


@router.get("/plan-context", response_model=PlanContextResponse)
def plan_context(
    meal_plan_id: str = Query(..., description="Meal plan UUID/id to build planning context for"),
) -> PlanContextResponse:
    """
    Returns a consolidated context bundle for UI + planning (read-only):
      - meal plan (source of truth: mealplans service)
      - shopping preview (computed by mealplans service)

    Polish:
      - Adds generated_at (UTC ISO)
      - Adds warnings list
      - Normalizes common error cases into 400/404/502
    """
    meal_plan_id = (meal_plan_id or "").strip()
    if not meal_plan_id:
        raise HTTPException(status_code=400, detail="meal_plan_id is required")

    warnings: List[str] = []

    # 1) Load meal plan
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

    # 2) Shopping preview
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

    # Convert Pydantic models (if returned) to dict
    meal_plan_dict = mp.model_dump() if hasattr(mp, "model_dump") else dict(mp)  # type: ignore[arg-type]
    shopping_preview_dict = sp.model_dump() if hasattr(sp, "model_dump") else dict(sp)  # type: ignore[arg-type]

    # Bubble up non-fatal warnings if preview already contains them
    try:
        if isinstance(shopping_preview_dict, dict):
            w = shopping_preview_dict.get("warnings")
            if isinstance(w, list):
                warnings.extend([str(x) for x in w if x is not None])
    except Exception:
        pass

    notes: List[str] = [
        "Context is read-only: no DB writes, no Grocy writes.",
        "Uses mealplans service as source of truth.",
        f"Admin protected via {_CANONICAL_ADMIN_HEADER}.",
    ]

    generated_at = datetime.now(timezone.utc).isoformat()

    return PlanContextResponse(
        meal_plan_id=meal_plan_id,
        generated_at=generated_at,
        meal_plan=meal_plan_dict,
        shopping_preview=shopping_preview_dict,
        warnings=warnings,
        notes=notes,
    )
