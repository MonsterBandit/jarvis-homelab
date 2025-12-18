from __future__ import annotations

import os
import socket
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import psutil
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from model_client import (
    GeminiModelClient,
    ModelClientError,
    OpenAICompatibleClient,
)
from services.homeassistant import HomeAssistantClient, HomeAssistantConfig
from services.grocy import GrocyClient, GrocyError, create_grocy_client
from services.barcodebuddy import (
    BarcodeBuddyClient,
    BarcodeBuddyError,
    create_barcodebuddy_client,
)

# ----------------------------
# Environment & configuration
# ----------------------------

DB_PATH = os.getenv("JARVIS_DB_PATH", "/app/data/jarvis_brain.db")

JARVIS_LLM_PROVIDER = os.getenv("JARVIS_LLM_PROVIDER", "openai-compatible")
JARVIS_LLM_MODEL = os.getenv("JARVIS_LLM_MODEL", "gpt-4.1-mini")

OPENAI_COMPATIBLE_API_BASE = os.getenv("OPENAI_COMPATIBLE_API_BASE", "").rstrip("/")
OPENAI_COMPATIBLE_API_KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY", "")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-1.5-flash")

API_KEY = os.getenv("JARVIS_API_KEY")

HOMEASSISTANT_BASE_URL = os.getenv("HOMEASSISTANT_BASE_URL", "")
HOMEASSISTANT_TOKEN = os.getenv("HOMEASSISTANT_TOKEN", "")

# ----------------------------
# Database helpers
# ----------------------------


def init_db() -> None:
    """
    Initialize the SQLite database with required tables.
    """
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()

        # Conversation log
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                user_message TEXT NOT NULL,
                jarvis_answer TEXT NOT NULL
            )
            """
        )

        # Shopping lists for Phase 6.5
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS shopping_list (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                household TEXT NOT NULL,
                item_name TEXT NOT NULL,
                quantity TEXT,
                source TEXT,
                completed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_shopping_list_household_completed
            ON shopping_list (household, completed)
            """
        )

        conn.commit()
    finally:
        conn.close()


def log_conversation(model: str, user_message: str, jarvis_answer: str) -> None:
    """
    Insert a single question/answer pair into the database.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO conversation_log (timestamp, model, user_message, jarvis_answer)
            VALUES (?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                model,
                user_message,
                jarvis_answer,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_recent_conversations(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Return the most recent N conversation entries, newest first.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT timestamp, model, user_message, jarvis_answer
            FROM conversation_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    items: List[Dict[str, Any]] = []
    for ts, model, user_msg, jarvis_msg in rows:
        items.append(
            {
                "timestamp": ts,
                "model": model,
                "user_message": user_msg,
                "jarvis_answer": jarvis_msg,
            }
        )
    return items


# Shopping list DB helpers
def add_shopping_list_item(
    household: str,
    item_name: str,
    quantity: Optional[str] = None,
    source: Optional[str] = None,
) -> int:
    """
    Insert a shopping list item and return its new ID.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO shopping_list (household, item_name, quantity, source, completed, created_at)
            VALUES (?, ?, ?, ?, 0, ?)
            """,
            (
                household,
                item_name,
                quantity,
                source,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_shopping_list_items(
    household: str,
    include_completed: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch shopping list items for a given household.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        if include_completed:
            cur.execute(
                """
                SELECT id, household, item_name, quantity, source, completed, created_at
                FROM shopping_list
                WHERE household = ?
                ORDER BY created_at ASC, id ASC
                """,
                (household,),
            )
        else:
            cur.execute(
                """
                SELECT id, household, item_name, quantity, source, completed, created_at
                FROM shopping_list
                WHERE household = ? AND completed = 0
                ORDER BY created_at ASC, id ASC
                """,
                (household,),
            )
        rows = cur.fetchall()
    finally:
        conn.close()

    items: List[Dict[str, Any]] = []
    for row in rows:
        (
            item_id,
            hh,
            name,
            quantity,
            source,
            completed,
            created_at,
        ) = row
        items.append(
            {
                "id": item_id,
                "household": hh,
                "name": name,
                "quantity": quantity,
                "source": source,
                "completed": bool(completed),
                "created_at": created_at,
            }
        )
    return items


def delete_shopping_list_item(household: str, item_id: int) -> bool:
    """
    Delete a single shopping list item for a given household.
    Returns True if something was deleted.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            DELETE FROM shopping_list
            WHERE household = ? AND id = ?
            """,
            (household, item_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def clear_shopping_list(household: str) -> int:
    """
    Delete all shopping list items for a given household.
    Returns number of rows deleted.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            DELETE FROM shopping_list
            WHERE household = ?
            """,
            (household,),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


# ----------------------------
# LLM client initialization
# ----------------------------


def create_model_client() -> GeminiModelClient | OpenAICompatibleClient:
    """
    Factory to create the configured LLM client.
    """
    provider = JARVIS_LLM_PROVIDER.lower()

    if provider == "openai-compatible":
        if not OPENAI_COMPATIBLE_API_BASE or not OPENAI_COMPATIBLE_API_KEY:
            raise RuntimeError(
                "OPENAI_COMPATIBLE_API_BASE and OPENAI_COMPATIBLE_API_KEY must be set "
                "for the openai-compatible provider."
            )
        return OpenAICompatibleClient(
            OPENAI_COMPATIBLE_API_BASE,
            OPENAI_COMPATIBLE_API_KEY,
            JARVIS_LLM_MODEL,
        )

    if provider == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY must be set when using the gemini provider."
            )
        return GeminiModelClient(
            api_key=GEMINI_API_KEY,
            model_name=GEMINI_MODEL_NAME,
        )

    raise RuntimeError(f"Unsupported JARVIS_LLM_PROVIDER: {JARVIS_LLM_PROVIDER}")


MODEL_CLIENT = create_model_client()
LLM_MODEL = JARVIS_LLM_MODEL

# ----------------------------
# FastAPI app setup
# ----------------------------

app = FastAPI(
    title="ISAC Brain",
    version="1.0.0",
)

# Allow CORS from anywhere for now. Adjust in production if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the DB on startup
init_db()


# ----------------------------
# API key dependency
# ----------------------------


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    Simple header-based API key check. If JARVIS_API_KEY is not set,
    this becomes a no-op (open access).
    """
    if not API_KEY:
        return

    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ----------------------------
# Request / response models
# ----------------------------


class AskRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None
    temperature: float = 0.2
    max_output_tokens: int = 512


class AskResponse(BaseModel):
    model: str
    answer: str


# ----------------------------
# Home Assistant client setup
# ----------------------------


def create_ha_client() -> Optional[HomeAssistantClient]:
    """
    Create a HomeAssistantClient from environment variables, or None if
    configuration is incomplete.
    """
    if not HOMEASSISTANT_BASE_URL or not HOMEASSISTANT_TOKEN:
        return None

    config = HomeAssistantConfig(
        base_url=HOMEASSISTANT_BASE_URL,
        token=HOMEASSISTANT_TOKEN,
    )
    return HomeAssistantClient(config)


HA_CLIENT = create_ha_client()

# ---------------------------------------------------------
# Grocy / household helpers (Phase 6.5 scaffolding)
# ---------------------------------------------------------


def _parse_location_ids(env_name: str) -> List[int]:
    raw = os.getenv(env_name, "") or ""
    ids: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return ids


GROCY_HOME_A_LOCATION_IDS = _parse_location_ids("GROCY_HOME_A_LOCATION_IDS")
GROCY_HOME_B_LOCATION_IDS = _parse_location_ids("GROCY_HOME_B_LOCATION_IDS")


def filter_stock_by_household(
    stock_items: List[Dict[str, Any]],
    household: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Filter stock items by household using Grocy location IDs.
    """
    if household in (None, "", "all"):
        return stock_items

    if household == "home_a" and GROCY_HOME_A_LOCATION_IDS:
        allowed = set(GROCY_HOME_A_LOCATION_IDS)
    elif household == "home_b" and GROCY_HOME_B_LOCATION_IDS:
        allowed = set(GROCY_HOME_B_LOCATION_IDS)
    else:
        return stock_items

    return [item for item in stock_items if item.get("location_id") in allowed]


def _require_household_query(household: Optional[str]) -> str:
    """
    Gate A + Phase 6.4 rule:
    For Grocy read endpoints, we do NOT guess intent.
    Callers must explicitly provide household=home_a|home_b|all.
    """
    if household is None:
        raise HTTPException(
            status_code=400,
            detail="Query parameter 'household' is required: home_a | home_b | all",
        )
    h = (household or "").strip().lower()
    if h not in {"home_a", "home_b", "all"}:
        raise HTTPException(
            status_code=400,
            detail="household must be 'home_a', 'home_b', or 'all'",
        )
    return h


def _require_household_write_scope(household: Optional[str]) -> str:
    """
    For operations that must be household-specific (no 'all' allowed).
    Phase 6.45 reasoning endpoints also use this to prevent ambiguity.
    """
    if household is None:
        raise HTTPException(
            status_code=400,
            detail="Query parameter 'household' is required: home_a | home_b",
        )
    h = (household or "").strip().lower()
    if h not in {"home_a", "home_b"}:
        raise HTTPException(
            status_code=400,
            detail="household must be 'home_a' or 'home_b'",
        )
    return h


# ---------------------------------------------------------
# Safe async method resolution (for Grocy + BarcodeBuddy)
# ---------------------------------------------------------


async def _acall_first_existing(
    obj: Any,
    method_names: List[str],
    *args: Any,
    **kwargs: Any,
) -> Any:
    for name in method_names:
        fn = getattr(obj, name, None)
        if fn is None:
            continue
        if callable(fn):
            return await fn(*args, **kwargs)

    available = [m for m in dir(obj) if not m.startswith("_")]
    raise HTTPException(
        status_code=500,
        detail={
            "error": "Expected service method not found",
            "tried": method_names,
            "service_type": type(obj).__name__,
            "available_sample": available[:60],
        },
    )


# ----------------------------
# Health router
# ----------------------------

health_router = APIRouter(prefix="/health", tags=["health"])


@health_router.get("/ping")
async def ping() -> Dict[str, Any]:
    return {"status": "ok", "message": "ISAC brain is alive"}


@health_router.get("/system")
async def system_health() -> Dict[str, Any]:
    hostname = socket.gethostname()
    boot_time = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc)
    now = datetime.now(timezone.utc)
    uptime_seconds = (now - boot_time).total_seconds()

    cpu_percent = psutil.cpu_percent(interval=0.2)
    virtual_mem = psutil.virtual_memory()
    swap_mem = psutil.swap_memory()
    disk_usage = psutil.disk_usage("/")

    return {
        "status": "ok",
        "hostname": hostname,
        "time_utc": now.isoformat(),
        "uptime_seconds": uptime_seconds,
        "cpu_percent": cpu_percent,
        "memory": {
            "total": virtual_mem.total,
            "available": virtual_mem.available,
            "used": virtual_mem.used,
            "percent": virtual_mem.percent,
        },
        "swap": {
            "total": swap_mem.total,
            "used": swap_mem.used,
            "free": swap_mem.free,
            "percent": swap_mem.percent,
        },
        "disk": {
            "total": disk_usage.total,
            "used": disk_usage.used,
            "free": disk_usage.free,
            "percent": disk_usage.percent,
        },
    }


@health_router.get("/database")
async def database_health() -> Dict[str, Any]:
    start = time.time()
    exists = Path(DB_PATH).is_file()
    duration = time.time() - start

    if not exists:
        return {
            "status": "error",
            "message": f"Database file not found at {DB_PATH}",
            "duration_seconds": duration,
        }

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1")
        conn.close()
    except sqlite3.Error as exc:
        return {
            "status": "error",
            "message": f"Error accessing DB: {exc}",
            "duration_seconds": duration,
        }

    return {
        "status": "ok",
        "message": "Database accessible",
        "duration_seconds": duration,
    }


@health_router.get("/homeassistant")
async def homeassistant_health() -> Dict[str, Any]:
    if not HA_CLIENT:
        return {
            "status": "disabled",
            "reason": "HOMEASSISTANT_BASE_URL or HOMEASSISTANT_TOKEN not configured",
        }

    try:
        info = await HA_CLIENT.get_config()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}

    return {
        "status": "ok",
        "instance": {
            "location_name": info.get("location_name"),
            "version": info.get("version"),
        },
    }


# ---------------------------------------------------------
# Home Assistant router & endpoints
# ---------------------------------------------------------

ha_router = APIRouter(
    prefix="/homeassistant",
    tags=["homeassistant"],
    dependencies=[Depends(require_api_key)],
)


def ensure_ha_client() -> HomeAssistantClient:
    if not HA_CLIENT:
        raise HTTPException(
            status_code=503,
            detail="Home Assistant client not configured or unavailable.",
        )
    return HA_CLIENT


@ha_router.get("/entities")
async def list_entities(
    domain: Optional[str] = Query(
        default=None,
        description="Filter entities by domain (e.g. 'light', 'switch', 'sensor').",
    ),
    search: Optional[str] = Query(
        default=None,
        description="Optional case-insensitive search across entity_id and friendly_name.",
    ),
) -> Dict[str, Any]:
    client = ensure_ha_client()
    states = await client.get_states()

    results: List[Dict[str, Any]] = []
    for state in states:
        entity_id = state.get("entity_id", "")
        attributes = state.get("attributes", {})
        friendly_name = attributes.get("friendly_name", "")

        if domain and not entity_id.startswith(domain + "."):
            continue

        if search:
            needle = search.lower()
            if needle not in entity_id.lower() and needle not in str(friendly_name).lower():
                continue

        results.append(
            {
                "entity_id": entity_id,
                "state": state.get("state"),
                "friendly_name": friendly_name,
                "domain": entity_id.split(".", 1)[0] if "." in entity_id else None,
            }
        )

    return {
        "status": "ok",
        "count": len(results),
        "entities": results,
    }


@ha_router.get("/states/{entity_id}")
async def get_entity_state(entity_id: str) -> Dict[str, Any]:
    client = ensure_ha_client()

    entity_id = entity_id.strip()
    if not entity_id:
        raise HTTPException(status_code=400, detail="Entity ID must not be empty")

    try:
        state = await client.get_state(entity_id)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching state for {entity_id}: {exc}",
        ) from exc

    return {
        "status": "ok",
        "entity_id": entity_id,
        "state": state,
    }


@ha_router.post("/services/{domain}/{service}")
async def call_service(
    domain: str,
    service: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    client = ensure_ha_client()

    domain = domain.strip()
    service = service.strip()
    if not domain or not service:
        raise HTTPException(
            status_code=400,
            detail="Domain and service must not be empty",
        )

    try:
        result = await client.call_service(domain=domain, service=service, data=data)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Error calling service {domain}.{service}: {exc}",
        ) from exc

    return {
        "status": "ok",
        "domain": domain,
        "service": service,
        "result": result,
    }


@ha_router.get("/summary")
async def ha_summary() -> Dict[str, Any]:
    client = ensure_ha_client()
    states = await client.get_states()

    total = len(states)
    by_domain: Dict[str, int] = {}
    lights: List[Dict[str, Any]] = []
    switches: List[Dict[str, Any]] = []
    problem_entities: List[Dict[str, Any]] = []

    for entry in states:
        entity_id = entry.get("entity_id", "")
        domain = entity_id.split(".", 1)[0] if "." in entity_id else "unknown"
        by_domain[domain] = by_domain.get(domain, 0) + 1

        state = entry.get("state")
        attributes = entry.get("attributes", {})
        friendly_name = attributes.get("friendly_name", entity_id)

        if domain == "light":
            lights.append(
                {
                    "entity_id": entity_id,
                    "friendly_name": friendly_name,
                    "state": state,
                }
            )
        elif domain == "switch":
            switches.append(
                {
                    "entity_id": entity_id,
                    "friendly_name": friendly_name,
                    "state": state,
                }
            )

        if state in ("unavailable", "unknown"):
            problem_entities.append(entry)

    now_utc = datetime.now(timezone.utc).isoformat()

    return {
        "status": "ok",
        "generated_at_utc": now_utc,
        "entity_counts": {
            "total": total,
            "by_domain": by_domain,
        },
        "domains": {
            "light": lights,
            "switch": switches,
        },
        "problem_entities": problem_entities,
    }


# ---------------------------------------------------------
# Grocy router & endpoints
# ---------------------------------------------------------

grocy_router = APIRouter(
    prefix="/grocy",
    tags=["grocy"],
    dependencies=[Depends(require_api_key)],
)


def _grocy_disabled_response(reason: Optional[str] = None) -> Dict[str, Any]:
    base_reason = (
        "Grocy not configured (missing GROCY_HOME_A_* and/or GROCY_HOME_B_* env vars, "
        "or client not initialized)"
    )
    return {
        "status": "disabled",
        "reason": reason or base_reason,
    }


async def get_grocy_client() -> Optional[GrocyClient]:
    """
    Dependency that returns a configured GrocyClient, or None if
    configuration is missing / invalid.
    """
    try:
        client = await create_grocy_client()
    except GrocyError:
        return None
    return client


# ----------------------------
# Phase 6.45 — Scan-Driven Product Knowledge (Read-only)
# ----------------------------

class ProductSuggestion(BaseModel):
    """
    A sparse suggestion object that can be progressively enriched over time.
    v1 starts mostly-empty; later phases populate from OpenFoodFacts / learned mappings.
    """
    name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    location_id: Optional[int] = None
    qu_id_purchase: Optional[int] = None
    qu_id_stock: Optional[int] = None
    confidence: Literal["low", "medium", "high"] = "low"
    notes: Optional[str] = None


class GrocyBarcodeInspectResponse(BaseModel):
    barcode: str
    household: Literal["home_a", "home_b"]
    found_in_grocy: bool
    product: Optional[Dict[str, Any]] = None
    suggestion: ProductSuggestion
    next_actions: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)


async def _try_grocy_lookup_by_barcode(
    client: GrocyClient,
    household: str,
    barcode: str,
) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
    """
    Attempt to look up a barcode in Grocy using whichever method the GrocyClient exposes.
    This keeps main.py stable even as GrocyClient evolves.
    """
    tried_methods = [
        "get_product_by_barcode",
        "get_product_for_barcode",
        "lookup_barcode",
        "barcode_lookup",
        "get_barcode_product",
        "get_product_from_barcode",
        "get_product_by_ean",
        "get_product_by_upc",
    ]

    reasons: List[str] = []
    fn = None
    for name in tried_methods:
        candidate = getattr(client, name, None)
        if callable(candidate):
            fn = candidate
            break

    if fn is None:
        reasons.append("grocy_client_has_no_barcode_lookup_method")
        return (False, None, reasons)

    try:
        result = await fn(household=household, barcode=barcode)  # type: ignore[misc]
    except TypeError:
        # Some implementations might use a different signature; try with minimal args.
        try:
            result = await fn(barcode=barcode)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"grocy_barcode_lookup_error: {exc}")
            return (False, None, reasons)
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"grocy_barcode_lookup_error: {exc}")
        return (False, None, reasons)

    if result is None:
        reasons.append("barcode_not_found_in_grocy")
        return (False, None, reasons)

    if isinstance(result, dict):
        return (True, result, reasons)

    # If the client returns a list or other shape, wrap it safely.
    return (True, {"result": result}, reasons)


@grocy_router.get(
    "/inspect-barcode",
    response_model=GrocyBarcodeInspectResponse,
    summary="Inspect a barcode (read-only) and return a suggestion object. No writes.",
)
async def grocy_inspect_barcode(
    barcode: str = Query(..., min_length=1, description="Barcode / UPC / EAN string"),
    household: Optional[str] = Query(
        default=None,
        description="REQUIRED. home_a | home_b (no 'all' allowed for scan reasoning).",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> GrocyBarcodeInspectResponse:
    if client is None:
        # Return disabled as a normal response_model payload with low confidence.
        # This keeps callers from crashing on 503s when services are intentionally off.
        hh = _require_household_write_scope(household)
        return GrocyBarcodeInspectResponse(
            barcode=barcode,
            household=hh,  # type: ignore[arg-type]
            found_in_grocy=False,
            product=None,
            suggestion=ProductSuggestion(
                confidence="low",
                notes="Grocy client not configured; cannot inspect barcode.",
            ),
            next_actions=["configure_grocy", "user_confirmation_required"],
            reasons=["grocy_disabled"],
        )

    hh = _require_household_write_scope(household)
    code = (barcode or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="barcode must not be empty")

    found, product, reasons = await _try_grocy_lookup_by_barcode(
        client=client,
        household=hh,
        barcode=code,
    )

    if found and product:
        # Minimal inference: if Grocy provides a name, surface it as high confidence.
        name = None
        for key in ("name", "product_name", "product", "title"):
            val = product.get(key) if isinstance(product, dict) else None
            if val:
                name = str(val)
                break

        suggestion = ProductSuggestion(
            name=name,
            confidence="high" if name else "medium",
            notes="Found existing product in Grocy for this barcode.",
        )

        return GrocyBarcodeInspectResponse(
            barcode=code,
            household=hh,  # type: ignore[arg-type]
            found_in_grocy=True,
            product=product,
            suggestion=suggestion,
            next_actions=["no_action_needed", "user_confirmation_required"],
            reasons=reasons,
        )

    # Not found (or lookup unavailable): return a sparse suggestion object.
    suggestion = ProductSuggestion(
        confidence="low",
        notes="No existing product found in Grocy for this barcode.",
    )

    next_actions = ["external_lookup", "user_confirmation_required"]
    if "grocy_client_has_no_barcode_lookup_method" in reasons:
        next_actions = ["add_grocy_barcode_lookup", "external_lookup", "user_confirmation_required"]
        suggestion.notes = (
            "Grocy client does not yet support barcode lookup; "
            "Phase 6.45 will add external lookup and (optionally) Grocy barcode mapping."
        )

    return GrocyBarcodeInspectResponse(
        barcode=code,
        household=hh,  # type: ignore[arg-type]
        found_in_grocy=False,
        product=None,
        suggestion=suggestion,
        next_actions=next_actions,
        reasons=reasons,
    )


@grocy_router.get("/health")
async def grocy_health(
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        return _grocy_disabled_response()

    try:
        info = await client.health()
        return {"status": "ok", "instances": info}
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}


@grocy_router.get("/stock")
async def grocy_stock(
    household: Optional[str] = Query(
        default=None,
        description="REQUIRED. Scope by household: 'home_a', 'home_b', or explicitly 'all'.",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        return _grocy_disabled_response()

    hh = _require_household_query(household)

    try:
        raw = await client.get_stock_overview(household=hh)
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}

    filtered_payload: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            filtered_payload[key] = filter_stock_by_household(value, hh)
        else:
            filtered_payload[key] = value

    return {
        "status": "ok",
        "household": hh,
        "stock": filtered_payload,
    }


@grocy_router.get("/shopping-list")
async def grocy_shopping_list(
    household: Optional[str] = Query(
        default=None,
        description="REQUIRED. Scope by household: 'home_a', 'home_b', or explicitly 'all'.",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        return _grocy_disabled_response()

    hh = _require_household_query(household)

    try:
        data = await client.get_shopping_list(household=hh)
        return {
            "status": "ok",
            "household": hh,
            "shopping_list": data,
        }
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}


@grocy_router.get("/products")
async def grocy_products(
    household: Optional[str] = Query(
        default=None,
        description="REQUIRED. Scope by household: 'home_a', 'home_b', or explicitly 'all'.",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        return _grocy_disabled_response()

    hh = _require_household_query(household)

    try:
        products = await client.get_products(household=hh)
        return {
            "status": "ok",
            "household": hh,
            "products": products,
        }
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}


@grocy_router.get("/locations")
async def grocy_locations(
    household: Optional[str] = Query(
        default=None,
        description="REQUIRED. Scope by household: 'home_a', 'home_b', or explicitly 'all'.",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        return _grocy_disabled_response()

    hh = _require_household_query(household)

    try:
        locations = await client.get_locations(household=hh)
        return {
            "status": "ok",
            "household": hh,
            "locations": locations,
        }
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}


# ----------------------------
# NEW: Quantity Units endpoint (Phase 6.4 Step 4)
# ----------------------------

@grocy_router.get("/quantity-units")
async def grocy_quantity_units(
    household: Optional[str] = Query(
        default=None,
        description="REQUIRED. Scope by household: 'home_a', 'home_b', or explicitly 'all'.",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    """
    List Grocy quantity units via /api/objects/quantity_units.
    Required for explicit unit selection during product creation.
    """
    if client is None:
        return _grocy_disabled_response()

    hh = _require_household_query(household)

    try:
        units = await client.get_quantity_units(household=hh)
        return {
            "status": "ok",
            "household": hh,
            "quantity_units": units,
        }
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}


@grocy_router.get("/summary")
async def grocy_summary(
    household: Optional[str] = Query(
        default=None,
        description="REQUIRED. Scope by household: 'home_a', 'home_b', or explicitly 'all'.",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        return _grocy_disabled_response()

    hh = _require_household_query(household)

    try:
        stock = await client.get_stock_overview(household=hh)
        shopping_list = await client.get_shopping_list(household=hh)
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}

    filtered_stock: Dict[str, Any] = {}
    for key, value in stock.items():
        if isinstance(value, list):
            filtered_stock[key] = filter_stock_by_household(value, hh)
        else:
            filtered_stock[key] = value

    return {
        "status": "ok",
        "household": hh,
        "stock": filtered_stock,
        "shopping_list": shopping_list,
    }


@grocy_router.get("/combined")
async def grocy_combined(
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        return _grocy_disabled_response()

    try:
        combined = await _acall_first_existing(
            client,
            [
                "get_combined",
                "combined_inventory",
                "get_combined_inventory",
                "combined",
            ],
        )
        return {"status": "ok", "combined": combined}
    except HTTPException:
        raise
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def _normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum() or ch.isspace()).strip()


def _build_inventory_index_with_names(
    stock_payload: Dict[str, Any],
    household: str,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    raw_items: List[Dict[str, Any]] = []
    for value in stock_payload.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    raw_items.append(item)

    filtered_items = filter_stock_by_household(raw_items, household)

    index: Dict[str, float] = {}
    display_names: Dict[str, str] = {}
    for item in filtered_items:
        product_name = item.get("product") or item.get("name")
        if not product_name:
            continue
        product_name_str = str(product_name)
        norm = _normalize_name(product_name_str)
        try:
            amount = float(item.get("amount", 0))
        except (TypeError, ValueError):
            amount = 0.0

        index[norm] = index.get(norm, 0.0) + amount
        if norm not in display_names:
            display_names[norm] = product_name_str

    return index, display_names


@grocy_router.get("/compare")
async def grocy_compare(
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        return _grocy_disabled_response()

    try:
        stock_a = await client.get_stock_overview(household="home_a")
        stock_b = await client.get_stock_overview(household="home_b")
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}

    inv_a, names_a = _build_inventory_index_with_names(stock_a, "home_a")
    inv_b, names_b = _build_inventory_index_with_names(stock_b, "home_b")

    all_keys = set(inv_a.keys()) | set(inv_b.keys())

    only_home_a: List[Dict[str, Any]] = []
    only_home_b: List[Dict[str, Any]] = []
    both: List[Dict[str, Any]] = []
    diffs: List[Dict[str, Any]] = []

    for key in all_keys:
        amount_a = inv_a.get(key, 0.0)
        amount_b = inv_b.get(key, 0.0)
        display_name = names_a.get(key) or names_b.get(key) or key

        in_a = amount_a > 0
        in_b = amount_b > 0

        if in_a and in_b:
            entry = {
                "name": display_name,
                "home_a_amount": amount_a,
                "home_b_amount": amount_b,
                "delta": amount_a - amount_b,
            }
            both.append(entry)
            if amount_a != amount_b:
                diffs.append(entry)
        elif in_a:
            only_home_a.append(
                {
                    "name": display_name,
                    "home_a_amount": amount_a,
                }
            )
        elif in_b:
            only_home_b.append(
                {
                    "name": display_name,
                    "home_b_amount": amount_b,
                }
            )

    def _sort_by_name(item: Dict[str, Any]) -> str:
        return str(item.get("name", "")).lower()

    only_home_a.sort(key=_sort_by_name)
    only_home_b.sort(key=_sort_by_name)
    both.sort(key=_sort_by_name)
    diffs.sort(key=_sort_by_name)

    return {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "households": ["home_a", "home_b"],
        "only_home_a": only_home_a,
        "only_home_b": only_home_b,
        "both": both,
        "diffs": diffs,
    }


# ----------------------------
# Grocy product write parity (Phase 6.4 Step 4) - Units Fix
# ----------------------------

class GrocyCreateProductRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Product name")
    location_id: int = Field(..., ge=1, description="Grocy location_id to assign to the product")
    qu_id_purchase: int = Field(..., ge=1, description="Grocy purchase unit ID (required)")
    qu_id_stock: int = Field(..., ge=1, description="Grocy stock unit ID (required)")


@grocy_router.post("/products/{household}")
async def grocy_create_product(
    household: str,
    body: GrocyCreateProductRequest,
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Grocy client not configured; create product unavailable.",
        )

    hh = (household or "").strip().lower()
    if hh not in {"home_a", "home_b"}:
        raise HTTPException(
            status_code=400,
            detail="household must be 'home_a' or 'home_b' for write operations",
        )

    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")

    try:
        created = await client.create_product(
            household=hh,
            name=name,
            location_id=body.location_id,
            qu_id_purchase=body.qu_id_purchase,
            qu_id_stock=body.qu_id_stock,
        )
    except GrocyError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Error creating product in Grocy: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error creating product: {exc}",
        ) from exc

    product_id: Optional[int] = None
    if isinstance(created, dict):
        pid = (
            created.get("id")
            or created.get("product_id")
            or created.get("created_object_id")
        )
        if pid is not None:
            try:
                product_id = int(pid)
            except (TypeError, ValueError):
                product_id = None

    return {
        "status": "ok",
        "household": hh,
        "product_id": product_id,
        "product": created,
    }


# ----------------------------
# Jarvis-managed shopping lists
# ----------------------------

VALID_LIST_HOUSEHOLDS = {"home_a", "home_b", "shared"}


def _normalize_list_household(household: str) -> str:
    h = (household or "").strip().lower()
    if h not in VALID_LIST_HOUSEHOLDS:
        raise HTTPException(
            status_code=400,
            detail="household must be 'home_a', 'home_b', or 'shared'",
        )
    return h


class ShoppingListCreateRequest(BaseModel):
    name: str
    quantity: Optional[str] = None
    source: Optional[str] = "manual"


@grocy_router.get("/list/{household}")
async def jarvis_shopping_list_get(
    household: str,
    include_completed: bool = Query(
        default=False,
        description="If true, include completed items as well.",
    ),
) -> Dict[str, Any]:
    hh = _normalize_list_household(household)
    items = get_shopping_list_items(hh, include_completed=include_completed)
    return {
        "status": "ok",
        "household": hh,
        "items": items,
    }


@grocy_router.post("/list/{household}")
async def jarvis_shopping_list_add(
    household: str,
    body: ShoppingListCreateRequest,
) -> Dict[str, Any]:
    hh = _normalize_list_household(household)
    item_id = add_shopping_list_item(
        household=hh,
        item_name=body.name.strip(),
        quantity=body.quantity.strip() if body.quantity else None,
        source=body.source,
    )
    items = get_shopping_list_items(hh, include_completed=False)
    return {
        "status": "ok",
        "household": hh,
        "created_id": item_id,
        "items": items,
    }


@grocy_router.delete("/list/{household}/{item_id}")
async def jarvis_shopping_list_delete_item(
    household: str,
    item_id: int,
) -> Dict[str, Any]:
    hh = _normalize_list_household(household)
    deleted = delete_shopping_list_item(hh, item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Item not found")

    items = get_shopping_list_items(hh, include_completed=False)
    return {
        "status": "ok",
        "household": hh,
        "items": items,
    }


@grocy_router.delete("/list/{household}")
async def jarvis_shopping_list_clear(
    household: str,
) -> Dict[str, Any]:
    hh = _normalize_list_household(household)
    deleted_count = clear_shopping_list(hh)
    return {
        "status": "ok",
        "household": hh,
        "deleted": deleted_count,
        "items": [],
    }


# ---------------------------------------------------------
# Meal planner router & endpoints (Phase 6.5)
# ---------------------------------------------------------


class MealPlanPreferences(BaseModel):
    max_meals: int = 7
    vegetarian_nights: int = 0
    avoid_duplicates: bool = True


class ShoppingListItem(BaseModel):
    name: str
    quantity: Optional[str] = None
    home: Optional[str] = None


class PlannedMeal(BaseModel):
    date: Optional[datetime] = None
    recipe_id: str
    recipe_name: str
    ready: bool
    missing_items: List[ShoppingListItem]


class MealPlanRequest(BaseModel):
    household: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    preferences: MealPlanPreferences = MealPlanPreferences()
    persist_to_list: bool = False
    list_household: Optional[str] = None


class MealPlanResponse(BaseModel):
    household: str
    plan: List[PlannedMeal]
    shopping_list: List[ShoppingListItem]
    list_household: Optional[str] = None
    week_layout: Dict[str, List[PlannedMeal]] = Field(default_factory=dict)


class HouseholdMealPlan(BaseModel):
    household: str
    plan: List[PlannedMeal]
    shopping_list: List[ShoppingListItem]
    list_household: Optional[str] = None
    week_layout: Dict[str, List[PlannedMeal]] = Field(default_factory=dict)


class MultiMealPlanRequest(BaseModel):
    households: List[str]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    preferences: MealPlanPreferences = MealPlanPreferences()
    persist_to_lists: bool = False
    list_household: Optional[str] = None


class MultiMealPlanResponse(BaseModel):
    households: List[str]
    plans: List[HouseholdMealPlan]
    combined_shopping_by_household: Dict[str, List[ShoppingListItem]]
    combined_shopping_all: List[ShoppingListItem]


RECIPE_LIBRARY: List[Dict[str, Any]] = [
    {
        "id": "pasta_bake",
        "name": "Pasta Bake",
        "tags": ["dinner"],
        "ingredients": [
            {"name": "Pasta", "quantity": "250g"},
            {"name": "Tomato Sauce", "quantity": "1 jar"},
            {"name": "Mozzarella", "quantity": "150g"},
        ],
    },
    {
        "id": "veggie_chili",
        "name": "Vegetarian Chili",
        "tags": ["dinner", "vegetarian"],
        "ingredients": [
            {"name": "Kidney Beans", "quantity": "1 can"},
            {"name": "Black Beans", "quantity": "1 can"},
            {"name": "Tomatoes", "quantity": "1 can"},
        ],
    },
    {
        "id": "chicken_rice_bowl",
        "name": "Chicken Rice Bowl",
        "tags": ["dinner"],
        "ingredients": [
            {"name": "Chicken Breast", "quantity": "300g"},
            {"name": "Rice", "quantity": "200g"},
            {"name": "Broccoli", "quantity": "150g"},
        ],
    },
]


def _build_inventory_index(
    stock_payload: Dict[str, Any],
    household: str,
) -> Dict[str, float]:
    raw_items: List[Dict[str, Any]] = []
    for value in stock_payload.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    raw_items.append(item)

    filtered_items = filter_stock_by_household(raw_items, household)

    index: Dict[str, float] = {}
    for item in filtered_items:
        product_name = item.get("product") or item.get("name")
        if not product_name:
            continue
        norm = _normalize_name(str(product_name))
        try:
            amount = float(item.get("amount", 0))
        except (TypeError, ValueError):
            amount = 0.0
        index[norm] = index.get(norm, 0.0) + amount

    return index


def _evaluate_recipe_against_inventory(
    recipe: Dict[str, Any],
    inventory_index: Dict[str, float],
    household: str,
) -> Dict[str, Any]:
    missing: List[ShoppingListItem] = []
    for ingredient in recipe.get("ingredients", []):
        ing_name = str(ingredient.get("name", "")).strip()
        if not ing_name:
            continue
        norm = _normalize_name(ing_name)
        available_amount = inventory_index.get(norm, 0.0)
        if available_amount <= 0:
            missing.append(
                ShoppingListItem(
                    name=ing_name,
                    quantity=str(ingredient.get("quantity") or ""),
                    home=household,
                )
            )

    ready = len(missing) == 0

    return {
        "ready": ready,
        "missing_items": missing,
    }


WEEKDAY_KEYS: List[str] = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def _build_week_layout(plan: List[PlannedMeal]) -> Dict[str, List[PlannedMeal]]:
    layout: Dict[str, List[PlannedMeal]] = {day: [] for day in WEEKDAY_KEYS}
    layout["unscheduled"] = []

    for meal in plan:
        if meal.date is None:
            layout["unscheduled"].append(meal)
            continue

        weekday_index = meal.date.weekday()
        if 0 <= weekday_index < len(WEEKDAY_KEYS):
            day_key = WEEKDAY_KEYS[weekday_index]
        else:
            day_key = "unscheduled"
        layout[day_key].append(meal)

    return layout


mealplanner_router = APIRouter(
    prefix="/mealplanner",
    tags=["mealplanner"],
    dependencies=[Depends(require_api_key)],
)


@mealplanner_router.post("/plan", response_model=MealPlanResponse)
async def generate_meal_plan(
    body: MealPlanRequest,
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> MealPlanResponse:
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Grocy client not configured; meal planning is unavailable.",
        )

    household = body.household.strip().lower()
    if household not in {"home_a", "home_b"}:
        raise HTTPException(
            status_code=400,
            detail="household must be 'home_a' or 'home_b'",
        )

    target_list_household: Optional[str] = None
    if body.persist_to_list:
        target_list_household = _normalize_list_household(body.list_household or household)

    try:
        stock_payload = await client.get_stock_overview(household=household)
    except GrocyError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Error fetching stock from Grocy: {exc}",
        ) from exc

    inventory_index = _build_inventory_index(stock_payload, household)
    preferences = body.preferences or MealPlanPreferences()

    evaluated_recipes: List[Dict[str, Any]] = []
    vegetarian_needed = max(preferences.vegetarian_nights, 0)
    veg_recipes: List[Dict[str, Any]] = []
    nonveg_recipes: List[Dict[str, Any]] = []

    for recipe in RECIPE_LIBRARY:
        eval_result = _evaluate_recipe_against_inventory(recipe, inventory_index, household)
        recipe_entry = {
            "recipe": recipe,
            "ready": eval_result["ready"],
            "missing_items": eval_result["missing_items"],
        }
        tags = {t.lower() for t in recipe.get("tags", [])}
        if "vegetarian" in tags:
            veg_recipes.append(recipe_entry)
        else:
            nonveg_recipes.append(recipe_entry)
        evaluated_recipes.append(recipe_entry)

    def _sort_key(entry: Dict[str, Any]) -> Any:
        return (
            0 if entry["ready"] else 1,
            len(entry["missing_items"]),
        )

    veg_recipes.sort(key=_sort_key)
    nonveg_recipes.sort(key=_sort_key)

    max_meals = max(preferences.max_meals, 1)
    chosen: List[Dict[str, Any]] = []

    for entry in veg_recipes:
        if len(chosen) >= max_meals or len(chosen) >= vegetarian_needed:
            break
        chosen.append(entry)

    pool = veg_recipes + nonveg_recipes
    if preferences.avoid_duplicates:
        chosen_ids = {e["recipe"]["id"] for e in chosen}
        pool = [e for e in pool if e["recipe"]["id"] not in chosen_ids]

    for entry in pool:
        if len(chosen) >= max_meals:
            break
        chosen.append(entry)

    planned_meals: List[PlannedMeal] = []
    if body.start_date and body.end_date:
        if body.end_date < body.start_date:
            raise HTTPException(
                status_code=400,
                detail="end_date must be on or after start_date",
            )
        total_days = (body.end_date - body.start_date).days + 1
        if total_days <= 0:
            total_days = 1
    else:
        total_days = 0

    raw_missing_items: List[ShoppingListItem] = []

    for idx, entry in enumerate(chosen):
        recipe = entry["recipe"]
        missing_items: List[ShoppingListItem] = entry["missing_items"]
        if body.start_date and body.end_date and total_days > 0:
            day_index = idx % total_days
            meal_date = body.start_date + timedelta(days=day_index)
        else:
            meal_date = None

        planned_meals.append(
            PlannedMeal(
                date=meal_date,
                recipe_id=str(recipe["id"]),
                recipe_name=str(recipe["name"]),
                ready=entry["ready"],
                missing_items=missing_items,
            )
        )

        raw_missing_items.extend(missing_items)

    deduped: Dict[Tuple[str, str], ShoppingListItem] = {}
    for item in raw_missing_items:
        item_name = (item.name or "").strip()
        if not item_name:
            continue
        home_tag = (item.home or household).strip().lower()
        key = (item_name.lower(), home_tag)
        if key not in deduped:
            item.home = home_tag
            deduped[key] = item

    shopping_list_items: List[ShoppingListItem] = list(deduped.values())

    if body.persist_to_list and target_list_household:
        for item in shopping_list_items:
            add_shopping_list_item(
                household=target_list_household,
                item_name=item.name,
                quantity=item.quantity,
                source="mealplanner",
            )

    week_layout = _build_week_layout(planned_meals)

    return MealPlanResponse(
        household=household,
        plan=planned_meals,
        shopping_list=shopping_list_items,
        list_household=target_list_household,
        week_layout=week_layout,
    )


@mealplanner_router.post("/plan-multi", response_model=MultiMealPlanResponse)
async def generate_multi_meal_plan(
    body: MultiMealPlanRequest,
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> MultiMealPlanResponse:
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Grocy client not configured; multi-household meal planning is unavailable.",
        )

    if not body.households:
        raise HTTPException(
            status_code=400,
            detail="households must contain at least one entry",
        )

    normalized_households: List[str] = []
    for hh in body.households:
        h = (hh or "").strip().lower()
        if h not in {"home_a", "home_b"}:
            raise HTTPException(
                status_code=400,
                detail="Each household must be 'home_a' or 'home_b'",
            )
        if h not in normalized_households:
            normalized_households.append(h)

    plans: List[HouseholdMealPlan] = []
    all_items: List[ShoppingListItem] = []

    for hh in normalized_households:
        single_body = MealPlanRequest(
            household=hh,
            start_date=body.start_date,
            end_date=body.end_date,
            preferences=body.preferences,
            persist_to_list=body.persist_to_lists,
            list_household=body.list_household or hh,
        )
        single_plan = await generate_meal_plan(single_body, client)

        plans.append(
            HouseholdMealPlan(
                household=single_plan.household,
                plan=single_plan.plan,
                shopping_list=single_plan.shopping_list,
                list_household=single_plan.list_household,
                week_layout=single_plan.week_layout,
            )
        )
        all_items.extend(single_plan.shopping_list)

    HOUSEHOLD_BUCKETS = ["home_a", "home_b", "shared"]
    combined_by_home: Dict[str, Dict[Tuple[str, str], ShoppingListItem]] = {
        bucket: {} for bucket in HOUSEHOLD_BUCKETS
    }

    for item in all_items:
        item_name = (item.name or "").strip()
        if not item_name:
            continue
        home_tag = (item.home or "").strip().lower()
        if home_tag not in {"home_a", "home_b"}:
            home_tag = "shared"

        key = (item_name.lower(), home_tag)
        if key not in combined_by_home[home_tag]:
            combined_by_home[home_tag][key] = item

    combined_shopping_by_household: Dict[str, List[ShoppingListItem]] = {}
    for home in HOUSEHOLD_BUCKETS:
        items_dict = combined_by_home.get(home, {})
        items_list = list(items_dict.values())
        items_list.sort(key=lambda i: (i.name or "").lower())
        combined_shopping_by_household[home] = items_list

    dedup_all: Dict[str, ShoppingListItem] = {}
    for item in all_items:
        name_key = (item.name or "").strip().lower()
        if not name_key:
            continue
        if name_key not in dedup_all:
            dedup_all[name_key] = item

    combined_shopping_all: List[ShoppingListItem] = list(dedup_all.values())
    combined_shopping_all.sort(key=lambda i: (i.name or "").lower())

    return MultiMealPlanResponse(
        households=normalized_households,
        plans=plans,
        combined_shopping_by_household=combined_shopping_by_household,
        combined_shopping_all=combined_shopping_all,
    )


# ---------------------------------------------------------
# BarcodeBuddy router & endpoints
# ---------------------------------------------------------

barcode_router = APIRouter(
    prefix="/barcodebuddy",
    tags=["barcodebuddy"],
    dependencies=[Depends(require_api_key)],
)


def _barcode_disabled_response(reason: Optional[str] = None) -> Dict[str, Any]:
    base_reason = (
        "BarcodeBuddy not configured (missing BARCODEBUDDY_BASE_URL or "
        "BARCODEBUDDY_API_KEY, or client not initialized)"
    )
    return {
        "status": "disabled",
        "reason": reason or base_reason,
    }


async def get_barcode_client() -> Optional[BarcodeBuddyClient]:
    try:
        client = await create_barcodebuddy_client()
    except BarcodeBuddyError:
        return None
    return client


class BarcodeScanRequest(BaseModel):
    barcode: str
    household: Optional[str] = None


class BarcodeCreateProductRequest(BaseModel):
    barcode: str = Field(..., min_length=1)
    household: str = Field(..., description="home_a | home_b")
    name: str = Field(..., min_length=1)
    location_id: int = Field(..., ge=1)

    # NEW: required unit IDs (Gate A: no defaults)
    qu_id_purchase: int = Field(..., ge=1, description="Grocy purchase unit ID (required)")
    qu_id_stock: int = Field(..., ge=1, description="Grocy stock unit ID (required)")

    add_stock: bool = False
    quantity: Optional[float] = Field(default=None, gt=0)

    best_before_date: Optional[str] = None
    purchased_date: Optional[str] = None
    price: Optional[float] = Field(default=None, gt=0)


@barcode_router.get("/health")
async def barcode_health(
    client: Optional[BarcodeBuddyClient] = Depends(get_barcode_client),
) -> Dict[str, Any]:
    if client is None:
        return _barcode_disabled_response()

    try:
        info = await client.health()
        return {"status": "ok", "info": info}
    except BarcodeBuddyError as exc:
        return {"status": "error", "error": str(exc)}


@barcode_router.get("/products")
async def barcode_products(
    client: Optional[BarcodeBuddyClient] = Depends(get_barcode_client),
) -> Dict[str, Any]:
    if client is None:
        return _barcode_disabled_response()

    return {
        "status": "error",
        "error": "Listing products is not implemented for this BarcodeBuddy client; use /barcodebuddy/scan or /barcodebuddy/product/{barcode}.",
    }


@barcode_router.get("/product/{barcode}")
async def barcode_product_lookup(
    barcode: str,
    client: Optional[BarcodeBuddyClient] = Depends(get_barcode_client),
) -> Dict[str, Any]:
    if client is None:
        return _barcode_disabled_response()

    try:
        raw = await client.scan_barcode(barcode=barcode)
        return {
            "status": "ok",
            "barcode": barcode,
            "result": raw,
        }
    except BarcodeBuddyError as exc:
        return {"status": "error", "error": str(exc)}


def _summarize_barcodebuddy_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    text_parts: List[str] = []

    data = raw.get("data")
    if isinstance(data, dict):
        msg = data.get("result")
        if msg:
            text_parts.append(str(msg))

    result_block = raw.get("result")
    if isinstance(result_block, dict):
        msg2 = result_block.get("result")
        if msg2:
            text_parts.append(str(msg2))

    full_text = " ".join(text_parts).strip()
    lower = full_text.lower()

    state = "unknown"
    can_offer_grocy_add = False

    if "unknown product already scanned" in lower:
        state = "unknown_already_scanned"
        can_offer_grocy_add = True
    elif "unknown product" in lower:
        state = "unknown"
        can_offer_grocy_add = True
    elif "increasing quantity" in lower or "quantity increased" in lower:
        state = "quantity_incremented"
    elif "added to stock" in lower or "added to inventory" in lower:
        state = "added_to_stock"
    elif "ok" in lower and not full_text:
        state = "ok"

    return {
        "state": state,
        "message": full_text or None,
        "can_offer_grocy_add": can_offer_grocy_add,
    }


@barcode_router.post("/scan")
async def barcode_scan(
    body: BarcodeScanRequest,
    client: Optional[BarcodeBuddyClient] = Depends(get_barcode_client),
) -> Dict[str, Any]:
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="BarcodeBuddy client not configured; scanning unavailable.",
        )

    barcode = (body.barcode or "").strip()
    if not barcode:
        raise HTTPException(
            status_code=400,
            detail="barcode must not be empty",
        )

    try:
        raw = await client.scan_barcode(barcode=barcode)
    except BarcodeBuddyError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Error querying BarcodeBuddy: {exc}",
        ) from exc

    summary: Dict[str, Any]
    if isinstance(raw, dict):
        summary = _summarize_barcodebuddy_result(raw)
    else:
        summary = {
            "state": "unknown",
            "message": None,
            "can_offer_grocy_add": False,
        }

    return {
        "status": "ok",
        "barcode": barcode,
        "household": body.household,
        "raw_result": raw,
        "summary": summary,
    }


@barcode_router.post("/create-product")
async def barcode_create_product(
    body: BarcodeCreateProductRequest,
    bb_client: Optional[BarcodeBuddyClient] = Depends(get_barcode_client),
    grocy_client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if bb_client is None:
        raise HTTPException(
            status_code=503,
            detail="BarcodeBuddy client not configured; create-product unavailable.",
        )
    if grocy_client is None:
        raise HTTPException(
            status_code=503,
            detail="Grocy client not configured; create-product unavailable.",
        )

    household = (body.household or "").strip().lower()
    if household not in {"home_a", "home_b"}:
        raise HTTPException(
            status_code=400,
            detail="household must be 'home_a' or 'home_b'",
        )

    if body.add_stock and body.quantity is None:
        raise HTTPException(
            status_code=400,
            detail="quantity is required when add_stock=true",
        )

    created = await _acall_first_existing(
        grocy_client,
        [
            "create_product",
            "create_product_in_household",
            "create_product_with_location",
            "create_product_and_link_barcode",
            "create_product_for_household",
        ],
        household=household,
        name=body.name,
        location_id=body.location_id,
        qu_id_purchase=body.qu_id_purchase,
        qu_id_stock=body.qu_id_stock,
    )

    product_id: Optional[int] = None
    if isinstance(created, dict):
        pid = (
            created.get("product_id")
            or created.get("id")
            or (created.get("product") or {}).get("id")
        )
        if pid is not None:
            try:
                product_id = int(pid)
            except (TypeError, ValueError):
                product_id = None

    if product_id is None:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Could not determine product_id from Grocy create response",
                "created": created,
            },
        )

    barcode_link = await _acall_first_existing(
        grocy_client,
        [
            "link_barcode",
            "link_barcode_to_product",
            "create_product_barcode",
            "create_product_barcode_link",
            "add_barcode",
        ],
        barcode=body.barcode,
        household=household,
        product_id=product_id,
    )

    stock_result: Optional[Any] = None
    if body.add_stock:
        stock_result = await _acall_first_existing(
            grocy_client,
            [
                "add_stock",
                "add_product_stock",
                "add_stock_to_product",
                "stock_add",
                "add_stock_for_product",
            ],
            household=household,
            product_id=product_id,
            quantity=body.quantity,
            best_before_date=body.best_before_date,
            purchased_date=body.purchased_date,
            price=body.price,
        )

    return {
        "status": "ok",
        "household": household,
        "barcode": body.barcode,
        "product_id": product_id,
        "product": created,
        "barcode_link": barcode_link,
        "stock_added": bool(body.add_stock),
        "stock_result": stock_result,
        "summary": {
            "household": household,
            "product_id": product_id,
            "barcode": body.barcode,
            "stock_added": bool(body.add_stock),
            "quantity": body.quantity if body.add_stock else None,
            "best_before_date": body.best_before_date,
            "price": body.price,
        },
    }


# ----------------------------
# Core ask/history routes
# ----------------------------


@app.get("/", include_in_schema=False)
async def root() -> Dict[str, Any]:
    return {
        "message": "ISAC brain is running",
        "llm_provider": JARVIS_LLM_PROVIDER,
        "llm_model": LLM_MODEL,
    }


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
async def ask_jarvis(body: AskRequest, request: Request) -> AskResponse:
    system_prompt = body.system_prompt or (
        "You are ISAC, a helpful assistant running inside a user's homelab. "
        "Be concise, clear, and practical. When the user references devices or "
        "services, assume they may exist in the homelab environment."
    )

    conversation_context = request.headers.get("X-Jarvis-Context", "")

    messages = []
    if conversation_context:
        messages.append({"role": "system", "content": conversation_context})
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": body.message})

    try:
        answer = await MODEL_CLIENT.generate_chat_completion(
            model=LLM_MODEL,
            messages=messages,
            temperature=body.temperature,
            max_output_tokens=body.max_output_tokens,
        )
    except ModelClientError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error from LLM provider: {exc}",
        ) from exc

    try:
        log_conversation(LLM_MODEL, body.message, answer)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Failed to log conversation: {e}")

    return {"model": LLM_MODEL, "answer": answer}


@app.get("/history", dependencies=[Depends(require_api_key)])
def history(limit: int = 20) -> Dict[str, Any]:
    items = fetch_recent_conversations(limit=limit)
    return {"count": len(items), "items": items}


# ----------------------------
# Exception handlers
# ----------------------------


@app.exception_handler(ModelClientError)
async def model_client_error_handler(
    request: Request, exc: ModelClientError
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": f"Model client error: {exc}"},
    )


# ----------------------------
# Mount routers
# ----------------------------

app.include_router(health_router)
app.include_router(ha_router)
app.include_router(grocy_router)
app.include_router(mealplanner_router)
app.include_router(barcode_router)
