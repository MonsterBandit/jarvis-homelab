from __future__ import annotations

import os
import socket
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from pydantic import BaseModel

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
    Initialize the SQLite database with a simple conversation log table.
    """
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
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
            # Ignore invalid entries instead of crashing
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

    - household="home_a" -> GROCY_HOME_A_LOCATION_IDS
    - household="home_b" -> GROCY_HOME_B_LOCATION_IDS
    - household=None or "all" -> no filtering

    If env vars are not set, returns stock unfiltered.
    """
    if household in (None, "", "all"):
        return stock_items

    if household == "home_a" and GROCY_HOME_A_LOCATION_IDS:
        allowed = set(GROCY_HOME_A_LOCATION_IDS)
    elif household == "home_b" and GROCY_HOME_B_LOCATION_IDS:
        allowed = set(GROCY_HOME_B_LOCATION_IDS)
    else:
        # Unknown household or no config; fall back to unfiltered
        return stock_items

    return [
        item
        for item in stock_items
        if item.get("location_id") in allowed
    ]


# ----------------------------
# Health router
# ----------------------------

health_router = APIRouter(prefix="/health", tags=["health"])


@health_router.get("/ping")
async def ping() -> Dict[str, Any]:
    """
    Simple liveness check.
    """
    return {"status": "ok", "message": "ISAC brain is alive"}


@health_router.get("/system")
async def system_health() -> Dict[str, Any]:
    """
    Basic system health metrics: hostname, uptime, CPU, memory, etc.
    """
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
    """
    Simple check that the SQLite DB file exists and is readable.
    """
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
    """
    Check connectivity and basic status for Home Assistant.
    """
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
    """
    List all entities (optionally filtered by domain or search text).
    """
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
            if needle not in entity_id.lower() and needle not in str(
                friendly_name
            ).lower():
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
    """
    Get full state details for a single entity.
    """
    client = ensure_ha_client()

    # Home Assistant expects the raw entity_id in the URL path, but we must ensure
    # it isn't empty or obviously malformed.
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
    """
    Call a Home Assistant service. The request body is passed directly as the service data.
    """
    client = ensure_ha_client()

    domain = domain.strip()
    service = service.strip()
    if not domain or not service:
        raise HTTPException(
            status_code=400, detail="Domain and service must not be empty"
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
    """
    High-level summary across lights, switches, and any obvious problem entities.
    """
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
        "Grocy not configured (missing GROCY_BASE_URL or GROCY_API_KEY, "
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


@grocy_router.get("/health")
async def grocy_health(
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    """
    Health check for Grocy connectivity.

    Returns per-household status for all configured Grocy instances.
    """
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
        default="all",
        description="Filter by household: 'home_a', 'home_b', or 'all'.",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    """
    Returns a stock overview (via /api/stock/volatile) and
    supports optional household filtering.

    household controls which Grocy instance(s) to query.
    Location-based filtering uses GROCY_HOME_A_LOCATION_IDS /
    GROCY_HOME_B_LOCATION_IDS to further narrow the result.
    """
    if client is None:
        return _grocy_disabled_response()

    try:
        raw = await client.get_stock_overview(household=household)
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}

    filtered_payload: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            filtered_payload[key] = filter_stock_by_household(value, household)
        else:
            filtered_payload[key] = value

    return {
        "status": "ok",
        "household": household or "all",
        "stock": filtered_payload,
    }


@grocy_router.get("/shopping-list")
async def grocy_shopping_list(
    household: Optional[str] = Query(
        default="all",
        description="Filter by household: 'home_a', 'home_b', or 'all'.",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    """
    Raw consolidated shopping list from Grocy for one or more households.
    """
    if client is None:
        return _grocy_disabled_response()

    try:
        data = await client.get_shopping_list(household=household)
        return {
            "status": "ok",
            "household": household or "all",
            "shopping_list": data,
        }
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}


@grocy_router.get("/products")
async def grocy_products(
    household: Optional[str] = Query(
        default="all",
        description="Filter by household: 'home_a', 'home_b', or 'all'.",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    """
    Raw products table from Grocy, optionally scoped by household.

    Each product in the list will include a 'household' field indicating
    which Grocy instance it came from.
    """
    if client is None:
        return _grocy_disabled_response()

    try:
        products = await client.get_products(household=household)
        return {
            "status": "ok",
            "household": household or "all",
            "products": products,
        }
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}


@grocy_router.get("/locations")
async def grocy_locations(
    household: Optional[str] = Query(
        default="all",
        description="Filter by household: 'home_a', 'home_b', or 'all'.",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    """
    Raw locations table from Grocy, optionally scoped by household.

    Each location in the list will include a 'household' field indicating
    which Grocy instance it came from.
    """
    if client is None:
        return _grocy_disabled_response()

    try:
        locations = await client.get_locations(household=household)
        return {
            "status": "ok",
            "household": household or "all",
            "locations": locations,
        }
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}


@grocy_router.get("/summary")
async def grocy_summary(
    household: Optional[str] = Query(
        default="all",
        description="Filter by household: 'home_a', 'home_b', or 'all'.",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    """
    High-level summary combining stock + shopping list,
    optionally scoped to a single household.
    """
    if client is None:
        return _grocy_disabled_response()

    try:
        stock = await client.get_stock_overview(household=household)
        shopping_list = await client.get_shopping_list(household=household)
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}

    filtered_stock: Dict[str, Any] = {}
    for key, value in stock.items():
        if isinstance(value, list):
            filtered_stock[key] = filter_stock_by_household(value, household)
        else:
            filtered_stock[key] = value

    return {
        "status": "ok",
        "household": household or "all",
        "stock": filtered_stock,
        "shopping_list": shopping_list,
    }


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


@barcode_router.get("/health")
async def barcode_health(
    client: Optional[BarcodeBuddyClient] = Depends(get_barcode_client),
) -> Dict[str, Any]:
    """
    Health check for BarcodeBuddy connectivity.
    """
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
    """
    Return all known products from BarcodeBuddy (name, barcode, etc.).
    """
    if client is None:
        return _barcode_disabled_response()

    try:
        products = await client.get_products()
        return {"status": "ok", "products": products}
    except BarcodeBuddyError as exc:
        return {"status": "error", "error": str(exc)}


@barcode_router.get("/product/{barcode}")
async def barcode_product_lookup(
    barcode: str,
    client: Optional[BarcodeBuddyClient] = Depends(get_barcode_client),
) -> Dict[str, Any]:
    """
    Look up a single product in BarcodeBuddy by barcode.
    """
    if client is None:
        return _barcode_disabled_response()

    try:
        product = await client.get_product(barcode=barcode)
        if product is None:
            return {"status": "not_found", "barcode": barcode}
        return {"status": "ok", "product": product}
    except BarcodeBuddyError as exc:
        return {"status": "error", "error": str(exc)}


# ----------------------------
# Core ask/history routes
# ----------------------------


@app.get("/", include_in_schema=False)
async def root() -> Dict[str, Any]:
    """
    Simple root endpoint.
    """
    return {
        "message": "ISAC brain is running",
        "llm_provider": JARVIS_LLM_PROVIDER,
        "llm_model": LLM_MODEL,
    }


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
async def ask_jarvis(body: AskRequest, request: Request) -> AskResponse:
    """
    Main endpoint where the UI sends user messages to be answered by the LLM.
    """
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

    # Log conversation to SQLite
    try:
        log_conversation(LLM_MODEL, body.message, answer)
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Failed to log conversation: {e}")

    return {"model": LLM_MODEL, "answer": answer}


@app.get("/history", dependencies=[Depends(require_api_key)])
def history(limit: int = 20) -> Dict[str, Any]:
    """
    Return the most recent conversation entries (user question + Jarvis reply).
    """
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
app.include_router(barcode_router)
