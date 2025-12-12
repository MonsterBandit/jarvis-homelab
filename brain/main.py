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
)
from fastapi.middleware.cors import CORSMiddleware
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

# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------

app = FastAPI(
    title="Jarvis Brain",
    description="Backend API for your personal Jarvis assistant",
    version="0.5.0",
)

# Allow the web UI (jarvis-agent / NGINX front) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # can be restricted later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Record when the service started (for uptime)
START_TIME = time.time()

# ---------------------------------------------------------
# Database setup (for conversation history)
# ---------------------------------------------------------

DATA_DIR = Path("/app/data")
DB_PATH = DATA_DIR / "jarvis_brain.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            model TEXT NOT NULL,
            user_message TEXT NOT NULL,
            jarvis_answer TEXT NOT NULL
        )
        """
    )
    return conn


def log_conversation(model: str, message: str, answer: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    conn = get_db_conn()
    try:
        conn.execute(
            """
            INSERT INTO conversations (ts_utc, model, user_message, jarvis_answer)
            VALUES (?, ?, ?, ?)
            """,
            (ts, model, message, answer),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_recent_conversations(limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_db_conn()
    try:
        cur = conn.execute(
            """
            SELECT id, ts_utc, model, user_message, jarvis_answer
            FROM conversations
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    rows.reverse()  # newest last for nicer display
    return [
        {
            "id": row[0],
            "ts_utc": row[1],
            "model": row[2],
            "user_message": row[3],
            "jarvis_answer": row[4],
        }
        for row in rows
    ]


# ---------------------------------------------------------
# LLM client configuration (model-agnostic brain)
# ---------------------------------------------------------

LLM_API_KEY = os.getenv("JARVIS_LLM_API_KEY")
LLM_BASE_URL = os.getenv("JARVIS_LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("JARVIS_LLM_MODEL", "gpt-4o-mini")
LLM_PROVIDER = os.getenv("JARVIS_LLM_PROVIDER", "openai").lower()

if not LLM_API_KEY:
    raise RuntimeError("JARVIS_LLM_API_KEY must be set in the environment.")

if LLM_PROVIDER == "openai":
    model_client = OpenAICompatibleClient(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
    )
elif LLM_PROVIDER == "gemini":
    model_client = GeminiModelClient(
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
    )
else:
    raise RuntimeError(f"Unsupported JARVIS_LLM_PROVIDER: {LLM_PROVIDER}")

# ---------------------------------------------------------
# API key security (optional)
# ---------------------------------------------------------

API_KEY = os.getenv("JARVIS_API_KEY")


def require_api_key(x_jarvis_api_key: str = Header(default="")) -> None:
    """
    Optional API key guard.

    If JARVIS_API_KEY is set in the environment, any endpoint that includes
    this as a dependency will require the client to send:
        X-Jarvis-Api-Key: <JARVIS_API_KEY>

    If JARVIS_API_KEY is not set, this check is effectively disabled.
    """
    if not API_KEY:
        return

    if x_jarvis_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------
# Home Assistant client configuration
# ---------------------------------------------------------

HA_BASE_URL = os.getenv("HOMEASSISTANT_BASE_URL")
HA_TOKEN = os.getenv("HOMEASSISTANT_TOKEN")
HA_TIMEOUT = float(os.getenv("HOMEASSISTANT_TIMEOUT", "8.0"))


def create_ha_client() -> Optional[HomeAssistantClient]:
    """
    Build a HomeAssistantClient from environment variables.

    Returns None if Home Assistant is not configured.
    """
    if not HA_BASE_URL or not HA_TOKEN:
        return None

    config = HomeAssistantConfig(
        base_url=HA_BASE_URL,
        token=HA_TOKEN,
        timeout=HA_TIMEOUT,
    )
    return HomeAssistantClient(config)


app.state.ha_client = create_ha_client()

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
        item for item in stock_items
        if item.get("location_id") in allowed
    ]


# ---------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------


class AskRequest(BaseModel):
    message: str


class BarcodeScanRequest(BaseModel):
    barcode: str


# ---------------------------------------------------------
# Basic health & identity endpoints
# ---------------------------------------------------------


@app.get("/health")
def health() -> Dict[str, Any]:
    """Simple health check endpoint."""
    return {"status": "ok", "service": "jarvis-brain"}


@app.get("/whoami")
def whoami() -> Dict[str, Any]:
    """Basic identity endpoint for Jarvis."""
    return {
        "name": "Jarvis",
        "role": "Personal homelab assistant",
        "host": socket.gethostname(),
        "version": "0.5.0",
        "description": (
            "I run in your homelab and help you monitor, automate, and query your systems."
        ),
    }


@app.get("/time")
def current_time() -> Dict[str, Any]:
    """Return the current server time in UTC and local."""
    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now()
    return {
        "utc_iso": now_utc.isoformat(),
        "local_iso": now_local.isoformat(),
    }


@app.get("/system-info")
def system_info() -> Dict[str, Any]:
    """Return basic system metrics from the host."""
    cpu_percent = psutil.cpu_percent(interval=0.5)
    virtual_mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    return {
        "cpu": {"percent": cpu_percent},
        "memory": {
            "total_bytes": virtual_mem.total,
            "used_bytes": virtual_mem.used,
            "available_bytes": virtual_mem.available,
            "percent": virtual_mem.percent,
        },
        "disk_root": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "percent": disk.percent,
        },
    }


@app.get("/uptime")
def uptime() -> Dict[str, Any]:
    """How long jarvis-brain has been running (and basic host uptime)."""
    service_uptime_seconds = time.time() - START_TIME

    boot_time = datetime.fromtimestamp(psutil.boot_time(), timezone.utc)
    host_uptime_seconds = (datetime.now(timezone.utc) - boot_time).total_seconds()

    return {
        "service_uptime_seconds": int(service_uptime_seconds),
        "host_uptime_seconds": int(host_uptime_seconds),
        "host_boot_time_utc": boot_time.isoformat(),
    }


# ---------------------------------------------------------
# Home Assistant endpoints
# ---------------------------------------------------------


@app.get("/homeassistant/health", dependencies=[Depends(require_api_key)])
async def homeassistant_health() -> Dict[str, Any]:
    """
    Health check for Home Assistant connectivity.
    """
    client = getattr(app.state, "ha_client", None)
    if client is None:
        return {
            "status": "disabled",
            "reason": (
                "Home Assistant not configured "
                "(missing HOMEASSISTANT_BASE_URL or HOMEASSISTANT_TOKEN)"
            ),
        }

    ok, details = client.health()
    return {
        "status": "ok" if ok else "error",
        "details": details,
    }


@app.get("/homeassistant/states", dependencies=[Depends(require_api_key)])
async def ha_list_states() -> Dict[str, Any]:
    """
    Return all entity states from Home Assistant.
    """
    ha: Optional[HomeAssistantClient] = getattr(app.state, "ha_client", None)
    if ha is None:
        return {
            "status": "disabled",
            "reason": (
                "Home Assistant not configured "
                "(missing HOMEASSISTANT_BASE_URL or HOMEASSISTANT_TOKEN)"
            ),
        }

    try:
        states = ha.list_states()
        return {"status": "ok", "states": states}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


@app.get(
    "/homeassistant/state/{entity_id}",
    dependencies=[Depends(require_api_key)],
)
async def ha_get_state(entity_id: str) -> Dict[str, Any]:
    """
    Return the state for a single Home Assistant entity.
    """
    ha: Optional[HomeAssistantClient] = getattr(app.state, "ha_client", None)
    if ha is None:
        return {
            "status": "disabled",
            "reason": (
                "Home Assistant not configured "
                "(missing HOMEASSISTANT_BASE_URL or HOMEASSISTANT_TOKEN)"
            ),
            "entity_id": entity_id,
        }

    try:
        state = ha.get_state(entity_id)
        return {"status": "ok", "state": state}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "entity_id": entity_id,
            "error": str(exc),
        }


@app.get("/homeassistant/services", dependencies=[Depends(require_api_key)])
async def ha_list_services() -> Dict[str, Any]:
    """
    Return all available Home Assistant services.
    """
    ha: Optional[HomeAssistantClient] = getattr(app.state, "ha_client", None)
    if ha is None:
        return {
            "status": "disabled",
            "reason": (
                "Home Assistant not configured "
                "(missing HOMEASSISTANT_BASE_URL or HOMEASSISTANT_TOKEN)"
            ),
        }

    try:
        services = ha.list_services()
        return {"status": "ok", "services": services}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


@app.post(
    "/homeassistant/service/{domain}/{service}",
    dependencies=[Depends(require_api_key)],
)
async def ha_call_service(
    domain: str,
    service: str,
    data: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Call a Home Assistant service, e.g. light.turn_on.
    """
    ha: Optional[HomeAssistantClient] = getattr(app.state, "ha_client", None)
    if ha is None:
        return {
            "status": "disabled",
            "reason": (
                "Home Assistant not configured "
                "(missing HOMEASSISTANT_BASE_URL or HOMEASSISTANT_TOKEN)"
            ),
            "domain": domain,
            "service": service,
        }

    try:
        result = ha.call_service(domain, service, data or {})
        return {"status": "ok", "result": result}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "domain": domain,
            "service": service,
            "error": str(exc),
        }


@app.get("/homeassistant/summary", dependencies=[Depends(require_api_key)])
async def ha_summary() -> Dict[str, Any]:
    """
    Return a high-level summary of Home Assistant state.
    """
    ha: Optional[HomeAssistantClient] = getattr(app.state, "ha_client", None)
    if ha is None:
        return {
            "status": "disabled",
            "reason": (
                "Home Assistant not configured "
                "(missing HOMEASSISTANT_BASE_URL or HOMEASSISTANT_TOKEN)"
            ),
        }

    try:
        states = ha.list_states()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}

    total = len(states)
    by_domain: Dict[str, int] = {}

    lights: List[Dict[str, Any]] = []
    switches: List[Dict[str, Any]] = []
    climates: List[Dict[str, Any]] = []
    media_players: List[Dict[str, Any]] = []
    problem_entities: List[Dict[str, Any]] = []

    for s in states:
        entity_id = s.get("entity_id")
        state = s.get("state")
        attrs = s.get("attributes") or {}

        if entity_id and "." in entity_id:
            domain = entity_id.split(".", 1)[0]
        else:
            domain = "unknown"

        by_domain[domain] = by_domain.get(domain, 0) + 1

        entry: Dict[str, Any] = {
            "entity_id": entity_id,
            "state": state,
        }
        if "friendly_name" in attrs:
            entry["name"] = attrs["friendly_name"]

        if domain == "light":
            for key in ("brightness", "color_temp", "hs_color"):
                if key in attrs:
                    entry[key] = attrs[key]
            lights.append(entry)

        elif domain == "switch":
            switches.append(entry)

        elif domain == "climate":
            for key in (
                "temperature",
                "current_temperature",
                "target_temp_low",
                "target_temp_high",
                "hvac_mode",
                "hvac_action",
            ):
                if key in attrs:
                    entry[key] = attrs[key]
            climates.append(entry)

        elif domain == "media_player":
            for key in ("volume_level", "source", "media_title", "media_artist"):
                if key in attrs:
                    entry[key] = attrs[key]
            media_players.append(entry)

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
            "climate": climates,
            "media_player": media_players,
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
    """
    if client is None:
        return _grocy_disabled_response()

    try:
        info = await client.health()
        return {"status": "ok", "info": info.get("info")}
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
    """
    if client is None:
        return _grocy_disabled_response()

    try:
        raw = await client.get_stock_overview()
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
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    """
    Raw consolidated shopping list from Grocy.
    """
    if client is None:
        return _grocy_disabled_response()

    try:
        data = await client.get_shopping_list()
        return {"status": "ok", "shopping_list": data}
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}


@grocy_router.get("/products")
async def grocy_products(
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    """
    Raw products table from Grocy.
    """
    if client is None:
        return _grocy_disabled_response()

    try:
        products = await client.get_products()
        return {"status": "ok", "products": products}
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}


@grocy_router.get("/locations")
async def grocy_locations(
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    """
    Locations table, useful for mapping to households.
    """
    if client is None:
        return _grocy_disabled_response()

    try:
        locations = await client.get_locations()
        return {"status": "ok", "locations": locations}
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
    High-level summary combining stock + shopping list.
    """
    if client is None:
        return _grocy_disabled_response()

    try:
        stock = await client.get_stock_overview()
        shopping_list = await client.get_shopping_list()
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

barcodebuddy_router = APIRouter(
    prefix="/barcodebuddy",
    tags=["barcodebuddy"],
    dependencies=[Depends(require_api_key)],
)


def _barcodebuddy_disabled_response(reason: Optional[str] = None) -> Dict[str, Any]:
    base_reason = "BarcodeBuddy not configured (missing BARCODEBUDDY_BASE_URL)"
    return {
        "status": "disabled",
        "reason": reason or base_reason,
    }


async def get_barcodebuddy_client() -> Optional[BarcodeBuddyClient]:
    """
    Dependency that returns a configured BarcodeBuddyClient, or None if
    configuration is missing / invalid.
    """
    try:
        client = await create_barcodebuddy_client()
    except BarcodeBuddyError:
        return None
    return client


@barcodebuddy_router.get("/health")
async def barcodebuddy_health(
    client: Optional[BarcodeBuddyClient] = Depends(get_barcodebuddy_client),
) -> Dict[str, Any]:
    """
    Check connectivity to Barcode Buddy.

    Returns system info from /api/system/info.
    """
    if client is None:
        # Graceful "off" state, just like Grocy/HA.
        return _barcodebuddy_disabled_response()

    try:
        info = await client.health()
        return {"status": "ok", "info": info}
    except BarcodeBuddyError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Unexpected BarcodeBuddy error: {e}")


@barcodebuddy_router.post("/scan")
async def barcodebuddy_scan(
    payload: BarcodeScanRequest,
    client: Optional[BarcodeBuddyClient] = Depends(get_barcodebuddy_client),
) -> Dict[str, Any]:
    """
    Pass a single barcode to Barcode Buddy.

    BB will handle resolving the product, creating it in Grocy if needed,
    and performing its configured action.
    """
    if client is None:
        return _barcodebuddy_disabled_response()

    try:
        result = await client.scan_barcode(payload.barcode)
        return {
            "status": "ok",
            "barcode": payload.barcode,
            "result": result,
        }
    except BarcodeBuddyError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Unexpected BarcodeBuddy error: {e}")


# Mount routers
app.include_router(grocy_router)
app.include_router(barcodebuddy_router)

# ---------------------------------------------------------
# Core ask/history endpoints
# ---------------------------------------------------------


@app.post("/ask", dependencies=[Depends(require_api_key)])
def ask_jarvis(body: AskRequest) -> Dict[str, Any]:
    """
    Proxy a simple question to a model backend and return the answer.
    Also logs the exchange in SQLite.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are Jarvis, a helpful personal assistant for the user's homelab. "
                "Be concise and practical."
            ),
        },
        {"role": "user", "content": body.message},
    ]

    try:
        data = model_client.chat(
            messages=messages,
            temperature=0.5,
        )
    except ModelClientError as e:
        raise HTTPException(status_code=502, detail=str(e))

    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(
            status_code=500,
            detail="Unexpected response from LLM provider.",
        )

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
