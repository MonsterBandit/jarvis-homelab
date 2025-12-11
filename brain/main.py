from __future__ import annotations

import os
import socket
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model_client import (
    OpenAICompatibleClient,
    GeminiModelClient,
    ModelClientError,
)
from services.homeassistant import HomeAssistantClient, HomeAssistantConfig


# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------

app = FastAPI(
    title="Jarvis Brain",
    description="Backend API for your personal Jarvis assistant",
    version="0.2.0",
)

# Allow the web UI (jarvis-agent / NGINX front) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can lock this down to specific domains later
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

    # newest last for nicer display
    rows.reverse()
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
    # NOTE: This will raise a clear ModelClientError if someone
    # actually tries to use it before we implement the HTTP calls.
    model_client = GeminiModelClient(
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
    )
else:
    raise RuntimeError(f"Unsupported JARVIS_LLM_PROVIDER: {LLM_PROVIDER}")


# ---------------------------------------------------------
# Home Assistant client configuration
# ---------------------------------------------------------

HA_BASE_URL = os.getenv("HOMEASSISTANT_BASE_URL")
HA_TOKEN = os.getenv("HOMEASSISTANT_TOKEN")


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
        timeout=5.0,
    )
    return HomeAssistantClient(config)


app.state.ha_client = create_ha_client()


# ---------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------

class AskRequest(BaseModel):
    message: str


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
        "version": "0.2.0",
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

@app.get("/homeassistant/health")
async def homeassistant_health() -> Dict[str, Any]:
    """
    Health check for Home Assistant connectivity.

    Uses the HomeAssistantClient to call HA's /api/ endpoint.
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


# ---------------------------------------------------------
# Home Assistant States & Services
# ---------------------------------------------------------

@app.get("/homeassistant/states")
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


@app.get("/homeassistant/state/{entity_id}")
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


@app.post("/homeassistant/service/{domain}/{service}")
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


# ---------------------------------------------------------
# Core ask/history endpoints
# ---------------------------------------------------------

@app.post("/ask")
def ask_jarvis(body: AskRequest) -> Dict[str, Any]:
    """
    Proxy a simple question to a model backend and return the answer.
    Also logs the exchange in SQLite.
    """
    # Prepare messages (unchanged logic, just made explicit)
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

    # Call through the model client instead of using requests directly
    try:
        data = model_client.chat(
            messages=messages,
            temperature=0.5,
        )
    except ModelClientError as e:
        # Uniform error surface for anything in the model backend
        raise HTTPException(status_code=502, detail=str(e))

    # Extract answer like before
    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(
            status_code=500,
            detail="Unexpected response from LLM provider.",
        )

    # Log to SQLite (unchanged)
    try:
        log_conversation(LLM_MODEL, body.message, answer)
    except Exception as e:  # noqa: BLE001
        # don't break the API if logging fails; just note it
        print(f"[WARN] Failed to log conversation: {e}")

    return {"model": LLM_MODEL, "answer": answer}


@app.get("/history")
def history(limit: int = 20) -> Dict[str, Any]:
    """
    Return the most recent conversation entries (user question + Jarvis reply).
    """
    items = fetch_recent_conversations(limit=limit)
    return {"count": len(items), "items": items}
