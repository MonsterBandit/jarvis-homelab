from __future__ import annotations

import os
import socket
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import httpx
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

# NEW: OpenFoodFacts (Phase 6.45 Step 2)
from services.openfoodfacts import (
    OpenFoodFactsError,
    create_openfoodfacts_client,
    extract_suggestion_from_off_payload,
)

from services.recipes import router as recipes_router  # Phase 6.75.1
from services.ingredient_parser import router as ingredient_parser_router  # Phase 6.75.2
from services.recipe_matcher import router as recipe_matcher_router  # Phase 6.75.3
from services.recipe_analyzer import router as recipe_analyzer_router  # Phase 6.75.4
from services.recipe_mappings import router as recipe_mappings_router  # Phase 6.75.5
from services.mealplans import router as mealplans_router
from services.mealplanner import router as mealplanner_context_router
# Alice (Phase 8) — read-only preview endpoint (NOT wired into /ask)
from alice_preview_router import router as alice_router


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
    conn.execute("PRAGMA foreign_keys = ON")
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

        # Recipe ingredient mappings (Phase 6.75.5)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recipe_ingredient_mapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                household TEXT, -- NULL = global mapping
                ingredient_norm TEXT NOT NULL,
                ingredient_display TEXT NOT NULL,
                product_id INTEGER NOT NULL,
                product_name TEXT,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_recipe_ing_map_household_norm
            ON recipe_ingredient_mapping (household, ingredient_norm)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_recipe_ing_map_norm
            ON recipe_ingredient_mapping (ingredient_norm)
            """
        )

        # -------------------------------------------------
        # Phase 6.9 — Calendar ownership & mapping (READ-ONLY)
        # -------------------------------------------------
        #
        # CONTRACT (LOCKED):
        # - These tables map calendar entities to owners/households only.
        # - We do NOT store calendar events in SQLite.
        # - We do NOT write any calendar-derived read results to SQLite.
        # - Calendar awareness is consumptive/descriptive only (context, not control).
        #

        # Maps a person/owner to a household
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS calendar_owner_household (
                owner_key TEXT PRIMARY KEY,
                household TEXT NOT NULL CHECK (household IN ('home_a', 'home_b')),
                display_name TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        # Maps Home Assistant calendar entities to an owner
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS calendar_entity_owner (
                entity_id TEXT PRIMARY KEY,
                owner_key TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                label TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (owner_key)
                    REFERENCES calendar_owner_household (owner_key)
                    ON DELETE CASCADE
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_calendar_entity_owner_owner
            ON calendar_entity_owner (owner_key)
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


# ---------------------------------------------------------
# Phase 6.9 — Calendar DB-read helpers (READ-ONLY)
# ---------------------------------------------------------

# Hard ignore list for calendar awareness reads (LOCKED)
# - Always ignored regardless of DB contents or HA discovery
CALENDAR_ENTITY_IGNORE: set[str] = {
    "calendar.kaitlyn",
    "calendar.laila",
    "calendar.apentalp_gmail_com",
}


def _normalize_calendar_household(household: Optional[str]) -> str:
    """
    Calendar awareness default = home_a.
    Only valid: home_a | home_b
    """
    h = (household or "home_a").strip().lower()
    if h not in {"home_a", "home_b"}:
        raise HTTPException(
            status_code=400,
            detail="household must be 'home_a' or 'home_b'",
        )
    return h


def get_enabled_calendar_entities_for_household(
    household: str,
) -> List[Dict[str, Any]]:
    """
    Return enabled calendar entities for a given household from DB.
    READ-ONLY. No DB writes.
    Excludes hard-ignored calendars.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                ceo.entity_id,
                ceo.owner_key,
                ceo.enabled,
                ceo.label,
                coh.household,
                coh.display_name
            FROM calendar_entity_owner ceo
            JOIN calendar_owner_household coh
                ON coh.owner_key = ceo.owner_key
            WHERE coh.household = ?
              AND ceo.enabled = 1
            ORDER BY ceo.entity_id ASC
            """,
            (household,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    results: List[Dict[str, Any]] = []
    for entity_id, owner_key, enabled, label, hh, display_name in rows:
        if not entity_id:
            continue
        if entity_id in CALENDAR_ENTITY_IGNORE:
            continue
        results.append(
            {
                "entity_id": entity_id,
                "owner_key": owner_key,
                "enabled": bool(enabled),
                "label": label,
                "household": hh,
                "owner_display_name": display_name,
            }
        )
    return results


# ---------------------------------------------------------
# Phase 6.9 — Calendar planning window helpers (pure logic)
# Server-time based, Sunday-first weeks, deterministic
# ---------------------------------------------------------

_SERVER_TZ: Optional[ZoneInfo] = None


def get_server_timezone() -> ZoneInfo:
    """
    Return a ZoneInfo representing the server's local timezone.
    Deterministic resolution order (no network, no HA, no DB):
      1) tzinfo key from datetime.now().astimezone()
      2) TZ environment variable if set
      3) Fallback to America/New_York (expected for this deployment)
    """
    global _SERVER_TZ

    if _SERVER_TZ is not None:
        return _SERVER_TZ

    # 1) Ask the OS what "local time" is, then attempt to recover a ZoneInfo key.
    try:
        local_tzinfo = datetime.now().astimezone().tzinfo
        tz_key = getattr(local_tzinfo, "key", None)
        if tz_key:
            _SERVER_TZ = ZoneInfo(str(tz_key))
            return _SERVER_TZ
    except Exception:
        pass

    # 2) TZ env var (common in containers)
    tz_env = (os.getenv("TZ") or "").strip()
    if tz_env:
        try:
            _SERVER_TZ = ZoneInfo(tz_env)
            return _SERVER_TZ
        except Exception:
            pass

    # 3) Locked expected fallback
    _SERVER_TZ = ZoneInfo("America/New_York")
    return _SERVER_TZ


def now_server(tz: Optional[ZoneInfo] = None) -> datetime:
    """
    Return current server-time as a timezone-aware datetime.
    """
    tz_final = tz or get_server_timezone()
    return datetime.now(tz_final)


def _ensure_server_aware(dt: datetime, tz: Optional[ZoneInfo] = None) -> datetime:
    """
    Normalize a datetime to server timezone.
    - If naive: interpret as server-local time (no guessing beyond that).
    - If aware: convert to server timezone.
    """
    tz_final = tz or get_server_timezone()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz_final)
    return dt.astimezone(tz_final)


def week_start_sunday(dt: datetime, tz: Optional[ZoneInfo] = None) -> datetime:
    """
    Given any datetime, return the Sunday 00:00:00 of its week in server time.
    Week definition: Sunday -> Saturday.
    """
    dt_local = _ensure_server_aware(dt, tz=tz)

    # Python weekday(): Monday=0 ... Sunday=6
    # We want Sunday=0 ... Saturday=6
    days_since_sunday = (dt_local.weekday() + 1) % 7

    start_date = (dt_local - timedelta(days=days_since_sunday)).date()
    return datetime(
        year=start_date.year,
        month=start_date.month,
        day=start_date.day,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        tzinfo=dt_local.tzinfo,
    )


def planning_window_week(
    offset_weeks: int = 0,
    anchor: Optional[datetime] = None,
    tz: Optional[ZoneInfo] = None,
) -> Tuple[datetime, datetime]:
    """
    Return a (start, end) tuple for a Sunday-first week window in server time.
    - start: Sunday 00:00:00
    - end: next Sunday 00:00:00 (end-exclusive)
    offset_weeks:
      0 = current week, 1 = next week, -1 = prior week, etc.
    anchor:
      If provided, the week is computed relative to that moment; otherwise uses now_server().
    """
    anchor_dt = anchor if anchor is not None else now_server(tz=tz)
    start = week_start_sunday(anchor_dt, tz=tz) + timedelta(weeks=int(offset_weeks))
    end = start + timedelta(days=7)
    return start, end
    

def planning_window_today(tz: ZoneInfo) -> Tuple[datetime, datetime]:
    """
    Return server-time 'today' window:
    [today 00:00:00 → tomorrow 00:00:00)
    """
    now = datetime.now(tz)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start, end   


def planning_window_current_week(
    anchor: Optional[datetime] = None,
    tz: Optional[ZoneInfo] = None,
) -> Tuple[datetime, datetime]:
    """
    Convenience wrapper for the current Sunday-start week in server time.
    """
    return planning_window_week(offset_weeks=0, anchor=anchor, tz=tz)


def planning_window_future_week(
    offset_weeks: int,
    anchor: Optional[datetime] = None,
    tz: Optional[ZoneInfo] = None,
) -> Tuple[datetime, datetime]:
    """
    Convenience wrapper for future/past weeks via offset.
    """
    return planning_window_week(offset_weeks=offset_weeks, anchor=anchor, tz=tz)


def planning_window_to_iso(
    window: Tuple[datetime, datetime],
) -> Dict[str, str]:
    """
    Convert a (start, end) window to ISO strings for logging/debugging/UI.
    Pure formatting helper (no side effects).
    """
    start, end = window
    return {"start": start.isoformat(), "end": end.isoformat()}


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


# ---------------------------------------------------------
# Phase 6.9 — Calendar awareness router (READ-ONLY)
# ---------------------------------------------------------
# CONTRACT (LOCKED):
# - Read-only signal layer. Context, not control.
# - No calendar writes. No scheduling. No reminders. No automation.
# - No Home Assistant actions (no service calls).
# - No database writes tied to calendar reads.
# - Ambiguity surfaces as notes/warnings, not assumptions.
#
# TIME RULES (LOCKED):
# - Server-time is authoritative.
# - Week starts on Sunday.
#
# ENTITY RULES (LOCKED):
# - Household-enabled entities come from DB mapping tables.
# - Hard ignore list always applies: CALENDAR_ENTITY_IGNORE
#
# Endpoints (READ-ONLY):
# - GET /calendar/entities
# - GET /calendar/today
# - GET /calendar/week
# - GET /calendar/busy/week
# ---------------------------------------------------------


calendar_router = APIRouter(
    prefix="/calendar",
    tags=["calendar"],
    dependencies=[Depends(require_api_key)],
)


def ensure_ha_configured_for_calendar_reads() -> None:
    """
    Calendar awareness requires HA base URL + token to do HA REST calendar reads.
    (This is read-only.)
    """
    if not HOMEASSISTANT_BASE_URL or not HOMEASSISTANT_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Home Assistant not configured (HOMEASSISTANT_BASE_URL / HOMEASSISTANT_TOKEN missing).",
        )


async def ha_calendar_get_events(
    entity_id: str,
    start: datetime,
    end: datetime,
) -> List[Dict[str, Any]]:
    """
    Read-only Home Assistant calendar events via REST:
      GET /api/calendars/{entity_id}?start=...&end=...
    """
    ensure_ha_configured_for_calendar_reads()

    base = HOMEASSISTANT_BASE_URL.rstrip("/")
    url = f"{base}/api/calendars/{entity_id}"

    headers = {
        "Authorization": f"Bearer {HOMEASSISTANT_TOKEN}",
        "Content-Type": "application/json",
    }

    # HA accepts RFC3339/ISO strings; we pass timezone-aware ISO.
    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
    }

    timeout = httpx.Timeout(20.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, headers=headers, params=params)

    if resp.status_code == 404:
        # Entity missing or calendar endpoint not available for this entity
        return []
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Home Assistant calendar read failed",
                "entity_id": entity_id,
                "status_code": resp.status_code,
                "body": resp.text,
            },
        )

    payload = resp.json()
    if isinstance(payload, list):
        return payload
    return []


def _coerce_ha_time_to_iso(
    value: Any,
    *,
    tz: ZoneInfo,
) -> Tuple[Optional[str], bool]:
    """
    Home Assistant calendar time fields can be:
      - {"dateTime": "..."}  (timed)
      - {"date": "YYYY-MM-DD"} (all-day)
      - "..." (string ISO-ish)

    Returns: (iso_string, is_all_day)
    - For all-day events, we normalize to midnight server tz (YYYY-MM-DDT00:00:00±HH:MM)
    - For timed events, we preserve the datetime in server tz.
    """
    if value is None:
        return (None, False)

    # Dict shape: {"dateTime": "..."} or {"date": "..."}
    if isinstance(value, dict):
        dt_val = value.get("dateTime")
        d_val = value.get("date")

        if isinstance(d_val, str) and d_val:
            # All-day: date only
            try:
                y, m, d = d_val.split("-")
                dt = datetime(int(y), int(m), int(d), 0, 0, 0, tzinfo=tz)
                return (dt.isoformat(), True)
            except Exception:
                # If parsing fails, fall through and stringify
                return (str(d_val), True)

        if isinstance(dt_val, str) and dt_val:
            # Timed: parse ISO -> convert to server tz if possible
            try:
                parsed = datetime.fromisoformat(dt_val.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc).astimezone(tz)
                else:
                    parsed = parsed.astimezone(tz)
                return (parsed.isoformat(), False)
            except Exception:
                return (dt_val, False)

    # String shape: best effort
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return (None, False)
        try:
            parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc).astimezone(tz)
            else:
                parsed = parsed.astimezone(tz)
            return (parsed.isoformat(), False)
        except Exception:
            # Could be a date-only string or non-ISO. Treat date-only as all-day.
            if len(s) == 10 and s[4] == "-" and s[7] == "-":
                try:
                    y, m, d = s.split("-")
                    dt = datetime(int(y), int(m), int(d), 0, 0, 0, tzinfo=tz)
                    return (dt.isoformat(), True)
                except Exception:
                    return (s, True)
            return (s, False)

    # Unknown type
    return (str(value), False)


def _parse_iso_to_dt(iso_str: Optional[str]) -> Optional[datetime]:
    """
    Best effort parse for sorting. Returns timezone-aware datetime when possible.
    """
    if not iso_str:
        return None
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _clip_interval_to_day(
    start: datetime,
    end: datetime,
    day_start: datetime,
    day_end: datetime,
) -> int:
    """
    Return overlap in minutes between [start, end) and a single day window.
    """
    latest_start = max(start, day_start)
    earliest_end = min(end, day_end)
    if earliest_end <= latest_start:
        return 0
    return int((earliest_end - latest_start).total_seconds() // 60)


def _busy_score_from_minutes(minutes: int) -> int:
    """
    Convert busy minutes to a 0–100 score.
    8h (480m) == 100, linear scale, capped.
    """
    return min(100, int((minutes / 480) * 100))        


@calendar_router.get("/entities")
async def calendar_entities(
    household: Optional[str] = Query(
        default="home_a",
        description="home_a | home_b (default: home_a)",
    ),
) -> Dict[str, Any]:
    """
    Read-only list of enabled calendar entities for the household.
    DB-read only. Hard-ignores kaitlyn/laila/apentalp.
    """
    hh = _normalize_calendar_household(household)
    entities = get_enabled_calendar_entities_for_household(hh)

    return {
        "status": "ok",
        "household": hh,
        "ignored": sorted(list(CALENDAR_ENTITY_IGNORE)),
        "count": len(entities),
        "entities": entities,
    }


@calendar_router.get("/week")
async def calendar_week(
    offset_weeks: int = Query(
        default=0,
        description="0=current week (Sunday-start), 1=next week, -1=previous week, etc.",
    ),
    household: Optional[str] = Query(
        default="home_a",
        description="home_a | home_b (default: home_a)",
    ),
    include_raw: bool = Query(
        default=False,
        description="If true, include raw HA event payloads. Default false.",
    ),
) -> Dict[str, Any]:
    """
    Read-only: Fetch calendar events for a Sunday-start planning week window
    using server timezone helpers (deterministic).

    POLISH A:
    - Normalizes start/end into start_iso/end_iso
    - Normalizes all_day boolean
    - Improves sorting by actual datetime
    - raw payload included only if include_raw=true
    """
    hh = _normalize_calendar_household(household)

    tz = get_server_timezone()
    window_start, window_end = planning_window_week(offset_weeks=offset_weeks, tz=tz)

    calendars = get_enabled_calendar_entities_for_household(hh)

    per_calendar: List[Dict[str, Any]] = []
    combined_events: List[Dict[str, Any]] = []

    for cal in calendars:
        entity_id = cal.get("entity_id")
        if not entity_id or entity_id in CALENDAR_ENTITY_IGNORE:
            continue

        events = await ha_calendar_get_events(
            entity_id=entity_id,
            start=window_start,
            end=window_end,
        )

        normalized_events: List[Dict[str, Any]] = []
        for ev in events:
            if not isinstance(ev, dict):
                continue

            start_iso, start_all_day = _coerce_ha_time_to_iso(ev.get("start"), tz=tz)
            end_iso, end_all_day = _coerce_ha_time_to_iso(ev.get("end"), tz=tz)
            all_day = bool(ev.get("all_day")) or start_all_day or end_all_day

            normalized: Dict[str, Any] = {
                "summary": ev.get("summary") or ev.get("title") or None,
                "location": ev.get("location"),
                "description": ev.get("description"),
                "all_day": all_day,
                "start_iso": start_iso,
                "end_iso": end_iso,
            }

            if include_raw:
                normalized["raw"] = ev

            normalized_events.append(normalized)

            combined_events.append(
                {
                    "entity_id": entity_id,
                    "owner_key": cal.get("owner_key"),
                    "owner_display_name": cal.get("owner_display_name"),
                    "label": cal.get("label"),
                    **normalized,
                }
            )

        per_calendar.append(
            {
                "entity_id": entity_id,
                "owner_key": cal.get("owner_key"),
                "owner_display_name": cal.get("owner_display_name"),
                "label": cal.get("label"),
                "event_count": len(normalized_events),
                "events": normalized_events,
            }
        )

    def _sort_key(ev: Dict[str, Any]) -> Tuple[datetime, int, str]:
        start_dt = _parse_iso_to_dt(ev.get("start_iso")) or datetime.max.replace(tzinfo=timezone.utc)
        all_day_rank = 0 if bool(ev.get("all_day")) else 1
        summary = str(ev.get("summary") or "")
        return (start_dt, all_day_rank, summary.lower())

    combined_events.sort(key=_sort_key)

    return {
        "status": "ok",
        "household": hh,
        "offset_weeks": int(offset_weeks),
        "timezone": getattr(tz, "key", str(tz)),
        "first_day_of_week": "sunday",
        "window": {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
        },
        "ignored": sorted(list(CALENDAR_ENTITY_IGNORE)),
        "include_raw": bool(include_raw),
        "calendars": {
            "count": len(per_calendar),
            "items": per_calendar,
        },
        "events": {
            "count": len(combined_events),
            "items": combined_events,
        },
        "notes": [
            "Read-only calendar awareness.",
            "No DB writes, no HA writes, no event creation.",
            "Week window is Sunday-start and derived from server time.",
            "POLISH A: normalized start/end fields and sorting; raw optional.",
        ],
    }


@calendar_router.get("/busy/week")
async def calendar_busy_week(
    offset_weeks: int = Query(
        default=0,
        description="0=current week (Sunday-start), 1=next week, -1=previous week",
    ),
    household: Optional[str] = Query(
        default="home_a",
        description="home_a | home_b",
    ),
) -> Dict[str, Any]:
    """
    Read-only calendar busyness aggregation.
    Produces per-day workload signals for planning awareness.
    """
    hh = _normalize_calendar_household(household)
    tz = get_server_timezone()

    window_start, window_end = planning_window_week(
        offset_weeks=offset_weeks, tz=tz
    )

    calendars = get_enabled_calendar_entities_for_household(hh)
    all_events: List[Dict[str, Any]] = []

    for cal in calendars:
        entity_id = cal.get("entity_id")
        if not entity_id or entity_id in CALENDAR_ENTITY_IGNORE:
            continue

        events = await ha_calendar_get_events(
            entity_id=entity_id,
            start=window_start,
            end=window_end,
        )

        for ev in events:
            start_iso, start_all_day = _coerce_ha_time_to_iso(ev.get("start"), tz=tz)
            end_iso, end_all_day = _coerce_ha_time_to_iso(ev.get("end"), tz=tz)

            start_dt = _parse_iso_to_dt(start_iso)
            end_dt = _parse_iso_to_dt(end_iso)

            if not start_dt or not end_dt:
                continue

            all_events.append(
                {
                    "start": start_dt,
                    "end": end_dt,
                    "all_day": bool(ev.get("all_day") or start_all_day or end_all_day),
                    "summary": ev.get("summary") or ev.get("title"),
                }
            )

    busy_by_day: List[Dict[str, Any]] = []

    for day_offset in range(7):
        day_start = window_start + timedelta(days=day_offset)
        day_end = day_start + timedelta(days=1)

        day_events = []
        busy_minutes = 0
        all_day_count = 0

        for ev in all_events:
            if ev["all_day"]:
                if ev["start"].date() == day_start.date():
                    all_day_count += 1
                continue

            minutes = _clip_interval_to_day(
                ev["start"], ev["end"], day_start, day_end
            )
            if minutes > 0:
                busy_minutes += minutes
                day_events.append(ev)

        summaries = [
            e["summary"] for e in day_events if e.get("summary")
        ][:3]

        busy_by_day.append(
            {
                "date": day_start.date().isoformat(),
                "event_count": len(day_events) + all_day_count,
                "all_day_count": all_day_count,
                "busy_minutes": busy_minutes,
                "busy_score": _busy_score_from_minutes(busy_minutes),
                "top_summaries": summaries,
            }
        )

    return {
        "status": "ok",
        "household": hh,
        "timezone": getattr(tz, "key", str(tz)),
        "window": {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
        },
        "busy_by_day": busy_by_day,
        "notes": [
            "Read-only calendar awareness signal",
            "No DB writes, no HA writes",
            "Designed for meal planning + UI hints",
        ],
    }


@calendar_router.get("/today")
async def calendar_today(
    household: Optional[str] = Query(
        default="home_a",
        description="home_a | home_b (default: home_a)",
    ),
    include_raw: bool = Query(
        default=False,
        description="If true, include raw HA event payloads. Default false.",
    ),
) -> Dict[str, Any]:
    """
    Read-only: Fetch calendar events for 'today' using server timezone.
    Same normalization and ignore rules as /calendar/week.
    """
    hh = _normalize_calendar_household(household)

    tz = get_server_timezone()
    window_start, window_end = planning_window_today(tz=tz)

    calendars = get_enabled_calendar_entities_for_household(hh)

    combined_events: List[Dict[str, Any]] = []

    for cal in calendars:
        entity_id = cal.get("entity_id")
        if not entity_id or entity_id in CALENDAR_ENTITY_IGNORE:
            continue

        events = await ha_calendar_get_events(
            entity_id=entity_id,
            start=window_start,
            end=window_end,
        )

        for ev in events:
            if not isinstance(ev, dict):
                continue

            start_iso, start_all_day = _coerce_ha_time_to_iso(ev.get("start"), tz=tz)
            end_iso, end_all_day = _coerce_ha_time_to_iso(ev.get("end"), tz=tz)
            all_day = bool(ev.get("all_day")) or start_all_day or end_all_day

            normalized: Dict[str, Any] = {
                "entity_id": entity_id,
                "owner_key": cal.get("owner_key"),
                "owner_display_name": cal.get("owner_display_name"),
                "label": cal.get("label"),
                "summary": ev.get("summary") or ev.get("title") or None,
                "location": ev.get("location"),
                "description": ev.get("description"),
                "all_day": all_day,
                "start_iso": start_iso,
                "end_iso": end_iso,
            }

            if include_raw:
                normalized["raw"] = ev

            combined_events.append(normalized)

    def _sort_key(ev: Dict[str, Any]) -> Tuple[datetime, int, str]:
        start_dt = _parse_iso_to_dt(ev.get("start_iso")) or datetime.max.replace(tzinfo=timezone.utc)
        all_day_rank = 0 if bool(ev.get("all_day")) else 1
        summary = str(ev.get("summary") or "")
        return (start_dt, all_day_rank, summary.lower())

    combined_events.sort(key=_sort_key)

    return {
        "status": "ok",
        "household": hh,
        "timezone": getattr(tz, "key", str(tz)),
        "window": {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
        },
        "ignored": sorted(list(CALENDAR_ENTITY_IGNORE)),
        "include_raw": bool(include_raw),
        "events": {
            "count": len(combined_events),
            "items": combined_events,
        },
        "notes": [
            "Read-only calendar awareness.",
            "Server-time 'today' window.",
            "No DB writes, no HA writes, no event creation.",
        ],
    }


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

    # NEW: external lookup payload (minimal/debuggable)
    external: Optional[Dict[str, Any]] = None

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


def _build_external_minimal_payload(source: str, off_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a minimal external payload suitable for debugging/UI without overwhelming.
    """
    product = off_payload.get("product") if isinstance(off_payload.get("product"), dict) else {}
    if not isinstance(product, dict):
        product = {}

    return {
        "source": source,
        "status": off_payload.get("status"),
        "code": off_payload.get("code"),
        "product_name": product.get("product_name") or product.get("product_name_en"),
        "brands": product.get("brands"),
        "categories": product.get("categories"),
        "image_url": product.get("image_url") or product.get("image_front_url"),
        "url": product.get("url"),
    }


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
    hh = _require_household_write_scope(household)

    code = (barcode or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="barcode must not be empty")

    if client is None:
        # Return disabled as a normal response_model payload with low confidence.
        # This keeps callers from crashing on 503s when services are intentionally off.
        return GrocyBarcodeInspectResponse(
            barcode=code,
            household=hh,  # type: ignore[arg-type]
            found_in_grocy=False,
            product=None,
            external=None,
            suggestion=ProductSuggestion(
                confidence="low",
                notes="Grocy client not configured; cannot inspect barcode.",
            ),
            next_actions=["configure_grocy", "external_lookup", "user_confirmation_required"],
            reasons=["grocy_disabled"],
        )

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
            external=None,
            suggestion=suggestion,
            next_actions=["no_action_needed", "user_confirmation_required"],
            reasons=reasons,
        )

    # ---------- NEW: External lookup fallback (OpenFoodFacts) ----------
    external_payload: Optional[Dict[str, Any]] = None
    external_suggestion: Dict[str, Optional[str]] = {"name": None, "brand": None, "category": None}

    try:
        off_client = create_openfoodfacts_client()
        off_raw = off_client.lookup_barcode(code)
        external_payload = _build_external_minimal_payload("openfoodfacts", off_raw)

        # OFF status 1 indicates product exists
        if isinstance(off_raw, dict) and off_raw.get("status") == 1:
            external_suggestion = extract_suggestion_from_off_payload(off_raw)
            reasons.append("found_in_openfoodfacts")
        else:
            reasons.append("not_found_in_openfoodfacts")

    except OpenFoodFactsError as exc:
        reasons.append(f"openfoodfacts_error: {exc}")
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"openfoodfacts_error: {exc}")

    # Build the suggestion object from external lookup, if any.
    name = external_suggestion.get("name")
    brand = external_suggestion.get("brand")
    category = external_suggestion.get("category")

    if name or brand or category:
        suggestion = ProductSuggestion(
            name=name,
            brand=brand,
            category=category,
            confidence="medium",
            notes="Not found in Grocy. External lookup suggests this product. Approval required before any creation.",
        )
        next_actions = ["user_confirmation_required"]
    else:
        # Not found (or lookup unavailable): return a sparse suggestion object.
        suggestion = ProductSuggestion(
            confidence="low",
            notes="No existing product found in Grocy for this barcode.",
        )
        next_actions = ["external_lookup", "user_confirmation_required"]

    if "grocy_client_has_no_barcode_lookup_method" in reasons:
        # Preserve the Step 1 signal, but now we *also* tried external lookup.
        suggestion.notes = (
            "Grocy client does not yet support barcode lookup; "
            "external lookup was attempted. Approval required before any creation."
        )
        # Make it explicit we still need Grocy mapping support later.
        if "add_grocy_barcode_lookup" not in next_actions:
            next_actions = ["add_grocy_barcode_lookup", "user_confirmation_required"]

    return GrocyBarcodeInspectResponse(
        barcode=code,
        household=hh,  # type: ignore[arg-type]
        found_in_grocy=False,
        product=None,
        external=external_payload,
        suggestion=suggestion,
        next_actions=next_actions,
        reasons=reasons,
    )


# ----------------------------
# Phase 6.45 — Step 4: Confirm + Create + Link + Optionally Add Stock (Explicit approval only)
# ----------------------------
    
class GrocyConfirmCreateAndOptionallyStockRequest(BaseModel):
    household: Literal["home_a", "home_b"] = Field(..., description="Target household (home_a | home_b)")
    barcode: str = Field(..., min_length=1, description="Barcode / UPC / EAN string")
    name: str = Field(..., min_length=1, description="Final authoritative product name")
    location_id: int = Field(..., ge=1, description="Grocy location_id to assign to the product")
    qu_id_purchase: int = Field(..., ge=1, description="Grocy purchase unit ID (required)")
    qu_id_stock: int = Field(..., ge=1, description="Grocy stock unit ID (required)")
    
    # Optional stock add (explicit)
    add_stock: bool = Field(False, description="If true, add stock after create+link")
    quantity: Optional[float] = Field(default=None, gt=0, description="Quantity to add (required if add_stock=true)")
    best_before_date: Optional[str] = Field(default=None, description="YYYY-MM-DD (optional)")
    purchased_date: Optional[str] = Field(default=None, description="YYYY-MM-DD (optional)")
    price: Optional[float] = Field(default=None, ge=0, description="Optional price (>= 0)")
    
    # Advisory only (not used for behavior yet)
    brand: Optional[str] = None
    category: Optional[str] = None
    
    
@grocy_router.post(
    "/confirm-create-and-optionally-stock",
    summary="Confirm-create product from inspection, link barcode, and optionally add stock (explicit only).",
)
async def grocy_confirm_create_and_optionally_stock(
    body: GrocyConfirmCreateAndOptionallyStockRequest,
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    """
    Phase 6.45 Step 4 (LOCKED):
    - Requires explicit final authoritative fields
    - Creates product in Grocy
    - Links barcode in Grocy
    - Optionally adds stock ONLY if add_stock=true and quantity provided
    - NO shopping list changes
    - NO inference
    """
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Grocy client not configured; confirm-create unavailable.",
        )
    
    household = body.household.strip().lower()
    if household not in {"home_a", "home_b"}:
        raise HTTPException(status_code=400, detail="household must be 'home_a' or 'home_b'")
    
    barcode = body.barcode.strip()
    if not barcode:
        raise HTTPException(status_code=400, detail="barcode must not be empty")
    
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")
    
    if body.add_stock and body.quantity is None:
        raise HTTPException(status_code=400, detail="quantity is required when add_stock=true")
    
    # 1) Create product
    try:
        created = await client.create_product(
            household=household,
            name=name,
            location_id=body.location_id,
            qu_id_purchase=body.qu_id_purchase,
            qu_id_stock=body.qu_id_stock,
        )
    except GrocyError as exc:
        raise HTTPException(status_code=502, detail=f"Error creating product in Grocy: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Unexpected error creating product: {exc}") from exc
    
    # Extract product_id robustly
    product_id: Optional[int] = None
    if isinstance(created, dict):
        pid = (
            created.get("product_id")
            or created.get("id")
            or created.get("created_object_id")
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
            detail={"error": "Could not determine product_id", "created": created},
        )
    
    # 2) Link barcode (explicit canonical method)
    try:
        barcode_link = await client.link_barcode_to_product(
            household=household,
            barcode=barcode,
            product_id=product_id,
        )
    except GrocyError as exc:
        raise HTTPException(status_code=502, detail=f"Error linking barcode in Grocy: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Unexpected error linking barcode: {exc}") from exc
    
    stock_result: Optional[Dict[str, Any]] = None
    
    # 3) Optional stock add (ONLY if explicitly requested)
    if body.add_stock:
        stock_payload: Dict[str, Any] = {"amount": float(body.quantity)}  # Grocy expects "amount"
    
        # Optional fields if provided
        if body.best_before_date:
            stock_payload["best_before_date"] = body.best_before_date
        if body.purchased_date:
            stock_payload["purchased_date"] = body.purchased_date
        if body.price is not None:
            stock_payload["price"] = float(body.price)
    
        try:
            stock_result = await client.add_stock(
                household=household,
                product_id=product_id,
                payload=stock_payload,
            )
        except GrocyError as exc:
            raise HTTPException(status_code=502, detail=f"Error adding stock in Grocy: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Unexpected error adding stock: {exc}") from exc
    
    return {
        "status": "ok",
        "household": household,
        "barcode": barcode,
        "product_id": product_id,
        "product": created,
        "barcode_link": barcode_link,
        "stock_added": bool(body.add_stock),
        "stock_result": stock_result,
        "advisory_metadata": {"brand": body.brand, "category": body.category},
        "notes": [
            "Phase 6.45 Step 4: explicit confirm-create + optional stock only",
            "No shopping list changes",
            "No inference",
        ],
    }
    

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
# NEW: Grocy location creation (explicit, Gate A)
# ----------------------------

class GrocyCreateLocationRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Location name")
    description: Optional[str] = None
    is_freezer: int = Field(0, description="1 if freezer, 0 otherwise")


@grocy_router.post("/locations/{household}")
async def grocy_create_location(
    household: str,
    body: GrocyCreateLocationRequest,
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Grocy client not configured; create location unavailable.",
        )

    hh = (household or "").strip().lower()
    if hh not in {"home_a", "home_b"}:
        raise HTTPException(
            status_code=400,
            detail="household must be 'home_a' or 'home_b'",
        )

    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")

    payload = {
        "name": name,
        "description": body.description,
        "is_freezer": 1 if body.is_freezer else 0,
    }

    try:
        created = await _acall_first_existing(
            client,
            [
                "create_location",
                "create_grocy_location",
                "add_location",
                "create_location_for_household",
            ],
            household=hh,
            payload=payload,
        )
    except HTTPException:
        raise
    except GrocyError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Error creating location in Grocy: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error creating location: {exc}",
        ) from exc

    return {
        "status": "ok",
        "household": hh,
        "location": created,
    }

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
# Meal planner (single source of truth)
# ---------------------------------------------------------
# All meal planner endpoints are defined in:
#   /opt/jarvis/brain/services/mealplanner.py
#
# Endpoints include:
#   - POST /mealplanner/plan
#   - POST /mealplanner/plan-multi
#   - GET  /mealplanner/plan-context
#
# This file intentionally defines NO /mealplanner routes
# to avoid router shadowing and OpenAPI collisions.


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

# Alice (Phase 8) — read-only preview endpoint
app.include_router(alice_router)

app.include_router(ha_router)
app.include_router(calendar_router)
app.include_router(grocy_router)

# Meal planner (single source of truth: services/mealplanner.py)
app.include_router(mealplanner_context_router)

app.include_router(barcode_router)
app.include_router(recipes_router)
app.include_router(ingredient_parser_router)
app.include_router(recipe_matcher_router)
app.include_router(recipe_analyzer_router)
app.include_router(recipe_mappings_router)

app.include_router(mealplans_router, prefix="/mealplans", tags=["mealplans"])

