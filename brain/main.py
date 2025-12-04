import os
import time
import socket
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import psutil
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Jarvis Brain",
    description="Backend API for your personal Jarvis assistant",
    version="0.2.0",
)

# CORS so the web UI can talk to us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can lock this down to your domains later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# record when the service started (for uptime)
START_TIME = time.time()

# ---- Database setup (for conversation history) ----

DATA_DIR = Path("/app/data")
DB_PATH = DATA_DIR / "jarvis_brain.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_db_conn():
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
            "INSERT INTO conversations (ts_utc, model, user_message, jarvis_answer) VALUES (?, ?, ?, ?)",
            (ts, model, message, answer),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_recent_conversations(limit: int = 20):
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


# ---- LLM config ----

LLM_API_KEY = os.getenv("JARVIS_LLM_API_KEY")
LLM_BASE_URL = os.getenv("JARVIS_LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("JARVIS_LLM_MODEL", "gpt-4o-mini")


class AskRequest(BaseModel):
    message: str


@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "jarvis-brain"}


@app.get("/whoami")
def whoami():
    """Basic identity endpoint for Jarvis."""
    return {
        "name": "Jarvis",
        "role": "Personal homelab assistant",
        "host": socket.gethostname(),
        "version": "0.2.0",
        "description": "I run in your homelab and help you monitor, automate, and query your systems.",
    }


@app.get("/time")
def current_time():
    """Return the current server time in UTC and local."""
    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now()
    return {
        "utc_iso": now_utc.isoformat(),
        "local_iso": now_local.isoformat(),
    }


@app.get("/system-info")
def system_info():
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
            "mountpoint": "/",
        },
    }


@app.get("/uptime")
def uptime():
    """How long jarvis-brain has been running (and basic host uptime)."""
    service_uptime_seconds = time.time() - START_TIME

    boot_time = datetime.fromtimestamp(psutil.boot_time(), timezone.utc)
    host_uptime_seconds = (datetime.now(timezone.utc) - boot_time).total_seconds()

    return {
        "service_uptime_seconds": int(service_uptime_seconds),
        "host_uptime_seconds": int(host_uptime_seconds),
        "host_boot_time_utc": boot_time.isoformat(),
    }


@app.post("/ask")
def ask_jarvis(body: AskRequest):
    """
    Proxy a simple question to an OpenAI-compatible LLM and return the answer.
    Also logs the exchange in SQLite.
    """
    if not LLM_API_KEY:
        raise HTTPException(status_code=500, detail="LLM API key not configured.")

    url = f"{LLM_BASE_URL.rstrip('/')}/chat/completions"

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are Jarvis, a helpful personal assistant for the user's homelab. "
                    "Be concise and practical."
                ),
            },
            {"role": "user", "content": body.message},
        ],
        "temperature": 0.5,
    }

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error contacting LLM provider: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()

    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise HTTPException(status_code=500, detail="Unexpected response from LLM provider.")

    # log to SQLite
    try:
        log_conversation(LLM_MODEL, body.message, answer)
    except Exception as e:
        # don't break the API if logging fails; just note it
        print(f"[WARN] Failed to log conversation: {e}")

    return {"model": LLM_MODEL, "answer": answer}


@app.get("/history")
def history(limit: int = 20):
    """
    Return the most recent conversation entries (user question + Jarvis reply).
    """
    items = fetch_recent_conversations(limit=limit)
    return {"count": len(items), "items": items}
