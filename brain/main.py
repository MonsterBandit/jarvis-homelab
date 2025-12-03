import time
import socket
from datetime import datetime, timezone

import psutil
from fastapi import FastAPI

app = FastAPI(
    title="Jarvis Brain",
    description="Backend API for your personal Jarvis assistant",
    version="0.1.0",
)

# record when the service started (for uptime)
START_TIME = time.time()


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
        "version": "0.1.0",
        "description": "I run in your homelab and help you monitor, automate, and query your systems.",
    }


@app.get("/time")
def current_time():
    """Return the current server time in UTC and local."""
    now_utc = datetime.now(timezone.utc)
    # naive local time (server's timezone)
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
        "cpu": {
            "percent": cpu_percent,
        },
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
