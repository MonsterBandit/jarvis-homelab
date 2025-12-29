# /opt/jarvis/brain/system_health_capture.py

from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import shutil


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ms_since(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _disk_usage(path: str) -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        usage = shutil.disk_usage(path)
        total = int(usage.total)
        free = int(usage.free)
        free_pct = (free / total * 100.0) if total > 0 else 0.0
        return {
            "ok": True,
            "latency_ms": _ms_since(t0),
            "details": {
                "path": path,
                "total_bytes": total,
                "free_bytes": free,
                "free_pct": round(free_pct, 2),
            },
        }
    except Exception as e:
        return {
            "ok": False,
            "latency_ms": _ms_since(t0),
            "error": f"disk_usage_failed: {type(e).__name__}",
            "details": {"path": path},
        }


def _docker_socket_present(sock_path: str = "/var/run/docker.sock") -> Dict[str, Any]:
    t0 = time.perf_counter()
    try:
        present = os.path.exists(sock_path)
        return {
            "ok": True,
            "latency_ms": _ms_since(t0),
            "details": {"socket_path": sock_path, "socket_present": bool(present)},
        }
    except Exception as e:
        return {
            "ok": False,
            "latency_ms": _ms_since(t0),
            "error": f"docker_socket_check_failed: {type(e).__name__}",
            "details": {"socket_path": sock_path},
        }


def _dns_resolve(hostname: str, timeout_s: float = 2.0) -> Dict[str, Any]:
    """
    DNS check (container-only):
    - resolves a hostname to confirm DNS path is working
    - does NOT attempt outbound HTTP
    """
    t0 = time.perf_counter()
    prior = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(timeout_s)
        infos = socket.getaddrinfo(hostname, None)
        # Sanitize: show at most first 2 resolved IPs (no huge dumps)
        ips = []
        for info in infos:
            addr = info[4][0]
            if addr not in ips:
                ips.append(addr)
            if len(ips) >= 2:
                break

        return {
            "ok": True,
            "latency_ms": _ms_since(t0),
            "details": {"hostname": hostname, "resolved": True, "ips": ips},
        }
    except Exception as e:
        return {
            "ok": False,
            "latency_ms": _ms_since(t0),
            "error": f"dns_resolve_failed: {type(e).__name__}",
            "details": {"hostname": hostname, "resolved": False},
        }
    finally:
        socket.setdefaulttimeout(prior)


async def capture_system_health(ha_client) -> Dict[str, Any]:
    """
    Read-only, non-persistent snapshot of system/external health.
    Designed to run entirely inside the jarvis-brain container.

    ha_client: HomeAssistantClient instance (already configured via env)
    """
    t0 = time.perf_counter()

    dns_host = os.getenv("ISAC_DNS_TEST_HOST", "example.com")

    checks: Dict[str, Any] = {}

    # Home Assistant (real API check)
    ha_t0 = time.perf_counter()
    try:
        ha_result = await ha_client.health_check()
        # Expect ha_result to already be sanitized
        ha_result.setdefault("latency_ms", _ms_since(ha_t0))
        checks["homeassistant"] = ha_result
    except Exception as e:
        checks["homeassistant"] = {
            "ok": False,
            "latency_ms": _ms_since(ha_t0),
            "error": f"ha_health_check_failed: {type(e).__name__}",
        }

    # Disk checks
    checks["disk"] = {
        "data_volume": _disk_usage("/app/data"),
        "container_root": _disk_usage("/"),
    }

    # Docker socket presence (no host coupling)
    checks["docker"] = _docker_socket_present()

    # DNS resolve
    checks["dns"] = _dns_resolve(dns_host)

    return {
        "domain": "system_external",
        "captured_at": _utc_now_iso(),
        "capture_duration_ms": _ms_since(t0),
        "checks": checks,
    }
