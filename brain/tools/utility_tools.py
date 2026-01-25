from __future__ import annotations

import ast
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from tools.types import ToolFailureClass, ToolProvenance, ToolRequest, ToolResult


# ----------------------------
# Helpers
# ----------------------------

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fail(req: ToolRequest, cls: ToolFailureClass, msg: str, started_at: str, t0: float) -> ToolResult:
    return ToolResult(
        ok=False,
        tool_name=req.tool_name,
        failure_class=cls,
        failure_message=msg,
        started_at=started_at,
        ended_at=_iso_now(),
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


# ----------------------------
# util.time
# ----------------------------

def _parse_utc_offset(offset: str) -> timezone:
    s = (offset or "").strip().upper()
    if s in {"Z", "UTC", "+00:00", "-00:00"}:
        return timezone.utc

    # Expected: ±HH:MM
    if len(s) != 6 or s[0] not in {"+", "-"} or s[3] != ":":
        raise ValueError("offset must be in format ±HH:MM (e.g. -05:00)")

    sign = 1 if s[0] == "+" else -1
    hh = int(s[1:3])
    mm = int(s[4:6])
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        raise ValueError("offset out of range")
    delta = timedelta(hours=hh, minutes=mm) * sign
    return timezone(delta)


def _util_time(req: ToolRequest, started_at: str, t0: float) -> ToolResult:
    args = req.args or {}
    offset = str(args.get("offset") or args.get("utc_offset") or "").strip()

    if not offset:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "util.time requires offset (±HH:MM)", started_at, t0)

    try:
        tz = _parse_utc_offset(offset)
    except Exception as e:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, f"Invalid offset: {e}", started_at, t0)

    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(tz)

    prov = ToolProvenance(
        sources=["server_clock"],
        retrieved_at=_iso_now(),
        notes=f"util.time offset={offset}",
    )

    return ToolResult(
        ok=True,
        tool_name=req.tool_name,
        primary={
            "offset": offset,
            "utc_now": now_utc.isoformat(),
            "local_now": now_local.isoformat(),
        },
        provenance=prov,
        started_at=started_at,
        ended_at=_iso_now(),
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


# ----------------------------
# util.calc
# ----------------------------

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
        v = _eval_ast(node.operand)
        return +v if isinstance(node.op, ast.UAdd) else -v

    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        a = _eval_ast(node.left)
        b = _eval_ast(node.right)

        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            if b == 0:
                raise ZeroDivisionError("division by zero")
            return a / b
        if isinstance(node.op, ast.FloorDiv):
            if b == 0:
                raise ZeroDivisionError("division by zero")
            return math.floor(a / b)
        if isinstance(node.op, ast.Mod):
            if b == 0:
                raise ZeroDivisionError("modulo by zero")
            return a % b
        if isinstance(node.op, ast.Pow):
            # Conservative cap to prevent huge computations
            if abs(b) > 1000:
                raise ValueError("exponent too large")
            return a ** b

    # Disallow everything else: names, calls, attributes, subscripts, etc.
    raise ValueError("expression contains unsupported syntax")


def _util_calc(req: ToolRequest, started_at: str, t0: float) -> ToolResult:
    args = req.args or {}
    expr = str(args.get("expression") or "").strip()

    if not expr:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "util.calc requires expression", started_at, t0)

    # Hard caps (defensive)
    if len(expr) > 200:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "expression too long (max 200 chars)", started_at, t0)

    try:
        tree = ast.parse(expr, mode="eval")
        value = _eval_ast(tree)
    except ZeroDivisionError as e:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, str(e), started_at, t0)
    except Exception as e:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, f"Invalid expression: {e}", started_at, t0)

    prov = ToolProvenance(
        sources=["deterministic_eval"],
        retrieved_at=_iso_now(),
        notes="util.calc (ast-based, no eval)",
    )

    return ToolResult(
        ok=True,
        tool_name=req.tool_name,
        primary={"expression": expr, "result": value},
        provenance=prov,
        started_at=started_at,
        ended_at=_iso_now(),
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


# ----------------------------
# util.weather
# ----------------------------

@dataclass
class _GeoHit:
    name: str
    latitude: float
    longitude: float
    country: Optional[str]
    admin1: Optional[str]


async def _open_meteo_geocode(location: str, started_at: str, t0: float, req: ToolRequest) -> Tuple[Optional[_GeoHit], List[str], Optional[ToolResult]]:
    # Open-Meteo geocoding: https://open-meteo.com/
    # Two-pass strategy (v4-A):
    #   Pass 1: normalized query + country bias (US)
    #   Pass 2: fallback to broader query (largest token), no country bias
    if not location or len(location) > 120:
        return None, [], _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "location must be 1..120 chars", started_at, t0)

    url = "https://geocoding-api.open-meteo.com/v1/search"
    timeout = httpx.Timeout(15.0, connect=10.0)
    headers = {"User-Agent": "ISAC/1.0", "Accept": "application/json"}

    # -------- Pass 1: normalized + country bias --------
    query1 = " ".join(location.replace(",", " ").split())
    params1 = {"name": query1, "count": 1, "language": "en", "format": "json", "country_code": "US"}
    prov_sources: List[str] = [f"{url}?name={query1}&country_code=US"]

    async def _try(params: Dict[str, Any]) -> Optional[_GeoHit]:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
            r = await client.get(url, params=params)
        if r.status_code >= 400:
            return None
        data = r.json()
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list) or not results:
            return None
        hit = results[0]
        try:
            return _GeoHit(
                name=str(hit.get("name") or params.get("name")),
                latitude=float(hit["latitude"]),
                longitude=float(hit["longitude"]),
                country=str(hit.get("country")) if hit.get("country") is not None else None,
                admin1=str(hit.get("admin1")) if hit.get("admin1") is not None else None,
            )
        except Exception:
            return None

    try:
        gh = await _try(params1)
        if gh:
            return gh, prov_sources, None
    except httpx.TimeoutException:
        return None, prov_sources, _fail(req, ToolFailureClass.TOOL_TIMEOUT, "Geocoding timeout", started_at, t0)
    except httpx.HTTPError as e:
        return None, prov_sources, _fail(req, ToolFailureClass.TOOL_UPSTREAM_ERROR, f"Geocoding HTTP error: {e}", started_at, t0)
    except Exception:
        pass

    # -------- Pass 2: fallback (largest token, no country bias) --------
    # Take the longest word as a conservative broadening strategy
    tokens = [t for t in query1.split(" ") if t]
    if tokens:
        broad = max(tokens, key=len)
    else:
        broad = query1

    params2 = {"name": broad, "count": 1, "language": "en", "format": "json"}
    prov_sources.append(f"{url}?name={broad}")

    try:
        gh2 = await _try(params2)
        if gh2:
            return gh2, prov_sources, None
    except httpx.TimeoutException:
        return None, prov_sources, _fail(req, ToolFailureClass.TOOL_TIMEOUT, "Geocoding timeout", started_at, t0)
    except httpx.HTTPError as e:
        return None, prov_sources, _fail(req, ToolFailureClass.TOOL_UPSTREAM_ERROR, f"Geocoding HTTP error: {e}", started_at, t0)
    except Exception:
        pass

    return None, prov_sources, _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "Location not found", started_at, t0)


async def _open_meteo_forecast(lat: float, lon: float, days: int, started_at: str, t0: float, req: ToolRequest) -> Tuple[Optional[Dict[str, Any]], List[str], Optional[ToolResult]]:
    # Open-Meteo forecast API (no key): https://open-meteo.com/
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
        "timezone": "auto",
    }
    # enforce days by trimming results; API supports forecast_days param but keep conservative
    params["forecast_days"] = int(days)

    prov_sources = [f"{url}?latitude={lat}&longitude={lon}"]

    timeout = httpx.Timeout(20.0, connect=10.0)
    headers = {"User-Agent": "ISAC/1.0", "Accept": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
            r = await client.get(url, params=params)
        if r.status_code >= 400:
            return None, prov_sources, _fail(req, ToolFailureClass.TOOL_UPSTREAM_ERROR, f"Forecast upstream returned {r.status_code}", started_at, t0)
        data = r.json()
    except httpx.TimeoutException:
        return None, prov_sources, _fail(req, ToolFailureClass.TOOL_TIMEOUT, "Forecast timeout", started_at, t0)
    except httpx.HTTPError as e:
        return None, prov_sources, _fail(req, ToolFailureClass.TOOL_UPSTREAM_ERROR, f"Forecast HTTP error: {e}", started_at, t0)
    except Exception as e:
        return None, prov_sources, _fail(req, ToolFailureClass.TOOL_PARSE_ERROR, f"Forecast parse error: {e}", started_at, t0)

    return data if isinstance(data, dict) else None, prov_sources, None


async def _util_weather(req: ToolRequest, started_at: str, t0: float) -> ToolResult:
    args = req.args or {}
    location = str(args.get("location") or "").strip()
    days = args.get("days", 7)

    try:
        days_i = int(days)
    except Exception:
        days_i = 7
    days_i = max(1, min(10, days_i))

    geo, prov_a, err = await _open_meteo_geocode(location, started_at, t0, req)
    if err:
        return err

    assert geo is not None

    forecast, prov_b, err2 = await _open_meteo_forecast(geo.latitude, geo.longitude, days_i, started_at, t0, req)
    if err2:
        return err2

    # Keep payload human-safe and bounded
    daily = (forecast or {}).get("daily") if isinstance(forecast, dict) else None
    primary: Dict[str, Any] = {
        "location": {
            "query": location,
            "name": geo.name,
            "admin1": geo.admin1,
            "country": geo.country,
            "latitude": geo.latitude,
            "longitude": geo.longitude,
        },
        "days": days_i,
        "daily": daily,
    }

    prov = ToolProvenance(
        sources=(prov_a + prov_b)[:10],
        retrieved_at=_iso_now(),
        notes="util.weather via Open-Meteo geocoding + forecast",
    )

    return ToolResult(
        ok=True,
        tool_name=req.tool_name,
        primary=primary,
        provenance=prov,
        started_at=started_at,
        ended_at=_iso_now(),
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


# ----------------------------
# Family entry point
# ----------------------------

async def run_utility_tool(req: ToolRequest) -> ToolResult:
    t0 = time.monotonic()
    started_at = _iso_now()

    try:
        if req.tool_name == "util.time":
            return _util_time(req, started_at, t0)
        if req.tool_name == "util.calc":
            return _util_calc(req, started_at, t0)
        if req.tool_name == "util.weather":
            return await _util_weather(req, started_at, t0)

        return _fail(req, ToolFailureClass.TOOL_NOT_ALLOWED, f"Unsupported utility tool: {req.tool_name}", started_at, t0)

    except Exception as e:
        return _fail(req, ToolFailureClass.TOOL_INTERNAL_ERROR, f"Internal error: {e}", started_at, t0)