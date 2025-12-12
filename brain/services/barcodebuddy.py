import os
from functools import lru_cache
from typing import Any, Dict, Optional

import httpx


class BarcodeBuddyError(Exception):
    """Errors talking to Barcode Buddy."""


class BarcodeBuddyClient:
    """
    Lightweight async client for Barcode Buddy.

    We primarily use:
      - GET /api/system/info           -> health
      - GET /api/action/scan?add=CODE -> pass a barcode to BB
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 10.0) -> None:
        if not base_url:
            raise BarcodeBuddyError("BARCODEBUDDY_BASE_URL is not configured")

        self.base_url = base_url.rstrip("/")  # e.g. https://barcode.plexmoose.com
        self.api_key = api_key or ""
        self.timeout = timeout

    def _build_headers_and_params(self, extra_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        headers: Dict[str, str] = {}
        params: Dict[str, Any] = dict(extra_params or {})

        # The API key can be sent as header **or** GET param.
        if self.api_key:
            headers["BBUDDY-API-KEY"] = self.api_key
            params.setdefault("apikey", self.api_key)

        return {"headers": headers, "params": params}

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = f"{self.base_url}/api{path}"

        hp = self._build_headers_and_params(params)
        headers = hp["headers"]
        query_params = hp["params"]

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.request(
                method=method.upper(),
                url=url,
                params=query_params,
                data=data,
            )

        if resp.status_code >= 400:
            snippet = resp.text[:200]
            raise BarcodeBuddyError(f"BarcodeBuddy returned HTTP {resp.status_code} for {url}: {snippet}")

        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            return resp.json()

        # Some endpoints may just return text / HTML
        return resp.text

    # ---------- High-level methods ----------

    async def health(self) -> Any:
        """Return Barcode Buddy system info (GET /api/system/info)."""
        return await self._request("GET", "/system/info")

    async def scan_barcode(self, barcode: str) -> Any:
        """
        Pass a single barcode to Barcode Buddy.

        Docs: /api/action/scan supports GET parameter `add`.
        Example:
          /api/action/scan?add=123456
        """
        if not barcode:
            raise BarcodeBuddyError("Barcode cannot be empty")

        return await self._request("GET", "/action/scan", params={"add": barcode})


# ---------- Factory helpers for FastAPI dependencies ----------


def _get_barcodebuddy_settings() -> Optional[Dict[str, str]]:
    """
    Read settings from environment.

    Return None if BARCODEBUDDY_BASE_URL is not set so the API
    can report 'disabled' instead of crashing.
    """
    base_url = os.getenv("BARCODEBUDDY_BASE_URL", "").strip()
    api_key = os.getenv("BARCODEBUDDY_API_KEY", "").strip()

    if not base_url:
        return None

    return {"base_url": base_url, "api_key": api_key}


@lru_cache
def _barcodebuddy_client_singleton() -> BarcodeBuddyClient:
    settings = _get_barcodebuddy_settings()
    if settings is None:
        raise BarcodeBuddyError("BarcodeBuddy not configured (missing BARCODEBUDDY_BASE_URL)")
    return BarcodeBuddyClient(
        base_url=settings["base_url"],
        api_key=settings["api_key"],
    )


async def create_barcodebuddy_client() -> BarcodeBuddyClient:
    """
    Kept for compatibility; returns a singleton client.
    """
    return _barcodebuddy_client_singleton()
