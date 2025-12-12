import os
from typing import Any, Dict, List, Optional

import httpx


class GrocyError(Exception):
    """Base error for Grocy client problems."""


class GrocyClient:
    """
    Minimal async Grocy API client.

    Uses:
      - GROCY_BASE_URL, e.g. https://grocy.plexmoose.com
      - GROCY_API_KEY
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "GROCY-API-KEY": self.api_key,
        }

    async def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Internal helper for GET requests."""
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(url, headers=self._headers, params=params)
        except httpx.RequestError as exc:
            raise GrocyError(f"Error connecting to Grocy at {url}: {exc}") from exc

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise GrocyError(
                f"Grocy returned HTTP {resp.status_code} for {url}: {resp.text}"
            ) from exc

        # Most Grocy endpoints are JSON
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            return resp.json()
        return resp.text

    # ---------- Public API methods ----------

    async def health(self) -> Dict[str, Any]:
        """
        Simple health check: call /api/system/info.
        If that works, Grocy is alive.
        """
        info = await self._get("/api/system/info")
        return {"status": "ok", "info": info}

    async def get_stock_overview(self) -> Dict[str, Any]:
        """
        Returns the "volatile" stock overview including due/overdue products.

        Shape typically includes keys like:
        - due_products
        - overdue_products
        - expired_products
        - missing_products
        """
        return await self._get("/api/stock/volatile")

    async def get_raw_stock(self) -> List[Dict[str, Any]]:
        """Raw stock list."""
        return await self._get("/api/stock")

    async def get_shopping_list(self) -> Any:
        """
        High-level shopping list.

        Grocy exposes shopping list entries via /api/objects/shopping_list
        for GET requests. The /api/stock/shoppinglist endpoints are meant
        for actions (add/remove) and will return HTTP 405 on GET.
        """
        return await self._get("/api/objects/shopping_list")

    async def get_products(self) -> List[Dict[str, Any]]:
        """All products from the products table."""
        return await self._get("/api/objects/products")

    async def get_locations(self) -> List[Dict[str, Any]]:
        """All locations from the locations table."""
        return await self._get("/api/objects/locations")


async def create_grocy_client() -> GrocyClient:
    """
    Factory used by FastAPI dependencies.
    Reads env vars and returns a configured client.
    """
    base_url = os.getenv("GROCY_BASE_URL")
    api_key = os.getenv("GROCY_API_KEY")

    if not base_url or not api_key:
        raise GrocyError(
            "GROCY_BASE_URL and GROCY_API_KEY must be set in the environment."
        )

    return GrocyClient(base_url=base_url, api_key=api_key)
