from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


class GrocyError(Exception):
    """Generic Grocy client error."""


@dataclass
class GrocyInstanceConfig:
    name: str
    base_url: str
    api_key: str


class _SingleGrocyClient:
    """
    Thin wrapper around a single Grocy instance.
    """

    def __init__(self, config: GrocyInstanceConfig, timeout: float = 10.0) -> None:
        self.config = config
        self.timeout = timeout

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = self.config.base_url.rstrip("/") + path
        headers = {
            "GROCY-API-KEY": self.config.api_key,
            "Accept": "application/json",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                )
            except httpx.RequestError as exc:  # type: ignore[attr-defined]
                raise GrocyError(
                    f"Error while requesting Grocy at {url}: {exc}"
                ) from exc

        if resp.status_code >= 400:
            raise GrocyError(
                f"Grocy API error {resp.status_code} for {url}: {resp.text}"
            )

        try:
            return resp.json()
        except ValueError as exc:
            raise GrocyError(
                f"Invalid JSON from Grocy at {url}: {resp.text[:200]}"
            ) from exc

    async def health(self) -> Dict[str, Any]:
        """
        Basic health info from /api/system/info.
        """
        return await self._request("GET", "/api/system/info")

    async def get_stock_overview(self) -> Dict[str, Any]:
        """
        Stock overview from /api/stock/volatile.
        """
        return await self._request("GET", "/api/stock/volatile")

    async def get_shopping_list(self) -> Any:
        """
        Consolidated shopping list from /api/stock/shoppinglist.
        """
        return await self._request("GET", "/api/stock/shoppinglist")

    async def get_products(self) -> Any:
        """
        All products from /api/objects/products.
        """
        return await self._request("GET", "/api/objects/products")

    async def get_locations(self) -> Any:
        """
        All locations from /api/objects/locations.
        """
        return await self._request("GET", "/api/objects/locations")


class GrocyClient:
    """
    High-level client which can talk to one or two Grocy instances.

    Instances are keyed by:
      - "home_a"
      - "home_b"

    At least one instance must be configured for the client to be usable.
    """

    def __init__(self, instances: Dict[str, _SingleGrocyClient]) -> None:
        if not instances:
            raise GrocyError("No Grocy instances configured")
        self.instances = instances

    def _select_instances(
        self,
        household: Optional[str],
    ) -> Dict[str, _SingleGrocyClient]:
        """
        Decide which instances to query based on requested household.

        household:
          - "home_a" -> only A
          - "home_b" -> only B
          - "all" / None / "" -> all configured
        """
        if not household or household == "all":
            return self.instances

        if household not in self.instances:
            raise GrocyError(f"Requested household '{household}' is not configured")

        return {household: self.instances[household]}

    async def health(self) -> Dict[str, Any]:
        """
        Return health info per configured instance.
        """
        results: Dict[str, Any] = {}
        for key, client in self.instances.items():
            try:
                info = await client.health()
                results[key] = {"status": "ok", "info": info}
            except GrocyError as exc:
                results[key] = {"status": "error", "error": str(exc)}

        return results

    async def get_stock_overview(
        self,
        household: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetch stock overview from one or more instances and merge the results.

        List-like values (e.g. stock items) are concatenated;
        non-list values are taken from the first instance queried.

        Each stock item gets an extra "household" field.
        """
        instances = self._select_instances(household)
        merged: Dict[str, Any] = {}
        first = True

        for name, client in instances.items():
            data = await client.get_stock_overview()
            for key, value in data.items():
                if isinstance(value, list):
                    items = []
                    for item in value:
                        if isinstance(item, dict):
                            # annotate each item with its household
                            item = {**item, "household": name}
                        items.append(item)
                    merged.setdefault(key, []).extend(items)
                else:
                    if first:
                        merged[key] = value
            first = False

        return merged

    async def get_shopping_list(
        self,
        household: Optional[str] = None,
    ) -> Any:
        """
        Fetch shopping list from one or more instances and merge.

        Behavior is similar to get_stock_overview:
        list-like values are concatenated, non-lists come from the first.
        """
        instances = self._select_instances(household)
        merged: Dict[str, Any] = {}
        first = True

        for name, client in instances.items():
            data = await client.get_shopping_list()

            if isinstance(data, list):
                # Simple list (older Grocy versions)
                items = []
                for item in data:
                    if isinstance(item, dict):
                        item = {**item, "household": name}
                    items.append(item)
                merged.setdefault("items", []).extend(items)

            elif isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, list):
                        items = []
                        for item in value:
                            if isinstance(item, dict):
                                item = {**item, "household": name}
                            items.append(item)
                        merged.setdefault(key, []).extend(items)
                    else:
                        if first:
                            merged[key] = value
            else:
                if first:
                    merged["raw"] = data

            first = False

        return merged

    async def get_products(
        self,
        household: Optional[str] = None,
    ) -> Any:
        """
        Fetch products from one or more instances and merge them into a single list.
        Each product gets an extra "household" field.
        """
        instances = self._select_instances(household)
        products: List[Any] = []

        for name, client in instances.items():
            data = await client.get_products()
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        item = {**item, "household": name}
                    products.append(item)
            else:
                products.append({"household": name, "data": data})

        return products

    async def get_locations(
        self,
        household: Optional[str] = None,
    ) -> Any:
        """
        Fetch locations from one or more instances and merge into a single list.
        Each location gets a "household" field.
        """
        instances = self._select_instances(household)
        locations: List[Any] = []

        for name, client in instances.items():
            data = await client.get_locations()
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        item = {**item, "household": name}
                    locations.append(item)
            else:
                locations.append({"household": name, "data": data})

        return locations


async def create_grocy_client() -> GrocyClient:
    """
    Factory that builds a GrocyClient from environment variables.

    Supported env vars:

      # Backwards-compatible primary instance (Household A)
      GROCY_BASE_URL
      GROCY_API_KEY

      # Explicit per-household config (preferred for Phase 6.5+)
      GROCY_HOME_A_BASE_URL
      GROCY_HOME_A_API_KEY
      GROCY_HOME_B_BASE_URL
      GROCY_HOME_B_API_KEY
    """
    instances: Dict[str, _SingleGrocyClient] = {}

    # Household A (or legacy single-instance)
    home_a_base = os.getenv("GROCY_HOME_A_BASE_URL") or os.getenv("GROCY_BASE_URL")
    home_a_key = os.getenv("GROCY_HOME_A_API_KEY") or os.getenv("GROCY_API_KEY")

    if home_a_base and home_a_key:
        instances["home_a"] = _SingleGrocyClient(
            GrocyInstanceConfig(
                name="home_a",
                base_url=home_a_base,
                api_key=home_a_key,
            )
        )

    # Household B (new in Phase 6.5)
    home_b_base = os.getenv("GROCY_HOME_B_BASE_URL")
    home_b_key = os.getenv("GROCY_HOME_B_API_KEY")

    if home_b_base and home_b_key:
        instances["home_b"] = _SingleGrocyClient(
            GrocyInstanceConfig(
                name="home_b",
                base_url=home_b_base,
                api_key=home_b_key,
            )
        )

    if not instances:
        raise GrocyError(
            "Grocy not configured. Set GROCY_BASE_URL/GROCY_API_KEY or "
            "GROCY_HOME_A_BASE_URL/GROCY_HOME_A_API_KEY at minimum."
        )

    return GrocyClient(instances)
