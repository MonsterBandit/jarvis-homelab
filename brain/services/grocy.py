from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx


class GrocyError(Exception):
    pass


def _env(name: str) -> str:
    return (os.getenv(name, "") or "").strip()


@dataclass(frozen=True)
class GrocyInstanceConfig:
    name: str
    base_url: str
    api_key: str


def _load_instance(name: str) -> Optional[GrocyInstanceConfig]:
    base_url = _env(f"GROCY_{name.upper()}_BASE_URL")
    api_key = _env(f"GROCY_{name.upper()}_API_KEY")
    if not base_url or not api_key:
        return None
    return GrocyInstanceConfig(name=name, base_url=base_url, api_key=api_key)


class _SingleGrocyClient:
    def __init__(self, config: GrocyInstanceConfig, timeout: float = 20.0) -> None:
        self.config = config
        self.timeout = timeout

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
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
                    json=json_body,
                )
            except httpx.RequestError as exc:
                raise GrocyError(
                    f"Request error talking to Grocy '{self.config.name}': {exc}"
                ) from exc

        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise GrocyError(
                f"Grocy '{self.config.name}' returned {resp.status_code} for {method} {path}: {body}"
            )

        if not resp.content:
            return None

        try:
            return resp.json()
        except Exception:
            return resp.text

    async def health(self) -> Any:
        return await self._request("GET", "/api/system/config")

    async def get_stock_volatile(self) -> Any:
        return await self._request("GET", "/api/stock/volatile")

    async def get_shopping_list(self) -> Any:
        try:
            return await self._request("GET", "/api/stock/shoppinglist")
        except GrocyError as exc:
            msg = str(exc)
            if "returned 405" not in msg and "returned 404" not in msg:
                raise

        try:
            return await self._request("POST", "/api/stock/shoppinglist", json_body={})
        except GrocyError as exc:
            msg = str(exc)
            if "returned 405" not in msg and "returned 404" not in msg:
                raise

        try:
            return await self._request("GET", "/api/objects/shopping_list")
        except GrocyError as exc:
            msg = str(exc)
            if "returned 404" not in msg:
                raise

        return await self._request("GET", "/api/objects/shopping_list_item")

    async def get_products(self) -> Any:
        return await self._request("GET", "/api/objects/products")

    async def get_locations(self) -> Any:
        return await self._request("GET", "/api/objects/locations")

    async def get_quantity_units(self) -> Any:
        return await self._request("GET", "/api/objects/quantity_units")

    async def create_product(self, payload: Dict[str, Any]) -> Any:
        return await self._request("POST", "/api/objects/products", json_body=payload)

    async def create_location(self, payload: Dict[str, Any]) -> Any:
        return await self._request("POST", "/api/objects/locations", json_body=payload)

    async def create_product_barcode(self, payload: Dict[str, Any]) -> Any:
        return await self._request(
            "POST", "/api/objects/product_barcodes", json_body=payload
        )

    async def add_stock(self, product_id: int, payload: Dict[str, Any]) -> Any:
        return await self._request(
            "POST", f"/api/stock/products/{product_id}/add", json_body=payload
        )


class GrocyClient:
    def __init__(self, instances: Dict[str, _SingleGrocyClient]) -> None:
        self.instances = instances

    def _select_instances(self, household: str) -> List[Tuple[str, _SingleGrocyClient]]:
        h = (household or "").strip().lower()
        if h not in {"home_a", "home_b", "all"}:
            raise GrocyError("household must be 'home_a', 'home_b', or 'all'")

        if h == "all":
            return list(self.instances.items())

        if h not in self.instances:
            raise GrocyError(f"Grocy instance for household '{h}' is not configured")
        return [(h, self.instances[h])]

    def _get_single_instance(self, household: str) -> _SingleGrocyClient:
        h = (household or "").strip().lower()
        if h not in {"home_a", "home_b"}:
            raise GrocyError("household must be 'home_a' or 'home_b' for write operations")
        if h not in self.instances:
            raise GrocyError(f"Grocy instance for household '{h}' is not configured")
        return self.instances[h]

    async def health(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for name, client in self.instances.items():
            try:
                info = await client.health()
                results[name] = {"status": "ok", "info": info}
            except GrocyError as exc:
                results[name] = {"status": "error", "error": str(exc)}
        return results

    async def get_stock_overview(self, household: str) -> Any:
        payload: Dict[str, Any] = {}
        for name, client in self._select_instances(household):
            payload[name] = await client.get_stock_volatile()
        return payload if household == "all" else payload[household]

    async def get_shopping_list(self, household: str) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for name, client in self._select_instances(household):
            data = await client.get_shopping_list()
            if isinstance(data, list):
                for item in data:
                    merged.append({**item, "household": name})
            else:
                merged.append({"value": data, "household": name})
        return merged

    async def get_products(self, household: str) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for name, client in self._select_instances(household):
            data = await client.get_products()
            for p in data:
                merged.append({**p, "household": name})
        return merged

    async def get_locations(self, household: str) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for name, client in self._select_instances(household):
            data = await client.get_locations()
            for loc in data:
                merged.append({**loc, "household": name})
        return merged

    async def get_quantity_units(self, household: str) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for name, client in self._select_instances(household):
            data = await client.get_quantity_units()
            for unit in data:
                merged.append({**unit, "household": name})
        return merged

    async def create_product(
        self,
        household: str,
        name: str,
        location_id: int,
        qu_id_purchase: int,
        qu_id_stock: int,
    ) -> Dict[str, Any]:
        client = self._get_single_instance(household)

        payload = {
            "name": name,
            "location_id": location_id,
            "qu_id_purchase": qu_id_purchase,
            "qu_id_stock": qu_id_stock,
        }

        created = await client.create_product(payload)
        return {
            "household": household,
            **(created if isinstance(created, dict) else {"result": created}),
        }

    async def create_location(
        self,
        household: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create a Grocy location (explicit, household-scoped).
        """
        client = self._get_single_instance(household)

        created = await client.create_location(payload)
        return {
            "household": household,
            **(created if isinstance(created, dict) else {"result": created}),
        }

    async def link_barcode_to_product(
        self,
        household: str,
        barcode: str,
        product_id: int,
    ) -> Dict[str, Any]:
        """
        Create a product_barcodes record in Grocy linking barcode -> product_id.
        Phase 6.45 Step 3 requires explicit linking, no inference.
        """
        client = self._get_single_instance(household)

        code = (barcode or "").strip()
        if not code:
            raise GrocyError("barcode must not be empty")

        payload = {
            "barcode": code,
            "product_id": int(product_id),
        }

        created = await client.create_product_barcode(payload)
        return {
            "household": household,
            "barcode": code,
            "product_id": int(product_id),
            **(created if isinstance(created, dict) else {"result": created}),
        }

    async def add_stock(
        self,
        household: str,
        product_id: int,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Add stock for a product (explicit, household-scoped).
        Phase 6.45 Step 4 uses this only when explicitly requested.
        """
        client = self._get_single_instance(household)

        created = await client.add_stock(int(product_id), payload)
        return {
            "household": household,
            "product_id": int(product_id),
            **(created if isinstance(created, dict) else {"result": created}),
        }


async def create_grocy_client() -> GrocyClient:
    instances: Dict[str, _SingleGrocyClient] = {}

    for household in ("home_a", "home_b"):
        cfg = _load_instance(household)
        if cfg:
            instances[household] = _SingleGrocyClient(cfg)

    if not instances:
        raise GrocyError(
            "No Grocy instances configured. "
            "Set GROCY_HOME_A_BASE_URL / GROCY_HOME_A_API_KEY "
            "and/or GROCY_HOME_B_BASE_URL / GROCY_HOME_B_API_KEY."
        )

    return GrocyClient(instances=instances)
