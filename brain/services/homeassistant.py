from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class HomeAssistantConfig:
    base_url: str
    token: str
    timeout: float = 8.0  # slightly more generous default

    @property
    def api_base(self) -> str:
        return self.base_url.rstrip("/") + "/api"


class HomeAssistantClient:
    """
    Home Assistant HTTP API client.

    NOTE ON ASYNC:
    This client uses `requests` internally (sync). For FastAPI usage we expose
    async wrappers that run sync methods in a thread via `asyncio.to_thread`.
    """

    def __init__(self, config: HomeAssistantConfig) -> None:
        self.config = config
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.config.token}",
                "Content-Type": "application/json",
            }
        )

    def _build_url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.config.api_base + path

    def _request(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        """
        Internal helper to send an HTTP request to Home Assistant
        with consistent timeout and error handling.
        """
        url = self._build_url(path)

        try:
            resp = self._session.request(
                method=method.upper(),
                url=url,
                json=json,
                timeout=self.config.timeout,
            )
        except requests.Timeout as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Home Assistant request to {url} timed out after {self.config.timeout} seconds"
            ) from exc
        except requests.RequestException as exc:  # noqa: BLE001
            raise RuntimeError(f"Error communicating with Home Assistant at {url}: {exc}") from exc

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:  # noqa: BLE001
            body_snippet = resp.text[:500]
            raise RuntimeError(
                f"Home Assistant returned HTTP {resp.status_code} for {url}: {body_snippet}"
            ) from exc

        return resp

    # --------------------------
    # Sync methods (requests)
    # --------------------------

    def health(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Legacy/simple health call: GET /api/ and verify we got a dict with "message".
        Returns (ok, payload).
        """
        url = self._build_url("/")
        try:
            resp = self._session.get(url, timeout=self.config.timeout)
        except Exception as exc:  # noqa: BLE001
            return False, {"error": str(exc), "url": url}

        try:
            data: Any = resp.json()
        except Exception:  # noqa: BLE001
            data = resp.text

        ok = resp.status_code == 200 and isinstance(data, dict) and "message" in data
        return ok, {"status_code": resp.status_code, "body": data, "url": url}

    def health_check_sync(self) -> Dict[str, Any]:
        """
        Canonical health check for observability capture:
        returns a single sanitized dict (no tuples).
        """
        ok, payload = self.health()
        result: Dict[str, Any] = {"ok": bool(ok)}

        # Preserve a few useful fields, but avoid huge dumps.
        if isinstance(payload, dict):
            if "url" in payload:
                result["url"] = payload.get("url")
            if "status_code" in payload:
                result["status_code"] = payload.get("status_code")

            body = payload.get("body")
            if isinstance(body, dict):
                msg = body.get("message")
                if isinstance(msg, str) and msg.strip():
                    result["message"] = msg.strip()

        # If legacy payload carried an error, bubble it up (sanitized).
        err = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(err, str) and err.strip():
            result["error"] = err.strip()

        return result

    def get_config_sync(self) -> Dict[str, Any]:
        resp = self._request("GET", "/config")
        data: Any = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("Unexpected Home Assistant /api/config response shape")
        return data

    def list_states_sync(self) -> List[Dict[str, Any]]:
        resp = self._request("GET", "/states")
        data: Any = resp.json()
        if not isinstance(data, list):
            raise RuntimeError("Unexpected Home Assistant /api/states response shape")
        return data

    def get_state_sync(self, entity_id: str) -> Dict[str, Any]:
        path = f"/states/{entity_id}"
        resp = self._request("GET", path)
        data: Any = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected Home Assistant /api/states/{entity_id} response shape")
        return data

    def list_services_sync(self) -> List[Dict[str, Any]]:
        resp = self._request("GET", "/services")
        data: Any = resp.json()
        if not isinstance(data, list):
            raise RuntimeError("Unexpected Home Assistant /api/services response shape")
        return data

    def call_service_sync(
        self,
        domain: str,
        service: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        path = f"/services/{domain}/{service}"
        payload: Dict[str, Any] = data or {}

        resp = self._request("POST", path, json=payload)

        try:
            body: Any = resp.json()
        except Exception:  # noqa: BLE001
            body = resp.text

        return {"status_code": resp.status_code, "body": body, "url": self._build_url(path)}

    # --------------------------
    # Async wrappers (FastAPI)
    # --------------------------

    async def health_check(self) -> Dict[str, Any]:
        return await asyncio.to_thread(self.health_check_sync)

    async def get_config(self) -> Dict[str, Any]:
        return await asyncio.to_thread(self.get_config_sync)

    async def get_states(self) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.list_states_sync)

    async def get_state(self, entity_id: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self.get_state_sync, entity_id)

    async def get_services(self) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.list_services_sync)

    async def call_service(
        self,
        domain: str,
        service: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(self.call_service_sync, domain, service, data)

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass
