from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


@dataclass
class HomeAssistantConfig:
    base_url: str
    token: str
    timeout: float = 8.0  # Phase 5.6: slightly more generous default

    @property
    def api_base(self) -> str:
        return self.base_url.rstrip("/") + "/api"


class HomeAssistantClient:
    """
    Minimal synchronous client for Home Assistant's HTTP API.

    We started with a simple health check and have been extending this
    with methods for entities, states, and service calls.
    Phase 5.6 adds better error handling and timeouts.
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
                f"Home Assistant request to {url} timed out after "
                f"{self.config.timeout} seconds"
            ) from exc
        except requests.RequestException as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Error communicating with Home Assistant at {url}: {exc}"
            ) from exc

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:  # noqa: BLE001
            # Include a small slice of the response body for diagnostics
            body_snippet = resp.text[:500]
            raise RuntimeError(
                f"Home Assistant returned HTTP {resp.status_code} for {url}: "
                f"{body_snippet}"
            ) from exc

        return resp

    def health(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Call HA /api/ endpoint to verify connectivity.

        Returns:
            (ok, details)

            ok:
                True if HA responds with 200 and a JSON body containing "message".
            details:
                dict with status_code, body, and/or error info.
        """
        url = self._build_url("/")
        try:
            resp = self._session.get(url, timeout=self.config.timeout)
        except Exception as exc:  # noqa: BLE001
            return False, {
                "error": str(exc),
                "url": url,
            }

        try:
            data: Any = resp.json()
        except Exception:  # noqa: BLE001
            data = resp.text

        ok = resp.status_code == 200 and isinstance(data, dict) and "message" in data

        return ok, {
            "status_code": resp.status_code,
            "body": data,
            "url": url,
        }

    # ==========================
    # States
    # ==========================

    def list_states(self) -> List[Dict[str, Any]]:
        """
        Return all entity states from Home Assistant (`GET /api/states`).
        """
        resp = self._request("GET", "/states")
        return resp.json()

    def get_state(self, entity_id: str) -> Dict[str, Any]:
        """
        Return the state for a single entity (`GET /api/states/{entity_id}`).
        """
        path = f"/states/{entity_id}"
        resp = self._request("GET", path)
        return resp.json()

    # ==========================
    # Services
    # ==========================

    def list_services(self) -> List[Dict[str, Any]]:
        """
        Return all available Home Assistant services (`GET /api/services`).
        """
        resp = self._request("GET", "/services")
        return resp.json()

    def call_service(
        self,
        domain: str,
        service: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call a Home Assistant service (`POST /api/services/{domain}/{service}`).

        Args:
            domain: e.g. "light", "switch", "script"
            service: e.g. "turn_on", "turn_off"
            data: payload dict passed through to Home Assistant.

        Returns:
            dict with status_code, body, and url.
        """
        path = f"/services/{domain}/{service}"
        payload: Dict[str, Any] = data or {}

        resp = self._request("POST", path, json=payload)

        try:
            body: Any = resp.json()
        except Exception:  # noqa: BLE001
            body = resp.text

        return {
            "status_code": resp.status_code,
            "body": body,
            "url": self._build_url(path),
        }

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            # Not critical; ignore cleanup errors
            pass
