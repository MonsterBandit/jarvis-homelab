from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import requests


@dataclass
class HomeAssistantConfig:
    base_url: str
    token: str
    timeout: float = 5.0

    @property
    def api_base(self) -> str:
        return self.base_url.rstrip("/") + "/api"


class HomeAssistantClient:
    """
    Minimal synchronous client for Home Assistant's HTTP API.

    Right now we only need a simple health check. We can extend this later
    with methods for entities, states, and service calls.
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

    def health(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Call HA /api/ endpoint to verify connectivity.

        Returns:
            (ok, details)
            ok: True if HA responds with 200 and a JSON body containing "message".
            details: dict with status_code, body, and/or error info.
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

    # =========================
    # New Phase 5.3 methods
    # =========================

    def list_states(self) -> List[Dict[str, Any]]:
        """
        Return all entity states from Home Assistant (`GET /api/states`).
        """
        url = self._build_url("/states")
        resp = self._session.get(url, timeout=self.config.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_state(self, entity_id: str) -> Dict[str, Any]:
        """
        Return the state for a single entity (`GET /api/states/{entity_id}`).
        """
        path = f"/states/{entity_id}"
        url = self._build_url(path)
        resp = self._session.get(url, timeout=self.config.timeout)
        resp.raise_for_status()
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
        url = self._build_url(path)
        payload: Dict[str, Any] = data or {}

        resp = self._session.post(
            url,
            json=payload,
            timeout=self.config.timeout,
        )
        resp.raise_for_status()

        try:
            body: Any = resp.json()
        except Exception:  # noqa: BLE001
            body = resp.text

        return {
            "status_code": resp.status_code,
            "body": body,
            "url": url,
        }

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            # Not critical; ignore cleanup errors
            pass
