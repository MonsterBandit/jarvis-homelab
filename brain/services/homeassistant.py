from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:
            # Not critical; ignore cleanup errors
            pass
