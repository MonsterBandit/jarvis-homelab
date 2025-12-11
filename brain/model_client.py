from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests


class ModelClientError(Exception):
    """Generic error from a model backend."""
    pass


class ModelClient(ABC):
    """
    Abstract base for any chat model backend (OpenAI, Gemini, local, etc).
    Everything else in the app should talk to THIS instead of requests/OpenAI directly.
    """

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Send a chat-style request and return the raw response dict.

        - messages: list of {"role": "...", "content": "..."}
        - temperature: sampling temperature
        - kwargs: for future extras (max_tokens, top_p, etc.)
        """
        raise NotImplementedError


class OpenAICompatibleClient(ModelClient):
    """
    Simple client for OpenAI-compatible /chat/completions HTTP APIs.

    It expects:
      - base_url like "https://api.openai.com/v1" or an OpenAI-compatible URL
      - api_key for Authorization: Bearer ...
      - model name like "gpt-4o-mini"
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str],
        model: str,
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ModelClientError("LLM API key not configured.")

        url = f"{self.base_url.rstrip('/')}/chat/completions"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        # Optional extra params (e.g. max_tokens) can be passed via kwargs["extra_params"]
        extra_params = kwargs.get("extra_params")
        if isinstance(extra_params, dict):
            payload.update(extra_params)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        except requests.RequestException as e:
            raise ModelClientError(f"Error contacting LLM provider: {e}") from e

        if resp.status_code != 200:
            raise ModelClientError(
                f"LLM provider returned {resp.status_code}: {resp.text}"
            )

        try:
            data: Dict[str, Any] = resp.json()
        except ValueError as e:
            raise ModelClientError(f"Invalid JSON from LLM provider: {e}") from e

        return data

# ======================================================================
# Gemini client (stub for future use)
# ======================================================================

class GeminiModelClient(ModelClient):
    """
    Placeholder Gemini client.

    This lets the rest of the app know "Gemini" is a valid provider,
    but the actual HTTP call is not implemented yet.

    Later we'll flesh this out to call the real Gemini API.
    """

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        timeout: int = 30,
        api_base: str = "https://generativelanguage.googleapis.com/v1beta",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.api_base = api_base

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # For now, this is intentionally not wired up.
        # If someone selects Gemini too early, fail loudly and clearly.
        raise ModelClientError(
            "GeminiModelClient is not implemented yet. "
            "Set JARVIS_LLM_PROVIDER=openai to use the current backend."
        )
