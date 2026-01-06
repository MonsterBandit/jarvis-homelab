from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests


class ModelClientError(Exception):
    """Generic error from a model backend."""


class ModelClient(ABC):
    """
    Abstract base for any chat model backend (OpenAI-compatible, Gemini, etc).
    The rest of the app should talk to THIS interface.
    """

    @abstractmethod
    async def generate_chat_completion(
        self,
        *,
        model: Optional[str],
        messages: List[Dict[str, Any]],
        temperature: float = 0.5,
        max_output_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        """
        Return the assistant's reply as a STRING.

        - model: optional model override
        - messages: list of {"role": "...", "content": "..."}
        - temperature: sampling temperature
        - max_output_tokens: output token cap
        - kwargs: reserved for future extras
        """
        raise NotImplementedError


def _normalize_api_base(api_base: str) -> str:
    """
    We treat api_base as a provider root (Option A):
      https://api.openai.com

    If someone accidentally sets:
      https://api.openai.com/v1
    we strip the trailing /v1 to prevent /v1/v1 paths.
    """
    base = (api_base or "").strip().rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base.rstrip("/")


def _extract_openai_compatible_text(payload: Dict[str, Any]) -> str:
    """
    Extract assistant text from OpenAI-compatible responses.

    Expected shape:
      {"choices":[{"message":{"content":"..."}}]}
    """
    try:
        choices = payload.get("choices") or []
        if not choices:
            raise KeyError("choices missing/empty")

        first = choices[0] or {}
        msg = first.get("message") or {}
        content = msg.get("content")

        if isinstance(content, str) and content.strip():
            return content

        # Some providers may return {"choices":[{"text":"..."}]} for non-chat
        text = first.get("text")
        if isinstance(text, str) and text.strip():
            return text

        raise KeyError("no assistant content found")
    except Exception as exc:
        # Keep this short and safe; don't dump giant payloads.
        snippet = str(payload)[:1200]
        raise ModelClientError(f"LLM response missing text content. Snippet: {snippet}") from exc


class OpenAICompatibleClient(ModelClient):
    """
    OpenAI-compatible client that calls:
      POST {api_base}/v1/chat/completions

    NOTE: api_base must be the provider root (no /v1). We sanitize if present.
    """

    def __init__(self, *, api_base: str, api_key: str, default_model: str) -> None:
        self.api_base = _normalize_api_base(api_base)
        self.api_key = (api_key or "").strip()
        self.default_model = (default_model or "").strip()

        if not self.api_base:
            raise ModelClientError("OpenAI-compatible api_base is empty")
        if not self.api_key:
            raise ModelClientError("OpenAI-compatible api_key is empty")
        if not self.default_model:
            raise ModelClientError("OpenAI-compatible default_model is empty")

    async def generate_chat_completion(
        self,
        *,
        model: Optional[str],
        messages: List[Dict[str, Any]],
        temperature: float = 0.5,
        max_output_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        chosen_model = (model or self.default_model).strip()
        if not chosen_model:
            raise ModelClientError("No model specified for OpenAI-compatible request")

        url = f"{self.api_base}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # OpenAI-compatible standard uses max_tokens.
        payload: Dict[str, Any] = {
            "model": chosen_model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_output_tokens),
        }

        # Allow future extensions without breaking callers
        # (but do not invent new required fields).
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]
        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
        except Exception as exc:
            raise ModelClientError(f"OpenAI-compatible request failed: {exc}") from exc

        if resp.status_code >= 400:
            body = (resp.text or "")[:1200]
            raise ModelClientError(
                f"OpenAI-compatible HTTP {resp.status_code} from {url}. Body: {body}"
            )

        try:
            data = resp.json()
        except Exception as exc:
            body = (resp.text or "")[:1200]
            raise ModelClientError(f"OpenAI-compatible response not JSON. Body: {body}") from exc

        return _extract_openai_compatible_text(data)


class GeminiModelClient(ModelClient):
    """
    Placeholder for Gemini until explicitly implemented.
    """

    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    async def generate_chat_completion(
        self,
        *,
        model: Optional[str],
        messages: List[Dict[str, Any]],
        temperature: float = 0.5,
        max_output_tokens: int = 512,
        **kwargs: Any,
    ) -> str:
        raise ModelClientError(
            "GeminiModelClient is not implemented yet. "
            "Set JARVIS_LLM_PROVIDER=openai-compatible to use the current backend."
        )


def build_model_client_from_env() -> ModelClient:
    """
    Factory used by main.py to build the configured client.

    Supported env (current stack):
      JARVIS_LLM_PROVIDER=openai-compatible
      JARVIS_LLM_API_KEY=...
      JARVIS_LLM_BASE_URL=https://api.openai.com
      JARVIS_LLM_MODEL=...

    Also supported (legacy/alternate):
      OPENAI_COMPATIBLE_API_KEY=...
      OPENAI_COMPATIBLE_API_BASE=https://api.openai.com
    """
    provider = (os.getenv("JARVIS_LLM_PROVIDER") or "").strip().lower()

    if provider in ("openai-compatible", "openai_compatible", "openai"):
        api_base = (
            os.getenv("OPENAI_COMPATIBLE_API_BASE")
            or os.getenv("JARVIS_LLM_BASE_URL")
            or ""
        )
        api_key = (
            os.getenv("OPENAI_COMPATIBLE_API_KEY")
            or os.getenv("JARVIS_LLM_API_KEY")
            or ""
        )
        model = os.getenv("JARVIS_LLM_MODEL") or ""
        return OpenAICompatibleClient(api_base=api_base, api_key=api_key, default_model=model)

    if provider == "gemini":
        # Not implemented yet; fail loudly and clearly.
        return GeminiModelClient()

    raise ModelClientError(
        f"Unknown JARVIS_LLM_PROVIDER={provider!r}. "
        "Expected 'openai-compatible' (or 'gemini' when implemented)."
    )
