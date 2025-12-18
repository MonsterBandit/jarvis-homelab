from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


class OpenFoodFactsError(Exception):
    pass


@dataclass
class OpenFoodFactsConfig:
    base_url: str = "https://world.openfoodfacts.org"


class OpenFoodFactsClient:
    """
    Minimal OpenFoodFacts client for barcode lookup (read-only).
    Uses OFF API v2: /api/v2/product/{barcode}.json

    Gate A compliant:
    - No writes, ever.
    - Pure lookup + parse.
    """

    def __init__(self, config: OpenFoodFactsConfig) -> None:
        self.config = config
        self.base_url = (config.base_url or "").rstrip("/")

    def _build_url(self, barcode: str) -> str:
        code = (barcode or "").strip()
        if not code:
            raise OpenFoodFactsError("barcode must not be empty")
        return f"{self.base_url}/api/v2/product/{code}.json"

    def lookup_barcode(self, barcode: str, timeout_seconds: float = 6.0) -> Dict[str, Any]:
        url = self._build_url(barcode)
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "ISAC-Homelab/1.0 (barcode lookup; contact: local)",
                "Accept": "application/json",
            },
            method="GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raise OpenFoodFactsError(f"HTTP {exc.code} from OpenFoodFacts") from exc
        except urllib.error.URLError as exc:
            raise OpenFoodFactsError(f"Network error calling OpenFoodFacts: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise OpenFoodFactsError(f"Unexpected error calling OpenFoodFacts: {exc}") from exc

        try:
            data = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            raise OpenFoodFactsError(f"Invalid JSON from OpenFoodFacts: {exc}") from exc

        if not isinstance(data, dict):
            raise OpenFoodFactsError("Unexpected response shape from OpenFoodFacts")

        return data


def create_openfoodfacts_client() -> OpenFoodFactsClient:
    # No env needed yet; keep it simple and deterministic.
    return OpenFoodFactsClient(OpenFoodFactsConfig())


def extract_suggestion_from_off_payload(payload: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Normalize OFF fields into suggestion-friendly values.
    Returns a small dict with keys: name, brand, category.
    """
    product = payload.get("product")
    if not isinstance(product, dict):
        return {"name": None, "brand": None, "category": None}

    name = product.get("product_name") or product.get("product_name_en") or product.get("generic_name")
    if isinstance(name, str):
        name = name.strip() or None
    else:
        name = None

    brand = product.get("brands")
    if isinstance(brand, str):
        brand = brand.strip() or None
    else:
        brand = None

    category = None
    # Prefer human-ish category; OFF commonly provides "categories_tags" and "categories".
    cats = product.get("categories")
    if isinstance(cats, str) and cats.strip():
        # Take first category label if comma-separated.
        category = cats.split(",")[0].strip() or None
    else:
        tags = product.get("categories_tags")
        if isinstance(tags, list) and tags:
            first = tags[0]
            if isinstance(first, str) and first.strip():
                category = first.strip()

    return {"name": name, "brand": brand, "category": category}
