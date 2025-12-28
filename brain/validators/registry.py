from __future__ import annotations

from typing import Dict, List

from validators.base import BaseValidator
from validators.presence import NonEmptyValidator
from validators.staleness import AgeStalenessValidator


class ValidatorRegistry:
    """
    Registry for domain-specific validator packs.

    This is intentionally code-defined in v0 to avoid premature configuration
    surfaces. Validators are instantiated explicitly with known parameters.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, List[BaseValidator]] = {
            # Example domains — expand as real usage begins
            "grocy": [
                NonEmptyValidator(),
                AgeStalenessValidator(
                    warn_after_seconds=3600,
                    invalid_after_seconds=86400,
                ),
            ],
            "homeassistant": [
                NonEmptyValidator(),
                AgeStalenessValidator(
                    warn_after_seconds=300,
                    invalid_after_seconds=1800,
                ),
            ],
            "finance": [
                NonEmptyValidator(),
                AgeStalenessValidator(
                    warn_after_seconds=86400,
                    invalid_after_seconds=None,  # finance data ages slowly
                ),
            ],
        }

    def get_validators(self, domain: str) -> List[BaseValidator]:
        """
        Return validators registered for a domain.

        Unknown domains return an empty list.
        """
        return list(self._registry.get(domain, []))
