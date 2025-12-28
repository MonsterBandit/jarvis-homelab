from dataclasses import dataclass, field
from typing import Dict, List

from .snapshot import ValidationSnapshot


@dataclass
class HealthAggregate:
    """
    In-memory aggregation of validation snapshots.

    Passive container:
    - No decisions
    - No persistence
    - No scoring
    """

    snapshots_by_domain: Dict[str, List[ValidationSnapshot]] = field(default_factory=dict)

    def add(self, snapshot: ValidationSnapshot) -> None:
        """
        Add a validation snapshot to the aggregate.
        """
        self.snapshots_by_domain.setdefault(snapshot.domain, []).append(snapshot)

    def get_domain(self, domain: str) -> List[ValidationSnapshot]:
        """
        Retrieve all snapshots for a given domain.
        """
        return self.snapshots_by_domain.get(domain, [])

    def domains(self) -> List[str]:
        """
        List domains currently represented in the aggregate.
        """
        return list(self.snapshots_by_domain.keys())
