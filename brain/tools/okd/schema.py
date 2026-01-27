from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal


ExpansionType = Literal["query_reformulation", "retry", "scope_expansion", "crawl"]


@dataclass
class ScopeBudget:
    queries: int = 4
    opens: int = 6
    crawl_depth: int = 2
    retries_per_action: int = 2
    # If provided, restrict to these domains (netloc). Empty means open scope.
    domains_allowlist: Optional[List[str]] = None


@dataclass
class ObservationPlan:
    intent: str
    user_prompt_excerpt: str
    expected_risk_tier: int
    initial_queries: List[str] = field(default_factory=list)
    planned_query_reformulations: List[str] = field(default_factory=list)
    scope_budget: ScopeBudget = field(default_factory=ScopeBudget)
    login_required: bool = False
    # Approval is handled as a separate artifact in v1 to keep artifacts append-only.
    approved_by_admin: bool = False


@dataclass
class ExpansionEvent:
    type: ExpansionType
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    count: int = 1


@dataclass
class UpdateBrief:
    facts: List[Dict[str, Any]] = field(default_factory=list)
    inferences: List[Dict[str, Any]] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    uncertainties: List[Dict[str, Any]] = field(default_factory=list)
    scope_used: Dict[str, Any] = field(default_factory=dict)
    login_used: bool = False
