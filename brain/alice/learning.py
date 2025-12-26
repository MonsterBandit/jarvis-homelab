"""
alice/learning.py

Learning gates & promotion mechanics for Alice v1.

IMPORTANT:
- Learning is disabled by default.
- This module provides structures and checks only.
- No persistence, no I/O, no implicit memory.

A future wiring layer may connect this to a datastore, but only with explicit
Admin authorization and clear rollback semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class LearningTier(str, Enum):
    """
    Canonical learning ladder.
    """
    TIER_0_EPHEMERAL = "tier0_ephemeral"
    TIER_1_PATTERN_CANDIDATE = "tier1_pattern_candidate"
    TIER_2_SOFT_PREFERENCE = "tier2_soft_preference"
    TIER_3_STRUCTURAL_SIGNAL = "tier3_structural_signal"


@dataclass(frozen=True)
class LearningItem:
    """
    A single learnable unit (preference or signal), always scoped and reversible.
    """
    key: str
    description: str
    tier: LearningTier
    domain_scope: str = "global"  # must remain explicit; no cross-domain leakage
    is_enabled: bool = False      # nothing is enabled by default
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LearningDecision:
    """
    Represents an explicit human decision about learning.
    """
    approved: bool
    tier: LearningTier
    reason: Optional[str] = None


@dataclass
class LearningState:
    """
    In-memory representation of learning. This is intentionally ephemeral until
    a future authorized persistence layer is introduced.

    For Alice v1, this should remain empty or disabled.
    """
    enabled: bool = False
    items: Dict[str, LearningItem] = field(default_factory=dict)


def allow_learning(state: LearningState) -> bool:
    """
    Global kill-switch. If disabled, Alice must not learn anything.
    """
    return bool(state and state.enabled)


def propose_pattern_candidate(key: str, description: str, domain_scope: str = "global") -> LearningItem:
    """
    Create a Tier 1 proposal. This does NOT store anything.
    """
    return LearningItem(
        key=key,
        description=description,
        tier=LearningTier.TIER_1_PATTERN_CANDIDATE,
        domain_scope=domain_scope,
        is_enabled=False,
    )


def apply_decision(state: LearningState, item: LearningItem, decision: LearningDecision) -> LearningState:
    """
    Apply an explicit decision to learning state.

    Rules:
    - If learning is disabled, never store anything.
    - If decision not approved, do not store anything.
    - Tier promotions must match decision tier (no silent promotion).
    - Tier 3 should be treated as Admin-only by policy; enforcement occurs in wiring layer.
    """
    if not allow_learning(state):
        return state

    if not decision.approved:
        # Declined means discard with no retention.
        return state

    if decision.tier != item.tier and decision.tier.value != item.tier.value:
        # No silent promotion or tier rewriting.
        return state

    # Store as enabled only if explicitly approved.
    stored = LearningItem(
        key=item.key,
        description=item.description,
        tier=item.tier,
        domain_scope=item.domain_scope,
        is_enabled=True,
        meta=dict(item.meta),
    )
    state.items[item.key] = stored
    return state


def remove_learning_item(state: LearningState, key: str) -> LearningState:
    """
    Immediate removal. No questions, no confirmation loops.
    """
    if not state:
        return state
    state.items.pop(key, None)
    return state


def reset_learning(state: LearningState) -> LearningState:
    """
    Full reset. This is always allowed.
    """
    if not state:
        return state
    state.items.clear()
    return state
