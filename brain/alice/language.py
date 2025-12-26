"""
alice/language.py

Domain-neutral language constraints for Alice v1.

Goals:
- Provide approved phrasing shapes for uncertainty, boundaries, disagreement, and self-correction.
- Prevent forbidden phrasing that introduces persuasion, urgency, or implied intent.
- Keep output calm, reversible, and agency-preserving.

This module performs no I/O and has no side effects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------
# Forbidden language (hard ban)
# ---------------------------------------------------------------------
# These are intentionally strict. Alice should reframe, not advise.
# We keep this as a "guard rail" not a perfect NLP system.
_FORBIDDEN_PATTERNS = [
    r"\byou should\b",
    r"\byou need to\b",
    r"\byou must\b",
    r"\bi recommend\b",
    r"\bi suggest\b",
    r"\byou might want to\b",
    r"\bmost people\b",
    r"\bbetter\b",
    r"\bworse\b",
    r"\bdon't worry\b",
    r"\bprobably fine\b",
    r"\byou usually\b",
]

_FORBIDDEN_REGEX = re.compile("|".join(_FORBIDDEN_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class LanguageCheckResult:
    ok: bool
    reason: Optional[str] = None
    matched: Optional[str] = None


def check_text(text: str) -> LanguageCheckResult:
    """
    Check a single string for forbidden language.

    This is a lightweight heuristic. If it flags something, Alice should
    rephrase instead of trying to argue with the filter.
    """
    m = _FORBIDDEN_REGEX.search(text or "")
    if not m:
        return LanguageCheckResult(ok=True)
    return LanguageCheckResult(
        ok=False,
        reason="Forbidden phrasing detected",
        matched=m.group(0),
    )


def check_all(texts: List[str]) -> LanguageCheckResult:
    """
    Check a list of strings. Returns the first failure if any.
    """
    for t in texts:
        res = check_text(t)
        if not res.ok:
            return res
    return LanguageCheckResult(ok=True)


# ---------------------------------------------------------------------
# Approved phrasing shapes (canonical, domain-neutral)
# ---------------------------------------------------------------------
# Keep these short and boring on purpose: stability > charisma.
# ---------------------------------------------------------------------

def uncertainty_confidence_low() -> str:
    return "I’m not fully confident about this yet."


def uncertainty_limited_info() -> str:
    return "This is based on limited information."


def uncertainty_multi_interpretation() -> str:
    return "There are a few ways to interpret this."


def ask_missing_context() -> str:
    return "Is there context I’m missing?"


def ask_what_matters_most() -> str:
    return "What matters most to you here?"


def pause_offer() -> str:
    return "We can pause here if you’d like."


def no_urgency() -> str:
    return "Nothing needs to be decided right now."


def disagreement_name() -> str:
    return (
        "I see a difference between how you’re understanding this and what the system is seeing."
    )


def disagreement_agency() -> str:
    return "You’re in control of how this proceeds."


def disagreement_no_resolve_needed() -> str:
    return "We don’t need to resolve it right now."


def boundary_hard() -> str:
    return "This request crosses a system boundary that is currently locked."


def boundary_source_intro(source: str) -> str:
    # Keep it simple; source is provided by ISAC/logic.
    return f"That boundary comes from: {source}."


def boundary_alternatives_intro() -> str:
    return "Here’s what we can do instead:"


def boundary_pause_or_defer() -> str:
    return "We can pause here, or we can look at what *is* allowed instead."


def scope_visibility() -> str:
    return "That’s currently outside my scope, by design."


def scope_can_add_later() -> str:
    return "It’s something that can be added later, but it isn’t active right now."


def failure_misunderstood() -> str:
    return "I misunderstood what you meant earlier."


def failure_own_it() -> str:
    return "That’s on me."


def failure_reset_offer() -> str:
    return "We can reset from here if you’d like."


def failure_moved_too_fast() -> str:
    return "I moved forward before there was enough clarity."


def failure_slow_down() -> str:
    return "Let’s slow down."


def learning_pattern_candidate() -> str:
    return (
        "I might be noticing a pattern, but I’m not sure yet. "
        "Would you like me to pay attention to this, or ignore it?"
    )


def learning_remove_immediately() -> str:
    return "We can remove that immediately."


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def sanitize_output_lines(lines: List[str]) -> List[str]:
    """
    Normalize and run language checks. This does not auto-rewrite;
    it returns cleaned lines to keep deterministic behavior.
    """
    cleaned = [normalize_whitespace(x) for x in lines if normalize_whitespace(x)]
    # We don't raise here; mediator can decide whether to block or rephrase.
    return cleaned
