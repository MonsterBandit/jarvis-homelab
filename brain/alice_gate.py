"""
Alice Advance Gate — v2 (Presence + Legitimacy)
----------------------------------------------
Pure governance logic enforcing Presence + Legitimacy.

Hard rule:
- NO wake words / name-prefix requirements.
- The gate consumes semantic intent (via flags), not magic strings.

This module must have:
- no side effects
- no I/O
- no model calls
- no framework dependencies

It answers one question only:
Is Alice allowed to advance in this moment?
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class GateDecision(str, Enum):
    RESTRAIN = "RESTRAIN"
    CLARIFY = "CLARIFY"
    ADVANCE_OK = "ADVANCE_OK"


# Intents that are *not* an invitation to engage.
# (Presence still exists, but speech is restrained.)
NO_INVITATION_INTENTS = {
    "ambient_test",      # e.g., "ping"
    "ambient_noise",     # background / stray text
    "thinking_aloud",    # e.g., "hmm", "..."
    "no_invitation",     # explicit classification from upstream
}


# Intents that *are* an invitation to engage.
INVITATION_INTENTS = {
    "invitation",        # e.g., "can you help me?"
    "direct_request",    # e.g., "explain admin unlock"
    "question",          # any clear question
    "command",           # "do X" (still gated by capability elsewhere)
    "continuation",      # conversational continuation
    "emotional_signal",  # e.g., "ugh" (often warrants response, but may clarify)
    "statement",         # plain statements often imply engagement in a dialog
}


def advance_gate(
    *,
    msg: str,
    ctx: List[Dict[str, str]],
    mode: str = "normal",
    flags: Optional[Dict[str, Any]] = None,
) -> Tuple[GateDecision, List[str], Optional[str]]:
    """
    Returns:
        decision: GateDecision
        reasons: list[str]
        question: optional[str] (only if CLARIFY)
    """

    flags = flags or {}
    reasons: List[str] = []

    # --- Hard Stops ---
    if flags.get("is_alignment_turn"):
        return GateDecision.RESTRAIN, ["ALIGNMENT_LOCK"], None

    if flags.get("is_reset_or_uncertain_state"):
        return GateDecision.RESTRAIN, ["RESET_UNCERTAIN"], None

    # --- Ingestion Ritual ---
    if mode == "ingestion":
        if flags.get("missing_required_input"):
            return (
                GateDecision.CLARIFY,
                ["INGESTION_MISSING_INPUT"],
                str(flags.get("clarifying_question") or "What information is missing?"),
            )

        return (
            GateDecision.ADVANCE_OK,
            ["INGESTION_INVITED", "SEQUENCE_OK", "UNDERSTANDING_OK"],
            None,
        )

    # --- Normal Mode (Presence-first) ---
    # Back-compat: older clients might still send explicit_invitation.
    explicit_invitation = bool(flags.get("explicit_invitation"))

    # Preferred path: semantic intent classification from upstream (backend-owned).
    intent = str(flags.get("intent") or flags.get("semantic_intent") or "unknown").strip().lower()

    if explicit_invitation:
        invited = True
        reasons.append("EXPLICIT_INVITATION")
    elif intent in NO_INVITATION_INTENTS:
        invited = False
        reasons.append(f"NO_INVITATION_INTENT:{intent}")
    elif intent in INVITATION_INTENTS:
        invited = True
        reasons.append(f"INVITATION_INTENT:{intent}")
    else:
        # Unknown intent: fail closed, but *not* by demanding a wake word.
        # Treat as not invited until clarified by a real request.
        invited = False
        reasons.append(f"UNKNOWN_INTENT:{intent}")

    if not invited:
        return GateDecision.RESTRAIN, (reasons or ["NO_INVITATION"]), None

    # If we are invited but still unclear, ask exactly one question.
    if flags.get("needs_clarification"):
        return (
            GateDecision.CLARIFY,
            (reasons + ["NEEDS_CLARIFICATION"]),
            str(flags.get("clarifying_question") or "What would you like me to do next?"),
        )

    return (
        GateDecision.ADVANCE_OK,
        (reasons + ["INVITED_ADVANCE", "SEQUENCE_OK", "UNDERSTANDING_OK"]),
        None,
    )