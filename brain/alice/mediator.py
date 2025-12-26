"""
alice/mediator.py

Domain-neutral mediation logic for Alice v1.

This is the "decision tree" that converts an IsacSignal into an AliceResponse.

Hard constraints:
- One question at a time (global rule)
- No recommendations / persuasion
- Explicit uncertainty disclosure
- Boundaries are revealed, not negotiated
- No I/O, no persistence, no runtime hooks
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional

from .types import (
    AliceMove,
    AliceResponse,
    AliceUtterance,
    ConfidenceLevel,
    IsacSignal,
    IsacSignalKind,
)
from . import language
from . import boundaries as boundary_render
from .learning import LearningState


def mediate(signal: IsacSignal, learning_state: Optional[LearningState] = None) -> AliceResponse:
    """
    Convert a structured ISAC signal into a structured Alice response.

    Note:
    - learning_state exists for future wiring but is not used to alter behavior here.
    - This function must remain deterministic and side-effect-free.
    """
    utter_lines: List[str] = []
    questions: List[str] = []
    options: List[str] = []
    boundary_echo = []

    # 0) If boundaries exist, render them first. Boundaries are high-salience.
    if signal.boundaries:
        b_lines, b_echo = boundary_render.render_boundaries(signal.boundaries)

        # Present boundary clearly and without emotional framing.
        utter_lines.append("There is a system boundary in effect right now.")
        utter_lines.extend(b_lines)

        boundary_echo.extend(b_echo)

        # Choose containment move based on risk.
        move = AliceMove.DEFER if signal.kind == IsacSignalKind.RISK else AliceMove.PAUSE

        # Add minimal context (title/summary) after boundary block.
        utter_lines.extend(_title_and_summary(signal))

        return _finalize(
            move=move,
            utter_lines=utter_lines,
            questions=[],
            options=[],
            boundary_echo=boundary_echo,
            signal=signal,
        )
        

    # 1) No boundaries: proceed by signal kind.
    utter_lines.extend(_title_and_summary(signal))

    if signal.kind == IsacSignalKind.INFORMATIONAL:
        utter_lines.extend(_confidence_disclosure_if_needed(signal))
        move = AliceMove.EXPLAIN

    elif signal.kind == IsacSignalKind.UNCERTAIN:
        utter_lines.extend(_confidence_disclosure_if_needed(signal, force=True))
        utter_lines.append(language.uncertainty_limited_info())
        # One question maximum.
        questions.append(language.ask_missing_context())
        move = AliceMove.ASK

    elif signal.kind == IsacSignalKind.TENSION:
        # Treat "tension" as disagreement needing safe containment.
        utter_lines.append(language.disagreement_name())
        utter_lines.append(language.disagreement_agency())
        utter_lines.append(language.disagreement_no_resolve_needed())
        utter_lines.extend(_confidence_disclosure_if_needed(signal))

        # Offer unranked paths (no preference). Ask which path they want.
        options.extend(
            [
                "Clarify your understanding (I’ll ask a few questions).",
                "Clarify what the system is seeing (step by step).",
                "Hold the disagreement for now (pause and revisit later).",
            ]
        )
        questions.append("Which path would you like to take?")
        move = AliceMove.OPTIONS

    elif signal.kind == IsacSignalKind.RISK:
        # Without explicit boundaries, treat as a soft caution: slow down and offer pause.
        utter_lines.extend(_confidence_disclosure_if_needed(signal, force=True))
        utter_lines.append("I want to slow down here because there may be risk or a constraint nearby.")
        utter_lines.append(language.pause_offer())
        questions.append("Do you want to pause, or should we continue carefully with more context?")
        move = AliceMove.ASK

    else:
        # Unknown kind: default to cautious ask.
        utter_lines.extend(_confidence_disclosure_if_needed(signal, force=True))
        utter_lines.append("I’m not sure how to classify this, so I’d like to slow down.")
        questions.append(language.ask_missing_context())
        move = AliceMove.ASK

    return _finalize(
        move=move,
        utter_lines=utter_lines,
        questions=questions,
        options=options,
        boundary_echo=boundary_echo,
        signal=signal,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _title_and_summary(signal: IsacSignal) -> List[str]:
    """
    Keep title/summary short and neutral.
    """
    lines: List[str] = []
    if signal.title:
        lines.append(signal.title.strip())
    if signal.summary:
        lines.append(signal.summary.strip())
    return language.sanitize_output_lines(lines)


def _confidence_disclosure_if_needed(signal: IsacSignal, force: bool = False) -> List[str]:
    """
    Add confidence disclosure for non-HIGH or when forced.
    """
    lvl = (signal.confidence.level if signal.confidence else ConfidenceLevel.UNKNOWN)

    if not force and lvl == ConfidenceLevel.HIGH:
        return []

    if lvl == ConfidenceLevel.LOW:
        return [language.uncertainty_confidence_low()]
    if lvl == ConfidenceLevel.MEDIUM:
        return ["My confidence here is medium."]
    if lvl == ConfidenceLevel.HIGH:
        return ["My confidence here is high."]
    return ["My confidence here is unknown."]


def _finalize(
    move: AliceMove,
    utter_lines: List[str],
    questions: List[str],
    options: List[str],
    boundary_echo,
    signal: IsacSignal,
) -> AliceResponse:
    """
    Normalize, enforce one-question rule, and check for forbidden language.
    """
    utter_lines = language.sanitize_output_lines(utter_lines)

    # Enforce one question at a time (global rule)
    questions = [q.strip() for q in (questions or []) if q and q.strip()]
    if len(questions) > 1:
        questions = questions[:1]

    options = [language.normalize_whitespace(o) for o in (options or []) if o and language.normalize_whitespace(o)]

    # Language checks (utterances + questions + options)
    res = language.check_all(utter_lines + questions + options)
    if not res.ok:
        # Fail closed to a safe, boring response.
        safe_lines = [
            "I want to be careful with how I phrase this.",
            "Let’s slow down and make sure we’re aligned.",
            language.pause_offer(),
        ]
        safe_lines = language.sanitize_output_lines(safe_lines)
        return AliceResponse(
            move=AliceMove.PAUSE,
            utterances=[AliceUtterance(text=t) for t in safe_lines],
            questions=[language.ask_missing_context()],
            options=[],
            boundary_echo=list(boundary_echo or []),
            meta=_meta(signal, flagged_language=res.matched),
        )

    return AliceResponse(
        move=move,
        utterances=[AliceUtterance(text=t) for t in utter_lines],
        questions=questions,
        options=options,
        boundary_echo=list(boundary_echo or []),
        meta=_meta(signal),
    )


def _meta(signal: IsacSignal, flagged_language: Optional[str] = None) -> dict:
    """
    Minimal metadata for future UI/logging (Gate D evidence).
    No secrets, no full raw data dump.
    """
    meta = {
        "trace_id": signal.trace_id,
        "kind": signal.kind.value,
        "confidence_level": signal.confidence.level.value if signal.confidence else "unknown",
    }
    if flagged_language:
        meta["flagged_language"] = flagged_language
    # Include title only; keep summary and data out of meta to reduce leakage risk.
    if signal.title:
        meta["title"] = signal.title
    return meta
