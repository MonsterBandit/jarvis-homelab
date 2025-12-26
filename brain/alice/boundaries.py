"""
alice/boundaries.py

Domain-neutral boundary rendering for Alice v1.

Responsibilities:
- Convert BoundaryNotice objects into human-facing lines that:
  - name the boundary
  - name the source
  - state what remains possible (unranked alternatives)
  - offer pause/defer

This module performs no I/O and has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .types import BoundaryNotice, BoundaryType
from . import language


@dataclass(frozen=True)
class RenderedBoundary:
    """
    Human-facing boundary output plus an audit echo.
    """
    lines: List[str]
    echo: BoundaryNotice


def render_boundary(b: BoundaryNotice) -> RenderedBoundary:
    """
    Render a single boundary notice into deterministic, calm language.

    Rules:
    - HARD boundaries must be clear and final, without threats.
    - SOFT boundaries emphasize timing/clarity and option to pause.
    - VISIBILITY boundaries emphasize intentional scope exclusion.
    - Alternatives are presented as unranked bullet-style options.
    """
    lines: List[str] = []

    if b.boundary_type == BoundaryType.HARD:
        lines.append(language.boundary_hard())
        lines.append(language.boundary_source_intro(b.source))
        if b.message:
            lines.append(b.message)

        if b.allowed_alternatives:
            lines.append(language.boundary_alternatives_intro())
            lines.extend(_render_alternatives(b.allowed_alternatives))

        lines.append(language.boundary_pause_or_defer())

    elif b.boundary_type == BoundaryType.SOFT:
        # Soft boundaries are hesitation due to clarity/confidence/risk thresholds.
        # We keep them non-urgent and collaborative.
        lines.append("I’m hesitant to go further because there isn’t enough clarity yet.")
        if b.message:
            lines.append(b.message)
        lines.append("With more information, this might look different.")

        if b.allowed_alternatives:
            lines.append(language.boundary_alternatives_intro())
            lines.extend(_render_alternatives(b.allowed_alternatives))

        lines.append(language.pause_offer())

    elif b.boundary_type == BoundaryType.VISIBILITY:
        # Visibility boundaries are intentional scope exclusions (e.g., finance excluded).
        lines.append(language.scope_visibility())
        if b.message:
            lines.append(b.message)
        lines.append(language.scope_can_add_later())
        if b.allowed_alternatives:
            lines.append(language.boundary_alternatives_intro())
            lines.extend(_render_alternatives(b.allowed_alternatives))
        lines.append(language.no_urgency())

    else:
        # Fallback: treat unknown boundary types as HARD for safety.
        lines.append(language.boundary_hard())
        lines.append(language.boundary_source_intro(b.source))
        if b.message:
            lines.append(b.message)
        if b.allowed_alternatives:
            lines.append(language.boundary_alternatives_intro())
            lines.extend(_render_alternatives(b.allowed_alternatives))
        lines.append(language.boundary_pause_or_defer())

    lines = language.sanitize_output_lines(lines)
    return RenderedBoundary(lines=lines, echo=b)


def render_boundaries(boundaries: List[BoundaryNotice]) -> Tuple[List[str], List[BoundaryNotice]]:
    """
    Render a list of boundaries into a single block of lines plus an echo list.

    Output rules:
    - Preserve order received (for auditability)
    - Separate boundaries with a blank line for readability
    """
    all_lines: List[str] = []
    echo: List[BoundaryNotice] = []

    for idx, b in enumerate(boundaries or []):
        rb = render_boundary(b)
        all_lines.extend(rb.lines)
        echo.append(rb.echo)
        if idx < len(boundaries) - 1:
            all_lines.append("")  # visual separator

    all_lines = language.sanitize_output_lines(all_lines)
    return all_lines, echo


def _render_alternatives(alts: List[str]) -> List[str]:
    """
    Render unranked alternatives. Keep them short, neutral, and reversible.
    """
    rendered: List[str] = []
    for a in alts:
        a_clean = language.normalize_whitespace(a)
        if a_clean:
            rendered.append(f"- {a_clean}")
    return rendered
