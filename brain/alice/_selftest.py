"""
alice/_selftest.py

Manual sanity checks for Alice v1 mediation logic.
Run directly with: python3 -m alice._selftest
(or from /opt/jarvis/brain: python3 -m alice._selftest)

No I/O beyond printing to stdout.
No persistence. No runtime wiring.
"""

from __future__ import annotations

from alice.mediator import mediate
from alice.types import (
    BoundaryNotice,
    BoundaryType,
    Confidence,
    ConfidenceLevel,
    IsacSignal,
    IsacSignalKind,
)


def _print_case(title: str, sig: IsacSignal) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    resp = mediate(sig)

    print(f"Move: {resp.move.value}")
    if resp.meta:
        print(f"Meta: {resp.meta}")

    print("\nUtterances:")
    for u in resp.utterances:
        print(f"- {u.text}")

    print("\nQuestions:")
    for q in resp.questions:
        print(f"- {q}")

    print("\nOptions:")
    for o in resp.options:
        print(f"- {o}")

    # Enforce your global rule: one question at a time.
    assert len(resp.questions) <= 1, "Violation: more than one question returned!"

    print("\n✅ PASS: one-question rule satisfied")


def main() -> None:
    # 1) Informational, high confidence
    _print_case(
        "CASE 1: INFORMATIONAL / HIGH",
        IsacSignal(
            kind=IsacSignalKind.INFORMATIONAL,
            title="System Status",
            summary="All monitored services report healthy status.",
            confidence=Confidence(level=ConfidenceLevel.HIGH, score=0.95),
        ),
    )

    # 2) Uncertain, low confidence (should ask one question)
    _print_case(
        "CASE 2: UNCERTAIN / LOW",
        IsacSignal(
            kind=IsacSignalKind.UNCERTAIN,
            title="Ambiguity Detected",
            summary="I see incomplete information for this request.",
            confidence=Confidence(level=ConfidenceLevel.LOW, score=0.22),
        ),
    )

    # 3) Tension (should present unranked options + one question)
    _print_case(
        "CASE 3: TENSION / MEDIUM",
        IsacSignal(
            kind=IsacSignalKind.TENSION,
            title="Conflicting Interpretations",
            summary="Two interpretations appear plausible based on available signals.",
            confidence=Confidence(level=ConfidenceLevel.MEDIUM, score=0.55),
        ),
    )

    # 4) Hard boundary (should render boundary block and avoid extra questions)
    _print_case(
        "CASE 4: RISK + HARD BOUNDARY",
        IsacSignal(
            kind=IsacSignalKind.RISK,
            title="Action Blocked",
            summary="A requested action would cross a locked boundary.",
            confidence=Confidence(level=ConfidenceLevel.HIGH, score=0.9),
            boundaries=[
                BoundaryNotice(
                    boundary_type=BoundaryType.HARD,
                    source="Global Execution Rules",
                    message="No execution and no runtime mutation are allowed in the current state.",
                    allowed_alternatives=[
                        "Draft the procedure instead (design-only).",
                        "Prepare a checklist for later authorization.",
                        "Pause and revisit after Gate D assessment.",
                    ],
                )
            ],
        ),
    )

    print("\n" + "=" * 72)
    print("ALL SELFTEST CASES PASSED ✅")
    print("=" * 72)


if __name__ == "__main__":
    main()
