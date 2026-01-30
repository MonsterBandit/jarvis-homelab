from __future__ import annotations

from typing import List


def code_review_rubric() -> List[str]:
    return [
        "Correctness: does it match requirements and edge cases?",
        "Safety: does it preserve guardrails and fail-closed behavior?",
        "Simplicity: minimal change surface?",
        "Clarity: readable, maintainable, well-scoped?",
        "Testability: can we verify with deterministic checks?",
        "Regression risk: what could break and how do we detect it?",
    ]
