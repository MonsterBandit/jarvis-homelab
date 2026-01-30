from __future__ import annotations

from typing import List


def propose_observe_analyze_propose_confirm_execute() -> List[str]:
    """
    Canonical coding workflow steps (data only).
    """
    return [
        "Observe: identify required files/logs/docs (no access yet).",
        "Analyze: reason about constraints and likely root causes.",
        "Propose: draft patch plan with minimal changes.",
        "Confirm: get explicit approval for any reads/writes/exec.",
        "Execute: perform bounded actions (outside toolkit).",
        "Verify: run tests/smoke checks and report results.",
    ]
