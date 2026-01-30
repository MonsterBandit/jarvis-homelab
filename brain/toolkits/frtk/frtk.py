from __future__ import annotations

from typing import Any, Dict, List

from .types import FRTKMode, FRTKRequest, FRTKResponse, ToolRequestSpec


def frtk_propose(req: FRTKRequest) -> FRTKResponse:
    mode = req.mode.value if isinstance(req.mode, FRTKMode) else str(req.mode)
    prompt = (req.prompt or "").strip()
    ctx: Dict[str, Any] = req.context or {}

    assumptions: List[str] = [
        "Finance execution is BLOCKED during LAP.",
        "FRTK operates in advisory, non-executing mode.",
    ]
    plan: List[str] = []
    verification: List[str] = []
    risks: List[str] = []
    notes: List[str] = []
    proposed_requests: List[ToolRequestSpec] = []

    if not prompt:
        return FRTKResponse(
            mode=mode,
            intent="empty_request",
            phase="unknown",
            plan=["Request a concrete finance goal or phase."],
            provenance={"frtk": "v1_suite_skeleton"},
        )

    plan.extend(
        [
            "Identify current finance phase (clean, learn, import, normalize, expand, automate).",
            "List data sources involved (UI, CSV, historical sets).",
            "Propose normalization or edit rules (data-only).",
            "Preview changes and require human confirmation.",
        ]
    )

    verification.extend(
        [
            "Preview imports/edits before apply.",
            "Verify balances and transaction counts post-change.",
            "Produce audit summary for review.",
        ]
    )

    risks.extend(
        [
            "Silent data mutation (forbidden).",
            "Rule conflicts or unintended scope.",
        ]
    )

    notes.append(f"Context keys received: {sorted(list(ctx.keys()))}")

    return FRTKResponse(
        mode=mode,
        intent="finance_reasoning_scaffold",
        phase="unspecified",
        assumptions=assumptions,
        plan=plan,
        proposed_requests=proposed_requests,
        verification=verification,
        risks=risks,
        notes=notes,
        provenance={"frtk": "v1_suite_skeleton"},
    )
