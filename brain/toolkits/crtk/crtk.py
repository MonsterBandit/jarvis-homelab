from __future__ import annotations

from typing import Any, Dict, List

from .types import CRTKMode, CRTKRequest, CRTKResponse, ToolRequestSpec


def crtk_propose(req: CRTKRequest) -> CRTKResponse:
    """
    CRTK proposal generator (inert).

    LOCKED (LAP):
    - No tool calls
    - No file reads
    - No writes
    - No network
    - No memory/identity learning

    CRTK MAY emit *ToolRequestSpec* entries describing what it would request later.
    """
    mode = req.mode.value if isinstance(req.mode, CRTKMode) else str(req.mode)
    prompt = (req.prompt or "").strip()
    ctx: Dict[str, Any] = req.context or {}

    assumptions: List[str] = []
    unknowns: List[str] = []
    plan: List[str] = []
    verification: List[str] = []
    risks: List[str] = []
    notes: List[str] = []
    proposed_requests: List[ToolRequestSpec] = []

    if not prompt:
        return CRTKResponse(
            mode=mode,
            intent="empty_request",
            unknowns=["No prompt provided."],
            plan=["Request a concrete coding goal and any relevant constraints."],
            provenance={"crtk": "v1_suite_skeleton"},
        )

    assumptions.append("CRTK is operating in LAP skeleton mode (no execution, no I/O).")

    # Minimal deterministic scaffold: capture goal and propose the standard loop.
    plan.extend(
        [
            "Restate the coding goal as a single sentence.",
            "Identify required observations (files, endpoints, configs) without accessing them.",
            "Produce a patch plan (what to change, where, why) as a proposal.",
            "Produce a verification plan (commands/checks) as a proposal.",
            "Request explicit approval before any observation or execution.",
        ]
    )

    # If user provided obvious hints, suggest tool request specs (non-executing).
    if "file" in prompt.lower() or "index.html" in prompt.lower():
        proposed_requests.append(
            ToolRequestSpec(
                tool_name="local.read_file",
                args={"path": "/opt/jarvis/data/index.html"},
                reason="User mentioned file/UI; would read target file under allowlist for accurate changes.",
                surface="toolbelt_local_read",
            )
        )

    verification.extend(
        [
            "Import test: ensure modules import with no side effects.",
            "Smoke test: run minimal health checks after rebuild (if applicable).",
        ]
    )

    risks.extend(
        [
            "Accidental scope creep into execution or I/O (forbidden in LAP).",
            "Ambiguous target files or requirements causing incorrect patches.",
        ]
    )

    notes.append(f"Context keys received: {sorted(list(ctx.keys()))}")

    return CRTKResponse(
        mode=mode,
        intent="coding_reasoning_scaffold",
        assumptions=assumptions,
        unknowns=unknowns,
        plan=plan,
        proposed_requests=proposed_requests,
        verification=verification,
        risks=risks,
        notes=notes,
        provenance={"crtk": "v1_suite_skeleton"},
    )
