from __future__ import annotations

from typing import Dict


def format_change_rationale(summary: str, risks: list[str], verification: list[str]) -> Dict[str, object]:
    return {
        "summary": summary,
        "risks": list(risks),
        "verification": list(verification),
        "notes": ["CRTK produces advisory artifacts only; execution is separate."],
    }
