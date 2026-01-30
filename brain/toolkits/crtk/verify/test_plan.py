from __future__ import annotations

from typing import List


def propose_minimal_verification_plan() -> List[str]:
    return [
        "Rebuild (no cache) if code changes touch runtime.",
        "Import test: ensure new modules import without side effects.",
        "Run unit tests if available.",
        "Run smoke test on key endpoints/flows.",
        "Tail logs to confirm no errors.",
    ]
