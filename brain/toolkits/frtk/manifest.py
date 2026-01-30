from __future__ import annotations

from .types import FRTKMode, ToolSpec

# FRTK v1 Manifest (inventory only; non-executing)
FRTK_MANIFEST_V1 = {
    "name": "FRTK",
    "version": "v1",
    "status": "LAP_SKELETON_INERT",
    "mode_allowed": [FRTKMode.LAP_INERT.value],
    "tools": [
        # INGESTION
        ToolSpec(
            name="import.csv",
            purpose="Import transactions from CSV/OFX/QIF.",
            surface="future",
            inputs=["file", "mapping"],
            outputs=["preview", "errors"],
            guardrails=["dry-run first", "human approval"],
        ),
        ToolSpec(
            name="import.replay",
            purpose="Re-run an import after rules change.",
            surface="future",
            inputs=["import_id"],
            outputs=["diff"],
            guardrails=["no silent overwrite"],
        ),
        # NORMALIZATION
        ToolSpec(
            name="normalize.payee",
            purpose="Canonicalize payee names via rules.",
            surface="internal",
            inputs=["transactions"],
            outputs=["rules"],
            guardrails=["no execution"],
        ),
        ToolSpec(
            name="normalize.category",
            purpose="Canonicalize categories/budgets.",
            surface="internal",
            inputs=["transactions"],
            outputs=["rules"],
            guardrails=["no execution"],
        ),
        # EDITS
        ToolSpec(
            name="edit.batch",
            purpose="Apply batch edits to transactions.",
            surface="future",
            inputs=["changes"],
            outputs=["preview"],
            guardrails=["preview required", "undoable"],
        ),
        # VERIFICATION
        ToolSpec(
            name="verify.integrity",
            purpose="Check balances and ledger consistency.",
            surface="internal",
            inputs=["dataset"],
            outputs=["report"],
            guardrails=["read-only"],
        ),
        # AUDIT
        ToolSpec(
            name="audit.summary",
            purpose="Produce human-legible audit summary.",
            surface="internal",
            inputs=["changes"],
            outputs=["summary"],
            guardrails=["no sensitive leakage"],
        ),
    ],
}
