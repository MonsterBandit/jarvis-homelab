from __future__ import annotations

from .types import CRTKMode, ToolSpec

# CRTK v1 Manifest (inventory only; non-executing)
# "All tools" here means the full specialist coding toolbox,
# including callable tool categories, expressed as declarative specs.

CRTK_MANIFEST_V1 = {
    "name": "CRTK",
    "version": "v1",
    "status": "LAP_SKELETON_INERT",
    "mode_allowed": [CRTKMode.LAP_INERT.value],
    "tools": [
        # -----------------------------
        # OBSERVE / READ (Local)
        # -----------------------------
        ToolSpec(
            name="local.read_file",
            purpose="Read a full text file (allowlisted, capped).",
            surface="toolbelt_local_read",
            inputs=["path"],
            outputs=["content", "sha256", "bytes"],
            guardrails=["allowlist enforced", "size capped", "utf-8 only", "read-only"],
        ),
        ToolSpec(
            name="local.read_snippet",
            purpose="Read a line-ranged snippet (allowlisted, capped).",
            surface="toolbelt_local_read",
            inputs=["path", "start_line", "end_line"],
            outputs=["lines", "sha256", "snippet_bytes", "total_lines"],
            guardrails=["allowlist enforced", "line cap", "byte cap", "read-only"],
        ),
        ToolSpec(
            name="local.list_tree",
            purpose="List directory tree for repo mapping (if/when added as a LOCAL_READ tool).",
            surface="future",
            inputs=["path", "max_depth"],
            outputs=["paths"],
            guardrails=["allowlist", "depth cap", "read-only"],
        ),
        ToolSpec(
            name="local.search_text",
            purpose="Search across allowlisted paths (grep-like) (if/when added).",
            surface="future",
            inputs=["query", "path_globs"],
            outputs=["matches"],
            guardrails=["allowlist", "match cap", "read-only"],
        ),

        # -----------------------------
        # OBSERVE / READ (Web under OKD)
        # -----------------------------
        ToolSpec(
            name="web.search_query",
            purpose="Public web search (governed, disclosed, cited).",
            surface="toolbelt_web_okd",
            inputs=["q", "recency", "domains"],
            outputs=["sources"],
            guardrails=["OKD plan/preview/execute", "scope budgets", "citations required"],
        ),
        ToolSpec(
            name="web.open",
            purpose="Open a retrieved page for extraction (governed).",
            surface="toolbelt_web_okd",
            inputs=["ref_id", "lineno"],
            outputs=["content excerpt"],
            guardrails=["OKD budgets", "disclose expansions/retries"],
        ),

        # -----------------------------
        # ANALYZE (Static analysis - callable later via runner/utility)
        # -----------------------------
        ToolSpec(
            name="lint",
            purpose="Run linter suite (ruff/eslint/etc.) to catch style/errors (runner-executed).",
            surface="runner",
            inputs=["command", "cwd"],
            outputs=["stdout", "stderr", "exit_code"],
            guardrails=["allowlisted commands only", "no secret output"],
        ),
        ToolSpec(
            name="typecheck",
            purpose="Run type checker (mypy/pyright/tsc) (runner-executed).",
            surface="runner",
            inputs=["command", "cwd"],
            outputs=["stdout", "stderr", "exit_code"],
            guardrails=["allowlisted commands only"],
        ),
        ToolSpec(
            name="format",
            purpose="Run formatter (black/prettier) (runner-executed).",
            surface="runner",
            inputs=["command", "cwd"],
            outputs=["stdout", "stderr", "exit_code"],
            guardrails=["allowlisted commands only"],
        ),

        # -----------------------------
        # CHANGE (Patch planning + governed writes)
        # -----------------------------
        ToolSpec(
            name="patch.plan",
            purpose="Produce a patch plan (files, blocks, rationale, verification) (internal).",
            surface="internal",
            inputs=["goal", "constraints", "observations"],
            outputs=["patch_plan"],
            guardrails=["no execution", "data only"],
        ),
        ToolSpec(
            name="write.apply_patch",
            purpose="Apply a bounded patch/write (governed write surface; not available in LAP).",
            surface="future",
            inputs=["diff_or_rewrite"],
            outputs=["verification_required"],
            guardrails=["explicit confirm", "dry-run default", "auto-verify", "allowlist"],
        ),

        # -----------------------------
        # VERIFY (Tests, builds, smoke checks)
        # -----------------------------
        ToolSpec(
            name="test.unit",
            purpose="Run unit tests (pytest/jest/etc.) (runner-executed).",
            surface="runner",
            inputs=["command", "cwd"],
            outputs=["stdout", "stderr", "exit_code"],
            guardrails=["allowlisted commands only"],
        ),
        ToolSpec(
            name="test.smoke",
            purpose="Run smoke checks (health endpoints, minimal flows) (runner-executed).",
            surface="runner",
            inputs=["command", "cwd"],
            outputs=["stdout", "stderr", "exit_code"],
            guardrails=["allowlisted commands only"],
        ),
        ToolSpec(
            name="build.container",
            purpose="Build containers / compile artifacts (runner-executed).",
            surface="runner",
            inputs=["command", "cwd"],
            outputs=["stdout", "stderr", "exit_code"],
            guardrails=["allowlisted commands only"],
        ),

        # -----------------------------
        # DEBUG (Logs, repro, diagnostics)
        # -----------------------------
        ToolSpec(
            name="logs.tail",
            purpose="Tail logs for diagnosis (runner or future tool).",
            surface="future",
            inputs=["service", "lines"],
            outputs=["log_lines"],
            guardrails=["read-only", "redact secrets"],
        ),
        ToolSpec(
            name="repro.plan",
            purpose="Generate minimal reproduction steps (internal).",
            surface="internal",
            inputs=["symptom", "context"],
            outputs=["steps"],
            guardrails=["no execution"],
        ),

        # -----------------------------
        # DOCS / COLLAB (Artifacts)
        # -----------------------------
        ToolSpec(
            name="change.summary",
            purpose="Generate change summary + rationale + verification notes (internal).",
            surface="internal",
            inputs=["patch_plan", "results"],
            outputs=["summary"],
            guardrails=["no sensitive data"],
        ),
        ToolSpec(
            name="incident.note",
            purpose="Produce incident-style notes for failures (internal).",
            surface="internal",
            inputs=["failure", "context"],
            outputs=["note"],
            guardrails=["no secrets", "human-legible"],
        ),
    ],
}
