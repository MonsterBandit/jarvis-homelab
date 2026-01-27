from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Set

from fastapi import HTTPException

from tools.executor import run_tool
from tools.types import ToolRequest
from tools.registry import is_tool_allowed

from tools.okd.schema import ExpansionEvent, ObservationPlan
from tools.okd.artifacts import (
    okd_append_expansion_event,
    okd_is_plan_approved,
    okd_load_latest_plan,
    okd_store_expansion_log,
    okd_store_update_brief,
)


def _brief_has_observables(brief: Dict[str, Any]) -> bool:
    """Treat any facts/citations as observable output."""
    facts = brief.get("facts") or []
    if isinstance(facts, list) and facts:
        return True
    try:
        for f in facts:
            c = (f or {}).get("citations") or []
            if c:
                return True
    except Exception:
        pass
    return False


async def execute_governed_observation(
    task_id: int,
    user_id: str,
    chat_id: str,
    enabled_tools: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Bundle 3: Governed Observation execution spine.

    PATCH: Enforce FAIL-CLOSED semantics.
    - If intent requires opening sources and opens budget prevents it -> 409 + no Update Brief.
    - If opens=0, suppress *all* observable outputs (facts/citations) -> 409 + no Update Brief.
    """
    plan_obj = okd_load_latest_plan(task_id)
    if not plan_obj:
        raise HTTPException(status_code=400, detail="Missing Observation Plan artifact (okd_observation_plan)")
    if not okd_is_plan_approved(task_id):
        raise HTTPException(status_code=409, detail="Observation Plan not approved")

    # Normalize plan artifact
    if isinstance(plan_obj, ObservationPlan):
        plan: Dict[str, Any] = asdict(plan_obj)
    else:
        plan = dict(plan_obj)

    scope = plan.get("scope_budget") or {}
    cap_queries = int(scope.get("queries", 4) or 4)
    cap_opens = int(scope.get("opens", 6) or 6)
    cap_depth = int(scope.get("crawl_depth", 2) or 2)
    cap_retries = int(scope.get("retries", 2) or 2)

    domains = scope.get("domains") or []
    domain_allowlist = [d for d in domains if isinstance(d, str) and d.strip()] or None

    intent = str(plan.get("intent") or "lookup").strip() or "lookup"
    excerpt = str(plan.get("user_prompt_excerpt") or "governed observation").strip()
    risk_tier = int(plan.get("expected_risk_tier") or 2)

    initial_queries = plan.get("initial_queries") or []
    if not isinstance(initial_queries, list):
        initial_queries = []
    initial_queries = [q for q in initial_queries if isinstance(q, str) and q.strip()]
    if not initial_queries:
        raise HTTPException(status_code=400, detail="Observation Plan missing initial_queries")

    # Bookkeeping
    queries_used = 0
    opens_used = 0
    crawl_depth_used = 0
    retries_used = 0

    expansion_events: list[Dict[str, Any]] = []

    def _log_event(ev_type: str, reason: str, details: Dict[str, Any]) -> None:
        ev = ExpansionEvent(type=ev_type, reason=reason, details=details, count=1)
        d = asdict(ev)
        expansion_events.append(d)
        okd_append_expansion_event(task_id, d)

    # ---- Search (planned) ----
    planned_queries = initial_queries[:cap_queries]
    if len(initial_queries) > len(planned_queries):
        _log_event(
            "scope_expansion",
            "Initial plan contained more queries than allowed; truncating to scope_budget.queries.",
            {"requested": len(initial_queries), "used": len(planned_queries), "cap_queries": cap_queries},
        )
    queries_used = len(planned_queries)

    if enabled_tools is not None and not is_tool_allowed("web.search", enabled_tools):
        raise HTTPException(status_code=403, detail="web.search not enabled")

    search_req = ToolRequest(
        tool_name="web.search",
        args={
            "queries": planned_queries,
            "max_results": 5,
            "domains": domain_allowlist,
            "intent": intent,
            "user_prompt_excerpt": excerpt,
            "risk_tier": risk_tier,
        },
        purpose=f"Governed observation: search sources for intent={intent}",
        task_id=task_id,
        step_id=None,
        user_id=user_id,
        chat_id=chat_id,
    )
    search_res = await run_tool(search_req, enabled_tools=enabled_tools)
    if not search_res.ok:
        _log_event(
            "retry",
            "Search failed (tool error). Failing closed.",
            {"failure_class": getattr(search_res.failure_class, "value", None), "message": search_res.failure_message},
        )
        okd_store_expansion_log(task_id, expansion_events)
        raise HTTPException(status_code=409, detail=f"Fail-closed: search failed: {search_res.failure_message or 'tool error'}")

    primary = search_res.primary if isinstance(search_res.primary, dict) else {}
    results = primary.get("results") or []

    # Conservative: intents that require opening sources
    requires_open = intent in {"verify", "locate-source", "compare"}

    # ---- FAIL-CLOSED (early) ----
    # If intent requires open but opens budget is 0, we must fail-closed (no partial brief).
    if requires_open and cap_opens <= 0:
        _log_event(
            "scope_violation",
            "Intent requires opening sources, but opens budget is 0. Failing closed.",
            {"intent": intent, "opens_cap": cap_opens},
        )
        okd_store_expansion_log(task_id, expansion_events)
        raise HTTPException(status_code=409, detail="Fail-closed: opens budget is 0 but intent requires opening sources")

    opened_pages: list[str] = []

    # Open first result if allowed
    if requires_open:
        if not results:
            _log_event("uncertainty", "No search results to open. Failing closed.", {"results_count": 0})
            okd_store_expansion_log(task_id, expansion_events)
            raise HTTPException(status_code=409, detail="Fail-closed: no search results to open")

        top = results[0] if isinstance(results, list) else None
        url = (top or {}).get("url") if isinstance(top, dict) else None
        if not isinstance(url, str) or not url.strip():
            _log_event("uncertainty", "Top search result has no openable URL. Failing closed.", {"top": top})
            okd_store_expansion_log(task_id, expansion_events)
            raise HTTPException(status_code=409, detail="Fail-closed: top result has no openable URL")

        if enabled_tools is not None and not is_tool_allowed("web.open", enabled_tools):
            raise HTTPException(status_code=403, detail="web.open not enabled")

        if opens_used + 1 > cap_opens:
            _log_event(
                "scope_violation",
                "Open would exceed opens budget. Failing closed.",
                {"opens_used": opens_used, "opens_cap": cap_opens},
            )
            okd_store_expansion_log(task_id, expansion_events)
            raise HTTPException(status_code=409, detail="Fail-closed: opens budget exceeded")

        opens_used += 1
        open_req = ToolRequest(
            tool_name="web.open",
            args={
                "url": url.strip(),
                "intent": intent,
                "user_prompt_excerpt": excerpt,
                "risk_tier": risk_tier,
            },
            purpose="Governed observation: open top source",
            task_id=task_id,
            step_id=None,
            user_id=user_id,
            chat_id=chat_id,
        )
        open_res = await run_tool(open_req, enabled_tools=enabled_tools)
        if not open_res.ok:
            _log_event(
                "retry",
                "Open failed (tool error). Failing closed.",
                {"failure_class": getattr(open_res.failure_class, "value", None), "message": open_res.failure_message},
            )
            okd_store_expansion_log(task_id, expansion_events)
            raise HTTPException(status_code=409, detail=f"Fail-closed: open failed: {open_res.failure_message or 'tool error'}")

        open_primary = open_res.primary if isinstance(open_res.primary, dict) else {}
        opened_pages.append(str(open_primary.get("url") or url))

    # ---- Build brief (success path only) ----
    facts: list[Dict[str, Any]] = []
    uncertainties: list[Dict[str, Any]] = []

    if opened_pages:
        facts.append(
            {
                "statement": f"Opened {len(opened_pages)} source page(s) for intent '{intent}'.",
                "citations": opened_pages[:5],
            }
        )
    else:
        uncertainties.append(
            {
                "statement": "No pages were opened.",
                "note": "Consider increasing opens budget or providing a domain allowlist.",
            }
        )

    brief: Dict[str, Any] = {
        "facts": facts,
        "inferences": [],
        "conflicts": [],
        "uncertainties": uncertainties,
        "scope_used": {
            "queries_used": queries_used,
            "opens_used": opens_used,
            "crawl_depth_used": crawl_depth_used,
            "retries_used": retries_used,
            "domains_allowlist": domain_allowlist,
        },
        "login_used": False,
        "opened_pages": opened_pages,
    }

    # ---- FAIL-CLOSED (late) ----
    # If opens=0, suppress all observables (facts/citations). This is the key patch for your failed test.
    if cap_opens <= 0 and _brief_has_observables(brief):
        _log_event(
            "scope_violation",
            "Observable output detected under opens=0 budget. Failing closed (no partial results).",
            {"opens_cap": cap_opens, "intent": intent},
        )
        okd_store_expansion_log(task_id, expansion_events)
        raise HTTPException(status_code=409, detail="Fail-closed: observable output not permitted under opens=0")

    okd_store_expansion_log(task_id, expansion_events)
    okd_store_update_brief(task_id, brief)

    return {"ok": True, "task_id": task_id, "update_brief_artifact": "okd_update_brief"}
