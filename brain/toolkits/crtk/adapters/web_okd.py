from __future__ import annotations

from ..types import ToolRequestSpec


def request_web_search(q: str, reason: str = "", recency: int | None = None, domains: list[str] | None = None) -> ToolRequestSpec:
    args = {"q": q}
    if recency is not None:
        args["recency"] = int(recency)
    if domains is not None:
        args["domains"] = list(domains)
    return ToolRequestSpec(
        tool_name="web.search_query",
        args=args,
        reason=reason or "Look up official docs / primary sources under OKD governance.",
        surface="toolbelt_web_okd",
    )


def request_web_open(ref_id: str, reason: str = "", lineno: int | None = None) -> ToolRequestSpec:
    args = {"ref_id": ref_id}
    if lineno is not None:
        args["lineno"] = int(lineno)
    return ToolRequestSpec(
        tool_name="web.open",
        args=args,
        reason=reason or "Open a sourced page for extraction under OKD governance.",
        surface="toolbelt_web_okd",
    )
