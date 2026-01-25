from __future__ import annotations

import base64
import html as _html
import re
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urljoin, urlparse

import httpx

from tools.types import ToolFailureClass, ToolProvenance, ToolRequest, ToolResult


# ---------------------------------------------------------
# Alice Browsing Guardrails v1 — mechanical enforcement
# ---------------------------------------------------------

_ALLOWED_INTENTS = {"lookup", "verify", "compare", "explain", "locate-source"}
_ALLOWED_RISK_TIERS = {2, 3}

MAX_SEARCH_QUERIES = 4
MAX_RESULTS_PER_QUERY = 10
MAX_OPENS_PER_CALL = 3
MAX_FIND_HITS = 20

_ALLOWED_SCHEMES = {"http", "https"}
_ALLOWED_CONTENT_TYPES = {"text/html", "text/plain", "application/pdf"}

_BLOCKED_EXTS = {
    ".zip", ".tar", ".gz", ".tgz", ".7z",
    ".exe", ".msi", ".dmg", ".pkg",
    ".apk", ".iso",
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _domain(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _fail(req: ToolRequest, cls: ToolFailureClass, msg: str, started_at: str, t0: float) -> ToolResult:
    return ToolResult(
        ok=False,
        tool_name=req.tool_name,
        failure_class=cls,
        failure_message=msg,
        started_at=started_at,
        ended_at=_iso_now(),
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


def _require_common(req: ToolRequest, started_at: str, t0: float) -> Optional[ToolResult]:
    args = req.args or {}

    intent = str(args.get("intent") or "").strip()
    excerpt = str(args.get("user_prompt_excerpt") or "").strip()
    risk_tier = args.get("risk_tier")

    if intent not in _ALLOWED_INTENTS:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "Missing/invalid intent", started_at, t0)
    if not excerpt:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "Missing user_prompt_excerpt", started_at, t0)

    try:
        rt = int(risk_tier)
    except Exception:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "Missing/invalid risk_tier", started_at, t0)
    if rt not in _ALLOWED_RISK_TIERS:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "risk_tier must be 2 or 3 for browsing tools", started_at, t0)

    return None


def _normalize_url(url: str) -> Optional[str]:
    u = (url or "").strip()
    if not u:
        return None
    try:
        p = urlparse(u)
    except Exception:
        return None
    if p.scheme not in _ALLOWED_SCHEMES:
        return None
    lower_path = (p.path or "").lower()
    for ext in _BLOCKED_EXTS:
        if lower_path.endswith(ext):
            return None
    return u


def _strip_html_to_text(html_text: str) -> str:
    s = html_text or ""
    s = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", s)
    s = re.sub(r"(?is)<br\s*/?>", "\n", s)
    s = re.sub(r"(?is)</p\s*>", "\n", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    s = _html.unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_title(html_text: str) -> Optional[str]:
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", html_text or "")
    if not m:
        return None
    t = re.sub(r"\s+", " ", _html.unescape(m.group(1))).strip()
    return t or None


def _extract_links(base_url: str, html_text: str, max_links: int = 80) -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    if not html_text:
        return links

    for m in re.finditer(r'(?is)<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html_text):
        href = (m.group(1) or "").strip()
        if not href or href.startswith("#"):
            continue

        abs_url = urljoin(base_url, href)
        abs_url_n = _normalize_url(abs_url)
        if not abs_url_n:
            continue

        inner = re.sub(r"(?is)<[^>]+>", " ", m.group(2) or "")
        inner = re.sub(r"\s+", " ", _html.unescape(inner)).strip()
        if not inner:
            inner = abs_url_n

        links.append({"text": inner[:180], "url": abs_url_n})
        if len(links) >= max_links:
            break

    out: List[Dict[str, Any]] = []
    for i, l in enumerate(links, start=1):
        out.append({"id": i, "text": l["text"], "url": l["url"]})
    return out


@dataclass
class _OpenedPage:
    ref_id: str
    url: str
    retrieved_at: str
    content_type: str
    title: Optional[str]
    text: Optional[str]
    raw_pdf_b64: Optional[str]  # in-memory only
    links: List[Dict[str, Any]]


_OPENED: Dict[Tuple[str, str, str], _OpenedPage] = {}


def _store_key(req: ToolRequest, ref_id: str) -> Tuple[str, str, str]:
    return (req.user_id or "unknown", req.chat_id or "unknown", ref_id)


def _store_put(req: ToolRequest, page: _OpenedPage) -> None:
    _OPENED[_store_key(req, page.ref_id)] = page


def _store_get(req: ToolRequest, ref_id: str) -> Optional[_OpenedPage]:
    return _OPENED.get(_store_key(req, ref_id))


async def run_web_tool(req: ToolRequest) -> ToolResult:
    t0 = time.monotonic()
    started_at = _iso_now()

    common_err = _require_common(req, started_at, t0)
    if common_err:
        return common_err

    try:
        if req.tool_name == "web.search":
            return await _web_search(req, started_at, t0)
        if req.tool_name == "web.open":
            return await _web_open(req, started_at, t0)
        if req.tool_name == "web.find":
            return _web_find(req, started_at, t0)
        if req.tool_name == "web.click":
            return await _web_click(req, started_at, t0)
        if req.tool_name == "web.screenshot_pdf":
            return _web_screenshot_pdf(req, started_at, t0)

        return _fail(req, ToolFailureClass.TOOL_NOT_ALLOWED, f"Unsupported web tool: {req.tool_name}", started_at, t0)

    except httpx.TimeoutException:
        return _fail(req, ToolFailureClass.TOOL_TIMEOUT, "Tool timeout", started_at, t0)
    except httpx.HTTPError as e:
        return _fail(req, ToolFailureClass.TOOL_UPSTREAM_ERROR, f"Upstream HTTP error: {e}", started_at, t0)
    except Exception as e:
        return _fail(req, ToolFailureClass.TOOL_INTERNAL_ERROR, f"Internal error: {e}", started_at, t0)


async def _web_search(req: ToolRequest, started_at: str, t0: float) -> ToolResult:
    args = req.args or {}
    queries = args.get("queries")

    if not isinstance(queries, list) or not queries or not all(isinstance(q, str) and q.strip() for q in queries):
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "web.search requires queries: list[str]", started_at, t0)
    if len(queries) > MAX_SEARCH_QUERIES:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, f"Too many queries (max {MAX_SEARCH_QUERIES})", started_at, t0)

    max_results = args.get("max_results", MAX_RESULTS_PER_QUERY)
    try:
        max_results = int(max_results)
    except Exception:
        max_results = MAX_RESULTS_PER_QUERY
    max_results = max(1, min(MAX_RESULTS_PER_QUERY, max_results))

    domains = args.get("domains")
    domain_allow: Optional[set[str]] = None
    if domains is not None:
        if not isinstance(domains, list) or not all(isinstance(d, str) for d in domains):
            return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "domains must be list[str]", started_at, t0)
        domain_allow = {d.strip().lower() for d in domains if d and d.strip()}

    results: List[Dict[str, Any]] = []
    prov_sources: List[str] = []
    seen: set[str] = set()

    def _maybe_add(url: str, title: Optional[str]) -> None:
        norm = _normalize_url(url)
        if not norm:
            return
        d = _domain(norm)
        if domain_allow is not None and d not in domain_allow:
            return
        if norm in seen:
            return
        seen.add(norm)
        results.append({"title": (title or norm)[:240], "url": norm, "domain": d, "snippet": None})

    def _extract_uddg(u: str) -> Optional[str]:
        try:
            p = urlparse(u)
            qs = dict(urllib.parse.parse_qsl(p.query or "", keep_blank_values=True))
            v = qs.get("uddg")
            if not v:
                return None
            return urllib.parse.unquote(v)
        except Exception:
            return None

    timeout = httpx.Timeout(15.0, connect=10.0)
    # NOTE: Some upstreams vary markup based on UA; use a browser-like UA to reduce empty SERPs.
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        for q in queries:
            q_clean = q.strip()
            url = "https://duckduckgo.com/html/?" + urlencode({"q": q_clean})
            prov_sources.append(url)

            r = await client.get(url)
            if r.status_code >= 400:
                return _fail(req, ToolFailureClass.TOOL_UPSTREAM_ERROR, f"Search upstream returned {r.status_code}", started_at, t0)

            html_text = r.text or ""

            # Pass 1: historical DDG markup (fast-path).
            for m in re.finditer(r'(?is)<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html_text):
                href = _html.unescape(m.group(1) or "").strip()
                title = re.sub(r"(?is)<[^>]+>", " ", m.group(2) or "")
                title = re.sub(r"\s+", " ", _html.unescape(title)).strip()

                # DDG sometimes returns redirect links (uddg=). Decode when present.
                uddg = _extract_uddg(href)
                if uddg:
                    href = uddg

                _maybe_add(href, title or None)
                if len(results) >= max_results:
                    break

            # Pass 2 (fallback): generic anchor extraction + DDG redirect decode.
            # This makes web.search resilient to DDG class-name churn.
            if len(results) < max_results:
                base = "https://duckduckgo.com"
                for m in re.finditer(r'(?is)<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html_text):
                    href = _html.unescape(m.group(1) or "").strip()
                    if not href or href.startswith("#"):
                        continue

                    # Decode DDG redirect form: /l/?uddg=<encoded>
                    if href.startswith("/l/") or href.startswith("/?") or href.startswith("/html/"):
                        abs_url = urljoin(base, href)
                        uddg = _extract_uddg(abs_url)
                        if uddg:
                            href = uddg
                        else:
                            # If it's still on DDG, it's not a destination result.
                            if _domain(abs_url).endswith("duckduckgo.com"):
                                continue
                            href = abs_url

                    # Ignore obvious DDG internal URLs.
                    if _domain(href).endswith("duckduckgo.com"):
                        continue

                    title = re.sub(r"(?is)<[^>]+>", " ", m.group(2) or "")
                    title = re.sub(r"\s+", " ", _html.unescape(title)).strip()

                    _maybe_add(href, title or None)
                    if len(results) >= max_results:
                        break

            if len(results) >= max_results:
                break

    results = results[:max_results]

    prov = ToolProvenance(
        sources=prov_sources[:10],
        retrieved_at=_iso_now(),
        notes="web.search (duckduckgo html; hardened parser)" + (" with domain pinning" if domain_allow else ""),
    )

    return ToolResult(
        ok=True,
        tool_name=req.tool_name,
        primary={"results": results, "count": len(results)},
        provenance=prov,
        started_at=started_at,
        ended_at=_iso_now(),
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


async def _web_open(req: ToolRequest, started_at: str, t0: float) -> ToolResult:
    args = req.args or {}
    url = args.get("url") or args.get("ref_url")
    ref_id = args.get("ref_id")

    if ref_id is not None:
        if not isinstance(ref_id, str) or not ref_id.strip():
            return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "ref_id must be a non-empty string", started_at, t0)
        cached = _store_get(req, ref_id.strip())
        if not cached:
            return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "ref_id not found (open the page first)", started_at, t0)

        prov = ToolProvenance(
            sources=[cached.url],
            retrieved_at=cached.retrieved_at,
            notes="web.open (cached)",
        )
        excerpt = (cached.text or "")[:800] if cached.text else None
        return ToolResult(
            ok=True,
            tool_name=req.tool_name,
            primary={
                "ref_id": cached.ref_id,
                "url": cached.url,
                "title": cached.title,
                "content_type": cached.content_type,
                "text_excerpt": excerpt,
                "links": cached.links[:20],
            },
            provenance=prov,
            started_at=started_at,
            ended_at=_iso_now(),
            latency_ms=int((time.monotonic() - t0) * 1000),
        )

    if not isinstance(url, str) or not url.strip():
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "web.open requires url/ref_url or ref_id", started_at, t0)

    norm = _normalize_url(url.strip())
    if not norm:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "URL is invalid or disallowed", started_at, t0)

    timeout = httpx.Timeout(20.0, connect=10.0)
    headers = {"User-Agent": "ISAC/1.0", "Accept": "*/*"}

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        r = await client.get(norm)
        if r.status_code >= 400:
            return _fail(req, ToolFailureClass.TOOL_UPSTREAM_ERROR, f"Open upstream returned {r.status_code}", started_at, t0)

        ct = (r.headers.get("content-type") or "").split(";")[0].strip().lower()
        if ct not in _ALLOWED_CONTENT_TYPES:
            return _fail(req, ToolFailureClass.TOOL_NOT_ALLOWED, f"Content type not allowed: {ct or 'unknown'}", started_at, t0)

        retrieved_at = _iso_now()
        ref = secrets.token_urlsafe(10)

        title: Optional[str] = None
        text: Optional[str] = None
        raw_pdf_b64: Optional[str] = None
        links: List[Dict[str, Any]] = []

        if ct == "application/pdf":
            pdf_bytes = r.content
            raw_pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                if doc.page_count > 0:
                    p0 = doc.load_page(0)
                    txt = p0.get_text("text") or ""
                    text = re.sub(r"\s+\n", "\n", txt).strip()
                doc.close()
            except Exception:
                text = None
                title = None

        elif ct in {"text/html", "text/plain"}:
            body = r.text or ""
            if ct == "text/html":
                title = _extract_title(body)
                links = _extract_links(norm, body, max_links=80)
                text = _strip_html_to_text(body)
            else:
                text = (body or "").strip()

        page = _OpenedPage(
            ref_id=ref,
            url=norm,
            retrieved_at=retrieved_at,
            content_type=ct,
            title=title,
            text=text,
            raw_pdf_b64=raw_pdf_b64,
            links=links,
        )
        _store_put(req, page)

        prov = ToolProvenance(
            sources=[norm],
            retrieved_at=retrieved_at,
            notes=f"web.open ({ct})",
        )
        excerpt = (text or "")[:800] if text else None

        return ToolResult(
            ok=True,
            tool_name=req.tool_name,
            primary={
                "ref_id": ref,
                "url": norm,
                "title": title,
                "content_type": ct,
                "text_excerpt": excerpt,
                "links": links[:20],
            },
            provenance=prov,
            started_at=started_at,
            ended_at=_iso_now(),
            latency_ms=int((time.monotonic() - t0) * 1000),
        )


def _web_find(req: ToolRequest, started_at: str, t0: float) -> ToolResult:
    args = req.args or {}
    ref_id = args.get("ref_id")
    pattern = args.get("pattern")

    if not isinstance(ref_id, str) or not ref_id.strip():
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "web.find requires ref_id", started_at, t0)
    if not isinstance(pattern, str) or not pattern.strip():
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "web.find requires pattern", started_at, t0)

    page = _store_get(req, ref_id.strip())
    if not page:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "ref_id not found (open the page first)", started_at, t0)
    if not page.text:
        return _fail(req, ToolFailureClass.TOOL_PARSE_ERROR, "No searchable text available for this ref_id", started_at, t0)

    max_hits = args.get("max_hits", MAX_FIND_HITS)
    try:
        max_hits = int(max_hits)
    except Exception:
        max_hits = MAX_FIND_HITS
    max_hits = max(1, min(MAX_FIND_HITS, max_hits))

    hits: List[Dict[str, Any]] = []
    text = page.text
    pat = pattern.strip()

    try:
        rx = re.compile(pat, re.IGNORECASE)
        for m in rx.finditer(text):
            if len(hits) >= max_hits:
                break
            s = max(m.start() - 80, 0)
            e = min(m.end() + 80, len(text))
            hits.append({"start": m.start(), "end": m.end(), "excerpt": text[s:e]})
    except re.error:
        idx = 0
        low = text.lower()
        needle = pat.lower()
        while True:
            j = low.find(needle, idx)
            if j == -1 or len(hits) >= max_hits:
                break
            s = max(j - 80, 0)
            e = min(j + len(needle) + 80, len(text))
            hits.append({"start": j, "end": j + len(needle), "excerpt": text[s:e]})
            idx = j + len(needle)

    prov = ToolProvenance(
        sources=[page.url],
        retrieved_at=page.retrieved_at,
        notes=f"web.find hits={len(hits)}",
    )

    return ToolResult(
        ok=True,
        tool_name=req.tool_name,
        primary={"ref_id": page.ref_id, "pattern": pat, "hits": hits, "hit_count": len(hits)},
        provenance=prov,
        started_at=started_at,
        ended_at=_iso_now(),
        latency_ms=int((time.monotonic() - t0) * 1000),
    )


async def _web_click(req: ToolRequest, started_at: str, t0: float) -> ToolResult:
    args = req.args or {}
    ref_id = args.get("ref_id")
    link_id = args.get("link_id")

    if not isinstance(ref_id, str) or not ref_id.strip():
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "web.click requires ref_id", started_at, t0)
    try:
        link_id_int = int(link_id)
    except Exception:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "web.click requires link_id (int)", started_at, t0)

    page = _store_get(req, ref_id.strip())
    if not page:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "ref_id not found (open the page first)", started_at, t0)

    target = next((l for l in page.links if int(l.get("id", -1)) == link_id_int), None)
    if not target:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "link_id not found on this page", started_at, t0)

    chain_budget = args.get("chain_budget", MAX_OPENS_PER_CALL)
    try:
        chain_budget = int(chain_budget)
    except Exception:
        chain_budget = MAX_OPENS_PER_CALL
    if chain_budget < 1:
        return _fail(req, ToolFailureClass.TOOL_NOT_ALLOWED, "web.click chain budget exhausted", started_at, t0)

    open_req = ToolRequest(
        tool_name="web.open",
        args={
            "url": str(target["url"]),
            "intent": args.get("intent"),
            "user_prompt_excerpt": args.get("user_prompt_excerpt"),
            "risk_tier": args.get("risk_tier"),
        },
        purpose=req.purpose,
        task_id=req.task_id,
        step_id=req.step_id,
        user_id=req.user_id,
        chat_id=req.chat_id,
    )
    return await _web_open(open_req, started_at, t0)


def _web_screenshot_pdf(req: ToolRequest, started_at: str, t0: float) -> ToolResult:
    args = req.args or {}
    ref_id = args.get("ref_id")
    pageno = args.get("pageno")

    if not isinstance(ref_id, str) or not ref_id.strip():
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "web.screenshot_pdf requires ref_id", started_at, t0)
    try:
        p = int(pageno)
    except Exception:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "web.screenshot_pdf requires pageno (int)", started_at, t0)
    if p < 0:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "pageno must be >= 0", started_at, t0)

    page = _store_get(req, ref_id.strip())
    if not page:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "ref_id not found (open the PDF first)", started_at, t0)
    if page.content_type != "application/pdf" or not page.raw_pdf_b64:
        return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, "ref_id is not a PDF", started_at, t0)

    try:
        import fitz  # PyMuPDF
    except Exception:
        return _fail(req, ToolFailureClass.TOOL_INTERNAL_ERROR, "PyMuPDF (fitz) not installed; cannot screenshot PDFs", started_at, t0)

    pdf_bytes = base64.b64decode(page.raw_pdf_b64.encode("utf-8"))
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if p >= doc.page_count:
            doc.close()
            return _fail(req, ToolFailureClass.TOOL_BAD_INPUT, f"pageno out of range (0..{doc.page_count-1})", started_at, t0)

        pg = doc.load_page(p)
        pix = pg.get_pixmap(dpi=150)
        png_bytes = pix.tobytes("png")
        doc.close()
    except Exception as e:
        return _fail(req, ToolFailureClass.TOOL_PARSE_ERROR, f"PDF render failed: {e}", started_at, t0)

    img_b64 = base64.b64encode(png_bytes).decode("utf-8")
    prov = ToolProvenance(
        sources=[page.url],
        retrieved_at=page.retrieved_at,
        notes=f"web.screenshot_pdf pageno={p}",
    )

    return ToolResult(
        ok=True,
        tool_name=req.tool_name,
        primary={"ref_id": page.ref_id, "pageno": p, "image_b64": img_b64, "mime": "image/png"},
        provenance=prov,
        started_at=started_at,
        ended_at=_iso_now(),
        latency_ms=int((time.monotonic() - t0) * 1000),
    )