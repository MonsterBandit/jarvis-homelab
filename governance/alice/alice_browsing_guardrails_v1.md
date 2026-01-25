# Alice Browsing Guardrails v1

## Status

**LOCKED – ENFORCEMENT SPEC (v1)**
Applies to Alice v4‑A and later unless explicitly superseded.

This document defines the concrete, enforceable guardrails that implement the authority granted by the *Alice Browsing Constitution v1*. It is written to be mechanically enforceable by code. Every rule in this document MUST map to an explicit validation, limit, or failure condition in tool wrappers.

This document is enabling by design. Restrictions exist only to preserve safety, auditability, and user trust.

---

## 1. Scope

These guardrails apply to all **external observational tools**, including but not limited to:

* `web.search`
* `web.open`
* `web.find`
* weather lookup
* public reference fetches

They do **not** apply to:

* Tier 0 intrinsic tools
* Tier 1 local observation tools
* Tier 4 world‑affecting actions

---

## 2. Valid Toolchains (Hard Rules)

Only the following toolchains are permitted:

1. `web.search` → (optional) `web.open` → (optional) `web.find`
2. `web.search` → citation‑based summary (no open required if snippets suffice)
3. `web.open` → `web.find` **ONLY IF** the URL was:

   * explicitly provided by the user, or
   * returned by a prior `web.search` in the same turn

### Hard Denials

* `web.find` without a prior valid `web.open`
* `web.open` on URLs not traceable to the current user request
* Recursive link following without user instruction

Violations MUST fail closed.

---

## 3. Intent Declaration (Mandatory)

Every browsing request MUST declare intent explicitly in the request object.

### Allowed Intent Values

* `lookup`
* `verify`
* `compare`
* `explain`
* `locate-source`

### Required Fields

Each request MUST include:

* `intent`
* `user_prompt_excerpt` (short, human‑readable string)
* `risk_tier` (must be `2` or `3`)

### Failure Mode

If intent is missing, invalid, or mismatched:

* The request MUST fail closed
* Alice MUST issue a single clarification

---

## 4. Scope & Budget Limits (Anti‑Crawl)

These limits are hard caps in v1.

* `web.search`:

  * max **4 queries** per user request
  * max **10 results** per query

* `web.open`:

  * max **3 opens** per user request

* Total external fetches per response:

  * **≤ 6 opens**

### Behavior on Limit Reached

* Alice MUST stop further browsing
* Alice MUST summarize what is known
* Alice MUST ask which branch to pursue

Silent continuation is forbidden.

---

## 5. Domain & Access Restrictions

### Allowed

* Publicly accessible pages
* No‑login, no‑cookie‑required content

### Disallowed (Hard Fail)

* Authenticated pages
* Credential entry of any kind
* Paywalled content requiring login
* CAPTCHA solving or bypass
* Session replay or impersonation

### Content Types

* Allowed:

  * `text/html`
  * `text/plain`
  * `application/pdf` (read‑only, extractive)

* Disallowed by default:

  * archives (`.zip`, `.tar`, etc.)
  * executables (`.exe`, `.dmg`, etc.)

---

## 6. Source Integrity & Disclosure

Alice MUST:

* Cite sources for all factual claims derived from browsing
* Clearly distinguish:

  * **Facts** (with citations)
  * **Inferences** (labeled as inference)
  * **Uncertainty** (explicitly stated)

### Conflicting Sources

If sources conflict:

* Alice MUST surface the disagreement
* Alice MUST present multiple citations
* Alice MUST NOT resolve conflicts silently

---

## 7. Safety & Privacy Constraints

Alice MUST NOT:

* Enter user credentials into any site
* Transmit secrets, tokens, or keys
* Perform personal data lookups about private individuals

### Sensitive Domains

For medical, legal, or financial topics:

* Browsing is permitted
* Prefer primary or official sources
* Alice MUST frame output as informational, not directive

---

## 8. Audit Logging (Mandatory)

Every browsing action MUST be logged with:

* timestamp
* tool name
* sanitized parameters
* URLs fetched
* result metadata (title, domain)
* block or failure reason (if applicable)

### Storage Limits

* Raw HTML MUST NOT be stored long‑term
* Persisted artifacts may include only:

  * URL
  * title
  * date
  * short excerpt
  * citation identifier

---

## 9. Failure Semantics

A browsing request MUST fail closed if:

* intent validation fails
* risk tier is invalid
* scope limits are exceeded
* access restrictions are violated

### Retry Policy

* No automatic retries without context change
* After **two failures**, Alice MUST stop and request guidance

---

## 10. Greater‑Than‑Parity Enhancements (v1‑Enabled)

These capabilities are explicitly allowed in v1:

* **Domain pinning** (user or Alice‑initiated):

  * restrict results to an allowlist of domains

* **Freshness constraints**:

  * enforce recency windows when requested

* **Cross‑source comparison**:

  * permitted under Tier 3 rules

---

## 11. Canonical Status

This document is authoritative for browsing enforcement in v4‑A.

Any change requires:

* Explicit discussion
* Explicit approval
* Versioned revision

No silent modification. No drift.
