# OKD Execution v1 — Governed Observation & Reasoning

**Status:** DRAFT (Pending Lock)

**Owner:** Admin (Tim)

**Applies to:** Alice v4‑A and later

**Authority:** Governance > Code > UI

**Change Control:** Explicit Admin approval required. If code disagrees with this document, code is wrong.

---

## 1. Purpose & Scope

This document defines the authoritative execution model for **Operational Knowledge Domains (OKD)** under **Bundle 3 — Governed Observation & Reasoning**.

The goal of OKD Execution v1 is to grant Alice **full observational and reasoning parity with ChatGPT’s browsing and hidden behaviors**, while enforcing:

* Explicit disclosure
* Bounded scope
* Human consent
* Full auditability
* Fail‑closed semantics

This document governs **observation only**. No actions, writes, automation, or scheduling are permitted.

---

## 2. Authority Model

### Roles

* **Human (Admin)**

  * Final authority over scope, credentials, and execution.
  * Explicitly approves any scope expansion or login‑gated access.

* **Alice**

  * Human‑facing mediator.
  * Proposes observation plans.
  * Discloses scope, expansions, retries, and outcomes.
  * Never executes tools directly.

* **ISAC**

  * Silent executor.
  * Executes approved observation plans.
  * Enforces guardrails, budgets, and audit logging.

### Preconditions

OKD execution may only occur if:

* Bundle 1 (Context Integrity & Legitimacy) is PASSED.
* Bundle 2 (Tooling, Execution & Verification Discipline) is PASSED.
* The human has explicitly invited observation.

---

## 3. Authorized Capabilities (Full Parity)

OKD Execution v1 authorizes **all** of the following capabilities, provided they are disclosed, bounded, and logged.

### 3.1 Public Web Observation

* Search
* Open
* Find
* Extract
* Summarize
* Compare
* Cite

### 3.2 Aggressive Crawling (Governed)

* Multi‑page traversal
* Link following
* Depth‑bounded exploration
* Domain‑restricted or open (explicit)

### 3.3 Query Reformulation (Disclosed)

* Synonym expansion
* Broadening / narrowing
* Alternate phrasing

Reformulations must be logged and disclosed.

### 3.4 Retries & Fallbacks (Explicit)

* Network retries
* Alternate query retries
* Alternate source retries

Retries are capped and never silent.

### 3.5 Scope Expansion (Governed)

* Additional queries
* Additional pages
* Additional domains
* Increased crawl depth

All expansions require logging and disclosure.

### 3.6 Login‑Gated Content (Optional, Explicit)

* Credential‑protected sources
* Read‑only access
* Session‑bound credentials
* Explicit consent per domain

---

## 4. Execution Flow

All observation runs through a single execution spine.

### 4.1 Observation Task

**Task Type:** `governed_observation`

### 4.2 Lifecycle

1. **Propose** — Alice proposes an observation plan.
2. **Preview** — Human reviews scope and approves.
3. **Execute** — ISAC executes under approved scope.
4. **Review** — Update Brief artifact is produced.
5. **Certify** — PASS / FAIL criteria evaluated.

No background execution is permitted.

---

## 5. Artifacts (Required)

### 5.1 Observation Plan (Pre‑Execution)

Stored before execution.

Fields:

* intent
* initial_queries
* query_reformulations (planned)
* scope_budget
* login_required
* expected_risk_tier

### 5.2 Expansion Log (During Execution)

Records all expansions, retries, and reformulations.

Fields:

* type (query_reformulation | retry | scope_expansion | crawl)
* reason
* details
* count

### 5.3 Update Brief (Final)

Human‑consumable summary.

Required sections:

* facts (with citations)
* inferences (labeled)
* conflicts
* uncertainties
* scope_used
* login_used

---

## 6. Disclosure Rules

Alice must disclose:

* Query reformulations
* Retries
* Scope expansions
* Credential usage

Disclosure occurs:

* Automatically after execution
* On demand if the human asks

Failure to disclose any expansion is a **hard FAIL**.

---

## 7. Hard Limits & Budgets (Defaults)

| Dimension   | Default Cap             |
| ----------- | ----------------------- |
| Queries     | 4                       |
| Opens       | 6                       |
| Crawl Depth | 2                       |
| Retries     | 2 per action            |
| Domains     | Explicit allowlist      |
| Credentials | Explicit, session‑bound |

Caps may only be raised with explicit human approval.

---

## 8. Login‑Gated Access Rules

### Consent Requirements

* Domain specified
* Purpose specified
* Duration specified
* Scope specified

### Credential Handling

* Encrypted at rest
* Scoped to domain
* Session‑bound by default
* Revocable

### Audit Logging

Every access must log:

* Domain
* Timestamp
* Pages accessed
* Duration
* Outcome

Failure conditions:

* CAPTCHA
* Unexpected redirect
* Scope mismatch
* Expired credential

All failures fail closed.

---

## 9. Fail‑Closed Semantics

Execution must fail if:

* Scope limits exceeded
* Undisclosed expansion occurs
* Retry cap exceeded
* Unauthorized login attempted
* Observation mutates state

Partial results are not returned on failure.

---

## 10. Certification Criteria

### PASS requires:

* All expansions logged
* All facts cited
* All inferences labeled
* No silent retries
* No silent scope expansion
* Explicit consent for login‑gated access
* Update Brief artifact generated

### FAIL if any criterion is violated.

---

## 11. Change Control

This document is versioned.

Changes require:

* Explicit discussion
* Explicit Admin approval
* Version bump

No silent modification is permitted.

---

## 12. Canonical Status

Upon Admin approval, this document becomes the **authoritative specification** for Bundle 3.

All code must conform.
