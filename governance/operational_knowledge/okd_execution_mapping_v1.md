# OKD Execution Mapping v1
## Browsing Constitution → Tooling → Enforcement

**Status:** DRAFT (Pending Admin Lock)  
**Authority:** Governance (Absolute once locked)  
**Depends On:**
- alice_browsing_constitution_v1.md
- okd_execution_v1.md

This document maps **governed observation policy** to **actual execution surfaces**.
It is the enforcement bridge between theory and code.

---

## 1. Scope

This document governs **observation only**.

It defines:
- which tools implement which browsing tiers
- what guardrails are enforced per tier
- how failures surface
- what artifacts are produced

It does **not** authorize:
- execution
- writes
- automation
- memory promotion

---

## 2. Risk Tier → Tool Mapping

### Tier 0 — Intrinsic / Zero Risk

**Definition:** Internal reasoning only

**Tools:**
- None (model-internal)

**Enforcement:**
- No logging required
- No artifacts
- No disclosure

---

### Tier 1 — Local Observation

**Definition:** Read-only access to owned system state

**Tools (ISAC-only):**
- `local.read_file`
- `local.read_snippet`
- `local.search_files`

**Guardrails:**
- Allowlist enforced
- Read-only
- No network
- Full audit logging

**Artifacts:**
- Tool output artifact (system card)
- No sources section

---

### Tier 2 — External Observation (Browsing Core)

**Definition:** Public, unauthenticated web observation

**Tools:**
- `web.search`
- `web.open`
- `web.find`

**Guardrails:**
- Explicit intent required
- No background execution
- No credentialed access
- Domain allowlist respected
- Requests and responses logged

**Artifacts:**
- Update Brief (facts + citations)
- Sources hidden unless requested

---

### Tier 3 — Interpretive / Aggregative

**Definition:** Synthesis across multiple observed sources

**Tools:**
- Same as Tier 2, plus aggregation logic

**Guardrails:**
- Inputs disclosed
- Inference vs fact labeled
- Uncertainty surfaced
- Output reversible

**Artifacts:**
- Update Brief with labeled sections:
  - facts
  - inferences
  - conflicts
  - uncertainties

---

### Tier 4 — World-Affecting

**Definition:** Any state-changing capability

**Status:** ❌ Explicitly excluded from OKD

**Handled Elsewhere:**
- Fix-It Protocol
- Execution Spine
- Admin confirmation flows

---

## 3. Execution Spine Binding

All OKD observation runs through:

```
POST /tools/call
```

### Required payload fields:
- `tool_name`
- `purpose`
- `args`
- `risk_tier`
- `user_prompt_excerpt`

### Enforcement:
- ISAC executes
- Alice narrates
- Tool budgets enforced per request
- No silent retries or expansions

---

## 4. Disclosure Rules (Binding)

Alice must disclose:
- query reformulations
- retries
- scope expansions
- credential usage (if any)

Disclosure occurs:
- automatically in the Update Brief
- on demand if the human asks

Failure to disclose → **hard FAIL**

---

## 5. Failure Semantics

If a tool request violates governance:
- execution halts
- no partial results returned
- Alice explains the boundary in human language
- system logs the failure

No auto-retry without scope change.

---

## 6. Audit Artifacts

Each governed observation produces:

### Required
- Observation Plan (pre-execution)
- Expansion Log (during execution)
- Update Brief (final)

Artifacts are:
- system-authored
- inspectable
- non-conversational

---

## 7. UI Binding (Non-Authoritative)

UI behavior must:
- render artifacts as system cards
- never embed artifacts in Alice prose
- hide sources unless requested
- preserve silence as a valid outcome

UI does not override governance.

---

## 8. LAP Status

During LAP:
- OKD execution is **permitted**
- Identity & Memory remain **fail-closed**
- Finance execution remains **blocked**

This document defines **how observation works now**.

---

## 9. Lock Criteria

This document may be locked when:
- Admin confirms alignment with OKD Execution v1
- Tooling matches described surfaces
- Smoke tests pass

Once locked:
- Code must conform
- UI must conform
- Deviations are incidents

---

**End of OKD Execution Mapping v1**
