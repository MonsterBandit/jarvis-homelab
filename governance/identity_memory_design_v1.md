# Identity & Memory Design v1 (LOCKED)

**Status:** Approved, Locked

This document is the canonical governance specification for Identity & Memory v1 within the Jarvis / ISAC / Alice system. It is binding and authoritative. Any deviation constitutes drift and must be treated as an incident.

---

## 1. Scope and Purpose

Identity & Memory v1 defines:

* How users are represented internally
* How memory is scoped, written, inspected, and destroyed
* The authority boundaries governing identity and memory actions
* The safety invariants required for Gate D eligibility

Identity & Memory are **inputs** to system stability, not rewards, not accelerants, and not implicit trust signals.

---

## 2. Identity Model

### 2.1 Canonical Identity

* **Internal users are canonical**
* External identities (HA users, OAuth, voice, etc.) may map to an internal user later, but are never primary
* All users share a **uniform schema** regardless of role or lifecycle stage

### 2.2 User Lifecycle States

1. **Provisional**

   * May be auto-created without admin involvement
   * Has a destruction timer
   * Limited memory eligibility

2. **Reinforced**

   * Achieved through repeated interaction or explicit reinforcement
   * Reinforcement extends or cancels destruction timer

3. **Permanent**

   * Explicitly promoted by Admin
   * Destruction timer permanently canceled
   * Eligible for broader (still bounded) memory suggestions

Lifecycle transitions are monotonic except for destruction.

### 2.3 Destruction Semantics

* Provisional users expire automatically when their timer elapses
* Expiry triggers:

  * User record deletion
  * All associated memory deletion
  * Audit log entry
* Destruction is irreversible

---

## 3. Memory Model

### 3.1 Core Properties

All memory is:

* Scoped to a user
* Typed
* Inspectable
* Auditable
* Explicitly consented

There are **no silent writes** under any circumstances.

### 3.2 Memory Tiers

#### Tier 0: Session Memory

* Ephemeral
* Exists only for the duration of the active session
* Automatically destroyed
* No consent required

#### Tier 1: Safe Profile Memory

* User-confirmed
* Non-admin gated
* Examples:

  * Preferences
  * Vocabulary / naming
  * Unit conventions
  * Intent patterns
* Explicitly excludes:

  * Finance
  * Calendar
  * Secrets
  * Transcripts
  * Authority expansion

#### Tier 2: Durable / Sensitive Memory

* Admin-gated
* Explicit admin confirmation required
* May include cross-user relevance
* Still inspectable and reversible

### 3.3 Memory Write Rules

* Alice may **suggest** memory writes
* Users must explicitly confirm Tier 1 writes
* Admin must explicitly confirm Tier 2 writes
* Absence of confirmation equals rejection

Rejected proposals must not be retried automatically.

---

## 4. Admin Controls

### 4.1 Admin Inbox

Admin receives notifications for:

* New provisional user creation
* User reinforcement events
* Impending user expiry
* Automatic user deletion

Inbox items are informational unless explicitly acted upon.

### 4.2 Promotion

* Admin may promote any user to Permanent
* Promotion:

  * Cancels destruction timer
  * Enables Alice to suggest broader Tier 1 memory
* Promotion does **not** grant authority expansion

### 4.3 Memory Hygiene

Admin capabilities:

* Show memory (per user, per tier)
* Forget specific entries
* Delete entire memory scopes
* Audit all memory writes and deletions

---

## 5. Safety and Drift

### 5.1 Drift Definition

Drift includes, but is not limited to:

* Silent memory writes
* Implicit authority expansion
* Memory outside defined tiers
* Identity lifecycle violations
* Unscoped or cross-user leakage

Drift is a **first-class incident type**.

### 5.2 Enforcement

* Any detected drift halts progression toward Gate D
* Drift must be investigated, explained, and resolved
* Repeated drift constitutes a hard stop

---

## 6. Gate D Relationship

* Identity & Memory are bounded inputs to Gate D
* They are not signals of readiness, trust, or autonomy
* Gate D evaluation consumes evidence derived from this model

Gate D **cannot** be run without full compliance with this document.

---

## 7. Change Control

* This document is LOCKED
* Amendments require:

  * Explicit Admin authorization
  * Versioned replacement
  * Migration and rollback plan

Unversioned changes are invalid.

---

**End of Identity & Memory Design v1**
