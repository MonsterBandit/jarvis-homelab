# Identity & Memory — Minimal API Surface v1 (DESIGN, LOCK-READY)

**Status:** Design-complete (pending explicit lock)
**Prerequisite:** identity_memory_design_v1.md (LOCKED)

This document defines the **minimal, sufficient API surface** required to implement Identity & Memory v1 **without expanding authority, scope, or semantics**.

No endpoint below implies permission to execute, infer, or automate.
All endpoints are auditable.

---

## 1. Design Principles (NON-NEGOTIABLE)

* Minimal surface area
* Explicit intent per call
* No combined read/write endpoints
* No silent defaults
* No inference from access
* All writes require explicit confirmation flags
* Identity & Memory are *inputs*, never rewards

---

## 2. Authentication & Authority

### 2.1 Required Headers

All endpoints:

* `X-API-Key` (base key, always required)

Admin-gated endpoints additionally require:

* `X-ISAC-ADMIN-TOKEN` (session-scoped, explicit unlock)

Absence or mismatch = hard failure.

---

## 3. User APIs

### 3.1 Resolve / Get Current User

```
GET /identity/me
```

**Purpose**

* Resolve the active internal user
* Auto-create provisional user if none exists

**Returns**

* user_id
* lifecycle_state: provisional | reinforced | permanent
* expiry_at (nullable)
* reinforced_at (nullable)

**Side effects**

* Provisional auto-creation is allowed
* MUST generate Admin Inbox notification if created

---

### 3.2 Get User (Admin)

```
GET /identity/users/{user_id}
```

**Admin-gated**

Returns full identity record + lifecycle metadata.

---

### 3.3 Promote User (Admin)

```
POST /identity/users/{user_id}/promote
```

**Admin-gated**

Effects:

* lifecycle_state → permanent
* expiry timer canceled
* audit event emitted

No other side effects allowed.

---

## 4. Memory APIs

### 4.1 List Memory (User or Admin)

```
GET /memory
```

Query params:

* tier=0|1|2
* user_id (admin only)

Returns:

* memory_id
* tier
* type
* value (human-readable)
* created_at
* created_by (user | admin)

---

### 4.2 Propose Memory Write

```
POST /memory/proposals
```

**Purpose**

* Create a *non-persistent* memory proposal

Payload:

* user_id
* tier (1 or 2)
* type
* value
* reason (human-readable)

Returns:

* proposal_id
* requires_admin (bool)

No persistence occurs here.

---

### 4.3 Confirm Memory Write

```
POST /memory/proposals/{proposal_id}/confirm
```

Rules:

* Tier 1: user confirmation required
* Tier 2: admin confirmation required

Payload:

* confirm: true

Effects:

* Memory entry created
* Audit log written

Absence of confirm=true = rejection.

---

### 4.4 Reject Memory Proposal

```
POST /memory/proposals/{proposal_id}/reject
```

Effects:

* Proposal destroyed
* Must not be auto-retried

---

### 4.5 Forget Memory

```
DELETE /memory/{memory_id}
```

Rules:

* User may delete Tier 1 memory
* Admin required for Tier 2

Deletion is irreversible.
Audit entry required.

---

## 5. Admin Inbox APIs

### 5.1 List Admin Inbox

```
GET /admin/inbox
```

**Admin-gated**

Event types:

* provisional_user_created
* user_reinforced
* user_expiring
* user_deleted

Inbox is informational.
No implicit actions.

---

### 5.2 Acknowledge Inbox Item

```
POST /admin/inbox/{event_id}/ack
```

Marks item as acknowledged.
No other effects.

---

## 6. Audit APIs (Read-only)

```
GET /audit/identity
GET /audit/memory
```

**Admin-gated**

Returns immutable audit records.

---

## 7. Explicitly Forbidden

The following are **not allowed**:

* Bulk memory writes
* Implicit confirmations
* Cross-user reads without admin
* Memory inference
* Authority expansion via memory
* Background reinforcement
* Silent lifecycle changes

---

## 8. Gate D Relevance

This API surface is the **only permitted interface** for Identity & Memory evidence consumed by Gate D.

Any additional endpoint constitutes drift.

---

**End Identity & Memory API Surface v1**
