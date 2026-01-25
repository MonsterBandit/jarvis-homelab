# Identity & Memory — UI Flows v1 (DESIGN, LOCK-READY)

**Status:** Design-complete (pending explicit lock)
**Depends on:**

* identity_memory_design_v1.md (LOCKED)
* identity_memory_api_surface_v1.md (LOCKED)

This document defines the **exact UI flows** required to surface Identity & Memory v1 **without authority leakage, pressure, or silent state change**.

Conversation remains the authority spine at all times.

---

## 1. Core UI Principles (NON-NEGOTIABLE)

* Conversation-first; UI elements are subordinate
* No dashboards by default
* No background learning
* No surprise persistence
* All memory actions are explicit, narrated, and reversible
* Silence is a valid outcome

---

## 2. Identity Chip Flow

### 2.1 When It Appears

The identity chip appears **only when relevant**, never persistently.

Triggers:

* New provisional user detected
* User promotion event (admin)
* User explicitly asks “who am I to you?” or similar

---

### 2.2 Identity Chip Content

Displayed inline, near the conversation margin:

* Display name (if known)
* Lifecycle state:

  * Provisional
  * Reinforced
  * Permanent
* Expiry indicator (provisional only, vague not alarming)

Example:

> Tim — Provisional

No calls to action by default.

---

### 2.3 Allowed Actions

* Admin-only: “Promote to permanent”
* User-only: none

Promotion always requires explicit admin confirmation via conversation.

---

## 3. Memory Proposal Flow (Tier 1)

### 3.1 Trigger

Alice may suggest memory **only after promotion OR sufficient reinforcement**.

Alice’s phrasing must be optional and low-pressure:

> “Would you like me to remember that you prefer metric units?”

No suggestion allowed mid-task execution.

---

### 3.2 Proposal Card

Inline card appears **only after user signals interest** (e.g., “yes”).

Card content:

* What will be remembered (plain language)
* Tier (Tier 1)
* Why it helps
* Buttons:

  * Confirm
  * Don’t remember

No default selection.

---

### 3.3 Outcomes

* Confirm → call `POST /memory/proposals` then `/confirm`
* Don’t remember → call `/reject`

Alice must verbally acknowledge the outcome.

---

## 4. Memory Proposal Flow (Tier 2 — Admin)

### 4.1 Trigger

Only when:

* Admin is present
* Admin explicitly requests or agrees

Never auto-suggested.

---

### 4.2 Admin Confirmation Card

Card content:

* Explicit tier warning
* Scope and impact
* Buttons:

  * Confirm (Admin)
  * Cancel

Admin token required.

---

## 5. View / Forget Memory Flow

### 5.1 Show Memory

Trigger phrases:

* “What do you remember about me?”
* “Show my memory”

Alice response:

* Summarizes first
* Offers expandable list by tier

---

### 5.2 Forget Memory

User selects memory item → “Forget this”

Flow:

* Confirmation prompt
* On confirm → `DELETE /memory/{id}`
* Verbal acknowledgment

Tier 2 deletion requires admin.

---

## 6. Admin Inbox Flow

### 6.1 When Surfaced

Never auto-opens.

Triggers:

* Admin says “show admin inbox”
* Admin navigates via explicit UI affordance

---

### 6.2 Inbox Item Card

Each item shows:

* Event type
* Affected user
* Timestamp

Actions:

* Acknowledge
* (Optional) Navigate to user

No inline execution.

---

## 7. Prohibited UI Patterns

Explicitly forbidden:

* Toasts implying memory saved
* Auto-dismiss confirmations
* Memory badges
* Gamification
* Nudging language
* Batch confirmations

---

## 8. Gate D Alignment

These UI flows produce:

* Explicit evidence of consent
* Clear audit trails
* Human-legible interaction records

They are sufficient and required for Gate D evaluation.

---

## 9. Change Control

This document is lockable.
Changes require versioning and explicit admin authorization.

---

**End Identity & Memory UI Flows v1**
