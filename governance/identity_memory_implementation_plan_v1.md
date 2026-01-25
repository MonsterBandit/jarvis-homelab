# Identity & Memory v1 — Implementation Plan (DESIGN)

**Status:** Implementation planning complete (pending explicit authorization to implement)
**Depends on (LOCKED):**

* /opt/jarvis/governance/identity_memory_design_v1.md
* /opt/jarvis/governance/identity_memory_api_surface_v1.md
* /opt/jarvis/governance/identity_memory_ui_flows_v1.md

**No execution is authorized by this document.**

---

## 0. Operator Check

Before any code changes, confirm where you are operating:

* Chromebook (VS Code Remote-SSH) or
* Server (local shell)

Implementation steps assume the canonical repo layout:

* Backend: `/opt/jarvis/brain/`
* DB: `/opt/jarvis/brain-data/jarvis_brain.db`
* Governance: `/opt/jarvis/governance/`

---

## 1. Data Model (SQLite)

### 1.1 Tables

#### `users`

* Canonical internal user records.

Columns:

* `id` INTEGER PRIMARY KEY AUTOINCREMENT
* `handle` TEXT NULL  (human-facing label if known)
* `lifecycle_state` TEXT NOT NULL CHECK IN ('provisional','reinforced','permanent')
* `created_at` TEXT NOT NULL (ISO8601)
* `updated_at` TEXT NOT NULL (ISO8601)
* `reinforced_at` TEXT NULL
* `expiry_at` TEXT NULL
* `is_deleted` INTEGER NOT NULL DEFAULT 0
* `deleted_at` TEXT NULL

Indexes:

* `idx_users_lifecycle_state`
* `idx_users_expiry_at`

#### `user_identity_links` (optional v1)

* Placeholder for future mapping (HA, OAuth, voice). Keep minimal.

Columns:

* `id` INTEGER PK
* `user_id` INTEGER NOT NULL FK(users.id)
* `provider` TEXT NOT NULL
* `provider_subject` TEXT NOT NULL
* `created_at` TEXT NOT NULL

Unique:

* (`provider`,`provider_subject`)

> If you want absolute minimal v1, omit this table and add later with a version bump.

#### `memory_entries`

Columns:

* `id` INTEGER PK
* `user_id` INTEGER NOT NULL FK(users.id)
* `tier` INTEGER NOT NULL CHECK IN (0,1,2)
* `type` TEXT NOT NULL
* `value_json` TEXT NOT NULL  (JSON string, schema per type)
* `value_text` TEXT NOT NULL  (human-readable)
* `created_at` TEXT NOT NULL
* `created_by` TEXT NOT NULL CHECK IN ('user','admin')
* `is_deleted` INTEGER NOT NULL DEFAULT 0
* `deleted_at` TEXT NULL

Indexes:

* `idx_memory_user_tier`
* `idx_memory_type`

#### `memory_proposals`

Columns:

* `id` TEXT PRIMARY KEY  (uuid)
* `user_id` INTEGER NOT NULL
* `tier` INTEGER NOT NULL CHECK IN (1,2)
* `type` TEXT NOT NULL
* `value_json` TEXT NOT NULL
* `value_text` TEXT NOT NULL
* `reason` TEXT NOT NULL
* `requires_admin` INTEGER NOT NULL CHECK IN (0,1)
* `status` TEXT NOT NULL CHECK IN ('pending','confirmed','rejected','expired')
* `created_at` TEXT NOT NULL
* `confirmed_at` TEXT NULL
* `rejected_at` TEXT NULL
* `expires_at` TEXT NOT NULL

Indexes:

* `idx_proposals_user_status`
* `idx_proposals_expires_at`

#### `admin_inbox_events`

Columns:

* `id` TEXT PRIMARY KEY (uuid)
* `event_type` TEXT NOT NULL CHECK IN ('provisional_user_created','user_reinforced','user_expiring','user_deleted')
* `user_id` INTEGER NOT NULL
* `created_at` TEXT NOT NULL
* `metadata_json` TEXT NOT NULL DEFAULT '{}'
* `acked_at` TEXT NULL
* `acked_by_user_id` INTEGER NULL

Indexes:

* `idx_inbox_acked`
* `idx_inbox_created_at`

#### `audit_log`

Columns:

* `id` TEXT PRIMARY KEY (uuid)
* `domain` TEXT NOT NULL CHECK IN ('identity','memory','admin_inbox')
* `action` TEXT NOT NULL
* `actor` TEXT NOT NULL CHECK IN ('system','user','admin')
* `actor_user_id` INTEGER NULL
* `subject_user_id` INTEGER NULL
* `target_id` TEXT NULL (proposal id, memory id, inbox id)
* `created_at` TEXT NOT NULL
* `summary` TEXT NOT NULL
* `metadata_json` TEXT NOT NULL DEFAULT '{}'

Indexes:

* `idx_audit_domain_created_at`
* `idx_audit_subject_created_at`

---

## 2. Minimal Business Logic

### 2.1 Provisional Creation (GET /identity/me)

* Resolve internal user by a deterministic session identity token (see Section 3).
* If absent: create provisional user with `expiry_at = now + PROVISIONAL_TTL`.
* Emit admin inbox event `provisional_user_created`.
* Audit: domain=identity, action=user_created_provisional, actor=system.

### 2.2 Reinforcement

* Reinforcement is explicit and bounded.
* In v1, reinforcement occurs only when a *recognized session identity* returns (not background).
* On reinforcement:

  * lifecycle_state: provisional → reinforced (if not permanent)
  * update reinforced_at
  * extend expiry_at by REINFORCEMENT_EXTEND (or cancel if policy says so)
  * emit admin inbox event `user_reinforced`
  * audit event

### 2.3 Expiry & Deletion Job

* A single periodic job (timer) runs inside jarvis-brain process.
* Frequency: every 15 minutes (configurable).
* Behavior:

  * Find provisional users where `expiry_at <= now` AND not deleted.
  * Soft-delete user (is_deleted=1, deleted_at).
  * Soft-delete their memory entries.
  * Mark any pending proposals as expired.
  * Emit admin inbox event `user_deleted`.
  * Audit.

> Note: The design says deletion is irreversible. We can still implement as soft-delete internally to preserve audit evidence. The UI must treat it as deleted.

### 2.4 Proposal Expiry Job

* Same periodic job:

  * Mark proposals as `expired` when expires_at <= now.
  * Audit optional.

---

## 3. Session Identity Token (v1 Minimal)

We need a deterministic way for `GET /identity/me` to map a browser session to an internal user.

**v1 choice (minimal):**

* UI generates and persists a random UUID in localStorage: `alice_session_id`.
* UI sends it on every request: `X-ALICE-SESSION-ID: <uuid>`
* Backend maps this to a user via a small table:

#### `session_identities`

* `session_id` TEXT PRIMARY KEY
* `user_id` INTEGER NOT NULL
* `created_at` TEXT NOT NULL
* `last_seen_at` TEXT NOT NULL

No PII.
No external identity.

This supports reinforcement without login.

---

## 4. API Implementation Mapping (FastAPI)

### 4.1 File Layout

Create new modules under `/opt/jarvis/brain/`:

* `identity_memory/`

  * `__init__.py`
  * `models.py` (pydantic DTOs)
  * `db.py` (SQL helpers)
  * `service.py` (business logic)
  * `router_identity.py`
  * `router_memory.py`
  * `router_admin.py`
  * `router_audit.py`
  * `jobs.py` (expiry/proposal sweeper)

Wire into `main.py` via router include, preserving existing auth patterns.

### 4.2 Endpoint Notes

* `GET /identity/me` requires `X-ALICE-SESSION-ID` and `X-API-Key`.
* Admin endpoints require `X-ISAC-ADMIN-TOKEN` in addition.
* Proposal confirm endpoint must enforce tier gating.

---

## 5. UI Implementation Hooks (index.html)

### 5.1 Minimal Additions

* Generate and persist `alice_session_id` in localStorage if missing.
* Ensure ALL calls include:

  * `X-API-Key`
  * `X-ALICE-SESSION-ID`
  * `X-ISAC-ADMIN-TOKEN` only when unlocked

### 5.2 Flow Wiring

* On initial load, call `GET /identity/me`.
* Identity chip surfaces only on triggers (per UI flows doc).
* Memory proposal UI must follow:

  * propose → confirm/reject
  * show/forget
* Admin inbox shown only on explicit command.

---

## 6. Configuration

Environment variables (jarvis-brain):

* `ALICE_PROVISIONAL_TTL_HOURS` (default 168)
* `ALICE_REINFORCEMENT_EXTEND_HOURS` (default 168)
* `ALICE_PROPOSAL_TTL_MINUTES` (default 60)
* `ALICE_SWEEPER_INTERVAL_SECONDS` (default 900)

All have safe defaults; no configuration required to boot.

---

## 7. Failure Semantics (No Raw Errors to User)

Backend returns structured errors:

* `code`: string
* `message`: safe
* `details`: optional

UI translates into Alice voice:

* missing session id
* missing base key
* admin token required
* proposal expired
* forbidden tier

All are logged.

---

## 8. Verification Plan (Pre-Gate D)

### 8.1 Unit-ish Smoke Tests (curl)

1. Create provisional (me)

* Expect lifecycle_state=provisional

2. Reinforcement

* Same session id second call
* Expect reinforced event or updated last_seen

3. Tier 1 proposal

* propose → confirm
* show memory
* forget memory

4. Admin inbox

* list → ack

5. Expiry

* shorten TTL in env
* wait for sweeper
* verify deletion + inbox event

### 8.2 Evidence Produced

* admin_inbox_events entries
* audit_log entries
* deterministic endpoint behavior

---

## 9. Implementation Order (Surgical)

1. DB migration (create tables)
2. session identity mapping + `GET /identity/me`
3. admin_inbox write path + list/ack
4. memory proposals + confirm/reject
5. memory list + delete
6. audit endpoints
7. sweeper job
8. UI wiring (identity chip + memory flows)

Stop after each step and verify.

---

**End Identity & Memory v1 Implementation Plan**
