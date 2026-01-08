# Alice v0 Scan Ingestion Flow (Phase 6.45)
Version: v0 (LOCKED)
Owner: Admin (Tim)
Last updated: 2026-01-07

## Purpose
Define the canonical, human-in-the-loop workflow for scan-driven product ingestion using Alice v0.
This document is authoritative over implementation. If code disagrees with this doc, code is wrong.

## Scope
In-scope (v0):
- Single-scan, single-product ingestion loop
- Manual scan intake (barcode + parsed name + quantity)
- Canonical naming proposal (Alice proposes, human confirms)
- Dependency creation (locations/units) with explicit approval
- Explicit final confirmation before any writes
- Visible success/failure and auditable outcomes

Out-of-scope / deferred:
- Product groups (secondary enrichment pass)
- Images, nutrition, weights, conversion factors, labels
- Background polling, queues, batch processing
- “Silent learning” or silent writes
- Any automation or scheduling
- Multi-house mixing in one session (prohibited)

## Roles & Authority
- Human (Admin): final authority for all writes. Explicit confirmation required.
- Alice v0: human-facing ingestion console. Collects inputs, asks questions, summarizes, requests confirmation.
- ISAC: reasons and proposes only. No autonomous writes. Implements write calls only after Admin confirmation.

## Non-Negotiables
- Governance before code.
- No silent writes.
- No background actions or polling.
- Ask one question at a time during ingestion.
- All writes must be visible, auditable, and reversible.
- Session is bound to exactly one household (home_a OR home_b).

## Session Model (House Lock)
At the beginning of each ingestion session:
1) Alice asks: which house? `home_a` or `home_b`
2) The choice is locked for the session.
3) Alice must NOT mix houses unless the user explicitly restarts the session.

Restart conditions:
- Admin explicitly says “restart session” (or equivalent)
- Alice confirms the new house before continuing

## Scan Intake (v0)
Input fields required:
- barcode (string)
- parsed_name (string; may be empty/unknown)
- scanned_quantity (number; defaults from scan, usually 1)

Notes:
- BarcodeBuddy context may have produced these values, but v0 intake is manual paste/type.
- No auto-pull of “last scan” in v0.

## Canonical Question Order (Single-Scan Loop)
Alice asks questions in this order, one at a time:

1) Canonical product name
   - Alice proposes a canonical name based on parsed_name and prior memory.
   - Admin confirms or edits.

2) Quantity confirmation
   - Default to scanned_quantity (usually 1).
   - Admin confirms or edits.

3) Storage location (house-scoped)
   - Ask: “Where is this stored?”
   - If missing location exists in Grocy: prompt to create (see Dependency Creation Rules).

4) Consume location (house-scoped)
   - Ask: “Is this used in a different location?”
   - If yes: capture the other location.
   - If missing: prompt to create.

5) Stock unit (house-scoped concept)
   - Ask: “What is the stock unit?” (e.g., bottle, jar, box, lb)
   - If missing: prompt to create.

6) Purchase/Consume unit confirmation
   - Default assumption: purchase unit == consume unit == stock unit.
   - Ask ONLY if different: “Is purchase or consume unit different than stock unit?”

7) Price unit
   - Ask: “What unit should price attach to?” (often same as stock, but not always)
   - If missing: prompt to create (if modeled as a unit concept) or capture as metadata per Grocy capability.

8) Final summary + explicit confirmation
   - Alice summarizes all resolved fields.
   - Ask: “Create product now?”
   - If no: stop without writing.

## Dependency Creation Rules (No Parking)
If a required dependency is missing (location/unit):
- Alice must ask immediately and explicitly:
  “That location/unit doesn’t exist yet. Create it now?”
- If Admin approves:
  - Create the dependency in Grocy (house-scoped where applicable).
  - Resume the ingestion flow automatically at the next step.
- If Admin declines:
  - Abort the ingestion flow for this scan (no writes).
  - Alice asks for an alternative (different existing location/unit) OR ends.

No “park and forget”:
- Alice cannot skip dependencies and continue. Either create now or abort.

## Write Boundary and Confirmation
Before ANY Grocy write:
- Alice must present a final summary that includes:
  - House (home_a/home_b)
  - Canonical name
  - Barcode
  - Quantity
  - Storage location
  - Consume location (if different)
  - Stock unit
  - Purchase/consume unit (if different)
  - Price unit
- Admin must explicitly confirm “Create product.”

Retry behavior:
- On failure: allow ONE retry (Admin-triggered).
- If retry fails: stop and present error details + next actions (no looping).

On success:
- Confirm the product was added.
- Prompt: “Ready for the next scan.”

## Deferred Enrichment (Explicitly Not Done Here)
Must NOT occur during scan ingestion:
- Product groups / categories
- Images, nutrition, labels
- Weights, tare, conversion factors
- Advanced inventory rules
- Multi-store intelligence

These occur in a secondary enrichment pass (separate governance doc later).

## Endpoint Map (High Level, No Code)
Alice UI:
- Manual intake → propose → confirm → write

ISAC / Alice memory:
- GET  /alice/memory/concepts
- POST /alice/memory/concepts  (Admin only; requires X-ISAC-ADMIN-KEY)
- GET  /alice/memory/aliases
- POST /alice/memory/aliases   (Admin only; requires X-ISAC-ADMIN-KEY)

Grocy (conceptual; implementation may vary):
- Create product
- Add initial stock
- Attach barcode
- Create missing location/unit dependencies (only when Admin approves)

Security:
- Admin writes require `X-ISAC-ADMIN-KEY`.
- Read vs Admin mode must be enforced server-side.
- No hardcoded keys. Keys must match real values from `.env`.

## Auditability & Reversibility
- All writes must be traceable (request, result, timestamps).
- If an ingestion creates incorrect data:
  - Admin can delete/undo using Grocy UI or a future controlled endpoint.
- Alice must never conceal failures or partial writes.

## Change Control
This document is LOCKED for v0.
Any change requires:
- A deliberate doc edit
- A commit message that includes “governance change”
- Explicit Admin approval in-session
