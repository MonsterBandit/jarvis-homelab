# ISAC Finance Read Contract (Firefly III)

## Status
**Locked – Read-Only**

## Purpose
This document defines how ISAC may access and reason about financial data
sourced from Firefly III. It exists to preserve financial authority boundaries,
auditability, and human trust.

Firefly III remains the system of record.
ISAC is a read-only observer and summarizer.

---

## Data Source

ISAC MUST read financial data exclusively from the local snapshot:

/opt/jarvis/brain-data/finance/snapshots/latest.json

This file is a symlink pointing to the most recent immutable snapshot
(e.g. firefly_snapshot_YYYY-MM-DD.json).

ISAC MUST NOT:
- Call the Firefly III API directly
- Store Firefly API tokens
- Modify snapshot files
- Create, edit, or delete financial records

---

## Snapshot Characteristics

Snapshots are:
- Generated out-of-band
- Date-stamped and immutable
- Append-only (historical snapshots are preserved)
- Auditable

Snapshot schema (v1):
- meta
- accounts
- transactions

---

## Permitted ISAC Actions

ISAC MAY:
- Read account balances
- Summarize spending and income
- Detect anomalies or trends
- Answer questions about historical data
- Ask the Admin for clarification or approval

ISAC MUST:
- Treat all financial output as advisory
- Defer to human confirmation for any action
- Preserve original data semantics

---

## Forbidden Actions

ISAC MUST NOT:
- Initiate transactions
- Modify categories, budgets, or accounts
- Reconcile or “fix” financial data
- Automate payments
- Write back to Firefly III or any finance system

---

## Future Changes

Any expansion beyond read-only behavior requires:
- Explicit Admin authorization
- Updated contract version
- Gate review (Gate D applicable)

Until then, this contract is binding.
