# ISAC Finance Contract (Phase 7)

## Purpose

This document defines the **explicit contract** for ISAC’s finance integration
during Phase 7 and beyond.

It exists to:
- Preserve auditability
- Prevent scope creep
- Clarify authority boundaries
- Serve as evidence for Gate D (Stability & Trust)

Firefly III remains the **sole system of record** for financial data.
ISAC is a **read-only analysis and observability layer**.

---

## Authority Model

### Source of Truth
- **Firefly III** is authoritative for all financial data.
- ISAC does not reconcile, correct, or override Firefly data.

### ISAC Capabilities (Allowed)
- Read snapshot files produced from Firefly data
- Summarize balances and cashflow
- Surface transaction history
- Perform health and integrity checks
- Provide analysis and insights (non-binding)

### ISAC Prohibitions (Hard Rules)
ISAC **must not**:
- Write to Firefly III
- Modify transactions, balances, categories, or budgets
- Initiate imports or exports
- Automate payments or transfers
- Reconcile discrepancies
- Create or alter financial records

Any future write capability requires:
- Explicit Admin authorization
- A new phase
- A new contract
- Gate re-evaluation

---

## Snapshot Model

### Snapshot Location
- Host: `/opt/jarvis/brain-data/finance/snapshots/latest.json`
- Container: `/app/data/finance/snapshots/latest.json`

`latest.json` is a symlink pointing to a dated snapshot:
