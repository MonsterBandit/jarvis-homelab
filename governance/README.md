# Governance Index (Authoritative)

This directory contains the canonical governance for the Jarvis / ISAC system.

Governance defines authority, constraints, phase behavior, and change rules.
It is not documentation of code, services, or infrastructure.
It exists to make system behavior explicit, reviewable, and bounded.

All code, automation, and operational procedures must comply with governance.
When governance and code disagree, governance wins.

---

## How Governance Works

Governance is layered:

1. **Global authority** establishes the maximum ceiling of what ISAC is ever allowed to do.
2. **Phase governance** overlays may narrow or further constrain authority.
3. **Execution artifacts** (code, scripts, runbooks) must operate strictly within these bounds.

Authority may be reduced by overlays.
Authority may never be expanded implicitly, temporarily, or silently.

---

## Canonical Authority & Contracts

These documents define binding authority and constraints.

### `isac-authority.md`
**Scope:** Global  
Defines the maximum authority ISAC may ever hold, the overlay model, and the
rules governing observation, suggestion, and action.

This document is the ceiling for all present and future behavior.

---

### `scan-driven-product-knowledge.md`
**Scope:** Phase 6.45 (Scanning & Ingestion)  
Defines scan-driven product knowledge, ingestion rules, and ISAC Ingestion
Authority V1 as a phase-specific overlay.

This document constrains how scanning, BarcodeBuddy, and Grocy ingestion behave.

---

### `finance_read_contract.md`
**Scope:** Phase 7 (Finance Awareness)  
Defines strict read-only access rules, prohibitions, and boundaries for
financial data.

ISAC may observe and summarize finance data only as explicitly permitted here.

---

## Governance Subdirectories

These directories contain governance-aligned artifacts and references.
They do not grant authority on their own.

### `runbooks/`
Human-executed procedures governed by authority.
Runbooks describe *how* approved actions are performed, not *whether* they are allowed.

---

### `irr/`
Incident Response & Recovery governance and artifacts.
Includes IRR stage definitions, scripts, and post-incident audit records.

---

### `incidents/`
Recorded incidents, narratives, and reviews.
Used for accountability, learning, and governance refinement.

---

### `backups/`
Governance-related backups and reference materials.
These are informational and do not imply execution authority.

---

### `bin/`
Governance-scoped scripts.
Scripts here must be non-autonomous and operate only within explicit authority.

---

## Status & Change Control

### `STATUS_CURRENT`
Represents the currently active and authoritative governance state.

If a rule or phase is not reflected here or in a canonical governance document,
it is not active.

---

### Change Rules

- Governance changes are explicit, reviewed, and committed intentionally.
- No silent edits.
- No retroactive authority expansion.
- Committed does not imply deployed or active unless governance says so.

---

## Adding New Governance

- New phases introduce new governance files.
- Authority changes must be explicit and documented.
- Governance must never be inferred from code behavior.

If behavior matters, it belongs here.
