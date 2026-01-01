# IRR Stage 4B — Shutdown Orchestrator (DESIGN-ONLY)

Status: DESIGN-ONLY (No execution, no timers, no hooks)  
Scope: Infrastructure Resilience & Recovery (IRR)  
Authority: Admin-only for cancellation; ISAC is observability + analysis only.

---

## Purpose

Define a safe, explainable, and auditable shutdown orchestration model for extended power events.
This document specifies **phases, invariants, gates, and authority**. It does **not** enable automation.

---

## Hard Safety Invariant (NON-NEGOTIABLE)

**ZFS pools must be cleanly exported or idle before host shutdown.**

If this invariant cannot be satisfied, shutdown **must not proceed**.

---

## Trigger Model (Power-First, Network-Aware)

- **UPS state is authoritative.**
- **WAN loss is contextual only** (early warning; never a trigger).
- Orchestration decisions are based on UPS status, runtime, and battery state.

---

## Escalation Phases

### Phase 0 — Observe
- UPS status: OL (On Line)
- No actions.
- Observability only.

### Phase 1 — Degraded
- UPS transitions to OB (On Battery) or related degraded states.
- WAN loss may occur earlier and is noted.
- No actions; monitoring continues.

### Phase 2 — Prepare
- Conditions indicate a prolonged outage is likely.
- **Design-only intent**:
  - Identify non-essential containers.
  - Identify VMs (e.g., Home Assistant).
  - Prepare verification checks.
- **No execution** in Stage 4B.

### Phase 3 — Shutdown
- Entered only after verification gate passes.
- Host shutdown is the final act.
- **Admin-only cancellation allowed prior to entry.**

---

## Verification Gate (Before Shutdown)

All must be true:

1. **ZFS idle**
   - No active I/O
   - No scrub or resilver in progress
   - Pools healthy

2. **Workloads halted**
   - Non-essential containers stopped
   - VMs halted

Failure of any check blocks shutdown.

---

## Power Flapping Rule

- Once **Prepare** begins:
  - Shutdown continues **unless** power is stable (OL) for a defined stability window.
- Stability window **pauses progression** but does not auto-cancel.

---

## Cancellation Authority

- **Admin-only manual intervention** may cancel shutdown.
- ISAC may:
  - Report state
  - Explain consequences
  - Surface risks
- ISAC may **not** cancel, pause, or initiate shutdown.

---

## Backup Policy (Contextual, Not a Gate)

- Local backups are preferred before shutdown when time permits.
- Backup completion is **not** a hard gate for shutdown.
- Data integrity (ZFS) overrides backup completeness.

---

## Audit & Explainability

Every phase transition must be explainable using:
- UPS logs and snapshots
- Observability summaries
- Verification gate outcomes

---

## Explicit Non-Goals

- No automatic shutdown
- No timers
- No cron/systemd hooks
- No ZFS command execution
- No VM or container control
- No cloud interactions

---

## Future Activation (Out of Scope)

Any move from DESIGN-ONLY to execution requires:
- Explicit Admin authorization
- Separate review and approval
- New IRR stage
- New safety review

