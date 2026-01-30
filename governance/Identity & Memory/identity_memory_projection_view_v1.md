# Identity & Memory — Human-Legible Projection View v1

**Status:** LOCKED  
**Authority:** Governance (Absolute)  
**Activation:** ❌ Inert (No execution authorized)  
**Canonical Storage:** Structured memory (database)  
**Filesystem Role:** Projection / Export / Inspection only  

---

## 1. Purpose

This document defines the **authoritative human-legible filesystem projection** of Identity & Memory for the Jarvis / ISAC / Alice system.

This projection exists to:

- Provide a human-inspectable representation of memory
- Enable deterministic export and migration in the future
- Prevent ambiguity between governance, memory, and runtime state
- Allow Alice to reason about filesystem-based memory safely when explicitly authorized

This document **does not authorize execution**, memory activation, or filesystem writes.

---

## 2. Projection vs Canonical Storage (Non-Negotiable)

- The filesystem layout described here is a **projection**, not canonical storage.
- Canonical memory storage remains **structured memory** as defined in:
  - identity_memory_design_v1.md
  - identity_memory_api_surface_v1.md
  - identity_memory_v1.sql
- No filesystem path described here may be treated as authoritative unless explicitly migrated by a future, approved plan.

Until such migration:
- No `/memory/` directory exists
- No `.md` files are written
- No filesystem reads are permitted

---

## 3. Human-Legible Memory Tree (Verbatim)

```
memory/
├── users/
│   ├── tim/
│   │   ├── core/
│   │   │   ├── personal_preferences.md
│   │   │   ├── communication_style.md
│   │   │   ├── reasoning_patterns.md
│   │   │   ├── decision_thresholds.md
│   │   │   └── project_understanding.md
│   │   │
│   │   ├── domains/
│   │   │   ├── finance/
│   │   │   │   ├── naming_conventions.md
│   │   │   │   ├── categorization_logic.md
│   │   │   │   ├── reconciliation_standards.md
│   │   │   │   ├── rule_patterns.md
│   │   │   │   ├── automation_preferences.md
│   │   │   │   └── open_questions.md
│   │   │
│   │   │   ├── grocy/
│   │   │   │   ├── product_naming.md
│   │   │   │   ├── unit_preferences.md
│   │   │   │   ├── location_conventions.md
│   │   │   │   └── workflow_preferences.md
│   │   │
│   │   │   ├── homeassistant/
│   │   │   │   ├── entity_naming.md
│   │   │   │   ├── room_mappings.md
│   │   │   │   ├── automation_style.md
│   │   │   │   └── safety_constraints.md
│   │   │
│   │   │   ├── myfin/
│   │   │   │   ├── goal_definitions.md
│   │   │   │   ├── budgeting_style.md
│   │   │   │   └── planning_preferences.md
│   │   │
│   │   │   └── future_domains/
│   │   │
│   │   └── relationships/
│   │       ├── kaitlyn.md
│   │       ├── shared_finance_context.md
│   │       └── household_dynamics.md
│   │
│   ├── kaitlyn/
│   │   ├── core/
│   │   │   ├── personal_preferences.md
│   │   │   ├── communication_style.md
│   │   │   └── reasoning_patterns.md
│   │   │
│   │   ├── domains/
│   │   │   ├── finance/
│   │   │   ├── grocy/
│   │   │   └── other_domains/
│   │   │
│   │   └── relationships/
│   │       ├── tim.md
│   │       └── shared_context.md
│   │
│   └── shared_patterns/
│       ├── finance_shared.md
│       ├── grocy_shared.md
│       ├── household_naming.md
│       └── cross_domain_conventions.md
│
├── alice/
│   ├── identity/
│   │   ├── self_definition.md
│   │   ├── voice_and_tone.md
│   │   ├── behavioral_contract.md
│   │   ├── refusal_and_pause_rules.md
│   │   └── trust_posture.md
│   │
│   ├── system/
│   │   ├── architecture_map.md
│   │   ├── file_structure_current.md
│   │   ├── file_structure_target.md
│   │   ├── integration_patterns.md
│   │   ├── service_boundaries.md
│   │   └── runtime_assumptions.md
│   │
│   ├── governance/
│   │   ├── autonomy_rules.md
│   │   ├── write_leash_protocol.md
│   │   ├── fix_it_protocol.md
│   │   ├── memory_rules.md
│   │   ├── okd_execution_rules.md
│   │   └── pre_go_live_state.md
│   │
│   └── developer_knowledge/
│       ├── firefly/
│       ├── grocy/
│       ├── homeassistant/
│       └── future_systems/
│
└── scratch/
    ├── session_only/
    ├── hypotheses/
    └── discarded_thoughts/
```

---

## 4. Mapping to Canonical Reality

### 4.1 Governance vs Memory

The following paths map to **governance**, not memory:

- memory/alice/identity/*
- memory/alice/system/*
- memory/alice/governance/*

These correspond to documents under:

```
/opt/jarvis/governance/
```

They must never be learned, overwritten, or deleted by Alice.

---

### 4.2 User Memory Mapping

Paths under:

- memory/users/<user>/core/*
- memory/users/<user>/domains/*

Map to structured memory entries:

- users.id ← <user>
- memory_entries.type ← relative path
- memory_entries.value_text ← file contents
- memory_entries.tier ← Tier 1 or Tier 2 (per design)

Filesystem export is a pure projection.

---

### 4.3 Relationship Mapping

Paths under:

- memory/users/<user>/relationships/*
- memory/users/shared_patterns/*

Map to relationship-scoped memory, not individual user memory.

No relationship files may be created without:
- Explicit relationship entity
- Explicit admin authorization
- Explicit scope definition

---

### 4.4 Scratch Mapping

Paths under:

- memory/scratch/*

Map to:
- Tier 0 session memory
- Sandbox reasoning artifacts
- Explicitly non-promotable content

Scratch is never exported unless explicitly requested.

---

## 5. Export / Import Semantics (Future)

- Export is one-way first
- Import requires validation, diff preview, and explicit admin approval
- No automatic synchronization is permitted

Filesystem projection must always be:
- auditable
- reversible
- non-authoritative unless explicitly migrated

---

## 6. Safety Constraints

- Alice may read this document
- Alice may propose actions based on it
- Alice may never execute filesystem changes unless:
  - Memory is active
  - Governed write capability is enabled
  - Admin authorization is explicit

Violation constitutes drift.

---

## 7. Change Control

This document is versioned and locked.

Changes require:
- Explicit Admin approval
- Version bump
- Migration and rollback plan

---

**End of Identity & Memory — Human-Legible Projection View v1**
