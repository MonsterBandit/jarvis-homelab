#Scan-Driven Product Knowledge

**Status:** CANONICAL

This document governs how products are introduced into Grocy during Phase 6.45. It is intentionally conservative, conversational, and human-led.

---

## 1. Purpose

Phase 6.45 exists to establish **clean, durable product identity**.

It explicitly does **not** attempt to optimize, enrich, or automate. Those concerns are deferred to later phases.

This phase answers only one question:

> **“What is this thing?”**

---

## 2. Scope

This governance applies to:

* Barcode scanning
* Manual product creation triggered by scans
* Quantity units and locations created to satisfy schema requirements
* ISAC involvement during ingestion

This governance does **not** apply to:

* Pricing
* Stores
* Shopping optimization
* Automation
* Recipes (beyond identity compatibility)

---

## 3. Core Principles (Non‑Negotiable)

1. **Products represent consumable units**

   * Not retail packaging
   * Not bundles
   * Not purchase events

2. **Humans are authoritative**

   * Lookups are advisory
   * ISAC proposes, never assumes

3. **Conversation before compression**

   * Slowness is a feature
   * Repetition is training data

4. **No enrichment during scan**

   * Identity first
   * Semantics later

5. **Consistency over realism**

   * Avoid premature unit algebra
   * Prefer reversible decisions

---

## 4. Scan Sessions (Location‑Scoped)

### Definition

A **Scan Session** is bounded by:

* One physical location
* Multiple products
* Shared context

Examples:

* Pantry
* Under Sink Cabinet
* Utility Closet
* Garage Storage

### Session Rules

* One location per session
* All unresolved barcodes belong to that location
* ISAC may assume location unless ambiguity arises
* All items must be resolved before session ends
* Logs may be cleared **only after** session completion

---

## 5. Canonical Scan Workflow

### Step 1 — Scan

* Phone app in **Purchase** mode
* Scan once
* Re‑scan only if barcode capture fails

### Step 2 — Observe Placement

* **Processed Barcodes** → already known, no action
* **New Barcodes** → normal path
* **Unknown Barcodes** → normal path

Do not manually remove rows.

### Step 3 — Lookup (Optional, Observational)

* “Search for barcode” may be used
* Results are never binding
* Ignore pack size, bundles, and marketing phrasing

### Step 4 — Decide Product Identity (Human‑Led)

Questions answered conversationally:

* What is this physically?
* What unit do I consume?
* Does form matter?

Rules:

* One product per physical form
* No pack size in product name
* Brand included only if meaningful

### Step 5 — Create Product (Minimal Schema)

**Allowed fields:**

* Product name
* Quantity unit (uniform across stock/purchase/consume/price)
* Default location (bulk)
* Default consume location (if relevant)
* Active = true

**Forbidden during Phase 6.45:**

* Tags
* Product groups
* Parent products
* Unit conversions
* Best‑before logic
* Automation flags
* Store or price data

### Step 6 — Return to BarcodeBuddy

* Confirm product dropdown populated
* Do nothing to resolved rows
* Future scans should land in **Processed**

Repeat until session complete.

---

## 6. ISAC Ingestion Authority — V1

### Allowed (with explicit confirmation)

* Create products
* Create base quantity units
* Create locations
* Set default location and consume location

All writes must be:

* Narrated
* Confirmed
* Reversible

### Explicitly Forbidden

* Unit conversions
* Product groups
* Parent products
* Tags
* Automation
* Stock movement
* Silent edits or merges

---

## 7. Conversational Requirement

* ISAC must explain reasoning
* ISAC must ask when meaning is at risk
* Compression is earned, never assumed
* Uncertainty requires more conversation, not less

---

## 8. Readiness Gate — “Ready to Scan”

Phase 6.45 is considered **Ready to Scan** when:

* Quantity units exist for common consumables
* Neutral or real locations exist
* Barcode scanning reliably produces New/Unknown rows
* Product creation round‑trips cleanly back to BarcodeBuddy

No scaling or bulk ingestion should occur until this gate is met.

---

## 9. Explicit Deferrals

The following are **out of scope** for Phase 6.45 and must not be introduced prematurely:

* Stores
* Prices
* Cost optimization
* Shopping strategies

These are deferred to **Phase 7 — Price & Store Intelligence (Maturity Phase)**.

---

## 10. Canonical Summary

> Phase 6.45 teaches the system what things *are*.
> It deliberately avoids teaching what things *cost* or *optimize*.
> Those concerns are deferred until trust and data maturity exist.
