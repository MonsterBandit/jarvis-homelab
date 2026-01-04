# Jarvis / ISAC Homelab

This repository contains the Jarvis / ISAC homelab system: a deliberately designed,
human-in-the-loop personal infrastructure platform.

The system integrates inventory, scanning, finance awareness, automation, and
observability under explicit authority and phase governance.

## Governance (Authoritative)

All authority, behavioral constraints, phase rules, and operating contracts
live under:

/governance/

Start here:
- /governance/README.md

Governance documents are authoritative over all code and services.
No runtime behavior may exceed or bypass governance constraints.

## Repository Structure (High Level)

- brain/         ISAC backend logic and APIs
- brain-data/    Persistent state (databases, finance, Grocy, IRR)
- data/          Frontend assets
- governance/    Canonical authority, contracts, runbooks
- archive/       Historical artifacts (non-active)

## Design Principles

- Slow, deliberate, human-in-the-loop evolution
- Observation before suggestion; suggestion before action
- Explicit authority ceilings with phase overlays
- No silent behavior changes

## Project Status

This system is under staged development.
Some phases are complete and locked; others are intentionally deferred.

Refer to governance documents for current phase status.
