# CRTK v1 — Coding Reasoning Toolkit

**Status:** LAP Skeleton (Inert)

CRTK is the complete specialist coding toolkit expressed as:
- A manifest of coding tools/capabilities (inventory + contracts)
- Internal playbooks/rubrics/templates that produce proposals and verification plans (data only)
- Adapters that emit *request specs* for callable tools (without executing anything)

## Hard Boundaries (LOCKED during LAP)
- No tool execution (no calls into the toolbelt or execution spine)
- No file or network I/O
- No memory or identity learning
- No finance logic

## Relationship to Toolbelt
- Callable tools live under `/opt/jarvis/brain/tools/*` and are executed via the canonical execution spine.
- CRTK may describe or request those tools via non-executing request specs only.

## Contents (v1 suite)
- Manifest: `brain/toolkits/crtk/manifest.py`
- Entry: `brain/toolkits/crtk/crtk.py` (`crtk_propose`)
- Adapters (request specs only): `brain/toolkits/crtk/adapters/*`
- Playbooks/rubrics: `brain/toolkits/crtk/propose/*`, `review/*`, `verify/*`, `explain/*`
