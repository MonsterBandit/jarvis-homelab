"""Identity & Memory v1 package.

Contract:
- Canonical persistence in SQLite (JARVIS_DB_PATH / DB_PATH).
- Auto-provision users from X-ISAC-USER-ID (provisional by default).
- Explicit writes only. Tier 1 (safe self-profile) is non-admin; Tier 2 is admin-gated.
- All actions are auditable.
"""
