# Grocy Snapshot Layer (Read-Only) — Contract

Phase: 6.4 execution (Grocy parity, read-only ingestion)
Status: Live (systemd timers + snapshots on disk)

## Purpose
ISAC consumes Grocy data strictly from local snapshots written on a schedule.
Grocy remains the system of record. ISAC performs no writes and no reconciliation.

This snapshot layer exists to:
- enable stable read-only analysis
- support Gate D observability
- prepare later phases (6.45 scan-driven product knowledge, 6.75 recipes, 6.95 routines)

## Instances (Dual-household)
Two separate snapshot tracks exist:

### Home A
- Base URL: `GROCY_HOME_A_BASE_URL` (from `/opt/jarvis/.env`)
- Snapshots:
  - Host: `/opt/jarvis/brain-data/grocy/home_a/snapshots/latest.json`
  - Container: `/app/data/grocy/home_a/snapshots/latest.json`

### Home B
- Base URL: `GROCY_HOME_B_BASE_URL` (from `/opt/jarvis/.env`)
- Snapshots:
  - Host: `/opt/jarvis/brain-data/grocy/home_b/snapshots/latest.json`
  - Container: `/app/data/grocy/home_b/snapshots/latest.json`

Each `latest.json` is a symlink to:
`grocy_snapshot_YYYY-MM-DD.json`

## Snapshot contents (raw, read-only)
Snapshot file includes:
- `meta` (home, generated_at_utc, base_url, notes)
- `system_info` (Grocy system info)
- `products` (raw `/objects/products`)
- `product_groups` (raw `/objects/product_groups`)
- `locations` (raw `/objects/locations`)
- `quantity_units` (raw `/objects/quantity_units`)
- `quantity_unit_conversions` (raw `/objects/quantity_unit_conversions`)
- `stock` (raw `/stock`)

**Important:** `/stock` is an array (not an object). ISAC should treat it as a list of entries.

## Refresh mechanism
Script:
- `/opt/jarvis/brain/services/grocy_snapshot_refresh.sh`
- Parameter: `home_a` or `home_b`
- Writes JSON via temp file + atomic move
- Updates `latest.json` via atomic symlink replacement
- Logs:
  - `/opt/jarvis/brain-data/grocy/logs/grocy_snapshot_home_a.log`
  - `/opt/jarvis/brain-data/grocy/logs/grocy_snapshot_home_b.log`

Systemd:
- Service (templated): `/etc/systemd/system/isac-grocy-snapshot@.service`
- Timer (templated): `/etc/systemd/system/isac-grocy-snapshot@.timer`
- Enabled:
  - `isac-grocy-snapshot@home_a.timer`
  - `isac-grocy-snapshot@home_b.timer`

## Non-goals / Guardrails
- No Grocy writes (no consumption, no purchases, no product creation)
- No reconciliation / deduping between households
- No automation (Barcode Buddy integration deferred)
- No “smart” inference that alters Grocy data

## Verification
Example jq sanity:
```sh
jq -r '.meta.home, .meta.generated_at_utc, (.products|length), (.stock|length)' \
  /opt/jarvis/brain-data/grocy/home_a/snapshots/latest.json
