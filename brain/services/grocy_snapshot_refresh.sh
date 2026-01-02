#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   grocy_snapshot_refresh.sh home_a
#   grocy_snapshot_refresh.sh home_b

HOME="${1:-}"
if [[ "$HOME" != "home_a" && "$HOME" != "home_b" ]]; then
  echo "Usage: $0 home_a|home_b" >&2
  exit 2
fi

DATA_ROOT="/opt/jarvis/brain-data/grocy"
SNAP_DIR="${DATA_ROOT}/${HOME}/snapshots"
LOG_DIR="${DATA_ROOT}/logs"
LOG_FILE="${LOG_DIR}/grocy_snapshot_${HOME}.log"

mkdir -p "$SNAP_DIR" "$LOG_DIR"

ts_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
date_utc="$(date -u +"%Y-%m-%d")"
snapshot_name="grocy_snapshot_${date_utc}.json"
snapshot_path="${SNAP_DIR}/${snapshot_name}"
tmp_file="$(mktemp "${SNAP_DIR}/.snapshot_tmp.XXXXXX.json")"

echo "[${ts_utc}] grocy snapshot refresh starting (${HOME})" | tee -a "$LOG_FILE"

# Pull Grocy config from env (do not echo keys)
# Expect env vars:
#   GROCY_HOME_A_BASE_URL / GROCY_HOME_A_API_KEY
#   GROCY_HOME_B_BASE_URL / GROCY_HOME_B_API_KEY
if [[ "$HOME" == "home_a" ]]; then
  : "${GROCY_HOME_A_BASE_URL:?Missing GROCY_HOME_A_BASE_URL}"
  : "${GROCY_HOME_A_API_KEY:?Missing GROCY_HOME_A_API_KEY}"
  BASE_URL="$GROCY_HOME_A_BASE_URL"
  API_KEY="$GROCY_HOME_A_API_KEY"
else
  : "${GROCY_HOME_B_BASE_URL:?Missing GROCY_HOME_B_BASE_URL}"
  : "${GROCY_HOME_B_API_KEY:?Missing GROCY_HOME_B_API_KEY}"
  BASE_URL="$GROCY_HOME_B_BASE_URL"
  API_KEY="$GROCY_HOME_B_API_KEY"
fi

# Helper: curl Grocy API
grocy_get() {
  local path="$1"
  curl -sS \
    -H "GROCY-API-KEY: ${API_KEY}" \
    -H "Accept: application/json" \
    "${BASE_URL%/}/api${path}"
}

# Fetch core datasets (read-only)
# Keep it minimal but useful for inventory normalization.
system_info_json="$(grocy_get "/system/info")"
products_json="$(grocy_get "/objects/products")"
product_groups_json="$(grocy_get "/objects/product_groups")"
locations_json="$(grocy_get "/objects/locations")"
quantity_units_json="$(grocy_get "/objects/quantity_units")"
quantity_unit_conversions_json="$(grocy_get "/objects/quantity_unit_conversions")"
stock_json="$(grocy_get "/stock")"

# Compose snapshot safely (avoid argv limits: write blobs to temp files, then slurpfile)
meta_tmp="$(mktemp)"
sys_tmp="$(mktemp)"
products_tmp="$(mktemp)"
groups_tmp="$(mktemp)"
locations_tmp="$(mktemp)"
units_tmp="$(mktemp)"
conversions_tmp="$(mktemp)"
stock_tmp="$(mktemp)"

cat > "$meta_tmp" <<EOF
{
  "home": "${HOME}",
  "generated_at_utc": "${ts_utc}",
  "base_url": "${BASE_URL}",
  "notes": [
    "Read-only Grocy snapshot.",
    "No writes. No reconciliation. Grocy remains system of record."
  ]
}
EOF

echo "$system_info_json" > "$sys_tmp"
echo "$products_json" > "$products_tmp"
echo "$product_groups_json" > "$groups_tmp"
echo "$locations_json" > "$locations_tmp"
echo "$quantity_units_json" > "$units_tmp"
echo "$quantity_unit_conversions_json" > "$conversions_tmp"
echo "$stock_json" > "$stock_tmp"

jq -n \
  --slurpfile meta "$meta_tmp" \
  --slurpfile system_info "$sys_tmp" \
  --slurpfile products "$products_tmp" \
  --slurpfile product_groups "$groups_tmp" \
  --slurpfile locations "$locations_tmp" \
  --slurpfile quantity_units "$units_tmp" \
  --slurpfile quantity_unit_conversions "$conversions_tmp" \
  --slurpfile stock "$stock_tmp" \
  '{
    meta: $meta[0],
    system_info: $system_info[0],
    products: $products[0],
    product_groups: $product_groups[0],
    locations: $locations[0],
    quantity_units: $quantity_units[0],
    quantity_unit_conversions: $quantity_unit_conversions[0],
    stock: $stock[0]
  }' > "$tmp_file"

rm -f "$meta_tmp" "$sys_tmp" "$products_tmp" "$groups_tmp" "$locations_tmp" "$units_tmp" "$conversions_tmp" "$stock_tmp"

# Validate JSON before publishing
jq -e . "$tmp_file" >/dev/null

# Atomic publish
mv -f "$tmp_file" "$snapshot_path"
ln -sfn "$snapshot_name" "${SNAP_DIR}/latest.json"

# Quick counts (for log only)
prod_count="$(jq -r '.products | length' "$snapshot_path" 2>/dev/null || echo "0")"
stock_count="$(jq -r '.stock.products | length' "$snapshot_path" 2>/dev/null || echo "0")"

echo "[${ts_utc}] wrote snapshot: ${snapshot_path} (products=${prod_count} stock_items=${stock_count})" | tee -a "$LOG_FILE"
echo "[${ts_utc}] updated symlink: ${SNAP_DIR}/latest.json -> ${snapshot_name}" | tee -a "$LOG_FILE"
echo "done" | tee -a "$LOG_FILE"
