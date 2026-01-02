#!/usr/bin/env bash
set -euo pipefail

# ---- Config ----
FINANCE_DIR="/opt/jarvis/brain-data/finance"
SNAP_DIR="${FINANCE_DIR}/snapshots"
LOG_DIR="${FINANCE_DIR}/logs"
ENV_FILE="${FINANCE_DIR}/firefly_readonly.env"

WINDOW_DAYS="${WINDOW_DAYS:-90}"     # Default window = last 90 days
ACCT_TYPES="${ACCT_TYPES:-asset}"    # Default = asset accounts

mkdir -p "$SNAP_DIR" "$LOG_DIR"

ts_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
date_utc="$(date -u +"%F")"
log_file="${LOG_DIR}/snapshot_refresh_${date_utc}.log"

echo "[$ts_utc] finance snapshot refresh starting" | tee -a "$log_file"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: $ENV_FILE" | tee -a "$log_file"
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

: "${FIREFLY_BASE_URL:?Missing FIREFLY_BASE_URL in ${ENV_FILE}}"
: "${FIREFLY_READONLY_API_TOKEN:?Missing FIREFLY_READONLY_API_TOKEN in ${ENV_FILE}}"

base="${FIREFLY_BASE_URL%/}"
token="$FIREFLY_READONLY_API_TOKEN"

start_date="$(date -u -d "${WINDOW_DAYS} days ago" +"%F")"
end_date="$(date -u +"%F")"

hdr_auth="Authorization: Bearer ${token}"
hdr_accept="Accept: application/json"

# ---- Fetch accounts (Firefly API) ----
# Firefly supports account listing; we filter to the needed types via query params.
accounts_json="$(curl -sS \
  -H "$hdr_auth" -H "$hdr_accept" \
  "${base}/api/v1/accounts?type=${ACCT_TYPES}" )"

# ---- Fetch transactions (Firefly API) ----
# Windowed fetch; page size at 500 to reduce pagination pressure.
transactions_json="$(curl -sS \
  -H "$hdr_auth" -H "$hdr_accept" \
  "${base}/api/v1/transactions?start=${start_date}&end=${end_date}&limit=500" )"

# ---- Transform accounts into the flat list our reader expects ----
# finance_reader.py expects:
#   snapshot["accounts"] = [{id,name,current_balance,currency}, ...]
# and:
#   snapshot["transactions"] = { "data": [...] } (Firefly-ish)
flat_accounts="$(jq -c '
  if (.data and (.data|type=="array")) then
    .data
    | map({
        id: .id,
        name: (.attributes.name // null),
        current_balance: (.attributes.current_balance // "0"),
        currency: (
          .attributes.currency_code
          // .attributes.currency_symbol
          // .attributes.currency_name
          // null
        )
      })
  else
    []
  end
' <<<"$accounts_json")"

tx_data_count="$(jq -r 'if .data and (.data|type=="array") then (.data|length) else 0 end' <<<"$transactions_json")"
acct_count="$(jq -r 'if .data and (.data|type=="array") then (.data|length) else 0 end' <<<"$accounts_json")"

meta="$(jq -c -n \
  --arg generated_at_utc "$ts_utc" \
  --arg base_url "$base" \
  --arg start "$start_date" \
  --arg end "$end_date" \
  --argjson account_count "$acct_count" \
  --argjson transaction_group_count "$tx_data_count" \
  '{
    generated_at_utc: $generated_at_utc,
    firefly_base_url: $base_url,
    window: { start: $start, end: $end, days: ( ($end | .) | 0 ) },  # days is informational only
    counts: { accounts: $account_count, transaction_groups: $transaction_group_count }
  }'
)"

# Build final snapshot
snapshot_tmp="$(mktemp "${SNAP_DIR}/.snapshot_tmp.XXXXXX.json")"
snapshot_name="firefly_snapshot_${date_utc}.json"
snapshot_path="${SNAP_DIR}/${snapshot_name}"

# Write large JSON blobs to temp files to avoid argv limits
meta_tmp="$(mktemp)"
accounts_tmp="$(mktemp)"
transactions_tmp="$(mktemp)"
  
echo "$meta" > "$meta_tmp"
echo "$flat_accounts" > "$accounts_tmp"
echo "$transactions_json" > "$transactions_tmp"
  
jq -n \
  --slurpfile meta "$meta_tmp" \
  --slurpfile accounts "$accounts_tmp" \
  --slurpfile transactions "$transactions_tmp" \
  '{
    meta: $meta[0],
    accounts: $accounts[0],
    transactions: $transactions[0]
  }' > "$snapshot_tmp"
  
rm -f "$meta_tmp" "$accounts_tmp" "$transactions_tmp"
  

# Validate JSON before publishing
jq -e . "$snapshot_tmp" >/dev/null

# Atomic publish: move temp file into place, then update latest.json symlink
mv -f "$snapshot_tmp" "$snapshot_path"
ln -sfn "$snapshot_name" "${SNAP_DIR}/latest.json"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] wrote snapshot: $snapshot_path" | tee -a "$log_file"
echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] updated symlink: ${SNAP_DIR}/latest.json -> $snapshot_name" | tee -a "$log_file"
echo "done" | tee -a "$log_file"
