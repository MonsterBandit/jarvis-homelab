#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

# Pull admin key from the running container env if not provided.
ADMIN_KEY="${ADMIN_KEY:-}"
if [[ -z "${ADMIN_KEY}" ]]; then
  ADMIN_KEY="$(
    docker inspect jarvis-brain --format '{{range .Config.Env}}{{println .}}{{end}}' \
      | awk -F= '/^ISAC_ADMIN_API_KEY=/{print substr($0,index($0,"=")+1)}' \
      | tail -n 1
  )"
fi

if [[ -z "${ADMIN_KEY}" ]]; then
  echo "ERROR: ADMIN_KEY is empty. Export ADMIN_KEY=... or ensure jarvis-brain has ISAC_ADMIN_API_KEY set."
  exit 1
fi

H_AUTH=(-H "X-ISAC-ADMIN-KEY: ${ADMIN_KEY}")
H_JSON=(-H "Content-Type: application/json")

tmp_body="$(mktemp)"
cleanup() { rm -f "${tmp_body}"; }
trap cleanup EXIT

request() {
  # Usage: request METHOD URL [DATA_JSON]
  local method="$1"
  local url="$2"
  local data="${3:-}"

  if [[ -n "${data}" ]]; then
    curl -sS -o "${tmp_body}" -w "%{http_code}" -X "${method}" "${url}" \
      "${H_AUTH[@]}" "${H_JSON[@]}" \
      -d "${data}"
  else
    curl -sS -o "${tmp_body}" -w "%{http_code}" -X "${method}" "${url}" \
      "${H_AUTH[@]}"
  fi
}

print_result() {
  local label="$1"
  local code="$2"
  echo "${label} -> HTTP ${code}"
  cat "${tmp_body}" || true
  echo
}

expect_ok_or_conflict() {
  # Treat 200/201/204 as OK, 409 as OK-ish (already exists), anything else = fail
  local label="$1"
  local code="$2"

  if [[ "${code}" == "200" || "${code}" == "201" || "${code}" == "204" ]]; then
    return 0
  fi

  if [[ "${code}" == "409" ]]; then
    echo "${label}: already exists (409) - OK for smoke run"
    return 0
  fi

  echo "FAIL: ${label} unexpected HTTP ${code}"
  cat "${tmp_body}" || true
  echo
  return 1
}

echo "== Health =="
code="$(request GET "${BASE_URL}/alice/memory/health")"
print_result "Health" "${code}"
expect_ok_or_conflict "Health" "${code}"
echo

echo "== Seed concept (idempotent-ish) =="
code="$(request POST "${BASE_URL}/alice/memory/concepts" '{
  "concept_key": "test.concept.seed.1",
  "display_name": "Test Concept Seed 1",
  "description": "Manually seeded concept to verify insert -> fetch round-trip.",
  "tags": "seed,test"
}')"
print_result "Seed concept" "${code}"
expect_ok_or_conflict "Seed concept" "${code}"
echo

echo "== List concepts =="
code="$(request GET "${BASE_URL}/alice/memory/concepts")"
print_result "List concepts" "${code}"
expect_ok_or_conflict "List concepts" "${code}"
echo

echo "== Seed alias for concept_id=1 user_id=admin (idempotent-ish) =="
code="$(request POST "${BASE_URL}/alice/memory/aliases" '{
  "concept_id": 1,
  "user_id": "admin",
  "preferred_name": "test concept",
  "confidence": 1.0,
  "locked": 1
}')"
print_result "Seed alias" "${code}"
expect_ok_or_conflict "Seed alias" "${code}"
echo

echo "== List aliases =="
code="$(request GET "${BASE_URL}/alice/memory/aliases")"
print_result "List aliases" "${code}"
expect_ok_or_conflict "List aliases" "${code}"
echo

echo "== List aliases by concept_id=1 =="
code="$(request GET "${BASE_URL}/alice/memory/aliases/by-concept/1")"
print_result "Aliases by concept" "${code}"
expect_ok_or_conflict "Aliases by concept" "${code}"
echo

echo "== List aliases by user_id=admin =="
code="$(request GET "${BASE_URL}/alice/memory/aliases/by-user/admin")"
print_result "Aliases by user" "${code}"
expect_ok_or_conflict "Aliases by user" "${code}"
echo

echo "OK: memory smoke completed"
