import os
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional


# Finance behavior contract:
# See services/finance_contract.md (Phase 7, read-only, snapshot-backed)


FINANCE_SNAPSHOT_PATH = Path(
    os.getenv("FINANCE_SNAPSHOT_PATH", "/app/data/finance/snapshots/latest.json")
)


class FinanceSnapshotError(Exception):
    """Raised when the finance snapshot cannot be read or is invalid."""


def read_finance_snapshot() -> Dict[str, Any]:
    """
    Read the latest finance snapshot from disk.

    Returns the raw parsed JSON structure.
    """
    if not FINANCE_SNAPSHOT_PATH.exists():
        raise FinanceSnapshotError(f"Finance snapshot not found at {FINANCE_SNAPSHOT_PATH}")

    try:
        with FINANCE_SNAPSHOT_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise FinanceSnapshotError("Finance snapshot is not valid JSON") from e


def summarize_finances() -> Dict[str, Any]:
    """
    Produce a minimal, read-only summary of financial state.

    This function performs NO writes and NO external calls.
    """
    snapshot = read_finance_snapshot()

    accounts = snapshot.get("accounts", [])
    transactions = snapshot.get("transactions", {}).get("data", [])

    balances = []
    total_balance = 0.0

    for acct in accounts:
        bal = float(acct.get("current_balance", 0.0))
        balances.append(
            {
                "id": acct.get("id"),
                "name": acct.get("name"),
                "balance": bal,
                "currency": acct.get("currency"),
            }
        )
        total_balance += bal

    inflow = 0.0
    outflow = 0.0

    for tx in transactions:
        for line in tx.get("attributes", {}).get("transactions", []):
            amount = float(line.get("amount", 0.0))
            if amount > 0:
                inflow += amount
            elif amount < 0:
                outflow += abs(amount)

    return {
        "meta": snapshot.get("meta", {}),
        "accounts": balances,
        "totals": {
            "balance": round(total_balance, 2),
            "inflow": round(inflow, 2),
            "outflow": round(outflow, 2),
        },
    }


def _parse_date_to_utc(date_str: Optional[str]) -> Optional[datetime]:
    """
    Best-effort parse for Firefly-ish dates.
    Accepts YYYY-MM-DD or ISO timestamps. Returns tz-aware UTC dt if possible.
    """
    if not date_str:
        return None
    s = str(date_str).strip()
    if not s:
        return None

    # Common case: "YYYY-MM-DD"
    try:
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            return dt
    except Exception:
        pass

    # ISO-ish
    try:
        # Handle trailing Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def list_transactions(days: int = 30, limit: int = 200, offset: int = 0) -> Dict[str, Any]:
    """
    Read-only transaction listing derived from the snapshot on disk.

    - Windowed by "days" (server UTC, best-effort date parse)
    - Paged by offset/limit (after filtering + sorting)
    - No external calls. No writes.
    """
    snapshot = read_finance_snapshot()
    tx_groups = snapshot.get("transactions", {}).get("data", [])

    now_utc = datetime.now(timezone.utc)
    if days < 1:
        days = 1
    if days > 365:
        days = 365

    cutoff = now_utc - timedelta(days=int(days))

    flat: List[Dict[str, Any]] = []

    for group in tx_groups:
        group_id = group.get("id")
        attrs = group.get("attributes", {}) if isinstance(group, dict) else {}
        lines = attrs.get("transactions", []) if isinstance(attrs, dict) else []
        if not isinstance(lines, list):
            continue

        for line in lines:
            if not isinstance(line, dict):
                continue

            dt = _parse_date_to_utc(line.get("date"))
            if dt is not None and dt < cutoff:
                continue

            flat.append(
                {
                    "group_id": group_id,
                    "date": line.get("date"),
                    "amount": line.get("amount"),
                    "currency_code": line.get("currency_code"),
                    "description": line.get("description"),
                    "type": line.get("type"),
                    "source_id": line.get("source_id"),
                    "source_name": line.get("source_name"),
                    "destination_id": line.get("destination_id"),
                    "destination_name": line.get("destination_name"),
                    "category_name": line.get("category_name"),
                    "budget_name": line.get("budget_name"),
                }
            )

    def _sort_key(item: Dict[str, Any]) -> str:
        # Sort newest-first using the raw date string when possible (ISO/YYYY-MM-DD sorts nicely)
        return str(item.get("date") or "")

    flat.sort(key=_sort_key, reverse=True)

    total = len(flat)

    if offset < 0:
        offset = 0
    if limit < 1:
        limit = 1
    if limit > 1000:
        limit = 1000

    page = flat[offset : offset + limit]

    return {
        "meta": snapshot.get("meta", {}),
        "window": {
            "days": int(days),
            "cutoff_utc": cutoff.isoformat(),
            "generated_at_utc": now_utc.isoformat(),
        },
        "paging": {
            "offset": int(offset),
            "limit": int(limit),
            "returned": len(page),
            "total_after_filter": total,
        },
        "transactions": page,
        "notes": [
            "Read-only snapshot-derived listing.",
            "No Firefly API calls. No writes.",
            "Sorted newest-first by date string; parsing is best-effort for window filtering.",
        ],
    }
