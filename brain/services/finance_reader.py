import os
import json
from pathlib import Path
from typing import Dict, Any


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
        raise FinanceSnapshotError(
            f"Finance snapshot not found at {FINANCE_SNAPSHOT_PATH}"
        )

    try:
        with FINANCE_SNAPSHOT_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise FinanceSnapshotError(
            "Finance snapshot is not valid JSON"
        ) from e


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
