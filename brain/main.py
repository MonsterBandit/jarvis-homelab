from __future__ import annotations

import json
import re
import os
import base64
import hashlib
import tempfile
import socket
import sqlite3
import time
import secrets
import hmac
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

# Service start time (monotonic) for /verify uptime
START_TIME = time.monotonic()

import httpx
import psutil
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from model_client import (
    GeminiModelClient,
    ModelClientError,
    OpenAICompatibleClient,
)
from services.homeassistant import HomeAssistantClient, HomeAssistantConfig
from services.grocy import GrocyClient, GrocyError, create_grocy_client
from services.barcodebuddy import (
    BarcodeBuddyClient,
    BarcodeBuddyError,
    create_barcodebuddy_client,
)

# NEW: OpenFoodFacts (Phase 6.45 Step 2)
from services.openfoodfacts import (
    OpenFoodFactsError,
    create_openfoodfacts_client,
    extract_suggestion_from_off_payload,
)

from services.recipes import router as recipes_router  # Phase 6.75.1
from services.ingredient_parser import router as ingredient_parser_router  # Phase 6.75.2
from services.recipe_matcher import router as recipe_matcher_router  # Phase 6.75.3
from services.recipe_analyzer import router as recipe_analyzer_router  # Phase 6.75.4
from services.recipe_mappings import router as recipe_mappings_router  # Phase 6.75.5
from services.mealplans import router as mealplans_router
from services.mealplanner import router as mealplanner_context_router
# Alice (Phase 8) — read-only preview endpoint (NOT wired into /ask)
from alice_preview_router import router as alice_router

# --- Alice v2 Advance Gate Router (minimal, fail-closed) ---
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

alice_gate_router = APIRouter(prefix="/alice/gate", tags=["alice-gate"])

class AliceGateAdvanceRequest(BaseModel):
    msg: str
    ctx: List[Dict[str, Any]] = []
    mode: str = "normal"
    flags: Dict[str, Any] = {}

class AliceGateAdvanceResponse(BaseModel):
    decision: str
    reasons: List[str]
    question: Optional[str] = None
    # Bundle 1 (B1): Context Integrity & Continuity (Alice-only)
    integrity: Dict[str, Any] = {}
    # Bundle 5 (B5): Human Trust (5A) + Oversight (5B)
    # Optional and only present when decision is influenced by Bundle 5.
    bundle5: Optional[Dict[str, Any]] = None


# --- Semantic intent detection (backend-owned; no wake words) ---
_ELONGATION_RE = re.compile(r"(.)\1{2,}")  # 3+ repeats -> 2
_TRAIL_PUNCT_RE = re.compile(r"[.!?…]+$")

def _normalize_human_token(msg: str) -> str:
    s = (msg or "").lower().strip()
    # Remove trailing punctuation runs (e.g., "hmmm..." -> "hmmm")
    s = _TRAIL_PUNCT_RE.sub("", s)
    # Collapse elongations (e.g., "hmmmmmm" -> "hhmm" shape -> effectively "hmm")
    s = _ELONGATION_RE.sub(r"\1\1", s)
    return s

def classify_semantic_intent(msg: str, ctx: List[Dict[str, Any]]) -> str:
    raw = (msg or "").strip()
    s = _normalize_human_token(raw)

    if not s:
        return "ambient_noise"

    # Thinking-aloud / non-invitation tokens (robust to elongation)
    if re.fullmatch(r"(h+m+|u+m+|u+h+)", s):
        return "thinking_aloud"
    if re.fullmatch(r"[.]+", raw.strip()):
        return "thinking_aloud"

    # Simple ambient tests
    if s in {"ping", "test"}:
        return "ambient_test"

    # Clear questions
    if raw.endswith("?"):
        return "question"

    # Invitation / direct requests
    if re.search(r"\b(can you|could you|help me|please|explain|how do i|what is)\b", s):
        return "direct_request"

    # Continuations
    if re.search(r"\b(ok|so|right|then)\b", s):
        return "continuation"

    return "statement"

    
# ---------------------------------------------------------
# Bundle 1 (B1) — Context Integrity & Continuity (Alice-only)
# Pure logic: no I/O, no DB, no tools, no model calls.
# ---------------------------------------------------------

def _b1_stable_ctx_hash(ctx: Any) -> str:
    """Best-effort stable hash for ctx (internal diagnostics only)."""
    try:
        payload = json.dumps(ctx, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
    except Exception:
        return "sha256:unavailable"


def _b1_continuity_state(chat_id: str, ctx_len: int) -> tuple[str, List[str]]:
    """Return (state, notes) where state is present|suspect|missing."""
    notes: List[str] = []
    chat_present = bool((chat_id or "").strip()) and (chat_id != "unknown")
    if not chat_present:
        notes.append("B1_CHAT_ID_MISSING")
        if ctx_len <= 0:
            notes.append("B1_CTX_EMPTY")
        return ("missing", notes)

    if ctx_len <= 0:
        notes.append("B1_CTX_MISSING_FOR_CHAT")
        return ("suspect", notes)

    return ("present", notes)


def _b1_notes_to_reason_tags(notes: List[str]) -> List[str]:
    """Convert internal notes to the public reason tags we locked (Option B)."""
    tags: List[str] = []
    for n in notes:
        if n == "B1_CHAT_ID_MISSING":
            tags.append("B1_CONTINUITY:CHAT_ID_MISSING")
        elif n == "B1_CTX_MISSING_FOR_CHAT":
            tags.append("B1_CONTINUITY:CTX_EMPTY_FOR_CHAT")
        elif n == "B1_CTX_EMPTY":
            tags.append("B1_CONTINUITY:CTX_EMPTY")
    return tags


def _b1_integrity_payload(
    *,
    endpoint: str,
    user_id: str,
    chat_id: str,
    ctx_len: int,
    notes: List[str],
    intent_value: str,
    intent_source: str,
) -> Dict[str, Any]:
    return {
        "cco_version": "B1.v1",
        "endpoint": endpoint,
        "continuity_state": ("present" if not notes else ("suspect" if "B1_CTX_MISSING_FOR_CHAT" in notes else "missing"
                            if "B1_CHAT_ID_MISSING" in notes else "present")),
        "notes": list(notes),
        "intent": {"value": intent_value, "source": intent_source},
        "identity": {
            "user_id_present": bool((user_id or "").strip()) and (user_id != "unknown"),
            "chat_id_present": bool((chat_id or "").strip()) and (chat_id != "unknown"),
        },
        "conversation": {"ctx_len": int(ctx_len)},
    }


# ---------------------------------------------------------
# Bundle 5 (B5) — Human Trust & Ethics (5A) + Oversight & Control (5B)
# Design-locked. Implementation is pure logic only (no I/O, no DB, no tools, no model calls).
# Applied ONLY when the v2 behavioral gate would otherwise allow ADVANCE_OK.
# ---------------------------------------------------------

_B5_RE_SQUEEZE_WS = re.compile(r"\s+")

def _b5_norm(s: str) -> str:
    return _B5_RE_SQUEEZE_WS.sub(" ", (s or "").strip().lower())

def _b5_detect_uncertainty(msg: str) -> str:
    s = _b5_norm(msg)
    if not s:
        return "high"
    high_markers = [
        "i don't know", "idk", "not sure", "whatever", "guess", "wild guess", "no idea",
        "doesn't matter if it's wrong", "even if it's wrong",
    ]
    med_markers = ["maybe", "might", "could", "i think", "probably", "sort of", "kinda", "unsure"]
    if any(x in s for x in high_markers):
        return "high"
    if any(x in s for x in med_markers):
        return "medium"
    return "low"

def _b5_classify_impact_tier(msg: str, intent: str) -> int:
    """
    Conservative impact classifier.
    Tier 0: informational only
    Tier 1: reversible low impact
    Tier 2: durable / cross-user / authority-sensitive
    Tier 3: high impact / irreversible / money / deletion / execution surfaces
    """
    s = _b5_norm(msg)
    # Tier 3 triggers (very conservative)
    tier3_terms = [
        "delete", "remove", "wipe", "truncate", "reset", "factory reset",
        "execute", "run", "apply", "resume", "abort", "go live",
        "write file", "replace file", "diff apply", "patch",
        "money", "transfer", "payment", "bank", "account", "wire", "zelle", "venmo", "paypal",
        "tax", "irs", "loan", "credit", "debt",
    ]
    if any(t in s for t in tier3_terms):
        return 3

    # Tier 2 triggers (durable/authority)
    tier2_terms = [
        "remember", "store", "save this", "save that", "persist", "set preference",
        "change", "modify", "configure", "enable", "disable", "reconfigure",
        "affects both", "for both of us", "my spouse", "kaitlyn",
        "admin", "unlock", "token", "permission", "role",
    ]
    if any(t in s for t in tier2_terms):
        return 2

    # Questions default to Tier 0 unless keywords above were found
    if (msg or "").strip().endswith("?") or intent in {"question", "direct_request"}:
        return 0

    return 1

def bundle5_ethics_check(*, msg: str, ctx: List[Dict[str, Any]], intent: str) -> Dict[str, Any]:
    """
    Bundle 5A: Hard prohibitions + bounded judgment (pre-action posture only).
    This evaluates the USER REQUEST posture (not a generated answer).
    """
    s = _b5_norm(msg)
    violations: List[str] = []
    explanation: List[str] = []

    # Hard prohibitions (request-pattern based)
    if ("if i don't respond" in s) or ("take that as a yes" in s) or ("silence" in s and "consent" in s):
        violations.append("B5A_NO_CONSENT_BY_SILENCE")
        explanation.append("I can’t treat silence or non-response as consent.")

    if ("don't ask" in s and "question" in s) or ("no questions" in s) or ("just do it" in s and "don't ask" in s):
        violations.append("B5A_NO_URGENCY_AS_AUTHORITY")
        explanation.append('Urgency or "no questions" doesn’t create permission.')

    if ("side with me" in s) or ("ignore them" in s) or ("my spouse is wrong" in s):
        violations.append("B5A_NO_PARTISANSHIP_WITHOUT_MANDATE")
        explanation.append("I can’t take sides between people without an explicit mandate.")

    # High-stakes / money transfer phrasing is always refused at this stage (no execution here).
    money_terms = ["transfer", "move money", "send money", "payment", "wire", "bank account", "zelle", "venmo", "paypal"]
    if any(t in s for t in money_terms):
        violations.append("B5A_NO_FINANCIAL_EXECUTION")
        explanation.append("I can’t perform or direct financial actions here without explicit authority and a governed execution lane.")

    if violations:
        return {
            "status": "REFUSE",
            "violations": violations,
            "explanation": explanation or ["I can’t proceed with that request as stated."],
        }

    # No violations: PASS (bounded judgment remains possible downstream)
    return {"status": "PASS", "violations": [], "explanation": []}


def bundle5_oversight_check(
    *,
    msg: str,
    intent: str,
    admin_unlocked: bool,
) -> Dict[str, Any]:
    """
    Bundle 5B: Tiered deference (role + impact + uncertainty).
    Produces posture only; does not execute.
    """
    impact_tier = _b5_classify_impact_tier(msg, intent)
    uncertainty = _b5_detect_uncertainty(msg)

    # Tier 3 always requires explicit admin involvement.
    if impact_tier >= 3:
        if not admin_unlocked:
            return {
                "posture": "ESCALATE_ADMIN",
                "impact_tier": impact_tier,
                "uncertainty": uncertainty,
                "required_role": "admin",
                "explanation": [
                    "This is high-impact. I need Admin unlocked before I can continue.",
                    "If you want to proceed, unlock Admin and repeat the request.",
                ],
            }
        # Admin already unlocked: require explicit confirmation (pause).
        return {
            "posture": "PAUSE",
            "impact_tier": impact_tier,
            "uncertainty": uncertainty,
            "required_role": "admin",
            "question": "Admin is unlocked. Confirm you want to proceed with this high-impact request (yes/no)?",
            "explanation": [
                "This is high-impact, so I’m pausing for explicit confirmation.",
            ],
        }

    # Tier 2 requires admin if uncertainty is medium/high, or always if not unlocked.
    if impact_tier == 2:
        if not admin_unlocked:
            return {
                "posture": "ESCALATE_ADMIN",
                "impact_tier": impact_tier,
                "uncertainty": uncertainty,
                "required_role": "admin",
                "explanation": [
                    "This is durable or authority-sensitive. I need Admin unlocked before continuing.",
                    "Unlock Admin and repeat the request.",
                ],
            }
        if uncertainty in {"medium", "high"}:
            return {
                "posture": "PAUSE",
                "impact_tier": impact_tier,
                "uncertainty": uncertainty,
                "required_role": "admin",
                "question": "Before I continue, what exactly should change, and what should stay untouched?",
                "explanation": ["This affects durable state; I need a tighter instruction."],
            }
        return {
            "posture": "PROCEED",
            "impact_tier": impact_tier,
            "uncertainty": uncertainty,
            "required_role": "admin",
            "explanation": [],
        }

    # Tier 0–1: proceed unless uncertainty is high (then pause).
    if uncertainty == "high":
        return {
            "posture": "PAUSE",
            "impact_tier": impact_tier,
            "uncertainty": uncertainty,
            "required_role": "any",
            "question": "What outcome are you aiming for here?",
            "explanation": ["I’m not confident I understand your intent yet."],
        }

    return {
        "posture": "PROCEED",
        "impact_tier": impact_tier,
        "uncertainty": uncertainty,
        "required_role": "any",
        "explanation": [],
    }

@alice_gate_router.post("/advance", response_model=AliceGateAdvanceResponse)
def alice_gate_advance(request: Request, req: AliceGateAdvanceRequest) -> AliceGateAdvanceResponse:
    """
    Alice Gate Advance — Bundle 1 wrapped.
    - Preserves pure governance gate semantics
    - Adds Context Integrity & Continuity signals (Alice-only)
    """
    endpoint = "/alice/gate/advance"

    # Headers (identity + session continuity carriers)
    user_id = (request.headers.get("X-ISAC-USER-ID") or "unknown").strip() or "unknown"
    chat_id = (request.headers.get("X-ISAC-CHAT-ID") or "unknown").strip() or "unknown"

    msg = req.msg
    ctx = req.ctx or []
    flags = dict(req.flags or {})

    # --- Hard schema checks (fail closed, return RESTRAIN) ---
    hard_notes: List[str] = []
    if not isinstance(msg, str) or not msg.strip():
        hard_notes.append("B1_EMPTY_MSG")
    if not isinstance(ctx, list):
        hard_notes.append("B1_CTX_BAD_SHAPE")
    if not isinstance(flags, dict):
        hard_notes.append("B1_FLAGS_BAD_SHAPE")

    continuity_state, continuity_notes = _b1_continuity_state(chat_id, len(ctx) if isinstance(ctx, list) else 0)

    # Intent normalization (preserve existing behavior)
    intent_source = "provided"
    intent_value = str(flags.get("intent") or flags.get("semantic_intent") or "").strip().lower()
    if not intent_value:
        intent_source = "derived"
        try:
            intent_value = classify_semantic_intent(msg if isinstance(msg, str) else "", ctx if isinstance(ctx, list) else [])
        except Exception:
            intent_value = "unknown"
        flags["intent"] = intent_value

    # Build integrity payload (minimal, public)
    notes = list(hard_notes) + list(continuity_notes)
    integrity = _b1_integrity_payload(
        endpoint=endpoint,
        user_id=user_id,
        chat_id=chat_id,
        ctx_len=(len(ctx) if isinstance(ctx, list) else 0),
        notes=notes,
        intent_value=intent_value or "unknown",
        intent_source=intent_source,
    )

    # If schema failed, RESTRAIN immediately (A0)
    if hard_notes:
        reasons = list(hard_notes) + _b1_notes_to_reason_tags(continuity_notes)
        return AliceGateAdvanceResponse(
            decision="RESTRAIN",
            reasons=reasons,
            question=None,
            integrity=integrity,
        )

    # --- Governance gate call (unchanged) ---
    decision, reasons, question = advance_gate(
        msg=msg,
        ctx=ctx,
        mode=req.mode,
        flags=flags,
    )

    # decision is an Enum; serialize to string
    decision_str = decision.value if hasattr(decision, "value") else str(decision)

    # Option B (locked): duplicate integrity notes into reasons as B1_CONTINUITY:* tags
    reasons_out = list(reasons or []) + _b1_notes_to_reason_tags(continuity_notes)

    

    # --- Bundle 5 (B5): Ethics + Oversight (applies only when ADVANCE_OK) ---
    bundle5: Optional[Dict[str, Any]] = None
    if decision_str.upper() == "ADVANCE_OK":
        # Admin is considered unlocked for this request if either:
        # - UI provided X-ISAC-ADMIN-TOKEN (preferred), or
        # - UI provided X-ISAC-ADMIN-KEY (legacy/emergency)
        admin_unlocked = bool((request.headers.get("X-ISAC-ADMIN-TOKEN") or "").strip()) or bool(
            (request.headers.get("X-ISAC-ADMIN-KEY") or "").strip()
        )

        ethics = bundle5_ethics_check(msg=msg, ctx=ctx, intent=intent_value or "unknown")
        if ethics.get("status") == "REFUSE":
            bundle5 = {
                "ethics": "REFUSE",
                "oversight": "REFUSE",
                "impact_tier": _b5_classify_impact_tier(msg, intent_value or ""),
                "uncertainty": _b5_detect_uncertainty(msg),
                "explanation": ethics.get("explanation") or ["I can’t proceed with that request as stated."],
                "violations": ethics.get("violations") or [],
            }
            # Override to a first-class refusal (UI must render this without calling /ask)
            decision_str = "REFUSE"
            question = None
            reasons_out = reasons_out + ["B5A_REFUSE"]
        else:
            os_res = bundle5_oversight_check(
                msg=msg,
                intent=intent_value or "unknown",
                admin_unlocked=admin_unlocked,
            )
            posture = str(os_res.get("posture") or "PROCEED").upper()
            bundle5 = {
                "ethics": "PASS",
                "oversight": posture,
                "impact_tier": int(os_res.get("impact_tier") or 0),
                "uncertainty": str(os_res.get("uncertainty") or "low"),
                "required_role": str(os_res.get("required_role") or "any"),
                "explanation": os_res.get("explanation") or [],
            }

            if posture == "ESCALATE_ADMIN":
                decision_str = "ESCALATE_ADMIN"
                question = None
                reasons_out = reasons_out + ["B5B_ESCALATE_ADMIN"]
            elif posture == "PAUSE":
                decision_str = "PAUSE"
                # Prefer oversight question, else fall back to existing gate question
                question = (
                    str(os_res.get("question") or question or "").strip()
                    or "What would you like me to do with that?"
                )
                reasons_out = reasons_out + ["B5B_PAUSE"]
            elif posture == "REFUSE":
                decision_str = "REFUSE"
                question = None
                reasons_out = reasons_out + ["B5B_REFUSE"]
            else:
                # PROCEED: leave decision as ADVANCE_OK
                pass

    return AliceGateAdvanceResponse(
        decision=decision_str,
        reasons=reasons_out,
        question=question,
        integrity=integrity,
        bundle5=bundle5,
    )



from alice_gate import advance_gate, GateDecision
from alice.memory_api import router as alice_memory_router
from alice.name_resolution import resolve_alias_to_concept

from health_capture import capture_internal_health
from system_health_capture import capture_system_health
from services.irr_ups import router as irr_router
from services.irr_narrative import router as irr_narrative_router
from services.finance_reader import summarize_finances, list_transactions, FinanceSnapshotError
from tools.task_ledger import create_task, add_step, add_artifact, create_tool_call_task
from tools.executor import run_tool
from tools.registry import TOOL_DEFS, is_known_tool, is_tool_allowed
from tools.types import ToolRequest
from tools.okd.artifacts import store_plan as okd_store_plan, store_plan_approval as okd_store_plan_approval
from tools.okd.artifacts import load_latest_plan as okd_load_latest_plan, is_plan_approved as okd_is_plan_approved
from tools.okd.governed_observation import execute_governed_observation



# ---------------------------------------------------------
# Escape Hatch v2 — Evidence & Environment Assumptions
# (LOCKED: v2 only; no templates, no diff-apply yet)
# ---------------------------------------------------------

class FailureClass(str):
    """Failure classification enum (string-friendly)."""
    TRANSIENT = "transient"
    CONFIG = "config"
    GUARDRAIL = "guardrail"
    VERIFY = "verify"
    WRITE = "write"
    EXECUTION = "execution"
    UNKNOWN = "unknown"


class ExecutionLockError(RuntimeError):
    pass


class ExecutionLock:
    """
    Cross-process single-execution concurrency guard.

    Uses an atomic lockfile create (O_EXCL). This works across multiple workers.
    The lock is held for the duration of any *execute* path (e.g., /tasks/*/resume).
    """

    def __init__(self, lock_path: str = "/app/data/locks/execution.lock") -> None:
        self.lock_path = lock_path
        self.fd: Optional[int] = None

    def acquire(self) -> None:
        Path(self.lock_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            payload = {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            os.write(self.fd, json.dumps(payload).encode("utf-8"))
        except FileExistsError as exc:
            # Surface helpful context if possible
            info: Any = None
            try:
                info = Path(self.lock_path).read_text(encoding="utf-8")
            except Exception:
                info = None
            raise ExecutionLockError(
                f"Another execution is already in progress (lock exists at {self.lock_path}). "
                f"Lock info: {info}"
            ) from exc

    def release(self) -> None:
        try:
            if self.fd is not None:
                try:
                    os.close(self.fd)
                except Exception:
                    pass
            self.fd = None
            try:
                Path(self.lock_path).unlink(missing_ok=True)
            except Exception:
                pass
        finally:
            self.fd = None


def _safe_bool_env(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


ISAC_DRY_RUN_DEFAULT = _safe_bool_env("ISAC_DRY_RUN_DEFAULT", True)


def tooling_probe_snapshot() -> Dict[str, Any]:
    """
    Gather a deterministic, read-only environment/tooling snapshot.
    This is the v2 'assumptions artifact' used later by verify templates.
    """
    now_utc = datetime.now(timezone.utc).isoformat()

    def which(cmd: str) -> Optional[str]:
        from shutil import which as _which
        return _which(cmd)

    # Minimal, practical probes. No network. No runner. No DB writes here.
    snapshot: Dict[str, Any] = {
        "captured_at_utc": now_utc,
        "service": "isac-brain",
        "db_path": DB_PATH,
        "runner_base_url": JARVIS_RUNNER_BASE_URL,
        "python": {
            "executable": which("python3") or which("python"),
            "version": ".".join(map(str, tuple(__import__("sys").version_info)[:3])),
        },
        "tools": {
            "curl": which("curl"),
            "jq": which("jq"),
            "git": which("git"),
        },
        "paths": {
            "brain_dir_exists": Path("/opt/jarvis/brain").exists(),
            "ui_index_exists": Path("/opt/jarvis/data/index.html").exists(),
            "db_file_exists": Path(DB_PATH).exists(),
        },
        "env_presence": {
            # Do not include secrets, only presence.
            "JARVIS_API_KEY": bool(os.getenv("JARVIS_API_KEY")),
            "ISAC_ADMIN_PIN": bool(os.getenv("ISAC_ADMIN_PIN")),
            "ISAC_ADMIN_KEY": bool(os.getenv("ISAC_ADMIN_KEY") or os.getenv("ISAC_ADMIN_API_KEY")),
            "JARVIS_RUNNER_BASE_URL": bool(os.getenv("JARVIS_RUNNER_BASE_URL")),
            "HOMEASSISTANT_BASE_URL": bool(os.getenv("HOMEASSISTANT_BASE_URL")),
            "GROCY_HOME_A_BASE_URL": bool(os.getenv("GROCY_HOME_A_BASE_URL")),
            "BARCODEBUDDY_BASE_URL": bool(os.getenv("BARCODEBUDDY_BASE_URL")),
            "FIREFLY_BASE_URL": bool(os.getenv("FIREFLY_BASE_URL")),
        },
        "dry_run": {
            "default": bool(ISAC_DRY_RUN_DEFAULT),
        },
    }

    # Optional: cheap system facts (no errors if psutil unavailable)
    try:
        snapshot["system"] = {
            "hostname": socket.gethostname(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "mem_total_bytes": getattr(psutil.virtual_memory(), "total", None),
        }
    except Exception:
        snapshot["system"] = {"hostname": socket.gethostname()}

    return snapshot


def _failure_class_from_exception(exc: Exception) -> str:
    msg = str(exc).lower()
    if isinstance(exc, ExecutionLockError):
        return FailureClass.GUARDRAIL
    if "not configured" in msg or "missing" in msg:
        return FailureClass.CONFIG
    if "timeout" in msg or "unreachable" in msg:
        return FailureClass.TRANSIENT
    return FailureClass.EXECUTION

# ----------------------------
# Environment & configuration
# ----------------------------

DB_PATH = os.getenv("JARVIS_DB_PATH", "/app/data/jarvis_brain.db")

# Identity & Memory v1: expose DB path to submodules
# (read-only reference; actual schema is migrated via SQL)
ENABLE_DIFF_APPLY = os.getenv("ENABLE_DIFF_APPLY", "0").strip() in ("1","true","TRUE","yes","YES")


JARVIS_LLM_PROVIDER = os.getenv("JARVIS_LLM_PROVIDER", "openai-compatible")
JARVIS_LLM_MODEL = os.getenv("JARVIS_LLM_MODEL", "gpt-4.1-mini")

OPENAI_COMPATIBLE_API_BASE = os.getenv("OPENAI_COMPATIBLE_API_BASE", "").rstrip("/")
OPENAI_COMPATIBLE_API_KEY = os.getenv("OPENAI_COMPATIBLE_API_KEY", "")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-1.5-flash")

API_KEY = os.getenv("JARVIS_API_KEY")

HOMEASSISTANT_BASE_URL = os.getenv("HOMEASSISTANT_BASE_URL", "")
HOMEASSISTANT_TOKEN = os.getenv("HOMEASSISTANT_TOKEN", "")

JARVIS_RUNNER_BASE_URL = os.getenv("JARVIS_RUNNER_BASE_URL", "http://jarvis-runner:8080")

# ----------------------------
# Alice Admin Unlock (Execution Spine v1.1)
# ----------------------------
# PIN unlock issues a short-lived admin token.
# Browser never needs to store X-ISAC-ADMIN-KEY.
ISAC_ADMIN_PIN = (os.getenv("ISAC_ADMIN_PIN") or "").replace("\r", "").strip()

try:
    ISAC_ADMIN_TOKEN_TTL_SECONDS = int(
        (os.getenv("ISAC_ADMIN_TOKEN_TTL_SECONDS") or "3600").strip()
    )
except Exception:
    ISAC_ADMIN_TOKEN_TTL_SECONDS = 3600

# ----------------------------
# Database helpers
# ----------------------------


def init_db() -> None:
    """
    Initialize the SQLite database with required tables.
    """
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        cur = conn.cursor()

        # Conversation log
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                user_message TEXT NOT NULL,
                jarvis_answer TEXT NOT NULL
            )
            """
        )

        # Shopping lists for Phase 6.5
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS shopping_list (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                household TEXT NOT NULL,
                item_name TEXT NOT NULL,
                quantity TEXT,
                source TEXT,
                completed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_shopping_list_household_completed
            ON shopping_list (household, completed)
            """
        )

        # Recipe ingredient mappings (Phase 6.75.5)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS recipe_ingredient_mapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                household TEXT, -- NULL = global mapping
                ingredient_norm TEXT NOT NULL,
                ingredient_display TEXT NOT NULL,
                product_id INTEGER NOT NULL,
                product_name TEXT,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_recipe_ing_map_household_norm
            ON recipe_ingredient_mapping (household, ingredient_norm)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_recipe_ing_map_norm
            ON recipe_ingredient_mapping (ingredient_norm)
            """
        )

        # -------------------------------------------------
        # Phase 6.9 — Calendar ownership & mapping (READ-ONLY)
        # -------------------------------------------------
        #
        # CONTRACT (LOCKED):
        # - These tables map calendar entities to owners/households only.
        # - We do NOT store calendar events in SQLite.
        # - We do NOT write any calendar-derived read results to SQLite.
        # - Calendar awareness is consumptive/descriptive only (context, not control).
        #

        # Maps a person/owner to a household
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS calendar_owner_household (
                owner_key TEXT PRIMARY KEY,
                household TEXT NOT NULL CHECK (household IN ('home_a', 'home_b')),
                display_name TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        # Maps Home Assistant calendar entities to an owner
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS calendar_entity_owner (
                entity_id TEXT PRIMARY KEY,
                owner_key TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                label TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (owner_key)
                    REFERENCES calendar_owner_household (owner_key)
                    ON DELETE CASCADE
            )
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_calendar_entity_owner_owner
            ON calendar_entity_owner (owner_key)
            """
        )

        
        # --- Escape Hatch v2 additions (evidence, lineage, timing, failures) ---
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_lineage (
                task_id INTEGER PRIMARY KEY,
                parent_task_id INTEGER,
                trigger_reason TEXT,
                verify_failures INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_timing (
                task_id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                last_verify_at TEXT,
                last_verify_ok INTEGER,
                last_verify_detail TEXT
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS execution_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER,
                step_id INTEGER,
                failure_class TEXT NOT NULL,
                message TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        conn.commit()
    finally:
        conn.close()


def _db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def lineage_init(task_id: int, parent_task_id: Optional[int] = None, trigger_reason: Optional[str] = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = _db_connect()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO execution_lineage (task_id, parent_task_id, trigger_reason, verify_failures, created_at, updated_at)
            VALUES (?, ?, ?, COALESCE((SELECT verify_failures FROM execution_lineage WHERE task_id=?), 0), COALESCE((SELECT created_at FROM execution_lineage WHERE task_id=?), ?), ?)
            """,
            (task_id, parent_task_id, trigger_reason, task_id, task_id, now, now),
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO execution_timing (task_id, created_at)
            VALUES (?, ?)
            """,
            (task_id, now),
        )
        conn.commit()
    finally:
        conn.close()


def timing_mark_started(task_id: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = _db_connect()
    try:
        conn.execute(
            """
            UPDATE execution_timing
            SET started_at = COALESCE(started_at, ?)
            WHERE task_id = ?
            """,
            (now, task_id),
        )
        conn.execute(
            """
            UPDATE execution_lineage
            SET updated_at = ?
            WHERE task_id = ?
            """,
            (now, task_id),
        )
        conn.commit()
    finally:
        conn.close()


def timing_mark_finished(task_id: int) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = _db_connect()
    try:
        conn.execute(
            """
            UPDATE execution_timing
            SET finished_at = ?
            WHERE task_id = ?
            """,
            (now, task_id),
        )
        conn.execute(
            """
            UPDATE execution_lineage
            SET updated_at = ?
            WHERE task_id = ?
            """,
            (now, task_id),
        )
        conn.commit()
    finally:
        conn.close()


def failures_record(task_id: int, step_id: Optional[int], failure_class: str, message: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = _db_connect()
    try:
        conn.execute(
            """
            INSERT INTO execution_failures (task_id, step_id, failure_class, message, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (task_id, step_id, failure_class, message[:5000], now),
        )
        conn.commit()
    finally:
        conn.close()


def lineage_get_verify_failures(task_id: int) -> int:
    conn = _db_connect()
    try:
        row = conn.execute(
            "SELECT verify_failures FROM execution_lineage WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    finally:
        conn.close()


def lineage_inc_verify_failures(task_id: int) -> int:
    """Increment and return verify_failures for this task's lineage."""
    now = datetime.now(timezone.utc).isoformat()
    conn = _db_connect()
    try:
        conn.execute(
            """
            UPDATE execution_lineage
            SET verify_failures = COALESCE(verify_failures, 0) + 1,
                updated_at = ?
            WHERE task_id = ?
            """,
            (now, task_id),
        )
        conn.commit()
        return lineage_get_verify_failures(task_id)
    finally:
        conn.close()

def log_conversation(model: str, user_message: str, jarvis_answer: str) -> None:
    """
    Insert a single question/answer pair into the database.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO conversation_log (timestamp, model, user_message, jarvis_answer)
            VALUES (?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                model,
                user_message,
                jarvis_answer,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_recent_conversations(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Return the most recent N conversation entries, newest first.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT timestamp, model, user_message, jarvis_answer
            FROM conversation_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    items: List[Dict[str, Any]] = []
    for ts, model, user_msg, jarvis_msg in rows:
        items.append(
            {
                "timestamp": ts,
                "model": model,
                "user_message": user_msg,
                "jarvis_answer": jarvis_msg,
            }
        )
    return items


# Shopping list DB helpers
def add_shopping_list_item(
    household: str,
    item_name: str,
    quantity: Optional[str] = None,
    source: Optional[str] = None,
) -> int:
    """
    Insert a shopping list item and return its new ID.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO shopping_list (household, item_name, quantity, source, completed, created_at)
            VALUES (?, ?, ?, ?, 0, ?)
            """,
            (
                household,
                item_name,
                quantity,
                source,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def get_shopping_list_items(
    household: str,
    include_completed: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch shopping list items for a given household.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        if include_completed:
            cur.execute(
                """
                SELECT id, household, item_name, quantity, source, completed, created_at
                FROM shopping_list
                WHERE household = ?
                ORDER BY created_at ASC, id ASC
                """,
                (household,),
            )
        else:
            cur.execute(
                """
                SELECT id, household, item_name, quantity, source, completed, created_at
                FROM shopping_list
                WHERE household = ? AND completed = 0
                ORDER BY created_at ASC, id ASC
                """,
                (household,),
            )
        rows = cur.fetchall()
    finally:
        conn.close()

    items: List[Dict[str, Any]] = []
    for row in rows:
        (
            item_id,
            hh,
            name,
            quantity,
            source,
            completed,
            created_at,
        ) = row
        items.append(
            {
                "id": item_id,
                "household": hh,
                "name": name,
                "quantity": quantity,
                "source": source,
                "completed": bool(completed),
                "created_at": created_at,
            }
        )
    return items


def delete_shopping_list_item(household: str, item_id: int) -> bool:
    """
    Delete a single shopping list item for a given household.
    Returns True if something was deleted.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            DELETE FROM shopping_list
            WHERE household = ? AND id = ?
            """,
            (household, item_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def clear_shopping_list(household: str) -> int:
    """
    Delete all shopping list items for a given household.
    Returns number of rows deleted.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            DELETE FROM shopping_list
            WHERE household = ?
            """,
            (household,),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


# ---------------------------------------------------------
# Phase 6.9 — Calendar DB-read helpers (READ-ONLY)
# ---------------------------------------------------------

# Hard ignore list for calendar awareness reads (LOCKED)
# - Always ignored regardless of DB contents or HA discovery
CALENDAR_ENTITY_IGNORE: set[str] = {
    "calendar.kaitlyn",
    "calendar.laila",
    "calendar.apentalp_gmail_com",
}


def _normalize_calendar_household(household: Optional[str]) -> str:
    """
    Calendar awareness default = home_a.
    Only valid: home_a | home_b
    """
    h = (household or "home_a").strip().lower()
    if h not in {"home_a", "home_b"}:
        raise HTTPException(
            status_code=400,
            detail="household must be 'home_a' or 'home_b'",
        )
    return h


def get_enabled_calendar_entities_for_household(
    household: str,
) -> List[Dict[str, Any]]:
    """
    Return enabled calendar entities for a given household from DB.
    READ-ONLY. No DB writes.
    Excludes hard-ignored calendars.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                ceo.entity_id,
                ceo.owner_key,
                ceo.enabled,
                ceo.label,
                coh.household,
                coh.display_name
            FROM calendar_entity_owner ceo
            JOIN calendar_owner_household coh
                ON coh.owner_key = ceo.owner_key
            WHERE coh.household = ?
              AND ceo.enabled = 1
            ORDER BY ceo.entity_id ASC
            """,
            (household,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    results: List[Dict[str, Any]] = []
    for entity_id, owner_key, enabled, label, hh, display_name in rows:
        if not entity_id:
            continue
        if entity_id in CALENDAR_ENTITY_IGNORE:
            continue
        results.append(
            {
                "entity_id": entity_id,
                "owner_key": owner_key,
                "enabled": bool(enabled),
                "label": label,
                "household": hh,
                "owner_display_name": display_name,
            }
        )
    return results


# ---------------------------------------------------------
# Phase 6.9 — Calendar planning window helpers (pure logic)
# Server-time based, Sunday-first weeks, deterministic
# ---------------------------------------------------------

_SERVER_TZ: Optional[ZoneInfo] = None


def get_server_timezone() -> ZoneInfo:
    """
    Return a ZoneInfo representing the server's local timezone.
    Deterministic resolution order (no network, no HA, no DB):
      1) tzinfo key from datetime.now().astimezone()
      2) TZ environment variable if set
      3) Fallback to America/New_York (expected for this deployment)
    """
    global _SERVER_TZ

    if _SERVER_TZ is not None:
        return _SERVER_TZ

    # 1) Ask the OS what "local time" is, then attempt to recover a ZoneInfo key.
    try:
        local_tzinfo = datetime.now().astimezone().tzinfo
        tz_key = getattr(local_tzinfo, "key", None)
        if tz_key:
            _SERVER_TZ = ZoneInfo(str(tz_key))
            return _SERVER_TZ
    except Exception:
        pass

    # 2) TZ env var (common in containers)
    tz_env = (os.getenv("TZ") or "").strip()
    if tz_env:
        try:
            _SERVER_TZ = ZoneInfo(tz_env)
            return _SERVER_TZ
        except Exception:
            pass

    # 3) Locked expected fallback
    _SERVER_TZ = ZoneInfo("America/New_York")
    return _SERVER_TZ


def now_server(tz: Optional[ZoneInfo] = None) -> datetime:
    """
    Return current server-time as a timezone-aware datetime.
    """
    tz_final = tz or get_server_timezone()
    return datetime.now(tz_final)


def _ensure_server_aware(dt: datetime, tz: Optional[ZoneInfo] = None) -> datetime:
    """
    Normalize a datetime to server timezone.
    - If naive: interpret as server-local time (no guessing beyond that).
    - If aware: convert to server timezone.
    """
    tz_final = tz or get_server_timezone()
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz_final)
    return dt.astimezone(tz_final)


def week_start_sunday(dt: datetime, tz: Optional[ZoneInfo] = None) -> datetime:
    """
    Given any datetime, return the Sunday 00:00:00 of its week in server time.
    Week definition: Sunday -> Saturday.
    """
    dt_local = _ensure_server_aware(dt, tz=tz)

    # Python weekday(): Monday=0 ... Sunday=6
    # We want Sunday=0 ... Saturday=6
    days_since_sunday = (dt_local.weekday() + 1) % 7

    start_date = (dt_local - timedelta(days=days_since_sunday)).date()
    return datetime(
        year=start_date.year,
        month=start_date.month,
        day=start_date.day,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
        tzinfo=dt_local.tzinfo,
    )


def planning_window_week(
    offset_weeks: int = 0,
    anchor: Optional[datetime] = None,
    tz: Optional[ZoneInfo] = None,
) -> Tuple[datetime, datetime]:
    """
    Return a (start, end) tuple for a Sunday-first week window in server time.
    - start: Sunday 00:00:00
    - end: next Sunday 00:00:00 (end-exclusive)
    offset_weeks:
      0 = current week, 1 = next week, -1 = prior week, etc.
    anchor:
      If provided, the week is computed relative to that moment; otherwise uses now_server().
    """
    anchor_dt = anchor if anchor is not None else now_server(tz=tz)
    start = week_start_sunday(anchor_dt, tz=tz) + timedelta(weeks=int(offset_weeks))
    end = start + timedelta(days=7)
    return start, end
    

def planning_window_today(tz: ZoneInfo) -> Tuple[datetime, datetime]:
    """
    Return server-time 'today' window:
    [today 00:00:00 → tomorrow 00:00:00)
    """
    now = datetime.now(tz)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start, end   


def planning_window_current_week(
    anchor: Optional[datetime] = None,
    tz: Optional[ZoneInfo] = None,
) -> Tuple[datetime, datetime]:
    """
    Convenience wrapper for the current Sunday-start week in server time.
    """
    return planning_window_week(offset_weeks=0, anchor=anchor, tz=tz)


def planning_window_future_week(
    offset_weeks: int,
    anchor: Optional[datetime] = None,
    tz: Optional[ZoneInfo] = None,
) -> Tuple[datetime, datetime]:
    """
    Convenience wrapper for future/past weeks via offset.
    """
    return planning_window_week(offset_weeks=offset_weeks, anchor=anchor, tz=tz)


def planning_window_to_iso(
    window: Tuple[datetime, datetime],
) -> Dict[str, str]:
    """
    Convert a (start, end) window to ISO strings for logging/debugging/UI.
    Pure formatting helper (no side effects).
    """
    start, end = window
    return {"start": start.isoformat(), "end": end.isoformat()}


# ----------------------------
# LLM client initialization
# ----------------------------


def create_model_client() -> GeminiModelClient | OpenAICompatibleClient:
    """
    Factory to create the configured LLM client.
    """
    provider = JARVIS_LLM_PROVIDER.lower()

    if provider == "openai-compatible":
        if not OPENAI_COMPATIBLE_API_BASE or not OPENAI_COMPATIBLE_API_KEY:
            raise RuntimeError(
                "OPENAI_COMPATIBLE_API_BASE and OPENAI_COMPATIBLE_API_KEY must be set "
                "for the openai-compatible provider."
            )
        return OpenAICompatibleClient(
            api_base=OPENAI_COMPATIBLE_API_BASE,
            api_key=OPENAI_COMPATIBLE_API_KEY,
            default_model=JARVIS_LLM_MODEL,
        )


    if provider == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY must be set when using the gemini provider."
            )
        return GeminiModelClient(
            api_key=GEMINI_API_KEY,
            model_name=GEMINI_MODEL_NAME,
        )

    raise RuntimeError(f"Unsupported JARVIS_LLM_PROVIDER: {JARVIS_LLM_PROVIDER}")


MODEL_CLIENT = create_model_client()
LLM_MODEL = JARVIS_LLM_MODEL

# ----------------------------
# FastAPI app setup
# ----------------------------

app = FastAPI(
    title="ISAC Brain",
    version="1.0.0",
)


# ---------------------------------------------------------
# Bundle 4 (Sandbox & Exploratory Reasoning) — Phase 1
# Mechanical guardrails: explicit sandbox trigger + hard blocks
# Trigger: header 'X-ISAC-SANDBOX: true|1|yes'
# ---------------------------------------------------------

_SANDBOX_HEADER = "x-isac-sandbox"

def _is_sandbox_request(request: Request) -> bool:
    try:
        v = (request.headers.get("X-ISAC-SANDBOX") or request.headers.get(_SANDBOX_HEADER) or "").strip()
        return v.lower() in {"1", "true", "yes"}
    except Exception:
        return False

def _sandbox_boundary_http(surface: str) -> None:
    # Fail-transparent: structured boundary signal only.
    raise HTTPException(
        status_code=403,
        detail={
            "ok": False,
            "error": "SANDBOX_BOUNDARY",
            "blocked": True,
            "blocked_surface": surface,
            "message": "Sandbox mode forbids tools, observation, execution, and memory access.",
            "next_allowed": ["exit_sandbox", "discard_sandbox", "summarize_sandbox"],
        },
    )

# Identity & Memory v1: shared DB path for routers
app.state.db_path = DB_PATH

# Allow CORS from anywhere for now. Adjust in production if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the DB on startup
init_db()


# ----------------------------
# API key dependency
# ----------------------------


def require_api_key(
    x_api_key: Optional[str] = Header(default=None),
    x_isac_readonly_key: Optional[str] = Header(default=None),
    x_isac_admin_key: Optional[str] = Header(default=None),
    x_isac_admin_token: Optional[str] = Header(default=None),
) -> None:
    """
    Unified read auth gate (LOCKED INTENT):
    - Keeps legacy support: X-API-Key == JARVIS_API_KEY
    - Adds Alice/ISAC support:
        X-ISAC-READONLY-KEY == ISAC_READONLY_API_KEY
        X-ISAC-ADMIN-KEY   == ISAC_ADMIN_API_KEY or ISAC_ADMIN_KEY
    NOTE (v6):
    - X-ISAC-ADMIN-TOKEN is *not* accepted as a read gate.
      Token proof is admin-only and handled by require_admin_if_configured.
      This enforces the invariant that the UI must always send the base API key.

    If none of these are configured, access is open (no-op).
    """

    def _clean(v: Optional[str]) -> str:
        return (v or "").replace("\r", "").strip()

    jarvis_key = _clean(os.getenv("JARVIS_API_KEY"))
    readonly_cfg = _clean(os.getenv("ISAC_READONLY_API_KEY"))
    admin_cfg = _clean(os.getenv("ISAC_ADMIN_API_KEY") or os.getenv("ISAC_ADMIN_KEY"))
    configured_pin = _clean(ISAC_ADMIN_PIN)

    # If nothing is configured, do not gate.
    if not jarvis_key and not readonly_cfg and not admin_cfg and not configured_pin:
        return
    # Accept any configured key presented via its corresponding header.
    if jarvis_key and _clean(x_api_key) == jarvis_key:
        return

    if readonly_cfg and _clean(x_isac_readonly_key) == readonly_cfg:
        return

    if admin_cfg and _clean(x_isac_admin_key) == admin_cfg:
        return

    raise HTTPException(status_code=401, detail="Invalid or missing API key")


def require_admin_if_configured(
    request: Request,
    x_isac_admin_key: Optional[str] = Header(default=None),
    x_isac_admin_token: Optional[str] = Header(default=None),
) -> None:
    """
    Admin-safe gating for sensitive endpoints.

    Accepted admin proofs (if configured):
    - X-ISAC-ADMIN-KEY matches ISAC_ADMIN_API_KEY or ISAC_ADMIN_KEY (legacy + emergency)
    - X-ISAC-ADMIN-TOKEN is a valid, unexpired token issued by POST /auth/admin/unlock
      (requires ISAC_ADMIN_PIN to be configured)

    If neither an admin key nor a PIN is configured, this gate is open (no-op).

    Additionally, stamps request.state.admin_* fields so endpoints can expose
    non-authoritative UI hints (e.g., "admin via token").
    """

    def _clean(v: Optional[str]) -> str:
        return (v or "").replace("\r", "").strip()

    configured_key = _clean(os.getenv("ISAC_ADMIN_API_KEY") or os.getenv("ISAC_ADMIN_KEY"))
    configured_pin = _clean(ISAC_ADMIN_PIN)

    # Default: not unlocked (only used for optional response hints)
    request.state.admin_unlocked = False
    request.state.admin_via = None

    # If nothing is configured, do not gate.
    if not configured_key and not configured_pin:
        return

    # 1) Legacy/admin key path (always allowed if configured)
    if configured_key and _clean(x_isac_admin_key) == configured_key:
        request.state.admin_unlocked = True
        request.state.admin_via = "key"
        return

    # 2) Token path (only valid if PIN configured)
    if configured_pin and _is_valid_admin_token(_clean(x_isac_admin_token)):
        request.state.admin_unlocked = True
        request.state.admin_via = "token"
        return

    raise HTTPException(status_code=401, detail="Invalid or missing admin credentials")

# ---------------------------------------------------------
# Alice Admin Unlock (Execution Spine v1.1)
# - PIN -> short-lived admin token
# - Tokens are in-memory (reset on brain restart), session-friendly
# ---------------------------------------------------------

_ADMIN_TOKENS: Dict[str, float] = {}  # token -> expires_at_epoch_seconds


def _cleanup_admin_tokens() -> None:
    now = time.time()
    expired = [tok for tok, exp in _ADMIN_TOKENS.items() if exp <= now]
    for tok in expired:
        _ADMIN_TOKENS.pop(tok, None)


def _issue_admin_token() -> Dict[str, Any]:
    _cleanup_admin_tokens()

    ttl = int(ISAC_ADMIN_TOKEN_TTL_SECONDS or 3600)
    if ttl < 60:
        ttl = 60
    if ttl > 24 * 3600:
        ttl = 24 * 3600

    token = secrets.token_urlsafe(32)
    exp = time.time() + ttl
    _ADMIN_TOKENS[token] = exp

    exp_dt = datetime.now(timezone.utc) + timedelta(seconds=ttl)
    return {
        "token": token,
        "ttl_seconds": ttl,
        "expires_at_utc": exp_dt.isoformat(),
    }


def _is_valid_admin_token(token: str) -> bool:
    if not token:
        return False
    _cleanup_admin_tokens()
    exp = _ADMIN_TOKENS.get(token)
    if exp is None:
        return False
    return exp > time.time()


class AdminUnlockRequest(BaseModel):
    pin: str = Field(..., min_length=1, description="Admin PIN/passphrase (server-side)")


@app.post("/auth/admin/unlock")
def auth_admin_unlock(body: AdminUnlockRequest) -> Dict[str, Any]:
    """
    Exchange a PIN for a short-lived admin token.

    Contract (LOCKED):
    - This endpoint is PIN-only (no API key required).
    - PIN is verified against ISAC_ADMIN_PIN env var.
    - Admin-gated endpoints still require the admin API key and/or a valid admin token.
    """
    expected = (ISAC_ADMIN_PIN or "").strip()
    provided = (body.pin or "").strip()

    if not expected:
        # Misconfiguration: no PIN set server-side.
        raise HTTPException(status_code=500, detail="Admin PIN not configured")

    # constant-time compare
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid PIN")

    issued = _issue_admin_token()
    return {"status": "ok", **issued}

# ---------------------------------------------------------
# Phase 7 — Finance (Firefly) READ-ONLY snapshot router
# ---------------------------------------------------------
finance_router = APIRouter(
    prefix="/finance",
    tags=["finance"],
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)


@finance_router.get("/summary")
def finance_summary() -> Dict[str, Any]:
    """
    READ-ONLY finance summary derived from the local snapshot file.
    No live API calls, no writes, no mutations.
    """
    try:
        return summarize_finances()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Finance snapshot not found: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Finance summary error: {exc}",
        ) from exc


@finance_router.get("/transactions")
def finance_transactions(
    days: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Window size in days (UTC). Default 30. Max 365.",
    ),
    limit: int = Query(
        default=200,
        ge=1,
        le=1000,
        description="Max items to return (after filtering). Default 200. Max 1000.",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Pagination offset (after filtering + sorting). Default 0.",
    ),
) -> Dict[str, Any]:
    """
    READ-ONLY transaction listing derived from the local snapshot file.
    No live API calls, no writes, no mutations.
    """
    try:
        return list_transactions(days=days, limit=limit, offset=offset)
    except FinanceSnapshotError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Finance transactions error: {exc}",
        ) from exc


@finance_router.get("/health")
def finance_health() -> Dict[str, Any]:
    """
    Admin-only finance health endpoint.
    Read-only diagnostics for snapshot presence, symlink integrity, freshness, and JSON validity.
    No live API calls. No writes.
    """
    snapshot_path = Path("/app/data/finance/snapshots/latest.json")
    now_utc = datetime.now(timezone.utc)

    exists = snapshot_path.exists()
    is_symlink = snapshot_path.is_symlink()

    resolved_path = snapshot_path.resolve(strict=False) if is_symlink else snapshot_path
    resolved_exists = resolved_path.exists()

    if not exists:
        raise HTTPException(
            status_code=503,
            detail=f"Finance snapshot not found at {snapshot_path}",
        )

    # Stat the actual file we can read (prefer resolved target for symlink)
    stat_path = resolved_path if resolved_exists else snapshot_path
    st = stat_path.stat()
    mtime_utc = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
    age_seconds = int((now_utc - mtime_utc).total_seconds())

    json_ok = False
    json_error: Optional[str] = None
    top_level_type: Optional[str] = None
    top_level_keys: Optional[int] = None

    try:
        raw = stat_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        json_ok = True
        top_level_type = type(parsed).__name__
        if isinstance(parsed, dict):
            top_level_keys = len(parsed.keys())
    except Exception as exc:  # noqa: BLE001
        json_ok = False
        json_error = str(exc)

    status = "ok"
    notes: List[str] = []

    if is_symlink and not resolved_exists:
        status = "degraded"
        notes.append("latest.json is a symlink but the resolved target does not exist (broken symlink).")

    if not json_ok:
        status = "degraded"
        notes.append("snapshot file exists but JSON failed to parse.")

    return {
        "status": status,
        "captured_at_utc": now_utc.isoformat(),
        "snapshot": {
            "path": str(snapshot_path),
            "exists": exists,
            "is_symlink": is_symlink,
            "resolved_path": str(resolved_path) if is_symlink else None,
            "resolved_exists": resolved_exists if is_symlink else None,
            "size_bytes": int(st.st_size),
            "mtime_utc": mtime_utc.isoformat(),
            "age_seconds": age_seconds,
            "json_ok": json_ok,
            "json_error": json_error,
            "top_level_type": top_level_type,
            "top_level_keys": top_level_keys,
        },
        "notes": notes,
    }

        


# ---------------------------------------------------------
# Phase 6.9 — Calendar awareness router (READ-ONLY)
# ---------------------------------------------------------
# CONTRACT (LOCKED):
# - Read-only signal layer. Context, not control.
# - No calendar writes. No scheduling. No reminders. No automation.
# - No Home Assistant actions (no service calls).
# - No database writes tied to calendar reads.
# - Ambiguity surfaces as notes/warnings, not assumptions.
#
# TIME RULES (LOCKED):
# - Server-time is authoritative.
# - Week starts on Sunday.
#
# ENTITY RULES (LOCKED):
# - Household-enabled entities come from DB mapping tables.
# - Hard ignore list always applies: CALENDAR_ENTITY_IGNORE
#
# Endpoints (READ-ONLY):
# - GET /calendar/entities
# - GET /calendar/today
# - GET /calendar/week
# - GET /calendar/busy/week
# ---------------------------------------------------------


calendar_router = APIRouter(
    prefix="/calendar",
    tags=["calendar"],
    dependencies=[Depends(require_api_key)],
)


def ensure_ha_configured_for_calendar_reads() -> None:
    """
    Calendar awareness requires HA base URL + token to do HA REST calendar reads.
    (This is read-only.)
    """
    if not HOMEASSISTANT_BASE_URL or not HOMEASSISTANT_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Home Assistant not configured (HOMEASSISTANT_BASE_URL / HOMEASSISTANT_TOKEN missing).",
        )


async def ha_calendar_get_events(
    entity_id: str,
    start: datetime,
    end: datetime,
) -> List[Dict[str, Any]]:
    """
    Read-only Home Assistant calendar events via REST:
      GET /api/calendars/{entity_id}?start=...&end=...
    """
    ensure_ha_configured_for_calendar_reads()

    base = HOMEASSISTANT_BASE_URL.rstrip("/")
    url = f"{base}/api/calendars/{entity_id}"

    headers = {
        "Authorization": f"Bearer {HOMEASSISTANT_TOKEN}",
        "Content-Type": "application/json",
    }

    # HA accepts RFC3339/ISO strings; we pass timezone-aware ISO.
    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
    }

    timeout = httpx.Timeout(20.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, headers=headers, params=params)

    if resp.status_code == 404:
        # Entity missing or calendar endpoint not available for this entity
        return []
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "Home Assistant calendar read failed",
                "entity_id": entity_id,
                "status_code": resp.status_code,
                "body": resp.text,
            },
        )

    payload = resp.json()
    if isinstance(payload, list):
        return payload
    return []


def _coerce_ha_time_to_iso(
    value: Any,
    *,
    tz: ZoneInfo,
) -> Tuple[Optional[str], bool]:
    """
    Home Assistant calendar time fields can be:
      - {"dateTime": "..."}  (timed)
      - {"date": "YYYY-MM-DD"} (all-day)
      - "..." (string ISO-ish)

    Returns: (iso_string, is_all_day)
    - For all-day events, we normalize to midnight server tz (YYYY-MM-DDT00:00:00±HH:MM)
    - For timed events, we preserve the datetime in server tz.
    """
    if value is None:
        return (None, False)

    # Dict shape: {"dateTime": "..."} or {"date": "..."}
    if isinstance(value, dict):
        dt_val = value.get("dateTime")
        d_val = value.get("date")

        if isinstance(d_val, str) and d_val:
            # All-day: date only
            try:
                y, m, d = d_val.split("-")
                dt = datetime(int(y), int(m), int(d), 0, 0, 0, tzinfo=tz)
                return (dt.isoformat(), True)
            except Exception:
                # If parsing fails, fall through and stringify
                return (str(d_val), True)

        if isinstance(dt_val, str) and dt_val:
            # Timed: parse ISO -> convert to server tz if possible
            try:
                parsed = datetime.fromisoformat(dt_val.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc).astimezone(tz)
                else:
                    parsed = parsed.astimezone(tz)
                return (parsed.isoformat(), False)
            except Exception:
                return (dt_val, False)

    # String shape: best effort
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return (None, False)
        try:
            parsed = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc).astimezone(tz)
            else:
                parsed = parsed.astimezone(tz)
            return (parsed.isoformat(), False)
        except Exception:
            # Could be a date-only string or non-ISO. Treat date-only as all-day.
            if len(s) == 10 and s[4] == "-" and s[7] == "-":
                try:
                    y, m, d = s.split("-")
                    dt = datetime(int(y), int(m), int(d), 0, 0, 0, tzinfo=tz)
                    return (dt.isoformat(), True)
                except Exception:
                    return (s, True)
            return (s, False)

    # Unknown type
    return (str(value), False)


def _parse_iso_to_dt(iso_str: Optional[str]) -> Optional[datetime]:
    """
    Best effort parse for sorting. Returns timezone-aware datetime when possible.
    """
    if not iso_str:
        return None


# Back-compat helper: earlier UI/task creators call lineage_link_task().
# v5 canonical implementation is lineage_init().
def lineage_link_task(task_id: int, parent_task_id=None, trigger_reason=None) -> None:
    try:
        lineage_init(task_id=task_id, parent_task_id=parent_task_id, trigger_reason=trigger_reason)
    except Exception:
        # lineage is best-effort; never fail task creation because of lineage logging
        return
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except Exception:
        return None


def _clip_interval_to_day(
    start: datetime,
    end: datetime,
    day_start: datetime,
    day_end: datetime,
) -> int:
    """
    Return overlap in minutes between [start, end) and a single day window.
    """
    latest_start = max(start, day_start)
    earliest_end = min(end, day_end)
    if earliest_end <= latest_start:
        return 0
    return int((earliest_end - latest_start).total_seconds() // 60)


def _busy_score_from_minutes(minutes: int) -> int:
    """
    Convert busy minutes to a 0–100 score.
    8h (480m) == 100, linear scale, capped.
    """
    return min(100, int((minutes / 480) * 100))        


@calendar_router.get("/entities")
async def calendar_entities(
    household: Optional[str] = Query(
        default="home_a",
        description="home_a | home_b (default: home_a)",
    ),
) -> Dict[str, Any]:
    """
    Read-only list of enabled calendar entities for the household.
    DB-read only. Hard-ignores kaitlyn/laila/apentalp.
    """
    hh = _normalize_calendar_household(household)
    entities = get_enabled_calendar_entities_for_household(hh)

    return {
        "status": "ok",
        "household": hh,
        "ignored": sorted(list(CALENDAR_ENTITY_IGNORE)),
        "count": len(entities),
        "entities": entities,
    }


@calendar_router.get("/week")
async def calendar_week(
    offset_weeks: int = Query(
        default=0,
        description="0=current week (Sunday-start), 1=next week, -1=previous week, etc.",
    ),
    household: Optional[str] = Query(
        default="home_a",
        description="home_a | home_b (default: home_a)",
    ),
    include_raw: bool = Query(
        default=False,
        description="If true, include raw HA event payloads. Default false.",
    ),
) -> Dict[str, Any]:
    """
    Read-only: Fetch calendar events for a Sunday-start planning week window
    using server timezone helpers (deterministic).

    POLISH A:
    - Normalizes start/end into start_iso/end_iso
    - Normalizes all_day boolean
    - Improves sorting by actual datetime
    - raw payload included only if include_raw=true
    """
    hh = _normalize_calendar_household(household)

    tz = get_server_timezone()
    window_start, window_end = planning_window_week(offset_weeks=offset_weeks, tz=tz)

    calendars = get_enabled_calendar_entities_for_household(hh)

    per_calendar: List[Dict[str, Any]] = []
    combined_events: List[Dict[str, Any]] = []

    for cal in calendars:
        entity_id = cal.get("entity_id")
        if not entity_id or entity_id in CALENDAR_ENTITY_IGNORE:
            continue

        events = await ha_calendar_get_events(
            entity_id=entity_id,
            start=window_start,
            end=window_end,
        )

        normalized_events: List[Dict[str, Any]] = []
        for ev in events:
            if not isinstance(ev, dict):
                continue

            start_iso, start_all_day = _coerce_ha_time_to_iso(ev.get("start"), tz=tz)
            end_iso, end_all_day = _coerce_ha_time_to_iso(ev.get("end"), tz=tz)
            all_day = bool(ev.get("all_day")) or start_all_day or end_all_day

            normalized: Dict[str, Any] = {
                "summary": ev.get("summary") or ev.get("title") or None,
                "location": ev.get("location"),
                "description": ev.get("description"),
                "all_day": all_day,
                "start_iso": start_iso,
                "end_iso": end_iso,
            }

            if include_raw:
                normalized["raw"] = ev

            normalized_events.append(normalized)

            combined_events.append(
                {
                    "entity_id": entity_id,
                    "owner_key": cal.get("owner_key"),
                    "owner_display_name": cal.get("owner_display_name"),
                    "label": cal.get("label"),
                    **normalized,
                }
            )

        per_calendar.append(
            {
                "entity_id": entity_id,
                "owner_key": cal.get("owner_key"),
                "owner_display_name": cal.get("owner_display_name"),
                "label": cal.get("label"),
                "event_count": len(normalized_events),
                "events": normalized_events,
            }
        )

    def _sort_key(ev: Dict[str, Any]) -> Tuple[datetime, int, str]:
        start_dt = _parse_iso_to_dt(ev.get("start_iso")) or datetime.max.replace(tzinfo=timezone.utc)
        all_day_rank = 0 if bool(ev.get("all_day")) else 1
        summary = str(ev.get("summary") or "")
        return (start_dt, all_day_rank, summary.lower())

    combined_events.sort(key=_sort_key)

    return {
        "status": "ok",
        "household": hh,
        "offset_weeks": int(offset_weeks),
        "timezone": getattr(tz, "key", str(tz)),
        "first_day_of_week": "sunday",
        "window": {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
        },
        "ignored": sorted(list(CALENDAR_ENTITY_IGNORE)),
        "include_raw": bool(include_raw),
        "calendars": {
            "count": len(per_calendar),
            "items": per_calendar,
        },
        "events": {
            "count": len(combined_events),
            "items": combined_events,
        },
        "notes": [
            "Read-only calendar awareness.",
            "No DB writes, no HA writes, no event creation.",
            "Week window is Sunday-start and derived from server time.",
            "POLISH A: normalized start/end fields and sorting; raw optional.",
        ],
    }


@calendar_router.get("/busy/week")
async def calendar_busy_week(
    offset_weeks: int = Query(
        default=0,
        description="0=current week (Sunday-start), 1=next week, -1=previous week",
    ),
    household: Optional[str] = Query(
        default="home_a",
        description="home_a | home_b",
    ),
) -> Dict[str, Any]:
    """
    Read-only calendar busyness aggregation.
    Produces per-day workload signals for planning awareness.
    """
    hh = _normalize_calendar_household(household)
    tz = get_server_timezone()

    window_start, window_end = planning_window_week(
        offset_weeks=offset_weeks, tz=tz
    )

    calendars = get_enabled_calendar_entities_for_household(hh)
    all_events: List[Dict[str, Any]] = []

    for cal in calendars:
        entity_id = cal.get("entity_id")
        if not entity_id or entity_id in CALENDAR_ENTITY_IGNORE:
            continue

        events = await ha_calendar_get_events(
            entity_id=entity_id,
            start=window_start,
            end=window_end,
        )

        for ev in events:
            start_iso, start_all_day = _coerce_ha_time_to_iso(ev.get("start"), tz=tz)
            end_iso, end_all_day = _coerce_ha_time_to_iso(ev.get("end"), tz=tz)

            start_dt = _parse_iso_to_dt(start_iso)
            end_dt = _parse_iso_to_dt(end_iso)

            if not start_dt or not end_dt:
                continue

            all_events.append(
                {
                    "start": start_dt,
                    "end": end_dt,
                    "all_day": bool(ev.get("all_day") or start_all_day or end_all_day),
                    "summary": ev.get("summary") or ev.get("title"),
                }
            )

    busy_by_day: List[Dict[str, Any]] = []

    for day_offset in range(7):
        day_start = window_start + timedelta(days=day_offset)
        day_end = day_start + timedelta(days=1)

        day_events = []
        busy_minutes = 0
        all_day_count = 0

        for ev in all_events:
            if ev["all_day"]:
                if ev["start"].date() == day_start.date():
                    all_day_count += 1
                continue

            minutes = _clip_interval_to_day(
                ev["start"], ev["end"], day_start, day_end
            )
            if minutes > 0:
                busy_minutes += minutes
                day_events.append(ev)

        summaries = [
            e["summary"] for e in day_events if e.get("summary")
        ][:3]

        busy_by_day.append(
            {
                "date": day_start.date().isoformat(),
                "event_count": len(day_events) + all_day_count,
                "all_day_count": all_day_count,
                "busy_minutes": busy_minutes,
                "busy_score": _busy_score_from_minutes(busy_minutes),
                "top_summaries": summaries,
            }
        )

    return {
        "status": "ok",
        "household": hh,
        "timezone": getattr(tz, "key", str(tz)),
        "window": {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
        },
        "busy_by_day": busy_by_day,
        "notes": [
            "Read-only calendar awareness signal",
            "No DB writes, no HA writes",
            "Designed for meal planning + UI hints",
        ],
    }


@calendar_router.get("/today")
async def calendar_today(
    household: Optional[str] = Query(
        default="home_a",
        description="home_a | home_b (default: home_a)",
    ),
    include_raw: bool = Query(
        default=False,
        description="If true, include raw HA event payloads. Default false.",
    ),
) -> Dict[str, Any]:
    """
    Read-only: Fetch calendar events for 'today' using server timezone.
    Same normalization and ignore rules as /calendar/week.
    """
    hh = _normalize_calendar_household(household)

    tz = get_server_timezone()
    window_start, window_end = planning_window_today(tz=tz)

    calendars = get_enabled_calendar_entities_for_household(hh)

    combined_events: List[Dict[str, Any]] = []

    for cal in calendars:
        entity_id = cal.get("entity_id")
        if not entity_id or entity_id in CALENDAR_ENTITY_IGNORE:
            continue

        events = await ha_calendar_get_events(
            entity_id=entity_id,
            start=window_start,
            end=window_end,
        )

        for ev in events:
            if not isinstance(ev, dict):
                continue

            start_iso, start_all_day = _coerce_ha_time_to_iso(ev.get("start"), tz=tz)
            end_iso, end_all_day = _coerce_ha_time_to_iso(ev.get("end"), tz=tz)
            all_day = bool(ev.get("all_day")) or start_all_day or end_all_day

            normalized: Dict[str, Any] = {
                "entity_id": entity_id,
                "owner_key": cal.get("owner_key"),
                "owner_display_name": cal.get("owner_display_name"),
                "label": cal.get("label"),
                "summary": ev.get("summary") or ev.get("title") or None,
                "location": ev.get("location"),
                "description": ev.get("description"),
                "all_day": all_day,
                "start_iso": start_iso,
                "end_iso": end_iso,
            }

            if include_raw:
                normalized["raw"] = ev

            combined_events.append(normalized)

    def _sort_key(ev: Dict[str, Any]) -> Tuple[datetime, int, str]:
        start_dt = _parse_iso_to_dt(ev.get("start_iso")) or datetime.max.replace(tzinfo=timezone.utc)
        all_day_rank = 0 if bool(ev.get("all_day")) else 1
        summary = str(ev.get("summary") or "")
        return (start_dt, all_day_rank, summary.lower())

    combined_events.sort(key=_sort_key)

    return {
        "status": "ok",
        "household": hh,
        "timezone": getattr(tz, "key", str(tz)),
        "window": {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
        },
        "ignored": sorted(list(CALENDAR_ENTITY_IGNORE)),
        "include_raw": bool(include_raw),
        "events": {
            "count": len(combined_events),
            "items": combined_events,
        },
        "notes": [
            "Read-only calendar awareness.",
            "Server-time 'today' window.",
            "No DB writes, no HA writes, no event creation.",
        ],
    }


# ----------------------------
# Request / response models
# ----------------------------


class AskRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = None
    temperature: float = 0.2
    max_output_tokens: int = 512


class AskResponse(BaseModel):
    model: str
    answer: str


# ----------------------------
# Home Assistant client setup
# ----------------------------


def create_ha_client() -> Optional[HomeAssistantClient]:
    """
    Create a HomeAssistantClient from environment variables, or None if
    configuration is incomplete.
    """
    if not HOMEASSISTANT_BASE_URL or not HOMEASSISTANT_TOKEN:
        return None

    config = HomeAssistantConfig(
        base_url=HOMEASSISTANT_BASE_URL,
        token=HOMEASSISTANT_TOKEN,
    )
    return HomeAssistantClient(config)


HA_CLIENT = create_ha_client()

# ---------------------------------------------------------
# Grocy / household helpers (Phase 6.5 scaffolding)
# ---------------------------------------------------------


def _parse_location_ids(env_name: str) -> List[int]:
    raw = os.getenv(env_name, "") or ""
    ids: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError:
            continue
    return ids


GROCY_HOME_A_LOCATION_IDS = _parse_location_ids("GROCY_HOME_A_LOCATION_IDS")
GROCY_HOME_B_LOCATION_IDS = _parse_location_ids("GROCY_HOME_B_LOCATION_IDS")


def filter_stock_by_household(
    stock_items: List[Dict[str, Any]],
    household: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Filter stock items by household using Grocy location IDs.
    """
    if household in (None, "", "all"):
        return stock_items

    if household == "home_a" and GROCY_HOME_A_LOCATION_IDS:
        allowed = set(GROCY_HOME_A_LOCATION_IDS)
    elif household == "home_b" and GROCY_HOME_B_LOCATION_IDS:
        allowed = set(GROCY_HOME_B_LOCATION_IDS)
    else:
        return stock_items

    return [item for item in stock_items if item.get("location_id") in allowed]


def _require_household_query(household: Optional[str]) -> str:
    """
    Gate A + Phase 6.4 rule:
    For Grocy read endpoints, we do NOT guess intent.
    Callers must explicitly provide household=home_a|home_b|all.
    """
    if household is None:
        raise HTTPException(
            status_code=400,
            detail="Query parameter 'household' is required: home_a | home_b | all",
        )
    h = (household or "").strip().lower()
    if h not in {"home_a", "home_b", "all"}:
        raise HTTPException(
            status_code=400,
            detail="household must be 'home_a', 'home_b', or 'all'",
        )
    return h


def _require_household_write_scope(household: Optional[str]) -> str:
    """
    For operations that must be household-specific (no 'all' allowed).
    Phase 6.45 reasoning endpoints also use this to prevent ambiguity.
    """
    if household is None:
        raise HTTPException(
            status_code=400,
            detail="Query parameter 'household' is required: home_a | home_b",
        )
    h = (household or "").strip().lower()
    if h not in {"home_a", "home_b"}:
        raise HTTPException(
            status_code=400,
            detail="household must be 'home_a' or 'home_b'",
        )
    return h


# ---------------------------------------------------------
# Safe async method resolution (for Grocy + BarcodeBuddy)
# ---------------------------------------------------------


async def _acall_first_existing(
    obj: Any,
    method_names: List[str],
    *args: Any,
    **kwargs: Any,
) -> Any:
    for name in method_names:
        fn = getattr(obj, name, None)
        if fn is None:
            continue
        if callable(fn):
            return await fn(*args, **kwargs)

    available = [m for m in dir(obj) if not m.startswith("_")]
    raise HTTPException(
        status_code=500,
        detail={
            "error": "Expected service method not found",
            "tried": method_names,
            "service_type": type(obj).__name__,
            "available_sample": available[:60],
        },
    )


# ----------------------------
# Health router
# ----------------------------

health_router = APIRouter(prefix="/health", tags=["health"])


@health_router.get("")
async def health_root() -> Dict[str, Any]:
    """
    Root health endpoint.
    NOTE: This maps to `/health` (no trailing slash).
    Keep it cheap: no external calls, no DB requirements.
    """
    return {
        "status": "ok",
        "service": "isac-brain",
        "version": app.version,
        "message": "ISAC brain is alive",
    }


@health_router.get("/ping")
async def ping() -> Dict[str, Any]:
    # Backward/explicit ping endpoint (`/health/ping`)
    return {"status": "ok", "message": "ISAC brain is alive"}


@app.get("/verify", dependencies=[Depends(require_api_key)])
async def verify() -> Dict[str, Any]:
    """
    Read-only verification endpoint.
    Designed for autonomy auto-verify and human confidence checks.
    No side effects. No external calls. No persistence.
    """
    uptime_seconds = int(time.monotonic() - START_TIME)

    return {
        "service": "isac-brain",
        "version": app.version,
        "commit": os.getenv("GIT_COMMIT") or os.getenv("CF_PAGES_COMMIT_SHA"),
        "uptime_seconds": uptime_seconds,
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
        "tool_allowlist": {
            "web": ["search", "open", "find", "click", "screenshot_pdf"],
            "local": ["read_file", "read_snippet"],
            "utility": ["time", "calc", "weather"],
        },
    }


@health_router.get("/isac_internal")
async def health_isac_internal(
    _admin_ok: None = Depends(require_admin_if_configured),
) -> Dict[str, Any]:
    """
    Read-only internal health snapshot captured at request time.
    - Explicit capture-on-request
    - No persistence
    - No decisions
    - Admin-safe (only gated if ISAC_ADMIN_KEY is configured)
    """
    snapshot = capture_internal_health()

    # Pydantic v2 -> model_dump; v1 -> dict; fallback -> raw object
    if hasattr(snapshot, "model_dump"):
        return snapshot.model_dump()
    if hasattr(snapshot, "dict"):
        return snapshot.dict()
    return {"snapshot": snapshot}


@health_router.get("/system")
async def system_health(
    _admin_ok: None = Depends(require_admin_if_configured),
) -> Dict[str, Any]:
    """
    Read-only external/system health snapshot captured at request time.
    - Explicit capture-on-request
    - No persistence
    - No decisions
    - Admin-safe (only gated if ISAC_ADMIN_API_KEY / ISAC_ADMIN_KEY is configured)
    """
    if HA_CLIENT is None:
        return {
            "domain": "system_external",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "capture_duration_ms": 0,
            "checks": {
                "homeassistant": {
                    "ok": False,
                    "error": "homeassistant_not_configured",
                }
            },
            "notes": [
                "HOMEASSISTANT_BASE_URL / HOMEASSISTANT_TOKEN missing; HA check skipped.",
                "All behavior is read-only and non-persistent.",
            ],
        }

    return await capture_system_health(HA_CLIENT)


@health_router.get("/database")
async def database_health() -> Dict[str, Any]:
    start = time.time()
    exists = Path(DB_PATH).is_file()
    duration = time.time() - start

    if not exists:
        return {
            "status": "error",
            "message": f"Database file not found at {DB_PATH}",
            "duration_seconds": duration,
        }

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1")
        conn.close()
    except sqlite3.Error as exc:
        return {
            "status": "error",
            "message": f"Error accessing DB: {exc}",
            "duration_seconds": duration,
        }

    return {
        "status": "ok",
        "message": "Database accessible",
        "duration_seconds": duration,
    }


@health_router.get("/homeassistant")
async def homeassistant_health() -> Dict[str, Any]:
    if not HA_CLIENT:
        return {
            "status": "disabled",
            "reason": "HOMEASSISTANT_BASE_URL or HOMEASSISTANT_TOKEN not configured",
        }

    try:
        info = await HA_CLIENT.get_config()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}

    return {
        "status": "ok",
        "instance": {
            "location_name": info.get("location_name"),
            "version": info.get("version"),
        },
    }


# ---------------------------------------------------------
# Runner router & endpoints (Execution Spine choke point)
# ---------------------------------------------------------

runner_router = APIRouter(
    prefix="/runner",
    tags=["runner"],
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)


def _runner_url(path: str) -> str:
    base = (JARVIS_RUNNER_BASE_URL or "").rstrip("/")
    if not base:
        # Should never happen because we default it, but keep it safe.
        raise HTTPException(status_code=500, detail="Runner base URL not configured.")
    if not path.startswith("/"):
        path = "/" + path
    return base + path


async def _runner_get(path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    url = _runner_url(path)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # Preserve upstream status code and body (best-effort)
        detail: Any
        try:
            detail = exc.response.json()
        except Exception:  # noqa: BLE001
            detail = {"error": exc.response.text}
        raise HTTPException(status_code=exc.response.status_code, detail=detail) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Runner unreachable: {exc}") from exc

    try:
        return resp.json()
    except Exception:  # noqa: BLE001
        return {"ok": True, "raw": resp.text}


@runner_router.get("/health")
async def runner_health() -> Dict[str, Any]:
    # Proxy to runner's own health endpoint
    return await _runner_get("/runner/health")


@runner_router.get("/fs/list")
async def runner_fs_list(path: str = Query(..., min_length=1, max_length=512)) -> Dict[str, Any]:
    # Proxy to runner's safe fs list endpoint
    return await _runner_get("/runner/fs/list", params={"path": path})

@runner_router.get("/fs/list_and_log")
async def runner_fs_list_and_log(
    path: str = Query("/opt/jarvis", min_length=1, max_length=512)
) -> Dict[str, Any]:
    # 1) Create task + step
    task_id = create_task(
        title="Runner FS list",
        resume_hint=f"List directory via runner: {path}",
    )

    lineage_init(task_id=task_id, parent_task_id=None, trigger_reason="manual: runner_fs_list_and_log")

    step_id = add_step(
        task_id=task_id,
        step_index=1,
        description=f"List directory: {path}",
    )

    # 2) Call runner
    result = await _runner_get(
        "/runner/fs/list",
        params={"path": path},
    )

    # 3) Store artifact
    add_artifact(
        task_id=task_id,
        step_id=step_id,
        artifact_type="runner_fs_list",
        metadata_json=json.dumps(
            {
                "path": path,
                "result": result,
            }
        ),
    )

    return {
        "ok": True,
        "task_id": task_id,
        "step_id": step_id,
        "runner": result,
    }

# ---------------------------------------------------------
# Task Ledger — READ API (Option A)
# Read-only visibility for Resume vs Abort / crash recovery
# ---------------------------------------------------------

def _task_db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _task_row_to_dict(row: tuple) -> Dict[str, Any]:
    # Expected minimum schema:
    # tasks(id, title, status, resume_hint, ...)
    return {
        "id": row[0],
        "title": row[1],
        "status": row[2],
        "resume_hint": row[3],
    }


def _task_step_row_to_dict(row: tuple) -> Dict[str, Any]:
    # Minimum schema:
    # task_steps(id, task_id, step_index, description, status, ...)
    return {
        "id": row[0],
        "task_id": row[1],
        "step_index": row[2],
        "description": row[3],
        "status": row[4],
    }


def _task_artifact_row_to_dict(row: tuple) -> Dict[str, Any]:
    # Minimum schema:
    # task_artifacts(id, task_id, step_id, artifact_type, metadata_json, ...)
    step_id = row[2]
    metadata_json = row[4]
    metadata: Optional[Dict[str, Any]] = None
    if metadata_json:
        try:
            metadata = json.loads(metadata_json)
        except Exception:
            metadata = None

    return {
        "id": row[0],
        "task_id": row[1],
        # IMPORTANT: allow null step_id (older artifacts or task-level artifacts)
        "step_id": int(step_id) if step_id is not None else None,
        "artifact_type": row[3],
        "metadata_json": metadata_json,
        "metadata": metadata,
    }
def list_tasks(
    limit: int = 25,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")

    conn = _task_db_connect()
    try:
        cur = conn.cursor()

        if status:
            # Explicit filter: return tasks matching exactly this status.
            cur.execute(
                """
                SELECT id, title, status, resume_hint
                FROM tasks
                WHERE status = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (status, limit),
            )
        else:
            # Default: unfinished only (v1 rule: "done" == completed).
            cur.execute(
                """
                SELECT id, title, status, resume_hint
                FROM tasks
                WHERE status != 'completed'
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )

        rows = cur.fetchall() or []
        return [_task_row_to_dict(r) for r in rows]
    finally:
        conn.close()

@app.get(
    "/tasks",
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)
def list_tasks_endpoint(
    request: Request,
    limit: int = Query(25, ge=1, le=100),
    status: Optional[str] = Query(None),
) -> Dict[str, Any]:
    tasks = list_tasks(limit=limit, status=status)

    admin_via = getattr(request.state, "admin_via", None)

    return {
        "count": len(tasks),
        "limit": limit,
        "status": status,
        "admin": True,
        "admin_via": admin_via,  # "token" | "key" | None
        "tasks": tasks,
    }

@app.get(
    "/tasks/{task_id}",
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)
def get_task(request: Request, task_id: int) -> Dict[str, Any]:
    if task_id < 1:
        raise HTTPException(status_code=400, detail="task_id must be >= 1")

    conn = _task_db_connect()
    try:
        cur = conn.cursor()

        cur.execute(
            """
            SELECT id, title, status, resume_hint
            FROM tasks
            WHERE id = ?
            """,
            (task_id,),
        )
        task_row = cur.fetchone()
        if not task_row:
            raise HTTPException(status_code=404, detail="Task not found")

        cur.execute(
            """
            SELECT id, task_id, step_index, description, status
            FROM task_steps
            WHERE task_id = ?
            ORDER BY step_index ASC, id ASC
            """,
            (task_id,),
        )
        step_rows = cur.fetchall() or []

        cur.execute(
            """
            SELECT id, task_id, step_id, artifact_type, metadata_json
            FROM task_artifacts
            WHERE task_id = ?
            ORDER BY id ASC
            """,
            (task_id,),
        )
        artifact_rows = cur.fetchall() or []

        admin_via = getattr(request.state, "admin_via", None)

        return {
            "admin": True,
            "admin_via": admin_via,  # "token" | "key" | None
            "task": _task_row_to_dict(task_row),
            "steps": [_task_step_row_to_dict(r) for r in step_rows],
            "artifacts": [_task_artifact_row_to_dict(r) for r in artifact_rows],
        }
    finally:
        conn.close()

# ---------------------------------------------------------
# Task Ledger — CREATE (public, safe)  (Option A)
# Creates a pending task that can later be Resumed (ISAC-only execution)
# NOTE: This does NOT call the runner. It only writes to the ledger.
# ---------------------------------------------------------

@app.post(
    "/tasks/create/runner_fs_list",
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)
def create_runner_fs_list_task(
    path: str = Query("/opt/jarvis", min_length=1, max_length=512),
    parent_task_id: Optional[int] = Query(None),
    trigger_reason: Optional[str] = Query(None, max_length=256),
) -> Dict[str, Any]:
    task_id = create_task(
        title="Runner FS list",
        resume_hint=f"List directory via runner: {path}",
    )

    # Optional: seed an initial step for visibility/audit trail
    add_step(
        task_id=task_id,
        step_index=1,
        description=f"Task created (pending). Resume will list directory: {path}",
    )


    return {
        "ok": True,
        "task_id": task_id,
        "title": "Runner FS list",
        "status": "pending",
        "resume_hint": f"List directory via runner: {path}",
    }

# ---------------------------------------------------------
# Task Ledger — ACTIONS (Abort / Resume)
# Execution Spine v1.3
# ---------------------------------------------------------

def _task_update_status(task_id: int, status: str) -> None:
    conn = _task_db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE tasks
            SET status = ?
            WHERE id = ?
            """,
            (status, task_id),
        )
        conn.commit()
        if cur.rowcount < 1:
            raise HTTPException(status_code=404, detail="Task not found")
    finally:
        conn.close()


def _task_mark_completed(task_id: int) -> None:
    # Canonical task status used throughout the spine
    _task_update_status(task_id, "completed")


def _task_get_min(task_id: int) -> Dict[str, Any]:
    conn = _task_db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, title, status, resume_hint
            FROM tasks
            WHERE id = ?
            """,
            (task_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        return _task_row_to_dict(row)
    finally:
        conn.close()

def _task_get_latest_artifact(task_id: int, artifact_type: str) -> Optional[Dict[str, Any]]:
    """Fetch the most recent artifact of a given type for a task."""
    conn = _task_db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, task_id, step_id, artifact_type, metadata_json
            FROM task_artifacts
            WHERE task_id = ? AND artifact_type = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (task_id, artifact_type),
        )
        row = cur.fetchone()
        if not row:
            return None
        return _task_artifact_row_to_dict(row)
    finally:
        conn.close()


def _is_path_allowed(path: str) -> bool:
    """Write allowlist (locked): /opt/jarvis/brain/** and /opt/jarvis/data/index.html"""
    if not path:
        return False
    norm = os.path.abspath(path)
    if norm.startswith("/opt/jarvis/brain/"):
        return True
    if norm == "/opt/jarvis/data/index.html":
        return True
    return False


def resolve_target_path(user_path: str, must_exist: bool = True) -> Optional[str]:
    """Resolve a user-facing path (host-style) to an on-container filesystem path.

    v4 note: The user is allowed to refer to host paths like /opt/jarvis/brain/**, but
    inside the container the code lives under /app/**. We attempt a conservative mapping.

    This resolver never expands scope: it only maps known prefixes and only returns an
    existing file (when must_exist=True).
    """
    p = (user_path or "").strip()
    if not p:
        return None

    candidates = [p]

    # Known mount: /opt/jarvis/brain -> /app
    if p == "/opt/jarvis/brain/main.py":
        candidates.append("/app/main.py")
    if p.startswith("/opt/jarvis/brain/"):
        rel = p[len("/opt/jarvis/brain/"):]
        candidates.append("/app/" + rel)
    if p == "/opt/jarvis/brain":
        candidates.append("/app")

    # As a last resort, if user passed a relative-ish path, try under /app
    if not p.startswith("/") and p:
        candidates.append("/app/" + p)

    for c in candidates:
        try:
            if must_exist:
                if Path(c).exists():
                    return c
            else:
                # For writes, allow resolving to a path under a known directory if the parent exists.
                parent = Path(c).parent
                if parent.exists():
                    return c
        except Exception:
            continue
    return None


def _atomic_write_text(path: str, content: str) -> None:
    """Atomic apply: temp -> fsync -> rename."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".isac_tmp_", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


async def _http_verify_template() -> Dict[str, Any]:
    """
    Auto-verify template (v3):
    - Backend health via http://127.0.0.1:8000/health
    - UI reachability via ISAC_UI_VERIFY_URL (default http://jarvis-agent/)
    HTTP only. No Node.
    """
    backend_url = os.getenv("ISAC_BACKEND_VERIFY_URL", "http://127.0.0.1:8000/health")
    ui_url = os.getenv("ISAC_UI_VERIFY_URL", "http://jarvis-agent/")

    results: Dict[str, Any] = {"backend": {}, "ui": {}}
    ok = True

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        # Backend
        try:
            r = await client.get(backend_url, headers={"Accept": "application/json"})
            results["backend"] = {"url": backend_url, "status_code": r.status_code}
            if r.status_code != 200:
                ok = False
        except Exception as e:
            results["backend"] = {"url": backend_url, "error": str(e)}
            ok = False

        # UI
        try:
            r = await client.get(ui_url, headers={"Accept": "text/html"})
            results["ui"] = {"url": ui_url, "status_code": r.status_code}
            if r.status_code != 200:
                ok = False
        except Exception as e:
            results["ui"] = {"url": ui_url, "error": str(e)}
            ok = False

    results["ok"] = ok
    return results

def _parse_diff_apply_resume_hint(resume_hint: Any) -> Optional[Dict[str, Any]]:
    """Parse resume_hint for Diff Apply tasks."""
    try:
        if resume_hint is None:
            return None
        if isinstance(resume_hint, str):
            resume_hint = resume_hint.strip()
            if not resume_hint:
                return None
            payload = json.loads(resume_hint)
        elif isinstance(resume_hint, dict):
            payload = resume_hint
        else:
            return None

        path = str(payload.get("path") or "").strip()
        unified_diff_b64 = str(payload.get("unified_diff_b64") or "").strip()
        dry_run = bool(payload.get("dry_run", True))

        if not path or not unified_diff_b64:
            return None

        return {"path": path, "unified_diff_b64": unified_diff_b64, "dry_run": dry_run}
    except Exception:
        return None


def _apply_unified_diff(original_text: str, diff_text: str) -> Dict[str, Any]:
    """
    Apply a unified diff to original_text.

    Constraints (v4):
    - Supports single-file diffs.
    - Requires all hunks to apply cleanly (no fuzzy matching).
    - Returns patched text plus stats (additions, deletions, hunks).
    """
    original_lines = original_text.splitlines(keepends=True)
    diff_lines = diff_text.splitlines(keepends=True)

    # Strip any leading noise before the first file header if present
    i = 0
    while i < len(diff_lines) and not diff_lines[i].startswith("--- "):
        i += 1
    diff_lines = diff_lines[i:]

    if len(diff_lines) < 2 or not diff_lines[0].startswith("--- ") or not diff_lines[1].startswith("+++ "):
        raise ValueError("diff must start with '---' and '+++' file headers")

    # Skip file headers
    j = 2

    out: List[str] = []
    src_idx = 0
    hunks = 0
    additions = 0
    deletions = 0

    hunk_re = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")

    while j < len(diff_lines):
        line = diff_lines[j]
        if not line.startswith("@@"):
            # Allow trailing metadata lines, but ignore
            j += 1
            continue

        m = hunk_re.match(line.rstrip("\n"))
        if not m:
            raise ValueError(f"invalid hunk header: {line.strip()}")

        hunks += 1
        old_start = int(m.group(1))
        old_count = int(m.group(2) or "1")

        # Copy unchanged content up to hunk start (1-based in diff)
        target_idx = max(old_start - 1, 0)
        if target_idx < src_idx:
            raise ValueError("hunk overlaps or is out of order")
        out.extend(original_lines[src_idx:target_idx])
        src_idx = target_idx

        j += 1  # move past hunk header

        # Apply hunk body
        while j < len(diff_lines):
            dl = diff_lines[j]
            if dl.startswith("@@"):
                break
            if dl.startswith("\\"):
                # e.g. "\ No newline at end of file" -> ignore
                j += 1
                continue

            if dl.startswith(" "):
                expected = dl[1:]
                if src_idx >= len(original_lines) or original_lines[src_idx] != expected:
                    raise ValueError("context mismatch while applying diff")
                out.append(original_lines[src_idx])
                src_idx += 1
            elif dl.startswith("-"):
                expected = dl[1:]
                if src_idx >= len(original_lines) or original_lines[src_idx] != expected:
                    raise ValueError("deletion mismatch while applying diff")
                # skip line (delete)
                src_idx += 1
                deletions += 1
            elif dl.startswith("+"):
                out.append(dl[1:])
                additions += 1
            else:
                raise ValueError(f"unexpected diff line: {dl.strip()}")

            j += 1

    # Append remainder
    out.extend(original_lines[src_idx:])

    patched_text = "".join(out)
    return {
        "patched_text": patched_text,
        "hunks": hunks,
        "additions": additions,
        "deletions": deletions,
    }


def _task_next_step_index(task_id: int) -> int:
    conn = _task_db_connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COALESCE(MAX(step_index), 0)
            FROM task_steps
            WHERE task_id = ?
            """,
            (task_id,),
        )
        max_idx = cur.fetchone()[0] or 0
        return int(max_idx) + 1
    finally:
        conn.close()


def _parse_runner_fs_list_resume_hint(resume_hint: Optional[str]) -> Optional[str]:
    """
    Supported v1 resume intent:
      "List directory via runner: /opt/jarvis"
    Returns extracted path or None.
    """
    if not resume_hint:
        return None
    prefix = "List directory via runner:"
    if not resume_hint.startswith(prefix):
        return None
    path = resume_hint[len(prefix):].strip()
    return path or None

@app.post(
    "/tasks/create/tooling_probe",
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)
def create_tooling_probe_task(
    parent_task_id: Optional[int] = Query(None),
    trigger_reason: Optional[str] = Query("manual", max_length=256),
) -> Dict[str, Any]:
    """
    v2 Evidence: Create a pending task that, when resumed, captures an environment/tooling
    assumptions artifact into the ledger.
    """
    task_id = create_task(
        title="Tooling probe",
        resume_hint="Capture environment/tooling assumptions snapshot (v2 evidence).",
    )
    lineage_init(task_id=task_id, parent_task_id=parent_task_id, trigger_reason=trigger_reason)
    add_step(
        task_id=task_id,
        step_index=1,
        description="Task created (pending). Resume will capture tooling_probe snapshot.",
    )
    return {"ok": True, "task_id": task_id}


@app.post(
    "/tasks/create/tool_call",
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)
def create_tool_call(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    v4-A: Create a pending tool_call task.

    Contract:
    - Admin-gated.
    - Creates a task ledger entry plus a tool_request artifact.
    - Execution occurs only via POST /tasks/{task_id}/resume.
    """
    tool_name = str(body.get("tool_name") or "").strip()
    purpose = str(body.get("purpose") or "").strip()
    args = body.get("args") or {}

    if not tool_name:
        raise HTTPException(status_code=400, detail="tool_name is required")
    if not purpose:
        raise HTTPException(status_code=400, detail="purpose is required")
    if not isinstance(args, dict):
        raise HTTPException(status_code=400, detail="args must be an object")

    task_id = create_tool_call_task(tool_name=tool_name, args=args, purpose=purpose)
    return {"ok": True, "task_id": task_id, "status": "pending", "tool_name": tool_name}



# ---------------------------------------------------------
# v4-A Retrieval Tool Call — READ-GATED (non-admin)
# - Executes retrieval-only tool families immediately (WEB/UTILITY/LOCAL_READ)
# - Still logs tool_request + tool_result to the task ledger for auditability
# - Does NOT use /tasks/{id}/resume (which is execution/admin-gated)
# ---------------------------------------------------------

@app.post(
    "/tools/call",
    dependencies=[Depends(require_api_key)],
)
async def call_tool_readonly(request: Request, body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read-gated tool execution for retrieval-only tool families.

    Contract (v4-A):
    - Read auth only (no admin required).
    - Allowed families: WEB, UTILITY, LOCAL_READ.
    - Full audit trail: creates a tool_call task, stores tool_request + tool_result artifacts.
    - No UI wiring implied; this is backend-only plumbing for later unified UI flow.
    """
    if _is_sandbox_request(request):
        _sandbox_boundary_http("tools.call")

    tool_name = str(body.get("tool_name") or "").strip()
    purpose = str(body.get("purpose") or "").strip()
    args = body.get("args") or {}

    if not tool_name:
        raise HTTPException(status_code=400, detail="tool_name is required")
    if not purpose:
        raise HTTPException(status_code=400, detail="purpose is required")
    if not isinstance(args, dict):
        raise HTTPException(status_code=400, detail="args must be an object")

    # Tool allowlist gate (v4-A baseline)
    if not is_known_tool(tool_name):
        raise HTTPException(status_code=403, detail="Unknown tool")
    if not is_tool_allowed(tool_name):
        raise HTTPException(status_code=403, detail="Tool not allowed in this phase")

    tdef = TOOL_DEFS[tool_name]
    if tdef.family.value not in {"web", "utility", "local_read"}:
        raise HTTPException(status_code=403, detail="Tool family is not allowed for read-gated calls")

    # Create task + store tool_request artifact (auditability)
    task_id = create_tool_call_task(tool_name=tool_name, args=args, purpose=purpose)

    # Best-effort evidence hooks (do not block execution lane)
    try:
        lineage_init(task_id=task_id, parent_task_id=None, trigger_reason="readonly: tools/call")
    except Exception:
        pass
    try:
        timing_mark_started(task_id)
    except Exception:
        pass

    step_id = add_step(
        task_id=task_id,
        step_index=_task_next_step_index(task_id),
        description=f"Execute (read-gated): tool_call → {tool_name}",
    )

    tool_req = ToolRequest(
        tool_name=tool_name,
        args=args,
        purpose=purpose,
        task_id=task_id,
        step_id=step_id,
        user_id=(request.headers.get("X-ISAC-USER-ID") or "unknown"),
        chat_id=(request.headers.get("X-ISAC-CHAT-ID") or "unknown"),
    )

    result = await run_tool(tool_req)

    add_artifact(
        task_id=task_id,
        step_id=step_id,
        artifact_type="tool_result",
        metadata_json=json.dumps(
            {
                "ok": result.ok,
                "tool_name": result.tool_name,
                "failure_class": (result.failure_class.value if result.failure_class else None),
                "failure_message": result.failure_message,
                "primary": result.primary,
                "provenance": (result.provenance.__dict__ if result.provenance else None),
                "started_at": result.started_at,
                "ended_at": result.ended_at,
                "latency_ms": result.latency_ms,
            }
        ),
    )

    _task_mark_completed(task_id)
    try:
        timing_mark_finished(task_id)
    except Exception:
        pass

    return {
        "ok": True,
        "task_id": task_id,
        "tool_name": tool_name,
        "result": {
            "ok": result.ok,
            "primary": result.primary,
            "failure_class": (result.failure_class.value if result.failure_class else None),
            "failure_message": result.failure_message,
            "provenance": (result.provenance.__dict__ if result.provenance else None),
        },
    }



# ---------------------------------------------------------
# Escape Hatch v3 — Templates: File Write (dry-run by default)
# ---------------------------------------------------------

class FileWriteTaskCreateRequest(BaseModel):
    path: str = Field(..., min_length=1, max_length=512)
    content_b64: str = Field(..., min_length=1)  # UTF-8 content, base64-encoded
    dry_run: bool = True
    parent_task_id: Optional[int] = None
    trigger_reason: Optional[str] = Field(None, max_length=256)


class DiffApplyTaskCreateRequest(BaseModel):
    path: str = Field(..., min_length=1)
    unified_diff_b64: str = Field(..., min_length=1, description="Base64-encoded unified diff text")
    dry_run: bool = True


# ---------------------------------------------------------
# Bundle 3 — OKD Governed Observation (v1)
# - Task type: "governed_observation"
# - Artifacts stored in task ledger (task_artifacts)
# - Preview required: plan must be approved via explicit admin endpoint
# ---------------------------------------------------------

@app.post(
    "/okd/observation/create",
    dependencies=[Depends(require_api_key)],
)
def okd_create_observation_task(request: Request, body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a governed observation task with an Observation Plan artifact (NOT approved yet).
    Execution occurs only via POST /tasks/{task_id}/resume after explicit approval.
    """
    # Identity/session carriers (Bundle 1 continuity)
    user_id = (request.headers.get("X-ISAC-USER-ID") or "unknown").strip() or "unknown"
    chat_id = (request.headers.get("X-ISAC-CHAT-ID") or "unknown").strip() or "unknown"

    intent = str(body.get("intent") or "").strip()
    user_prompt_excerpt = str(body.get("user_prompt_excerpt") or "").strip()
    expected_risk_tier = body.get("expected_risk_tier")
    initial_queries = body.get("initial_queries") or []
    planned_reforms = body.get("planned_query_reformulations") or []
    scope_budget = body.get("scope_budget") or {}
    login_required = bool(body.get("login_required", False))

    if not intent:
        raise HTTPException(status_code=422, detail="intent is required")
    if not user_prompt_excerpt:
        raise HTTPException(status_code=422, detail="user_prompt_excerpt is required")
    try:
        expected_risk_tier = int(expected_risk_tier)
    except Exception:
        raise HTTPException(status_code=422, detail="expected_risk_tier must be int")

    if not isinstance(initial_queries, list) or not all(isinstance(q, str) and q.strip() for q in initial_queries):
        raise HTTPException(status_code=422, detail="initial_queries must be list[str] (non-empty strings)")

    if planned_reforms and (not isinstance(planned_reforms, list) or not all(isinstance(q, str) for q in planned_reforms)):
        raise HTTPException(status_code=422, detail="planned_query_reformulations must be list[str]")

    if scope_budget and not isinstance(scope_budget, dict):
        raise HTTPException(status_code=422, detail="scope_budget must be object")

    plan = {
        "intent": intent,
        "user_prompt_excerpt": user_prompt_excerpt,
        "expected_risk_tier": expected_risk_tier,
        "initial_queries": [q.strip() for q in initial_queries if q and q.strip()],
        "planned_query_reformulations": [q.strip() for q in planned_reforms if isinstance(q, str) and q.strip()],
        "scope_budget": scope_budget or {},
        "login_required": login_required,
        "approved_by_admin": False,
        "created_by": user_id,
        "created_in_chat": chat_id,
    }

    task_id = create_task(
        title="governed_observation",
        resume_hint="OKD governed observation (requires approval)",
    )
    add_step(
        task_id=task_id,
        step_index=1,
        description="Observation Plan stored. Awaiting explicit Admin approval before execution.",
    )
    okd_store_plan(task_id, plan, step_id=None)

    return {"ok": True, "task_id": task_id, "status": "pending", "approved": False}


@app.post(
    "/okd/observation/{task_id}/approve",
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)
def okd_approve_observation_plan(task_id: int, request: Request) -> Dict[str, Any]:
    """
    Explicit Admin approval for an Observation Plan.
    This is the Preview step in OKD Execution v1.
    """
    if task_id < 1:
        raise HTTPException(status_code=400, detail="task_id must be >= 1")

    user_id = (request.headers.get("X-ISAC-USER-ID") or "unknown").strip() or "unknown"

    plan = okd_load_latest_plan(task_id)
    if not plan:
        raise HTTPException(status_code=404, detail="Observation Plan not found for this task")

    if okd_is_plan_approved(task_id):
        return {"ok": True, "task_id": task_id, "approved": True, "message": "Already approved"}

    add_step(
        task_id=task_id,
        step_index=_task_next_step_index(task_id),
        description=f"Admin approved Observation Plan (Preview complete). approved_by={user_id}",
    )
    okd_store_plan_approval(task_id, approved_by=user_id, step_id=None)

    return {"ok": True, "task_id": task_id, "approved": True}



@app.post(
    "/tasks/create/file_write",
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)
def create_file_write_task(body: FileWriteTaskCreateRequest) -> Dict[str, Any]:
    # Hard caps (v3 conservative defaults)
    MAX_CONTENT_BYTES = int(os.getenv("ISAC_FILE_WRITE_MAX_BYTES", "200000"))  # 200 KB default

    try:
        raw = base64.b64decode(body.content_b64.encode("utf-8"), validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="content_b64 must be valid base64")

    if len(raw) > MAX_CONTENT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"content exceeds cap ({len(raw)} bytes > {MAX_CONTENT_BYTES} bytes)",
        )

    # Validate UTF-8 for predictable behavior
    try:
        content_text = raw.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="content must be UTF-8 text")

    sha256 = hashlib.sha256(raw).hexdigest()

    task_id = create_task(
        title="File write",
        resume_hint=f"Write file (v3 template): {body.path}",
    )

    # Linkage (v2+)
    if body.parent_task_id is not None or body.trigger_reason is not None:
        lineage_link_task(
            task_id=task_id,
            parent_task_id=body.parent_task_id,
            trigger_reason=body.trigger_reason,
        )

    add_artifact(
        task_id=task_id,
        artifact_type="file_write_proposal",
        metadata_json=json.dumps(
            {
                "path": body.path,
                "dry_run": bool(body.dry_run),
                "content_b64": body.content_b64,
                "bytes": len(raw),
                "sha256": sha256,
                "trigger_reason": body.trigger_reason,
                "parent_task_id": body.parent_task_id,
            }
        ),
    )

    add_step(
        task_id=task_id,
        step_index=1,
        description=f"Created pending file write task for {body.path} (dry_run={bool(body.dry_run)})",
    )

    return {"status": "ok", "task_id": task_id, "sha256": sha256, "bytes": len(raw), "dry_run": bool(body.dry_run)}


@app.post(
    "/tasks/create/file_replace",
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)
def create_file_replace_task(body: FileWriteTaskCreateRequest) -> Dict[str, Any]:
    # Hard caps (v3 conservative defaults)
    MAX_CONTENT_BYTES = int(os.getenv("ISAC_FILE_WRITE_MAX_BYTES", "200000"))  # 200 KB default

    try:
        raw = base64.b64decode(body.content_b64.encode("utf-8"), validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="content_b64 must be valid base64")

    if len(raw) > MAX_CONTENT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"content exceeds cap ({len(raw)} bytes > {MAX_CONTENT_BYTES} bytes)",
        )

    # Validate UTF-8 for predictable behavior
    try:
        content_text = raw.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="content must be UTF-8 text")

    sha256 = hashlib.sha256(raw).hexdigest()

    task_id = create_task(
        title="File replace",
        resume_hint=f"Write file (v3 template): {body.path}",
    )

    # Linkage (v2+)
    if body.parent_task_id is not None or body.trigger_reason is not None:
        lineage_link_task(
            task_id=task_id,
            parent_task_id=body.parent_task_id,
            trigger_reason=body.trigger_reason,
        )

    add_artifact(
        task_id=task_id,
        artifact_type="file_write_proposal",
        metadata_json=json.dumps(
            {
                "path": body.path,
                "dry_run": bool(body.dry_run),
                "content_b64": body.content_b64,
                "bytes": len(raw),
                "sha256": sha256,
                "content_b64": body.content_b64,
            }
        ),
    )

    add_step(
        task_id=task_id,
        step_index=1,
        description=f"Created pending file replace task for {body.path} (dry_run={bool(body.dry_run)})",
    )

    return {"status": "ok", "task_id": task_id, "sha256": sha256, "bytes": len(raw), "dry_run": bool(body.dry_run)}


@app.post(
    "/tasks/create/diff_apply",
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)
def create_diff_apply_task(body: DiffApplyTaskCreateRequest) -> Dict[str, Any]:
    if not ENABLE_DIFF_APPLY:
        raise HTTPException(status_code=403, detail="Diff-apply is disabled in v5 (ENABLE_DIFF_APPLY=1 to enable). Use file replace tasks instead.")

    # Hard caps (v4 conservative defaults)
    MAX_DIFF_BYTES = int(os.getenv("ISAC_DIFF_APPLY_MAX_BYTES", "200000"))  # 200 KB default

    path = body.path.strip()
    if not _is_path_allowed(path):
        raise HTTPException(status_code=403, detail="path not in write allowlist")

    resolved_path = resolve_target_path(path, must_exist=True)
    if not resolved_path:
        raise HTTPException(status_code=400, detail=f"unable to read target file: no such file for path '{path}'")

    try:
        diff_raw = base64.b64decode(body.unified_diff_b64.encode("utf-8"), validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="unified_diff_b64 must be valid base64")

    if len(diff_raw) > MAX_DIFF_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"diff exceeds cap ({len(diff_raw)} bytes > {MAX_DIFF_BYTES} bytes)",
        )

    try:
        diff_text = diff_raw.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="diff must decode as UTF-8")

    # Pre-compute before/after hashes for preview + audit
    try:
        before_text = Path(resolved_path).read_text(encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"unable to read target file: {e}")

    before_sha = hashlib.sha256(before_text.encode("utf-8")).hexdigest()

    # Validate the diff can apply cleanly (preview correctness guarantee)
    try:
        applied = _apply_unified_diff(before_text, diff_text)
    except Exception as e:
        raise HTTPException(status_code=409, detail=f"diff does not apply cleanly: {e}")

    after_text = applied["patched_text"]
    after_sha = hashlib.sha256(after_text.encode("utf-8")).hexdigest()

    task_id = create_task(
        title="Diff Apply",
        resume_hint=json.dumps(
            {
                "path": path,
                "resolved_path": resolved_path,
                "dry_run": bool(body.dry_run),
                "unified_diff_b64": body.unified_diff_b64,
                "before_sha256": before_sha,
                "after_sha256": after_sha,
                "hunks": int(applied.get("hunks") or 0),
                "additions": int(applied.get("additions") or 0),
                "deletions": int(applied.get("deletions") or 0),
            }
        ),
    )

    add_step(
        task_id=task_id,
        step_index=1,
        description=f"Created pending diff-apply task for {path} (dry_run={bool(body.dry_run)})",
    )

    add_artifact(
        task_id=task_id,
        step_id=None,
        artifact_type="diff_apply_proposal",
        metadata_json=json.dumps(
            {
                "path": path,
                "dry_run": bool(body.dry_run),
                "unified_diff_b64": body.unified_diff_b64,
                "before_sha256": before_sha,
                "after_sha256": after_sha,
                "hunks": int(applied.get("hunks") or 0),
                "additions": int(applied.get("additions") or 0),
                "deletions": int(applied.get("deletions") or 0),
                "diff_bytes": len(diff_raw),
            }
        ),
    )

    return {
        "status": "ok",
        "task_id": task_id,
        "path": path,
        "dry_run": bool(body.dry_run),
                "unified_diff_b64": body.unified_diff_b64,
        "before_sha256": before_sha,
        "after_sha256": after_sha,
        "hunks": int(applied.get("hunks") or 0),
        "additions": int(applied.get("additions") or 0),
        "deletions": int(applied.get("deletions") or 0),
        "diff_bytes": len(diff_raw),
    }

@app.post(
    "/tasks/{task_id}/abort",
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)
def abort_task(task_id: int) -> Dict[str, Any]:
    """
    Ledger-only abort:
    - No runner execution
    - Just marks task status = 'aborted'
    """
    if task_id < 1:
        raise HTTPException(status_code=400, detail="task_id must be >= 1")

    task = _task_get_min(task_id)

    # If already completed, don't mutate silently.
    if task.get("status") == "completed":
        raise HTTPException(status_code=409, detail="Task is already completed")

    _task_update_status(task_id, "aborted")
    return {"status": "ok", "task_id": task_id, "new_status": "aborted"}


@app.post(
    "/tasks/{task_id}/resume/preview",
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)
def resume_task_preview(task_id: int) -> Dict[str, Any]:
    """
    Preview what resume would do (no execution).
    """
    if task_id < 1:
        raise HTTPException(status_code=400, detail="task_id must be >= 1")

    task = _task_get_min(task_id)
    title = str(task.get("title") or "")
    resume_hint = task.get("resume_hint")

    if title == "Runner FS list":
        path = _parse_runner_fs_list_resume_hint(resume_hint)
        if not path:
            return {
                "status": "ok",
                "task_id": task_id,
                "supported": False,
                "message": "Runner FS list task is missing its path (resume_hint parse failed).",
                "task": task,
            }
        return {
            "status": "ok",
            "task_id": task_id,
            "supported": True,
            "message": "Will re-run runner fs list for the same path, log a new step+artifact, then mark the task completed.",
            "task": task,
            "plan": {"action": "runner_fs_list", "path": path},
        }

    if title == "governed_observation":
        plan = okd_load_latest_plan(task_id)
        approved = okd_is_plan_approved(task_id)
        if not plan:
            return {
                "status": "ok",
                "task_id": task_id,
                "supported": False,
                "message": "Governed observation task is missing its Observation Plan artifact.",
                "task": task,
            }
        if not approved:
            return {
                "status": "ok",
                "task_id": task_id,
                "supported": True,
                "message": "Governed observation is awaiting Admin approval. Use POST /okd/observation/{task_id}/approve, then resume to execute.",
                "task": task,
                "plan": {"action": "governed_observation", "approved": False, "observation_plan": plan},
            }
        return {
            "status": "ok",
            "task_id": task_id,
            "supported": True,
            "message": "Will execute governed observation under the approved plan, produce Expansion Log + Update Brief artifacts, then mark task completed (or fail-closed).",
            "task": task,
            "plan": {"action": "governed_observation", "approved": True, "observation_plan": plan},
        }

    if title == "Tooling probe":
        return {
            "status": "ok",
            "task_id": task_id,
            "supported": True,
            "message": "Will capture a tooling_probe snapshot (read-only) and store it as an artifact, then mark the task completed.",
            "task": task,
            "plan": {"action": "tooling_probe"},
        }

    if title in ("File write", "File replace"):
        try:
            payload = json.loads(resume_hint) if isinstance(resume_hint, str) else (resume_hint or {})
            path = str(payload.get("path") or "").strip()
            dry_run = bool(payload.get("dry_run", True))
            bytes_len = int(payload.get("bytes") or 0)
            sha256 = str(payload.get("sha256") or "")
        except Exception:
            path = ""
            dry_run = True
            bytes_len = 0
            sha256 = ""

        if not path:
            return {
                "status": "ok",
                "task_id": task_id,
                "supported": False,
                "message": "File Write task is missing required resume_hint fields.",
                "task": task,
            }

        return {
            "status": "ok",
            "task_id": task_id,
            "supported": True,
            "message": "Will replace the target file atomically (whole-file), then run HTTP auto-verify.",
            "task": task,
            "plan": {"action": "file_write", "path": path, "dry_run": dry_run, "bytes": bytes_len, "sha256": sha256},
        }

    if title == "Diff Apply":
        if not ENABLE_DIFF_APPLY:
            return {
                "status": "ok",
                "task_id": task_id,
                "supported": False,
                "message": "Diff Apply is disabled in v5. Set ENABLE_DIFF_APPLY=1 to enable, or use File Replace tasks.",
                "task": task,
            }

        parsed = _parse_diff_apply_resume_hint(resume_hint)
        if not parsed:
            return {
                "status": "ok",
                "task_id": task_id,
                "supported": False,
                "message": "Diff Apply task is missing required resume_hint fields.",
                "task": task,
            }

        path = parsed["path"]
        dry_run = bool(parsed["dry_run"])

        # Pull precomputed stats if available (created at task creation time)
        try:
            payload = json.loads(resume_hint) if isinstance(resume_hint, str) else (resume_hint or {})
            hunks = int(payload.get("hunks") or 0)
            additions = int(payload.get("additions") or 0)
            deletions = int(payload.get("deletions") or 0)
            before_sha = str(payload.get("before_sha256") or "")
            after_sha = str(payload.get("after_sha256") or "")
        except Exception:
            hunks = additions = deletions = 0
            before_sha = after_sha = ""

        return {
            "status": "ok",
            "task_id": task_id,
            "supported": True,
            "message": "Will apply the unified diff atomically to the target file, then run HTTP auto-verify.",
            "task": task,
            "plan": {
                "action": "diff_apply",
                "path": path,
                "dry_run": dry_run,
                "hunks": hunks,
                "additions": additions,
                "deletions": deletions,
                "before_sha256": before_sha,
                "after_sha256": after_sha,
            },
        }

    return {
        "status": "ok",
        "task_id": task_id,
        "supported": False,
        "message": "Resume not supported for this task type yet.",
        "task": task,
    }
@app.post(
    "/tasks/{task_id}/resume/plan",
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)
def resume_task_plan(task_id: int):
    # Alias for UI compatibility
    return resume_task_preview(task_id)

@app.post(
    "/tasks/{task_id}/resume",
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)
async def resume_task(task_id: int, request: Request) -> Dict[str, Any]:
    """
    Resume (Execution Spine):
    - Enforces single-execution concurrency guard (v2)
    - Supports:
        * Runner FS list (v1.3)
        * Tooling probe (v2 evidence)
        * File write/replace (v3 template; whole-file; dry-run supported)
        * Diff apply (v4; disabled by default in v5)
        * Governed observation (Bundle 3: OKD execution v1)
    """
    if _is_sandbox_request(request):
        _sandbox_boundary_http("tasks.resume")

    if task_id < 1:
        raise HTTPException(status_code=400, detail="task_id must be >= 1")

    task = _task_get_min(task_id)
    title = str(task.get("title") or "")
    status = str(task.get("status") or "")
    resume_hint = task.get("resume_hint")

    if status == "completed":
        raise HTTPException(status_code=409, detail="Task is already completed")
    if status == "aborted":
        raise HTTPException(status_code=409, detail="Task is aborted; create a new task instead.")

    # Human exit rule scaffold (per-lineage): stop after 2 verify failures.
    if lineage_get_verify_failures(task_id) >= 2:
        raise HTTPException(
            status_code=409,
            detail="Lineage has reached the verify-failure cap (>=2). Human intervention required before continuing.",
        )

    lock = ExecutionLock()
    try:
        lock.acquire()
    except ExecutionLockError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    timing_mark_started(task_id)

    try:
        # ---- Tool call (v4-1) ----
        if title == "tool_call":
            req_art = _task_get_latest_artifact(task_id, "tool_request")
            if not req_art:
                raise HTTPException(status_code=400, detail="Missing tool_request artifact")

            meta = req_art.get("metadata") or {}
            tool_name = str(meta.get("tool_name") or "")
            args = meta.get("args") or {}
            purpose = str(meta.get("purpose") or "")

            step_id = add_step(
                task_id=task_id,
                step_index=_task_next_step_index(task_id),
                description=f"Resume: tool_call → {tool_name}",
            )

            tool_req = ToolRequest(
                tool_name=tool_name,
                args=args,
                purpose=purpose,
                task_id=task_id,
                step_id=step_id,
                user_id=(request.headers.get("X-ISAC-USER-ID") or "unknown"),
                chat_id=(request.headers.get("X-ISAC-CHAT-ID") or "unknown"),
            )

            result = await run_tool(tool_req)

            add_artifact(
                task_id=task_id,
                step_id=step_id,
                artifact_type="tool_result",
                metadata_json=json.dumps(
                    {
                        "ok": result.ok,
                        "tool_name": result.tool_name,
                        "failure_class": (result.failure_class.value if result.failure_class else None),
                        "failure_message": result.failure_message,
                        "primary": result.primary,
                        "provenance": (result.provenance.__dict__ if result.provenance else None),
                    }
                ),
            )

            _task_mark_completed(task_id)
            timing_mark_finished(task_id)
            return {"ok": True, "task_id": task_id}

        # ---- Governed observation (Bundle 3) ----
        if title == "governed_observation":
            # Enforce Preview requirement (plan + approval) before any observation.
            plan = okd_load_latest_plan(task_id)
            if not plan:
                raise HTTPException(status_code=400, detail="Missing Observation Plan artifact (okd_observation_plan)")
            if not okd_is_plan_approved(task_id):
                raise HTTPException(status_code=409, detail="Observation Plan not approved. Run POST /okd/observation/{task_id}/approve first.")

            step_id = add_step(
                task_id=task_id,
                step_index=_task_next_step_index(task_id),
                description="Execute: governed observation under approved plan (Bundle 3).",
            )

            user_id = (request.headers.get("X-ISAC-USER-ID") or "unknown").strip() or "unknown"
            chat_id = (request.headers.get("X-ISAC-CHAT-ID") or "unknown").strip() or "unknown"

            # execute_governed_observation is responsible for:
            # - expansion log artifact
            # - update brief artifact
            # - fail-closed semantics by raising on governance violations
            result = await execute_governed_observation(
                task_id=task_id,
                user_id=user_id,
                chat_id=chat_id,
                enabled_tools=None,
            )

            add_artifact(
                task_id=task_id,
                step_id=step_id,
                artifact_type="okd_execution_result",
                metadata_json=json.dumps({"ok": True, "result": result}),
            )

            _task_mark_completed(task_id)
            timing_mark_finished(task_id)
            return {"status": "ok", "task_id": task_id, "result": result}

        # ---- File write (v3 template) ----
        if title in ("File write", "File replace"):
            proposal = _task_get_latest_artifact(task_id, "file_write_proposal")
            if not proposal:
                raise HTTPException(status_code=400, detail="Missing file_write_proposal artifact")

            meta = proposal.get("metadata") or {}
            path = str(meta.get("path") or "")
            dry_run = bool(meta.get("dry_run", True))
            content_b64 = str(meta.get("content_b64") or "")
            sha256 = str(meta.get("sha256") or "")
            declared_bytes = int(meta.get("bytes") or 0)

            if not _is_path_allowed(path):
                raise HTTPException(status_code=403, detail="Target path is not in write allowlist")

            try:
                raw = base64.b64decode(content_b64.encode("utf-8"), validate=True)
            except Exception:
                raise HTTPException(status_code=400, detail="Stored content_b64 is invalid base64")

            if declared_bytes and declared_bytes != len(raw):
                add_step(
                    task_id=task_id,
                    step_index=_task_next_step_index(task_id),
                    description=f"Warning: stored byte count mismatch (declared={declared_bytes}, actual={len(raw)})",
                )

            if hashlib.sha256(raw).hexdigest() != sha256:
                raise HTTPException(status_code=409, detail="Content hash mismatch; refusing to proceed")

            content_text = raw.decode("utf-8")

            next_idx = _task_next_step_index(task_id)
            step_id = add_step(
                task_id=task_id,
                step_index=next_idx,
                description=f"Resume: file write template → {path} (dry_run={dry_run})",
            )

            if dry_run:
                add_artifact(
                    task_id=task_id,
                    step_id=step_id,
                    artifact_type="file_write_dry_run",
                    metadata_json=json.dumps({"path": path, "sha256": sha256, "bytes": len(raw), "would_write": True}),
                )
            else:
                try:
                    _atomic_write_text(path, content_text)
                except Exception as e:
                    failures_record(task_id, step_id=step_id, failure_class=FailureClass.WRITE.value, message=str(e))
                    raise
                add_artifact(
                    task_id=task_id,
                    step_id=step_id,
                    artifact_type="file_write_applied",
                    metadata_json=json.dumps({"path": path, "sha256": sha256, "bytes": len(raw)}),
                )

            verify = await _http_verify_template()
            add_artifact(
                task_id=task_id,
                step_id=step_id,
                artifact_type="verify_http",
                metadata_json=json.dumps(verify),
            )

            if not verify.get("ok"):
                failures_record(task_id, step_id=step_id, failure_class=FailureClass.VERIFY.value, message="HTTP verify template failed")
                lineage_inc_verify_failures(task_id)
                raise HTTPException(status_code=409, detail="Verify failed; refusing to mark task completed")

            _task_update_status(task_id, "completed")
            timing_mark_finished(task_id)
            return {"status": "ok", "task_id": task_id, "dry_run": dry_run, "verify": verify}

        # ---- Runner FS list ----
        if title == "Runner FS list":
            path = _parse_runner_fs_list_resume_hint(resume_hint)
            if not path:
                raise HTTPException(status_code=400, detail="Unable to parse path from resume_hint")

            next_idx = _task_next_step_index(task_id)
            step_id = add_step(
                task_id=task_id,
                step_index=next_idx,
                description=f"Resume: list directory: {path}",
            )

            result = await _runner_get("/runner/fs/list", params={"path": path})

            add_artifact(
                task_id=task_id,
                step_id=step_id,
                artifact_type="runner_fs_list",
                metadata_json=json.dumps({"path": path, "result": result}),
            )

            _task_update_status(task_id, "completed")
            timing_mark_finished(task_id)

            return {
                "status": "ok",
                "task_id": task_id,
                "action": "runner_fs_list",
                "completed": True,
            }

        # ---- Diff Apply (v4) ----
        if title == "Diff Apply":
            proposal = _task_get_latest_artifact(task_id, "diff_apply_proposal")
            if not proposal:
                raise HTTPException(status_code=400, detail="Missing diff_apply_proposal artifact")

            meta = proposal.get("metadata") or {}
            path = str(meta.get("path") or "").strip()
            dry_run = bool(meta.get("dry_run", True))
            unified_diff_b64 = str(meta.get("unified_diff_b64") or "").strip()

            if not path or not unified_diff_b64:
                raise HTTPException(status_code=400, detail="Diff Apply proposal missing required fields")

            if not _is_path_allowed(path):
                raise HTTPException(status_code=403, detail="path not in write allowlist")

            try:
                diff_raw = base64.b64decode(unified_diff_b64.encode("utf-8"), validate=True)
                diff_text = diff_raw.decode("utf-8")
            except Exception:
                raise HTTPException(status_code=400, detail="unified_diff_b64 must be valid base64 UTF-8 diff")

            try:
                resolved_path = str(meta.get("resolved_path") or "").strip() or resolve_target_path(path, must_exist=True)
                if not resolved_path:
                    raise HTTPException(status_code=400, detail=f"unable to read target file: no such file for path '{path}'")
                before_text = Path(resolved_path).read_text(encoding="utf-8")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"unable to read target file: {e}")

            before_sha = hashlib.sha256(before_text.encode("utf-8")).hexdigest()

            try:
                applied = _apply_unified_diff(before_text, diff_text)
            except Exception as e:
                failures_record(task_id=task_id, step_id=None, failure_class=FailureClass.WRITE.value, message=f"diff apply failed: {e}")
                raise HTTPException(status_code=409, detail=f"diff does not apply cleanly: {e}")

            after_text = applied["patched_text"]
            after_sha = hashlib.sha256(after_text.encode("utf-8")).hexdigest()

            next_idx = _task_next_step_index(task_id)
            step_id = add_step(
                task_id=task_id,
                step_index=next_idx,
                description=f"Resume: diff-apply → {path} (dry_run={dry_run})",
            )

            if dry_run:
                add_artifact(
                    task_id=task_id,
                    step_id=step_id,
                    artifact_type="diff_apply_dry_run",
                    metadata_json=json.dumps(
                        {
                            "path": path,
                            "dry_run": True,
                            "before_sha256": before_sha,
                            "after_sha256": after_sha,
                            "hunks": int(applied.get("hunks") or 0),
                            "additions": int(applied.get("additions") or 0),
                            "deletions": int(applied.get("deletions") or 0),
                            "diff_bytes": len(diff_raw),
                            "would_apply": True,
                        }
                    ),
                )
            else:
                try:
                    _atomic_write_text(path, after_text)
                except Exception as e:
                    failures_record(task_id=task_id, step_id=step_id, failure_class=FailureClass.WRITE.value, message=str(e))
                    raise

                add_artifact(
                    task_id=task_id,
                    step_id=step_id,
                    artifact_type="diff_apply_applied",
                    metadata_json=json.dumps(
                        {
                            "path": path,
                            "dry_run": False,
                            "before_sha256": before_sha,
                            "after_sha256": after_sha,
                            "hunks": int(applied.get("hunks") or 0),
                            "additions": int(applied.get("additions") or 0),
                            "deletions": int(applied.get("deletions") or 0),
                            "diff_bytes": len(diff_raw),
                        }
                    ),
                )

            verify = await _http_verify_template()
            add_artifact(
                task_id=task_id,
                step_id=step_id,
                artifact_type="verify_http",
                metadata_json=json.dumps(verify),
            )

            if not verify.get("ok"):
                failures_record(task_id, step_id=step_id, failure_class=FailureClass.VERIFY.value, message="HTTP verify template failed")
                lineage_inc_verify_failures(task_id)
                raise HTTPException(status_code=409, detail="Verify failed; refusing to mark task completed")

            _task_update_status(task_id, "completed")
            timing_mark_finished(task_id)
            return {
                "status": "ok",
                "task_id": task_id,
                "completed": True,
                "action": "diff_apply",
                "dry_run": dry_run,
                "verify": verify,
            }

        # ---- Tooling probe (v2 evidence) ----
        if title == "Tooling probe":
            next_idx = _task_next_step_index(task_id)
            step_id = add_step(
                task_id=task_id,
                step_index=next_idx,
                description="Execute: capture tooling_probe snapshot (read-only).",
            )

            snapshot = tooling_probe_snapshot()

            add_artifact(
                task_id=task_id,
                step_id=step_id,
                artifact_type="tooling_probe",
                metadata_json=json.dumps(snapshot),
            )

            _task_update_status(task_id, "completed")
            timing_mark_finished(task_id)
            return {
                "status": "ok",
                "task_id": task_id,
                "action": "tooling_probe",
                "completed": True,
            }

        # Unknown task type
        raise HTTPException(status_code=400, detail="Resume not supported for this task type yet.")

    except HTTPException:
        failures_record(task_id=task_id, step_id=None, failure_class=FailureClass.GUARDRAIL, message="HTTPException raised during resume")
        raise
    except Exception as exc:
        fc = _failure_class_from_exception(exc)
        failures_record(task_id=task_id, step_id=None, failure_class=str(fc), message=str(exc))
        raise HTTPException(status_code=500, detail=f"Resume failed: {exc}")
    finally:
        try:
            lock.release()
        finally:
            pass
@app.post(
    "/tasks/{task_id}/resume/confirm",
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)
async def resume_task_confirm(task_id: int, request: Request):
    # UI compatibility shim: confirm → execute
    return await resume_task(task_id, request)

# ---------------------------------------------------------
# Home Assistant router & endpoints
# ---------------------------------------------------------

ha_router = APIRouter(
    prefix="/homeassistant",
    tags=["homeassistant"],
    dependencies=[Depends(require_api_key)],
)


def ensure_ha_client() -> HomeAssistantClient:
    if not HA_CLIENT:
        raise HTTPException(
            status_code=503,
            detail="Home Assistant client not configured or unavailable.",
        )
    return HA_CLIENT


@ha_router.get("/entities")
async def list_entities(
    domain: Optional[str] = Query(
        default=None,
        description="Filter entities by domain (e.g. 'light', 'switch', 'sensor').",
    ),
    search: Optional[str] = Query(
        default=None,
        description="Optional case-insensitive search across entity_id and friendly_name.",
    ),
) -> Dict[str, Any]:
    client = ensure_ha_client()
    states = await client.get_states()

    results: List[Dict[str, Any]] = []
    for state in states:
        entity_id = state.get("entity_id", "")
        attributes = state.get("attributes", {})
        friendly_name = attributes.get("friendly_name", "")

        if domain and not entity_id.startswith(domain + "."):
            continue

        if search:
            needle = search.lower()
            if needle not in entity_id.lower() and needle not in str(friendly_name).lower():
                continue

        results.append(
            {
                "entity_id": entity_id,
                "state": state.get("state"),
                "friendly_name": friendly_name,
                "domain": entity_id.split(".", 1)[0] if "." in entity_id else None,
            }
        )

    return {
        "status": "ok",
        "count": len(results),
        "entities": results,
    }


@ha_router.get("/states/{entity_id}")
async def get_entity_state(entity_id: str) -> Dict[str, Any]:
    client = ensure_ha_client()

    entity_id = entity_id.strip()
    if not entity_id:
        raise HTTPException(status_code=400, detail="Entity ID must not be empty")

    try:
        state = await client.get_state(entity_id)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching state for {entity_id}: {exc}",
        ) from exc

    return {
        "status": "ok",
        "entity_id": entity_id,
        "state": state,
    }


@ha_router.post("/services/{domain}/{service}")
async def call_service(
    domain: str,
    service: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    client = ensure_ha_client()

    domain = domain.strip()
    service = service.strip()
    if not domain or not service:
        raise HTTPException(
            status_code=400,
            detail="Domain and service must not be empty",
        )

    try:
        result = await client.call_service(domain=domain, service=service, data=data)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Error calling service {domain}.{service}: {exc}",
        ) from exc

    return {
        "status": "ok",
        "domain": domain,
        "service": service,
        "result": result,
    }


@ha_router.get("/summary")
async def ha_summary() -> Dict[str, Any]:
    client = ensure_ha_client()
    states = await client.get_states()

    total = len(states)
    by_domain: Dict[str, int] = {}
    lights: List[Dict[str, Any]] = []
    switches: List[Dict[str, Any]] = []
    problem_entities: List[Dict[str, Any]] = []

    for entry in states:
        entity_id = entry.get("entity_id", "")
        domain = entity_id.split(".", 1)[0] if "." in entity_id else "unknown"
        by_domain[domain] = by_domain.get(domain, 0) + 1

        state = entry.get("state")
        attributes = entry.get("attributes", {})
        friendly_name = attributes.get("friendly_name", entity_id)

        if domain == "light":
            lights.append(
                {
                    "entity_id": entity_id,
                    "friendly_name": friendly_name,
                    "state": state,
                }
            )
        elif domain == "switch":
            switches.append(
                {
                    "entity_id": entity_id,
                    "friendly_name": friendly_name,
                    "state": state,
                }
            )

        if state in ("unavailable", "unknown"):
            problem_entities.append(entry)

    now_utc = datetime.now(timezone.utc).isoformat()

    return {
        "status": "ok",
        "generated_at_utc": now_utc,
        "entity_counts": {
            "total": total,
            "by_domain": by_domain,
        },
        "domains": {
            "light": lights,
            "switch": switches,
        },
        "problem_entities": problem_entities,
    }


# ---------------------------------------------------------
# Grocy router & endpoints
# ---------------------------------------------------------

grocy_router = APIRouter(
    prefix="/grocy",
    tags=["grocy"],
    dependencies=[Depends(require_api_key)],
)


def _grocy_disabled_response(reason: Optional[str] = None) -> Dict[str, Any]:
    base_reason = (
        "Grocy not configured (missing GROCY_HOME_A_* and/or GROCY_HOME_B_* env vars, "
        "or client not initialized)"
    )
    return {
        "status": "disabled",
        "reason": reason or base_reason,
    }


async def get_grocy_client() -> Optional[GrocyClient]:
    """
    Dependency that returns a configured GrocyClient, or None if
    configuration is missing / invalid.
    """
    try:
        client = await create_grocy_client()
    except GrocyError:
        return None
    return client


# ----------------------------
# Phase 6.45 — Scan-Driven Product Knowledge (Read-only)
# ----------------------------

class ProductSuggestion(BaseModel):
    """
    A sparse suggestion object that can be progressively enriched over time.
    v1 starts mostly-empty; later phases populate from OpenFoodFacts / learned mappings.
    """
    name: Optional[str] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    location_id: Optional[int] = None
    qu_id_purchase: Optional[int] = None
    qu_id_stock: Optional[int] = None
    confidence: Literal["low", "medium", "high"] = "low"
    notes: Optional[str] = None


class GrocyBarcodeInspectResponse(BaseModel):
    barcode: str
    household: Literal["home_a", "home_b"]
    found_in_grocy: bool
    product: Optional[Dict[str, Any]] = None

    # NEW: external lookup payload (minimal/debuggable)
    external: Optional[Dict[str, Any]] = None

    suggestion: ProductSuggestion
    next_actions: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)


async def _try_grocy_lookup_by_barcode(
    client: GrocyClient,
    household: str,
    barcode: str,
) -> Tuple[bool, Optional[Dict[str, Any]], List[str]]:
    """
    Attempt to look up a barcode in Grocy using whichever method the GrocyClient exposes.
    This keeps main.py stable even as GrocyClient evolves.
    """
    tried_methods = [
        "get_product_by_barcode",
        "get_product_for_barcode",
        "lookup_barcode",
        "barcode_lookup",
        "get_barcode_product",
        "get_product_from_barcode",
        "get_product_by_ean",
        "get_product_by_upc",
    ]

    reasons: List[str] = []
    fn = None
    for name in tried_methods:
        candidate = getattr(client, name, None)
        if callable(candidate):
            fn = candidate
            break

    if fn is None:
        reasons.append("grocy_client_has_no_barcode_lookup_method")
        return (False, None, reasons)

    try:
        result = await fn(household=household, barcode=barcode)  # type: ignore[misc]
    except TypeError:
        # Some implementations might use a different signature; try with minimal args.
        try:
            result = await fn(barcode=barcode)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"grocy_barcode_lookup_error: {exc}")
            return (False, None, reasons)
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"grocy_barcode_lookup_error: {exc}")
        return (False, None, reasons)

    if result is None:
        reasons.append("barcode_not_found_in_grocy")
        return (False, None, reasons)

    if isinstance(result, dict):
        return (True, result, reasons)

    # If the client returns a list or other shape, wrap it safely.
    return (True, {"result": result}, reasons)


def _build_external_minimal_payload(source: str, off_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a minimal external payload suitable for debugging/UI without overwhelming.
    """
    product = off_payload.get("product") if isinstance(off_payload.get("product"), dict) else {}
    if not isinstance(product, dict):
        product = {}

    return {
        "source": source,
        "status": off_payload.get("status"),
        "code": off_payload.get("code"),
        "product_name": product.get("product_name") or product.get("product_name_en"),
        "brands": product.get("brands"),
        "categories": product.get("categories"),
        "image_url": product.get("image_url") or product.get("image_front_url"),
        "url": product.get("url"),
    }


@grocy_router.get(
    "/inspect-barcode",
    response_model=GrocyBarcodeInspectResponse,
    summary="Inspect a barcode (read-only) and return a suggestion object. No writes.",
)
async def grocy_inspect_barcode(
    barcode: str = Query(..., min_length=1, description="Barcode / UPC / EAN string"),
    household: Optional[str] = Query(
        default=None,
        description="REQUIRED. home_a | home_b (no 'all' allowed for scan reasoning).",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> GrocyBarcodeInspectResponse:
    hh = _require_household_write_scope(household)

    code = (barcode or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="barcode must not be empty")

    if client is None:
        # Return disabled as a normal response_model payload with low confidence.
        # This keeps callers from crashing on 503s when services are intentionally off.
        return GrocyBarcodeInspectResponse(
            barcode=code,
            household=hh,  # type: ignore[arg-type]
            found_in_grocy=False,
            product=None,
            external=None,
            suggestion=ProductSuggestion(
                confidence="low",
                notes="Grocy client not configured; cannot inspect barcode.",
            ),
            next_actions=["configure_grocy", "external_lookup", "user_confirmation_required"],
            reasons=["grocy_disabled"],
        )

    found, product, reasons = await _try_grocy_lookup_by_barcode(
        client=client,
        household=hh,
        barcode=code,
    )

    if found and product:
        # Minimal inference: if Grocy provides a name, surface it as high confidence.
        name = None
        for key in ("name", "product_name", "product", "title"):
            val = product.get(key) if isinstance(product, dict) else None
            if val:
                name = str(val)
                break

        suggestion = ProductSuggestion(
            name=name,
            confidence="high" if name else "medium",
            notes="Found existing product in Grocy for this barcode.",
        )

        return GrocyBarcodeInspectResponse(
            barcode=code,
            household=hh,  # type: ignore[arg-type]
            found_in_grocy=True,
            product=product,
            external=None,
            suggestion=suggestion,
            next_actions=["no_action_needed", "user_confirmation_required"],
            reasons=reasons,
        )

    # ---------- NEW: External lookup fallback (OpenFoodFacts) ----------
    external_payload: Optional[Dict[str, Any]] = None
    external_suggestion: Dict[str, Optional[str]] = {"name": None, "brand": None, "category": None}

    try:
        off_client = create_openfoodfacts_client()
        off_raw = off_client.lookup_barcode(code)
        external_payload = _build_external_minimal_payload("openfoodfacts", off_raw)

        # OFF status 1 indicates product exists
        if isinstance(off_raw, dict) and off_raw.get("status") == 1:
            external_suggestion = extract_suggestion_from_off_payload(off_raw)
            reasons.append("found_in_openfoodfacts")
        else:
            reasons.append("not_found_in_openfoodfacts")

    except OpenFoodFactsError as exc:
        reasons.append(f"openfoodfacts_error: {exc}")
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"openfoodfacts_error: {exc}")

    # Build the suggestion object from external lookup, if any.
    name = external_suggestion.get("name")
    brand = external_suggestion.get("brand")
    category = external_suggestion.get("category")

    if name or brand or category:
        suggestion = ProductSuggestion(
            name=name,
            brand=brand,
            category=category,
            confidence="medium",
            notes="Not found in Grocy. External lookup suggests this product. Approval required before any creation.",
        )
        next_actions = ["user_confirmation_required"]
    else:
        # Not found (or lookup unavailable): return a sparse suggestion object.
        suggestion = ProductSuggestion(
            confidence="low",
            notes="No existing product found in Grocy for this barcode.",
        )
        next_actions = ["external_lookup", "user_confirmation_required"]

    if "grocy_client_has_no_barcode_lookup_method" in reasons:
        # Preserve the Step 1 signal, but now we *also* tried external lookup.
        suggestion.notes = (
            "Grocy client does not yet support barcode lookup; "
            "external lookup was attempted. Approval required before any creation."
        )
        # Make it explicit we still need Grocy mapping support later.
        if "add_grocy_barcode_lookup" not in next_actions:
            next_actions = ["add_grocy_barcode_lookup", "user_confirmation_required"]

    return GrocyBarcodeInspectResponse(
        barcode=code,
        household=hh,  # type: ignore[arg-type]
        found_in_grocy=False,
        product=None,
        external=external_payload,
        suggestion=suggestion,
        next_actions=next_actions,
        reasons=reasons,
    )


# ----------------------------
# Phase 6.45 — Step 4: Confirm + Create + Link + Optionally Add Stock (Explicit approval only)
# ----------------------------
    
class GrocyConfirmCreateAndOptionallyStockRequest(BaseModel):
    household: Literal["home_a", "home_b"] = Field(..., description="Target household (home_a | home_b)")
    barcode: str = Field(..., min_length=1, description="Barcode / UPC / EAN string")
    name: str = Field(..., min_length=1, description="Final authoritative product name")
    location_id: int = Field(..., ge=1, description="Grocy location_id to assign to the product")
    qu_id_purchase: int = Field(..., ge=1, description="Grocy purchase unit ID (required)")
    qu_id_stock: int = Field(..., ge=1, description="Grocy stock unit ID (required)")
    
    # Optional stock add (explicit)
    add_stock: bool = Field(False, description="If true, add stock after create+link")
    quantity: Optional[float] = Field(default=None, gt=0, description="Quantity to add (required if add_stock=true)")
    best_before_date: Optional[str] = Field(default=None, description="YYYY-MM-DD (optional)")
    purchased_date: Optional[str] = Field(default=None, description="YYYY-MM-DD (optional)")
    price: Optional[float] = Field(default=None, ge=0, description="Optional price (>= 0)")
    
    # Advisory only (not used for behavior yet)
    brand: Optional[str] = None
    category: Optional[str] = None
    
    
@grocy_router.post(
    "/confirm-create-and-optionally-stock",
    summary="Confirm-create product from inspection, link barcode, and optionally add stock (explicit only).",
)
async def grocy_confirm_create_and_optionally_stock(
    body: GrocyConfirmCreateAndOptionallyStockRequest,
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    """
    Phase 6.45 Step 4 (LOCKED):
    - Requires explicit final authoritative fields
    - Creates product in Grocy
    - Links barcode in Grocy
    - Optionally adds stock ONLY if add_stock=true and quantity provided
    - NO shopping list changes
    - NO inference
    """
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Grocy client not configured; confirm-create unavailable.",
        )
    
    household = body.household.strip().lower()
    if household not in {"home_a", "home_b"}:
        raise HTTPException(status_code=400, detail="household must be 'home_a' or 'home_b'")
    
    barcode = body.barcode.strip()
    if not barcode:
        raise HTTPException(status_code=400, detail="barcode must not be empty")
    
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")
    
    if body.add_stock and body.quantity is None:
        raise HTTPException(status_code=400, detail="quantity is required when add_stock=true")
    
    # 1) Create product
    try:
        created = await client.create_product(
            household=household,
            name=name,
            location_id=body.location_id,
            qu_id_purchase=body.qu_id_purchase,
            qu_id_stock=body.qu_id_stock,
        )
    except GrocyError as exc:
        raise HTTPException(status_code=502, detail=f"Error creating product in Grocy: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Unexpected error creating product: {exc}") from exc
    
    # Extract product_id robustly
    product_id: Optional[int] = None
    if isinstance(created, dict):
        pid = (
            created.get("product_id")
            or created.get("id")
            or created.get("created_object_id")
            or (created.get("product") or {}).get("id")
        )
        if pid is not None:
            try:
                product_id = int(pid)
            except (TypeError, ValueError):
                product_id = None
    
    if product_id is None:
        raise HTTPException(
            status_code=500,
            detail={"error": "Could not determine product_id", "created": created},
        )
    
    # 2) Link barcode (explicit canonical method)
    try:
        barcode_link = await client.link_barcode_to_product(
            household=household,
            barcode=barcode,
            product_id=product_id,
        )
    except GrocyError as exc:
        raise HTTPException(status_code=502, detail=f"Error linking barcode in Grocy: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Unexpected error linking barcode: {exc}") from exc
    
    stock_result: Optional[Dict[str, Any]] = None
    
    # 3) Optional stock add (ONLY if explicitly requested)
    if body.add_stock:
        stock_payload: Dict[str, Any] = {"amount": float(body.quantity)}  # Grocy expects "amount"
    
        # Optional fields if provided
        if body.best_before_date:
            stock_payload["best_before_date"] = body.best_before_date
        if body.purchased_date:
            stock_payload["purchased_date"] = body.purchased_date
        if body.price is not None:
            stock_payload["price"] = float(body.price)
    
        try:
            stock_result = await client.add_stock(
                household=household,
                product_id=product_id,
                payload=stock_payload,
            )
        except GrocyError as exc:
            raise HTTPException(status_code=502, detail=f"Error adding stock in Grocy: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Unexpected error adding stock: {exc}") from exc
    
    return {
        "status": "ok",
        "household": household,
        "barcode": barcode,
        "product_id": product_id,
        "product": created,
        "barcode_link": barcode_link,
        "stock_added": bool(body.add_stock),
        "stock_result": stock_result,
        "advisory_metadata": {"brand": body.brand, "category": body.category},
        "notes": [
            "Phase 6.45 Step 4: explicit confirm-create + optional stock only",
            "No shopping list changes",
            "No inference",
        ],
    }
    

@grocy_router.get("/health")
async def grocy_health(
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        return _grocy_disabled_response()

    try:
        info = await client.health()
        return {"status": "ok", "instances": info}
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}


@grocy_router.get("/stock")
async def grocy_stock(
    household: Optional[str] = Query(
        default=None,
        description="REQUIRED. Scope by household: 'home_a', 'home_b', or explicitly 'all'.",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        return _grocy_disabled_response()

    hh = _require_household_query(household)

    try:
        raw = await client.get_stock_overview(household=hh)
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}

    filtered_payload: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            filtered_payload[key] = filter_stock_by_household(value, hh)
        else:
            filtered_payload[key] = value

    return {
        "status": "ok",
        "household": hh,
        "stock": filtered_payload,
    }


@grocy_router.get("/shopping-list")
async def grocy_shopping_list(
    household: Optional[str] = Query(
        default=None,
        description="REQUIRED. Scope by household: 'home_a', 'home_b', or explicitly 'all'.",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        return _grocy_disabled_response()

    hh = _require_household_query(household)

    try:
        data = await client.get_shopping_list(household=hh)
        return {
            "status": "ok",
            "household": hh,
            "shopping_list": data,
        }
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}


@grocy_router.get("/products")
async def grocy_products(
    household: Optional[str] = Query(
        default=None,
        description="REQUIRED. Scope by household: 'home_a', 'home_b', or explicitly 'all'.",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        return _grocy_disabled_response()

    hh = _require_household_query(household)

    try:
        products = await client.get_products(household=hh)
        return {
            "status": "ok",
            "household": hh,
            "products": products,
        }
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}


@grocy_router.get("/locations")
async def grocy_locations(
    household: Optional[str] = Query(
        default=None,
        description="REQUIRED. Scope by household: 'home_a', 'home_b', or explicitly 'all'.",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        return _grocy_disabled_response()

    hh = _require_household_query(household)

    try:
        locations = await client.get_locations(household=hh)
        return {
            "status": "ok",
            "household": hh,
            "locations": locations,
        }
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}

# ----------------------------
# NEW: Grocy location creation (explicit, Gate A)
# ----------------------------

class GrocyCreateLocationRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Location name")
    description: Optional[str] = None
    is_freezer: int = Field(0, description="1 if freezer, 0 otherwise")


@grocy_router.post("/locations/{household}")
async def grocy_create_location(
    household: str,
    body: GrocyCreateLocationRequest,
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Grocy client not configured; create location unavailable.",
        )

    hh = (household or "").strip().lower()
    if hh not in {"home_a", "home_b"}:
        raise HTTPException(
            status_code=400,
            detail="household must be 'home_a' or 'home_b'",
        )

    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")

    payload = {
        "name": name,
        "description": body.description,
        "is_freezer": 1 if body.is_freezer else 0,
    }

    try:
        created = await _acall_first_existing(
            client,
            [
                "create_location",
                "create_grocy_location",
                "add_location",
                "create_location_for_household",
            ],
            household=hh,
            payload=payload,
        )
    except HTTPException:
        raise
    except GrocyError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Error creating location in Grocy: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error creating location: {exc}",
        ) from exc

    return {
        "status": "ok",
        "household": hh,
        "location": created,
    }

# ----------------------------
# NEW: Quantity Units endpoint (Phase 6.4 Step 4)
# ----------------------------

@grocy_router.get("/quantity-units")
async def grocy_quantity_units(
    household: Optional[str] = Query(
        default=None,
        description="REQUIRED. Scope by household: 'home_a', 'home_b', or explicitly 'all'.",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    """
    List Grocy quantity units via /api/objects/quantity_units.
    Required for explicit unit selection during product creation.
    """
    if client is None:
        return _grocy_disabled_response()

    hh = _require_household_query(household)

    try:
        units = await client.get_quantity_units(household=hh)
        return {
            "status": "ok",
            "household": hh,
            "quantity_units": units,
        }
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}


@grocy_router.get("/summary")
async def grocy_summary(
    household: Optional[str] = Query(
        default=None,
        description="REQUIRED. Scope by household: 'home_a', 'home_b', or explicitly 'all'.",
    ),
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        return _grocy_disabled_response()

    hh = _require_household_query(household)

    try:
        stock = await client.get_stock_overview(household=hh)
        shopping_list = await client.get_shopping_list(household=hh)
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}

    filtered_stock: Dict[str, Any] = {}
    for key, value in stock.items():
        if isinstance(value, list):
            filtered_stock[key] = filter_stock_by_household(value, hh)
        else:
            filtered_stock[key] = value

    return {
        "status": "ok",
        "household": hh,
        "stock": filtered_stock,
        "shopping_list": shopping_list,
    }


@grocy_router.get("/combined")
async def grocy_combined(
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        return _grocy_disabled_response()

    try:
        combined = await _acall_first_existing(
            client,
            [
                "get_combined",
                "combined_inventory",
                "get_combined_inventory",
                "combined",
            ],
        )
        return {"status": "ok", "combined": combined}
    except HTTPException:
        raise
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def _normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum() or ch.isspace()).strip()


def _build_inventory_index_with_names(
    stock_payload: Dict[str, Any],
    household: str,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    raw_items: List[Dict[str, Any]] = []
    for value in stock_payload.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    raw_items.append(item)

    filtered_items = filter_stock_by_household(raw_items, household)

    index: Dict[str, float] = {}
    display_names: Dict[str, str] = {}
    for item in filtered_items:
        product_name = item.get("product") or item.get("name")
        if not product_name:
            continue
        product_name_str = str(product_name)
        norm = _normalize_name(product_name_str)
        try:
            amount = float(item.get("amount", 0))
        except (TypeError, ValueError):
            amount = 0.0

        index[norm] = index.get(norm, 0.0) + amount
        if norm not in display_names:
            display_names[norm] = product_name_str

    return index, display_names


@grocy_router.get("/compare")
async def grocy_compare(
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        return _grocy_disabled_response()

    try:
        stock_a = await client.get_stock_overview(household="home_a")
        stock_b = await client.get_stock_overview(household="home_b")
    except GrocyError as exc:
        return {"status": "error", "error": str(exc)}

    inv_a, names_a = _build_inventory_index_with_names(stock_a, "home_a")
    inv_b, names_b = _build_inventory_index_with_names(stock_b, "home_b")

    all_keys = set(inv_a.keys()) | set(inv_b.keys())

    only_home_a: List[Dict[str, Any]] = []
    only_home_b: List[Dict[str, Any]] = []
    both: List[Dict[str, Any]] = []
    diffs: List[Dict[str, Any]] = []

    for key in all_keys:
        amount_a = inv_a.get(key, 0.0)
        amount_b = inv_b.get(key, 0.0)
        display_name = names_a.get(key) or names_b.get(key) or key

        in_a = amount_a > 0
        in_b = amount_b > 0

        if in_a and in_b:
            entry = {
                "name": display_name,
                "home_a_amount": amount_a,
                "home_b_amount": amount_b,
                "delta": amount_a - amount_b,
            }
            both.append(entry)
            if amount_a != amount_b:
                diffs.append(entry)
        elif in_a:
            only_home_a.append(
                {
                    "name": display_name,
                    "home_a_amount": amount_a,
                }
            )
        elif in_b:
            only_home_b.append(
                {
                    "name": display_name,
                    "home_b_amount": amount_b,
                }
            )

    def _sort_by_name(item: Dict[str, Any]) -> str:
        return str(item.get("name", "")).lower()

    only_home_a.sort(key=_sort_by_name)
    only_home_b.sort(key=_sort_by_name)
    both.sort(key=_sort_by_name)
    diffs.sort(key=_sort_by_name)

    return {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "households": ["home_a", "home_b"],
        "only_home_a": only_home_a,
        "only_home_b": only_home_b,
        "both": both,
        "diffs": diffs,
    }


# ----------------------------
# Grocy product write parity (Phase 6.4 Step 4) - Units Fix
# ----------------------------

class GrocyCreateProductRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Product name")
    location_id: int = Field(..., ge=1, description="Grocy location_id to assign to the product")
    qu_id_purchase: int = Field(..., ge=1, description="Grocy purchase unit ID (required)")
    qu_id_stock: int = Field(..., ge=1, description="Grocy stock unit ID (required)")


@grocy_router.post("/products/{household}")
async def grocy_create_product(
    household: str,
    body: GrocyCreateProductRequest,
    client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Grocy client not configured; create product unavailable.",
        )

    hh = (household or "").strip().lower()
    if hh not in {"home_a", "home_b"}:
        raise HTTPException(
            status_code=400,
            detail="household must be 'home_a' or 'home_b' for write operations",
        )

    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name must not be empty")

    try:
        created = await client.create_product(
            household=hh,
            name=name,
            location_id=body.location_id,
            qu_id_purchase=body.qu_id_purchase,
            qu_id_stock=body.qu_id_stock,
        )
    except GrocyError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Error creating product in Grocy: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error creating product: {exc}",
        ) from exc

    product_id: Optional[int] = None
    if isinstance(created, dict):
        pid = (
            created.get("id")
            or created.get("product_id")
            or created.get("created_object_id")
        )
        if pid is not None:
            try:
                product_id = int(pid)
            except (TypeError, ValueError):
                product_id = None

    return {
        "status": "ok",
        "household": hh,
        "product_id": product_id,
        "product": created,
    }


# ----------------------------
# Jarvis-managed shopping lists
# ----------------------------

VALID_LIST_HOUSEHOLDS = {"home_a", "home_b", "shared"}


def _normalize_list_household(household: str) -> str:
    h = (household or "").strip().lower()
    if h not in VALID_LIST_HOUSEHOLDS:
        raise HTTPException(
            status_code=400,
            detail="household must be 'home_a', 'home_b', or 'shared'",
        )
    return h


class ShoppingListCreateRequest(BaseModel):
    name: str
    quantity: Optional[str] = None
    source: Optional[str] = "manual"


@grocy_router.get("/list/{household}")
async def jarvis_shopping_list_get(
    household: str,
    include_completed: bool = Query(
        default=False,
        description="If true, include completed items as well.",
    ),
) -> Dict[str, Any]:
    hh = _normalize_list_household(household)
    items = get_shopping_list_items(hh, include_completed=include_completed)
    return {
        "status": "ok",
        "household": hh,
        "items": items,
    }


@grocy_router.post("/list/{household}")
async def jarvis_shopping_list_add(
    household: str,
    body: ShoppingListCreateRequest,
) -> Dict[str, Any]:
    hh = _normalize_list_household(household)
    item_id = add_shopping_list_item(
        household=hh,
        item_name=body.name.strip(),
        quantity=body.quantity.strip() if body.quantity else None,
        source=body.source,
    )
    items = get_shopping_list_items(hh, include_completed=False)
    return {
        "status": "ok",
        "household": hh,
        "created_id": item_id,
        "items": items,
    }


@grocy_router.delete("/list/{household}/{item_id}")
async def jarvis_shopping_list_delete_item(
    household: str,
    item_id: int,
) -> Dict[str, Any]:
    hh = _normalize_list_household(household)
    deleted = delete_shopping_list_item(hh, item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Item not found")

    items = get_shopping_list_items(hh, include_completed=False)
    return {
        "status": "ok",
        "household": hh,
        "items": items,
    }


@grocy_router.delete("/list/{household}")
async def jarvis_shopping_list_clear(
    household: str,
) -> Dict[str, Any]:
    hh = _normalize_list_household(household)
    deleted_count = clear_shopping_list(hh)
    return {
        "status": "ok",
        "household": hh,
        "deleted": deleted_count,
        "items": [],
    }


# ---------------------------------------------------------
# Meal planner (single source of truth)
# ---------------------------------------------------------
# All meal planner endpoints are defined in:
#   /opt/jarvis/brain/services/mealplanner.py
#
# Endpoints include:
#   - POST /mealplanner/plan
#   - POST /mealplanner/plan-multi
#   - GET  /mealplanner/plan-context
#
# This file intentionally defines NO /mealplanner routes
# to avoid router shadowing and OpenAPI collisions.


# ---------------------------------------------------------
# BarcodeBuddy router & endpoints
# ---------------------------------------------------------

barcode_router = APIRouter(
    prefix="/barcodebuddy",
    tags=["barcodebuddy"],
    dependencies=[Depends(require_api_key)],
)


def _barcode_disabled_response(reason: Optional[str] = None) -> Dict[str, Any]:
    base_reason = (
        "BarcodeBuddy not configured (missing BARCODEBUDDY_BASE_URL or "
        "BARCODEBUDDY_API_KEY, or client not initialized)"
    )
    return {
        "status": "disabled",
        "reason": reason or base_reason,
    }


async def get_barcode_client() -> Optional[BarcodeBuddyClient]:
    try:
        client = await create_barcodebuddy_client()
    except BarcodeBuddyError:
        return None
    return client


class BarcodeScanRequest(BaseModel):
    barcode: str
    household: Optional[str] = None


class BarcodeCreateProductRequest(BaseModel):
    barcode: str = Field(..., min_length=1)
    household: str = Field(..., description="home_a | home_b")
    name: str = Field(..., min_length=1)
    location_id: int = Field(..., ge=1)

    # NEW: required unit IDs (Gate A: no defaults)
    qu_id_purchase: int = Field(..., ge=1, description="Grocy purchase unit ID (required)")
    qu_id_stock: int = Field(..., ge=1, description="Grocy stock unit ID (required)")

    add_stock: bool = False
    quantity: Optional[float] = Field(default=None, gt=0)

    best_before_date: Optional[str] = None
    purchased_date: Optional[str] = None
    price: Optional[float] = Field(default=None, gt=0)


@barcode_router.get("/health")
async def barcode_health(
    client: Optional[BarcodeBuddyClient] = Depends(get_barcode_client),
) -> Dict[str, Any]:
    if client is None:
        return _barcode_disabled_response()

    try:
        info = await client.health()
        return {"status": "ok", "info": info}
    except BarcodeBuddyError as exc:
        return {"status": "error", "error": str(exc)}


@barcode_router.get("/products")
async def barcode_products(
    client: Optional[BarcodeBuddyClient] = Depends(get_barcode_client),
) -> Dict[str, Any]:
    if client is None:
        return _barcode_disabled_response()

    return {
        "status": "error",
        "error": "Listing products is not implemented for this BarcodeBuddy client; use /barcodebuddy/scan or /barcodebuddy/product/{barcode}.",
    }


@barcode_router.get("/product/{barcode}")
async def barcode_product_lookup(
    barcode: str,
    client: Optional[BarcodeBuddyClient] = Depends(get_barcode_client),
) -> Dict[str, Any]:
    if client is None:
        return _barcode_disabled_response()

    try:
        raw = await client.scan_barcode(barcode=barcode)
        return {
            "status": "ok",
            "barcode": barcode,
            "result": raw,
        }
    except BarcodeBuddyError as exc:
        return {"status": "error", "error": str(exc)}


def _summarize_barcodebuddy_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    text_parts: List[str] = []

    data = raw.get("data")
    if isinstance(data, dict):
        msg = data.get("result")
        if msg:
            text_parts.append(str(msg))

    result_block = raw.get("result")
    if isinstance(result_block, dict):
        msg2 = result_block.get("result")
        if msg2:
            text_parts.append(str(msg2))

    full_text = " ".join(text_parts).strip()
    lower = full_text.lower()

    state = "unknown"
    can_offer_grocy_add = False

    if "unknown product already scanned" in lower:
        state = "unknown_already_scanned"
        can_offer_grocy_add = True
    elif "unknown product" in lower:
        state = "unknown"
        can_offer_grocy_add = True
    elif "increasing quantity" in lower or "quantity increased" in lower:
        state = "quantity_incremented"
    elif "added to stock" in lower or "added to inventory" in lower:
        state = "added_to_stock"
    elif "ok" in lower and not full_text:
        state = "ok"

    return {
        "state": state,
        "message": full_text or None,
        "can_offer_grocy_add": can_offer_grocy_add,
    }


@barcode_router.post("/scan")
async def barcode_scan(
    body: BarcodeScanRequest,
    client: Optional[BarcodeBuddyClient] = Depends(get_barcode_client),
) -> Dict[str, Any]:
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="BarcodeBuddy client not configured; scanning unavailable.",
        )

    barcode = (body.barcode or "").strip()
    if not barcode:
        raise HTTPException(
            status_code=400,
            detail="barcode must not be empty",
        )

    try:
        raw = await client.scan_barcode(barcode=barcode)
    except BarcodeBuddyError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Error querying BarcodeBuddy: {exc}",
        ) from exc

    summary: Dict[str, Any]
    if isinstance(raw, dict):
        summary = _summarize_barcodebuddy_result(raw)
    else:
        summary = {
            "state": "unknown",
            "message": None,
            "can_offer_grocy_add": False,
        }

    return {
        "status": "ok",
        "barcode": barcode,
        "household": body.household,
        "raw_result": raw,
        "summary": summary,
    }


@barcode_router.post("/create-product")
async def barcode_create_product(
    body: BarcodeCreateProductRequest,
    bb_client: Optional[BarcodeBuddyClient] = Depends(get_barcode_client),
    grocy_client: Optional[GrocyClient] = Depends(get_grocy_client),
) -> Dict[str, Any]:
    if bb_client is None:
        raise HTTPException(
            status_code=503,
            detail="BarcodeBuddy client not configured; create-product unavailable.",
        )
    if grocy_client is None:
        raise HTTPException(
            status_code=503,
            detail="Grocy client not configured; create-product unavailable.",
        )

    household = (body.household or "").strip().lower()
    if household not in {"home_a", "home_b"}:
        raise HTTPException(
            status_code=400,
            detail="household must be 'home_a' or 'home_b'",
        )

    if body.add_stock and body.quantity is None:
        raise HTTPException(
            status_code=400,
            detail="quantity is required when add_stock=true",
        )

    created = await _acall_first_existing(
        grocy_client,
        [
            "create_product",
            "create_product_in_household",
            "create_product_with_location",
            "create_product_and_link_barcode",
            "create_product_for_household",
        ],
        household=household,
        name=body.name,
        location_id=body.location_id,
        qu_id_purchase=body.qu_id_purchase,
        qu_id_stock=body.qu_id_stock,
    )

    product_id: Optional[int] = None
    if isinstance(created, dict):
        pid = (
            created.get("product_id")
            or created.get("id")
            or (created.get("product") or {}).get("id")
        )
        if pid is not None:
            try:
                product_id = int(pid)
            except (TypeError, ValueError):
                product_id = None

    if product_id is None:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Could not determine product_id from Grocy create response",
                "created": created,
            },
        )

    barcode_link = await _acall_first_existing(
        grocy_client,
        [
            "link_barcode",
            "link_barcode_to_product",
            "create_product_barcode",
            "create_product_barcode_link",
            "add_barcode",
        ],
        barcode=body.barcode,
        household=household,
        product_id=product_id,
    )

    stock_result: Optional[Any] = None
    if body.add_stock:
        stock_result = await _acall_first_existing(
            grocy_client,
            [
                "add_stock",
                "add_product_stock",
                "add_stock_to_product",
                "stock_add",
                "add_stock_for_product",
            ],
            household=household,
            product_id=product_id,
            quantity=body.quantity,
            best_before_date=body.best_before_date,
            purchased_date=body.purchased_date,
            price=body.price,
        )

    return {
        "status": "ok",
        "household": household,
        "barcode": body.barcode,
        "product_id": product_id,
        "product": created,
        "barcode_link": barcode_link,
        "stock_added": bool(body.add_stock),
        "stock_result": stock_result,
        "summary": {
            "household": household,
            "product_id": product_id,
            "barcode": body.barcode,
            "stock_added": bool(body.add_stock),
            "quantity": body.quantity if body.add_stock else None,
            "best_before_date": body.best_before_date,
            "price": body.price,
        },
    }


# ----------------------------
# Core ask/history routes
# ----------------------------


@app.get("/", include_in_schema=False)
async def root() -> Dict[str, Any]:
    return {
        "message": "ISAC brain is running",
        "llm_provider": JARVIS_LLM_PROVIDER,
        "llm_model": LLM_MODEL,
    }


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
async def ask_jarvis(body: AskRequest, request: Request) -> AskResponse:
    system_prompt = body.system_prompt or (
        "You are ISAC, a helpful assistant running inside a user's homelab. "
        "Be concise, clear, and practical. When the user references devices or "
        "services, assume they may exist in the homelab environment."
    )
    # Bundle 4 Phase 2 — Sandbox language enforcement for /ask
    # Sandbox is exploratory-only: no facts, no commitments, no observation claims, no memory.
    sandbox = _is_sandbox_request(request)

    SANDBOX_SYSTEM = (
        "You are Alice. You are in SANDBOX mode.\n"
        "Rules (hard):\n"
        "- Speak only in hypotheticals and clearly labeled assumptions.\n"
        "- Do not claim you observed anything (no checking, searching, verifying, reading).\n"
        "- Do not commit to actions or imply you will do anything.\n"
        "- Do not write memory or imply remembering.\n"
        "- Do not promote sandbox content to truth.\n"
        "Always end by offering: summarize, discard, or exit sandbox.\n"
    )

    def _sandbox_language_violation(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        bad = [
            r"\b(i\s*(?:checked|found|saw|verified|looked\s*up|searched|pulled|retrieved))\b",
            r"\b(i\s*(?:will|i'll|we'll|i\s+can|i\s+am\s+going\s+to))\b",
            r"\b(saved|remembered|stored|wrote\s+to\s+memory|added\s+to\s+memory|i\s+noted)\b",
            r"\b(executed|ran|called|triggered)\b",
        ]
        for rx in bad:
            if re.search(rx, t, flags=re.IGNORECASE):
                return True
        return False

    def _sandbox_boundary_answer() -> str:
        return (
            "Sandbox boundary. I started to speak as if I had observed, committed, or stored something.\n\n"
            "If you want, say: “discard”, “summarize”, or “exit sandbox”."
        )



    conversation_context = request.headers.get("X-Jarvis-Context", "")
    
    

    # SANDBOX_PROMOTION_GUARD (Bundle 4 Phase 3)
    if _is_sandbox_request(request):
        raw = (body.message or "").strip().lower()
        if raw in {"do it","make this real","make it real","apply it","run it","execute","go live","promote"}:
            return {
                "model": LLM_MODEL,
                "answer": (
                    "Sandbox boundary. I can’t promote or act on sandbox reasoning implicitly.\n\n"
                    "Say “request promotion” to start a promotion ceremony, or “exit sandbox” to leave sandbox."
                ),
            }


# Phase 8 (Unified): read-only name resolution (Option A + B)
    # - Single resolution pass
    # - No writes, no learning, no behavior changes
    hidden_resolution_context: Optional[str] = None

    try:
        nr_user_id = request.headers.get("X-ISAC-USER-ID")
        nr_result = None

        if nr_user_id:
            nr_result = resolve_alias_to_concept(
                text=body.message,
                user_id=nr_user_id,
            )

            # Option A: logging only
            if nr_result:
                print(
                    "[ASK][NAME_RESOLUTION] match "
                    f"user_id={nr_user_id!r} "
                    f"alias_id={nr_result.get('alias_id')} "
                    f"concept_id={nr_result.get('concept_id')} "
                    f"concept_key={nr_result.get('concept_key')!r} "
                    f"preferred_name={nr_result.get('preferred_name')!r} "
                    f"confidence={nr_result.get('confidence')} "
                    f"source={nr_result.get('source')!r}"
                )

                # Option B: hidden advisory context
                hidden_resolution_context = (
                    "Internal note (read-only): The user's message exactly matches a known "
                    "user-specific alias. This may refer to concept "
                    f"'{nr_result.get('concept_key')}' "
                    f"(preferred name: '{nr_result.get('preferred_name')}'). "
                    "Treat this as contextual awareness only. "
                    "Do not assume correctness and do not state this explicitly unless "
                    "the user naturally confirms or asks."
                )
            else:
                print(
                    "[ASK][NAME_RESOLUTION] no_match "
                    f"user_id={nr_user_id!r} "
                    f"text={body.message!r}"
                )
        else:
            print("[ASK][NAME_RESOLUTION] skipped missing X-ISAC-USER-ID")

    except Exception as exc:  # noqa: BLE001
        print(f"[ASK][NAME_RESOLUTION] error: {exc}")

    
    # SANDBOX_SYSTEM_INSTRUCTION (Bundle 4 Phase 2)
    sandbox_sys = (
        "SANDBOX MODE: You are exploring hypotheticals only. "
        "Do NOT claim observation, truth, completion, or commitment. "
        "Do NOT suggest or imply tool usage. "
        "Do NOT write or imply memory. "
        "Use explicit markers like 'Assumption:' and 'Hypothetical:'. "
        "End by offering: summarize / discard / exit sandbox."
    )

    messages = []
    if sandbox:
        messages.append({"role": "system", "content": SANDBOX_SYSTEM})

    if conversation_context:
        messages.append({"role": "system", "content": conversation_context})

    
    if _is_sandbox_request(request):
        messages.append({"role": "system", "content": sandbox_sys})
    if hidden_resolution_context:
        messages.append({"role": "system", "content": hidden_resolution_context})

    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": body.message})

    try:
        answer = await MODEL_CLIENT.generate_chat_completion(
            model=LLM_MODEL,
            messages=messages,
            temperature=body.temperature,
            max_output_tokens=body.max_output_tokens,
        )

        if sandbox and _sandbox_language_violation(answer):
            answer = _sandbox_boundary_answer()
    except ModelClientError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error from LLM provider: {exc}",
        ) from exc
    if not sandbox:


        if not _is_sandbox_request(request):
            try:
                log_conversation(LLM_MODEL, body.message, answer)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Failed to log conversation: {e}")

    return {"model": LLM_MODEL, "answer": answer}


@app.get("/history", dependencies=[Depends(require_api_key)])
def history(limit: int = 20) -> Dict[str, Any]:
    items = fetch_recent_conversations(limit=limit)
    return {"count": len(items), "items": items}


# ----------------------------
# Exception handlers
# ----------------------------


@app.exception_handler(ModelClientError)
async def model_client_error_handler(
    request: Request, exc: ModelClientError
) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": f"Model client error: {exc}"},
    )

@app.exception_handler(sqlite3.IntegrityError)
async def sqlite_integrity_error_handler(
    request: Request, exc: sqlite3.IntegrityError
) -> JSONResponse:
    # Common case for our memory system: idempotent-ish seeds collide with UNIQUE constraints.
    # Return a clean 409 instead of a scary 500.
    msg = str(exc)
    detail = "Conflict: integrity constraint violation."
    # Keep it helpful but not overly leaky.
    if "UNIQUE" in msg.upper():
        detail = "Conflict: item already exists (unique constraint)."
    return JSONResponse(
        status_code=409,
        content={"detail": detail},
    )



# ---------------------------------------------------------
# Bundle 1 (B1) — Alice identity completeness dependency
# Enforced only on /alice/* prefixed routers (Alice-only scope).
# Pure logic: no DB reads/writes, no tools.
# ---------------------------------------------------------

def require_user_identity_for_alice(request: Request) -> None:
    user_id = (request.headers.get("X-ISAC-USER-ID") or "").replace("\r", "").strip()
    chat_id = (request.headers.get("X-ISAC-CHAT-ID") or "").replace("\r", "").strip()

    user_present = bool(user_id)
    chat_present = bool(chat_id)

    notes: List[str] = []
    if not user_present:
        notes.append("B1_USER_ID_MISSING")
    if not chat_present:
        notes.append("B1_CHAT_ID_MISSING")

    if user_present:
        return

    # Unified B1 Integrity Error Format (minimal)
    detail = {
        "error": {
            "code": "B1_USER_ID_MISSING",
            "message": "User identity is required for this endpoint.",
            "hint": "Send X-ISAC-USER-ID header.",
        },
        "integrity": {
            "cco_version": "B1.v1",
            "endpoint": request.url.path,
            "continuity_state": ("present" if chat_present else "missing"),
            "notes": notes,
            "identity": {"user_id_present": False, "chat_id_present": chat_present},
        },
    }
    raise HTTPException(status_code=400, detail=detail)


# ----------------------------
# Mount routers
# ----------------------------

# Identity & Memory v1 routers (post-auth definitions; avoids circular imports)
from identity_memory.router import (
    identity_router as identity_v1_router,
    memory_router as memory_v1_router,
    admin_identity_router as identity_v1_admin_router,
    admin_memory_router as memory_v1_admin_router,
    admin_inbox_router as admin_inbox_v1_router,
)


app.include_router(health_router)
app.include_router(runner_router)
# Alice (Phase 8) — read-only preview endpoint
app.include_router(alice_router)
app.include_router(alice_memory_router)
app.include_router(alice_gate_router)

# Identity & Memory v1 (auto-provision + explicit consent)
app.include_router(identity_v1_router, dependencies=[Depends(require_api_key)])
app.include_router(memory_v1_router, dependencies=[Depends(require_api_key)])

app.include_router(
    identity_v1_admin_router,
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)
app.include_router(
    memory_v1_admin_router,
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)
app.include_router(
    admin_inbox_v1_router,
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)

# Identity & Memory v1 — /alice aliases (for front-door routing)
app.include_router(identity_v1_router, prefix="/alice", dependencies=[Depends(require_api_key), Depends(require_user_identity_for_alice)])
app.include_router(memory_v1_router, prefix="/alice", dependencies=[Depends(require_api_key), Depends(require_user_identity_for_alice)])

app.include_router(
    identity_v1_admin_router,
    prefix="/alice",
    dependencies=[Depends(require_api_key), Depends(require_user_identity_for_alice), Depends(require_admin_if_configured)],
)
app.include_router(
    memory_v1_admin_router,
    prefix="/alice",
    dependencies=[Depends(require_api_key), Depends(require_user_identity_for_alice), Depends(require_admin_if_configured)],
)
app.include_router(
    admin_inbox_v1_router,
    prefix="/alice",
    dependencies=[Depends(require_api_key), Depends(require_user_identity_for_alice), Depends(require_admin_if_configured)],
)

app.include_router(ha_router)
app.include_router(calendar_router)
app.include_router(grocy_router)
app.include_router(finance_router)


# Meal planner (single source of truth: services/mealplanner.py)
app.include_router(mealplanner_context_router)

app.include_router(barcode_router)
app.include_router(recipes_router)
app.include_router(ingredient_parser_router)
app.include_router(recipe_matcher_router)
app.include_router(recipe_analyzer_router)
app.include_router(recipe_mappings_router)

app.include_router(mealplans_router, prefix="/mealplans", tags=["mealplans"])

app.include_router(
    irr_router,
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)

app.include_router(
    irr_narrative_router,
    dependencies=[Depends(require_api_key), Depends(require_admin_if_configured)],
)