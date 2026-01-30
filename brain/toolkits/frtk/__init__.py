"""
FRTK: Finance Reasoning Toolkit (v1 suite, skeleton, inert).

FRTK is the complete specialist finance toolkit, expressed as:
- A manifest of finance tools/capabilities (inventory + contracts)
- Playbooks for phased ingestion, normalization, correction, and audit
- Adapters that emit *request specs* for callable finance actions (without executing)

LOCKED (LAP):
- Finance execution is BLOCKED
- No execution, no I/O
- Outputs are advisory and data-only
"""
from .types import FRTKRequest, FRTKResponse, FRTKMode
from .frtk import frtk_propose
from .manifest import FRTK_MANIFEST_V1
