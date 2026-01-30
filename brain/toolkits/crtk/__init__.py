"""
CRTK: Coding Reasoning Toolkit (v1 suite, skeleton, inert).

CRTK is the complete specialist coding toolkit, expressed as:
- A manifest of tools/capabilities (inventory + contracts)
- Playbooks/rubrics/templates for planning, review, and verification
- Adapters that emit *request specs* for callable tools (without executing anything)

LOCKED (LAP):
- No execution, no I/O, no tool calls
- Outputs are advisory and data-only
"""
from .types import CRTKRequest, CRTKResponse, CRTKMode
from .crtk import crtk_propose
from .manifest import CRTK_MANIFEST_V1
