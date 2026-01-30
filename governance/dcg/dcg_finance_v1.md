# DCG Finance v1 — Domain Command Grammar (Finance)

**Status:** LAP Skeleton (Inert, Non-Executable)

## Authority
This document is authoritative governance. In any conflict, governance overrides code.
Nothing in this document authorizes execution.

## Purpose
Define the complete, self-contained command grammar used by Alice/ISAC for the Finance domain.
This is a structural contract only. It defines language, order, boundaries, and prohibitions.

## Scope
Applies exclusively to the Finance domain.
Does not apply to Coding, Identity, Memory, Automation, or Household systems.

## Hard Prohibitions (LAP)
- Finance execution is BLOCKED
- No execution
- No tool calls
- No filesystem or network I/O
- No memory or identity learning
- No automation
- No background actions

## Relationship to FRTK
- FRTK v1 is the specialist reasoning toolkit for Finance.
- DCG defines the *grammar*; FRTK defines the *reasoning content*.
- FRTK may only emit non-executing request specifications.
- DCG does not import, invoke, or execute FRTK.

## Relationship to Finance Systems
- Firefly III is the system of record
- No writes, imports, syncs, or rule changes are permitted during LAP
- All future writes require explicit unblocking and governance approval

## Relationship to Toolbelt & Execution Spine
- Callable tools live under /opt/jarvis/brain/tools/*
- Execution is mediated by the execution spine and runner
- DCG v1 does not authorize access to either

## Command Grammar (Structural Only)

All Finance interactions MUST conform to the following ordered phases.
Phases may not be skipped, merged, or reordered.

### 1. Observe
Purpose:
- Describe the current financial state
- Identify accounts, transactions, balances, rules, and gaps

Allowed:
- Read-only descriptions
- Snapshot interpretation
- Clarifying questions

Prohibited:
- Recommendations
- Normalization
- Corrections
- Tool usage

### 2. Analyze
Purpose:
- Reason about observed financial data
- Identify inconsistencies, risks, and opportunities

Allowed:
- Pattern analysis
- Categorization discussion (conceptual)
- Risk articulation

Prohibited:
- Prescriptive changes
- Rule authoring
- Execution planning

### 3. Propose
Purpose:
- Present candidate financial actions for human review

Allowed:
- Conceptual proposals
- Rule or batch-change descriptions (non-executable)
- Verification approaches (conceptual)

Prohibited:
- Any execution
- Any write intent

### 4. Confirm
Purpose:
- Obtain explicit human authorization

Allowed:
- Clarifying questions
- Proposal restatement

Prohibited:
- Any action without confirmation

### 5. Execute
Status:
- Defined but INACTIVE

Notes:
- Execution semantics are governed elsewhere
- This phase exists only to complete the grammar

## Failure Semantics
- Any violation of order or boundary must fail closed
- Failures must be explicit, human-legible, and auditable

## Inert State Declaration
DCG Finance v1 is inert until Finance is explicitly unblocked by governance.
