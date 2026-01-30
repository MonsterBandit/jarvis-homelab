# DCG Coding v1 — Domain Command Grammar (Coding)

**Status:** LAP Skeleton (Inert, Non-Executable)

## Authority
This document is authoritative governance. In any conflict, governance overrides code.
Nothing in this document authorizes execution.

## Purpose
Define the complete, self-contained command grammar used by Alice/ISAC for the Coding domain.
This is a structural contract only. It defines language, order, boundaries, and prohibitions.

## Scope
Applies exclusively to the Coding domain.
Does not apply to Finance, Identity, Memory, or Automation.

## Hard Prohibitions (LAP)
- No execution
- No tool calls
- No filesystem or network I/O
- No memory or identity learning
- No finance logic
- No automation
- No background actions

## Relationship to CRTK
- CRTK v1 is the specialist reasoning toolkit for Coding.
- DCG defines the *grammar*; CRTK defines the *reasoning content*.
- CRTK may only emit non-executing request specifications.
- DCG does not import, invoke, or execute CRTK.

## Relationship to Toolbelt & Execution Spine
- Callable tools live under /opt/jarvis/brain/tools/*
- Execution is mediated by the execution spine and runner
- DCG v1 does not authorize access to either

## Command Grammar (Structural Only)

All Coding interactions MUST conform to the following ordered phases.
Phases may not be skipped, merged, or reordered.

### 1. Observe
Purpose:
- Describe the current state of code or system
- Identify files, structures, versions, and constraints

Allowed:
- Read-only descriptions
- Questions to clarify state

Prohibited:
- Recommendations
- Changes
- Tool usage

### 2. Analyze
Purpose:
- Reason about the observed state
- Identify risks, inconsistencies, or design tensions

Allowed:
- Logical analysis
- Tradeoff discussion
- Risk articulation

Prohibited:
- Prescriptive steps
- Patch drafting

### 3. Propose
Purpose:
- Present one or more candidate solutions

Allowed:
- Structured proposals
- Patch descriptions (conceptual)
- Verification strategies (conceptual)

Prohibited:
- Execution
- File modification

### 4. Confirm
Purpose:
- Obtain explicit human authorization

Allowed:
- Clarifying questions
- Restatement of proposal

Prohibited:
- Any action without confirmation

### 5. Execute
Status:
- Defined but INACTIVE during LAP

Notes:
- Execution semantics are governed elsewhere
- This phase exists only to complete the grammar

## Failure Semantics
- Any violation of order or boundary must fail closed
- Failure must be surfaced explicitly and human-legibly

## Inert State Declaration
DCG Coding v1 is inert until explicitly activated by governance.
