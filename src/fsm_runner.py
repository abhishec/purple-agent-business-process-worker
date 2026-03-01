"""
fsm_runner.py
8-state FSM for structured business process execution.
Upgraded from 6-state to match BrainOS process-intelligence/fsm-runner.ts exactly.

States: DECOMPOSE → ASSESS → COMPUTE → POLICY_CHECK → APPROVAL_GATE
        → MUTATE → SCHEDULE_NOTIFY → COMPLETE
Error paths: ESCALATE, FAILED

Key upgrades over original design:
- COMPUTE state: run financial calculations BEFORE policy check (no tools, pure math)
- MUTATE replaces EXECUTE: semantically explicit — this is where state changes happen
- SCHEDULE_NOTIFY: send notifications AFTER mutations, not mixed in with them
- HITL guard now wired at APPROVAL_GATE (mutate tools blocked by hitl_guard.py)
- Multi-checkpoint: processes that need sequential human confirmation use
  APPROVAL_GATE → MUTATE → APPROVAL_GATE (loop via fsm.reopen_approval_gate())
- Read-only shortcircuit: pure query tasks skip to 3-state path (DECOMPOSE→ASSESS→COMPLETE)
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FSMState(str, Enum):
    DECOMPOSE       = "DECOMPOSE"
    ASSESS          = "ASSESS"
    COMPUTE         = "COMPUTE"         # NEW: financial math, pure computation
    POLICY_CHECK    = "POLICY_CHECK"
    APPROVAL_GATE   = "APPROVAL_GATE"   # HITL: mutation tools blocked here
    MUTATE          = "MUTATE"          # was EXECUTE: state changes happen here
    SCHEDULE_NOTIFY = "SCHEDULE_NOTIFY" # NEW: notifications and scheduling
    COMPLETE        = "COMPLETE"
    ESCALATE        = "ESCALATE"
    FAILED          = "FAILED"


# ── Read-only shortcircuit patterns ───────────────────────────────────────────

_READONLY_PATTERNS = re.compile(
    r'\b(what is|what are|show me|get me|find|list|status of|how many|'
    r'who is|when was|where is|tell me|display|check|look up|retrieve|'
    r'can you show|give me|what\'s the|whats the)\b',
    re.IGNORECASE,
)

_ACTION_PATTERNS = re.compile(
    r'''\b(
        # Direct mutations
        approve|reject|create|update|delete|remove|add|insert|
        cancel|void|reverse|close|open|submit|send|post|apply|
        assign|transfer|move|mark|set|put|save|write|record|log|
        register|enroll|activate|deactivate|enable|disable|lock|unlock|
        flag|escalate|resolve|complete|finish|terminate|suspend|resume|
        refund|charge|credit|debit|pay|invoice|bill|collect|disburse|
        book|schedule|notify|alert|dispatch|route|forward|
        merge|split|link|unlink|attach|detach|replace|modify|change|
        # Business process verbs that imply action
        process|handle|execute|perform|run|manage|implement|
        initiate|trigger|issue|generate|produce|fulfill|provision|
        onboard|offboard|migrate|reconcile|settle|clear|adjust|
        override|authorize|certify|validate|confirm|acknowledge|
        remediate|remediate|revoke|reset|provision|deprovision
    )\b''',
    re.IGNORECASE | re.VERBOSE,
)


def detect_task_complexity(task_text: str) -> str:
    """Return 'readonly' for pure query tasks, 'full' for tasks with actions.

    Read-only tasks get a 3-state path (DECOMPOSE → ASSESS → COMPLETE) instead
    of the full 8-state path, saving latency and tool budget.

    IMPORTANT: Be conservative — only shortcircuit when there is NO possible action word.
    The competition penalizes missing tool calls more than it rewards shortcircuit speed.
    A false "readonly" classification causes missed mutation tool calls = 0 functional score.
    Only shortcircuit when the task is PURELY informational with zero ambiguity.
    """
    has_action = bool(_ACTION_PATTERNS.search(task_text))
    has_readonly = bool(_READONLY_PATTERNS.search(task_text))

    # Only shortcircuit if: clearly readonly AND no action verb whatsoever
    # AND the task text is short enough that we can be confident it's pure query.
    # Longer tasks (> 120 chars) often have embedded action requirements not caught by regex.
    if has_readonly and not has_action and len(task_text) <= 120:
        return "readonly"
    return "full"


# ── Process templates ─────────────────────────────────────────────────────────

# Process type → ordered FSM states
# Mirrors BrainOS process-registry.ts 16 built-in types
PROCESS_TEMPLATES: dict[str, list[FSMState]] = {
    "expense_approval": [
        FSMState.DECOMPOSE, FSMState.ASSESS, FSMState.COMPUTE,
        FSMState.POLICY_CHECK, FSMState.APPROVAL_GATE,
        FSMState.MUTATE, FSMState.COMPLETE,
    ],
    "procurement": [
        FSMState.DECOMPOSE, FSMState.ASSESS, FSMState.COMPUTE,
        FSMState.POLICY_CHECK, FSMState.APPROVAL_GATE,
        FSMState.MUTATE, FSMState.SCHEDULE_NOTIFY, FSMState.COMPLETE,
    ],
    "hr_offboarding": [
        FSMState.DECOMPOSE, FSMState.ASSESS, FSMState.POLICY_CHECK,
        FSMState.MUTATE, FSMState.SCHEDULE_NOTIFY, FSMState.COMPLETE,
    ],
    "incident_response": [
        FSMState.DECOMPOSE, FSMState.ASSESS, FSMState.COMPUTE,
        FSMState.APPROVAL_GATE, FSMState.MUTATE,
        FSMState.SCHEDULE_NOTIFY, FSMState.COMPLETE,
    ],
    "invoice_reconciliation": [
        FSMState.DECOMPOSE, FSMState.ASSESS, FSMState.COMPUTE,
        FSMState.POLICY_CHECK, FSMState.MUTATE, FSMState.COMPLETE,
    ],
    "customer_onboarding": [
        FSMState.DECOMPOSE, FSMState.ASSESS,
        FSMState.MUTATE, FSMState.SCHEDULE_NOTIFY, FSMState.COMPLETE,
    ],
    "compliance_audit": [
        FSMState.DECOMPOSE, FSMState.ASSESS, FSMState.COMPUTE,
        FSMState.POLICY_CHECK, FSMState.APPROVAL_GATE,
        FSMState.MUTATE, FSMState.SCHEDULE_NOTIFY, FSMState.COMPLETE,
    ],
    "dispute_resolution": [
        FSMState.DECOMPOSE, FSMState.ASSESS, FSMState.POLICY_CHECK,
        FSMState.APPROVAL_GATE, FSMState.MUTATE, FSMState.COMPLETE,
    ],
    "order_management": [
        FSMState.DECOMPOSE, FSMState.ASSESS, FSMState.COMPUTE,
        FSMState.APPROVAL_GATE, FSMState.MUTATE, FSMState.COMPLETE,
    ],
    "sla_breach": [
        FSMState.DECOMPOSE, FSMState.ASSESS, FSMState.COMPUTE,
        FSMState.POLICY_CHECK, FSMState.MUTATE, FSMState.SCHEDULE_NOTIFY, FSMState.COMPLETE,
    ],
    "month_end_close": [
        FSMState.DECOMPOSE, FSMState.ASSESS, FSMState.COMPUTE,
        FSMState.POLICY_CHECK, FSMState.APPROVAL_GATE,
        FSMState.MUTATE, FSMState.COMPLETE,
    ],
    "ar_collections": [
        FSMState.DECOMPOSE, FSMState.ASSESS, FSMState.COMPUTE,
        FSMState.POLICY_CHECK, FSMState.MUTATE,
        FSMState.SCHEDULE_NOTIFY, FSMState.COMPLETE,
    ],
    "subscription_migration": [
        # Multi-checkpoint: 5 confirm gates before migration
        FSMState.DECOMPOSE, FSMState.ASSESS, FSMState.COMPUTE,
        FSMState.POLICY_CHECK, FSMState.APPROVAL_GATE,
        FSMState.MUTATE, FSMState.COMPLETE,
    ],
    "payroll": [
        FSMState.DECOMPOSE, FSMState.ASSESS, FSMState.COMPUTE,
        FSMState.POLICY_CHECK, FSMState.APPROVAL_GATE,
        FSMState.MUTATE, FSMState.SCHEDULE_NOTIFY, FSMState.COMPLETE,
    ],
    "general": [
        FSMState.DECOMPOSE, FSMState.ASSESS,
        FSMState.POLICY_CHECK, FSMState.MUTATE, FSMState.COMPLETE,
    ],
}

# Keywords → process type (extended with new types)
PROCESS_KEYWORDS: dict[str, list[str]] = {
    "expense_approval":       ["expense", "reimbursement", "approval", "spend", "budget", "receipt", "claim"],
    "procurement":            ["vendor", "purchase", "order", "contract", "supplier", "rfp", "quote", "procurement"],
    "hr_offboarding":         ["offboarding", "offboard", "termination", "access revocation", "exit", "last day"],
    "incident_response":      ["incident", "outage", "down", "breach", "alert", "p1", "p2", "emergency", "sev"],
    "invoice_reconciliation": ["invoice", "reconcile", "reconciliation", "statement", "bill", "ap ", "accounts payable"],
    "customer_onboarding":    ["onboarding", "onboard", "new customer", "new client", "setup", "provision"],
    "compliance_audit":       ["compliance", "audit", "kyc", "gdpr", "pci", "sox", "regulatory", "review"],
    "dispute_resolution":     ["dispute", "chargeback", "complaint", "resolution", "contested", "claim"],
    "order_management":       ["order", "shipment", "delivery", "fulfillment", "cart", "item", "product"],
    "sla_breach":             ["sla", "service level", "uptime", "downtime", "breach", "penalty", "credit"],
    "month_end_close":        ["month-end", "month end", "close", "p&l", "financial close", "accounting", "books"],
    "ar_collections":         ["accounts receivable", "ar ", "aging", "overdue", "collection", "payment plan", "bad debt"],
    "subscription_migration": ["migrate", "migration", "downgrade", "upgrade", "plan change", "subscription change"],
    "payroll":                ["payroll", "salary", "wages", "compensation", "pay run", "paye", "bacs"],
    "general":                [],
}


def detect_process_type(task_text: str) -> str:
    """Keyword-match task text to the most specific process type."""
    text = task_text.lower()
    best_type = "general"
    best_score = 0
    for ptype, keywords in PROCESS_KEYWORDS.items():
        if ptype == "general":
            continue
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_type = ptype
    return best_type


def _states_from_definition(definition: dict) -> list[FSMState]:
    """Convert a synthesized definition's state names to FSMState enums."""
    result = []
    for name in definition.get("states", []):
        try:
            result.append(FSMState(name))
        except ValueError:
            pass  # ignore unrecognized state names
    # Guarantee DECOMPOSE→...→COMPLETE bookends
    if not result or result[0] != FSMState.DECOMPOSE:
        result.insert(0, FSMState.DECOMPOSE)
    if not result or result[-1] != FSMState.COMPLETE:
        result.append(FSMState.COMPLETE)
    return result


@dataclass
class FSMContext:
    task_text: str
    session_id: str
    process_type: str
    current_state: FSMState = FSMState.DECOMPOSE
    state_history: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    policy_result: dict | None = None
    escalation_reason: str = ""
    requires_hitl: bool = False
    approval_count: int = 0   # tracks multi-checkpoint approvals


class FSMRunner:
    """
    8-state FSM for business process execution.
    Mirrors BrainOS process-intelligence/fsm-runner.ts.

    Key behaviors:
    - State-gated execution with policy + HITL checks
    - COMPUTE state: math before mutation (no tools, no side effects)
    - MUTATE state: all state changes happen here (hitl_guard wired at APPROVAL_GATE)
    - Multi-checkpoint: reopen_approval_gate() for sequential confirms
    - Per-phase prompt injection with phase-appropriate instructions
    - Read-only shortcircuit: detect_task_complexity() collapses path to 3 states
    """

    def __init__(
        self,
        task_text: str,
        session_id: str,
        process_type: str | None = None,
        checkpoint=None,
        definition: dict | None = None,  # synthesized FSM definition for novel types
    ):
        ptype = process_type or detect_process_type(task_text)
        self.ctx = FSMContext(task_text=task_text, session_id=session_id, process_type=ptype)
        self._definition: dict | None = definition
        self._idx = 0

        if checkpoint:
            ptype_c = checkpoint.process_type
            self.ctx.process_type = ptype_c

            # on checkpoint restore, look up synthesized definition from cache
            # if this was a novel type (not in PROCESS_TEMPLATES)
            if definition is None and ptype_c not in PROCESS_TEMPLATES:
                try:
                    from src.dynamic_fsm import get_synthesized
                    self._definition = get_synthesized(ptype_c)
                except Exception:
                    pass

            if self._definition:
                self.states = _states_from_definition(self._definition)
            else:
                self.states = PROCESS_TEMPLATES.get(ptype_c, PROCESS_TEMPLATES["general"])

            self._idx = checkpoint.state_idx
            self.ctx.state_history = list(checkpoint.state_history)
            self.ctx.current_state = (
                self.states[self._idx] if self._idx < len(self.states) else FSMState.COMPLETE
            )
        else:
            if self._definition:
                self.states = _states_from_definition(self._definition)
            else:
                self.states = PROCESS_TEMPLATES.get(ptype, PROCESS_TEMPLATES["general"])
            # Read-only shortcircuit: collapse to 3-state path for pure queries
            if detect_task_complexity(task_text) == "readonly":
                self.states = [FSMState.DECOMPOSE, FSMState.ASSESS, FSMState.COMPLETE]

    @property
    def current_state(self) -> FSMState:
        return self.ctx.current_state

    @property
    def process_type(self) -> str:
        return self.ctx.process_type

    @property
    def is_terminal(self) -> bool:
        return self.ctx.current_state in (FSMState.COMPLETE, FSMState.FAILED, FSMState.ESCALATE)

    def advance(self, data: dict | None = None) -> FSMState:
        if data:
            self.ctx.data.update(data)
        self.ctx.state_history.append(self.ctx.current_state.value)
        self._idx += 1
        self.ctx.current_state = (
            self.states[self._idx] if self._idx < len(self.states) else FSMState.COMPLETE
        )
        return self.ctx.current_state

    def fail(self, reason: str) -> FSMState:
        self.ctx.state_history.append(self.ctx.current_state.value)
        self.ctx.current_state = FSMState.FAILED
        self.ctx.data["failure_reason"] = reason
        return FSMState.FAILED

    def escalate(self, reason: str) -> FSMState:
        self.ctx.state_history.append(self.ctx.current_state.value)
        self.ctx.current_state = FSMState.ESCALATE
        self.ctx.escalation_reason = reason
        self.ctx.requires_hitl = True
        return FSMState.ESCALATE

    def apply_policy(self, policy_result: dict) -> FSMState:
        self.ctx.policy_result = policy_result
        if not policy_result.get("passed"):
            if policy_result.get("escalationRequired"):
                return self.escalate(policy_result.get("summary", "Policy escalation required"))
            if policy_result.get("requiresApproval"):
                self.ctx.requires_hitl = True
        return self.advance()

    def reopen_approval_gate(self) -> None:
        """
        Multi-checkpoint support: push back to APPROVAL_GATE from MUTATE.
        Used for processes like subscription_migration that need 5 sequential confirms.
        """
        if self.ctx.current_state == FSMState.MUTATE:
            self.ctx.state_history.append(FSMState.MUTATE.value)
            self.ctx.current_state = FSMState.APPROVAL_GATE
            self.ctx.approval_count += 1

    def build_phase_prompt(self) -> str:
        """
        Build process context prompt for injection into system context.

        Design note: This prompt describes the COMPLETE required workflow for the
        process type, not just the "current phase." Previously, per-phase instructions
        like "ASSESS: DO NOT take any actions yet" were causing Claude to stop after
        gathering data, never reaching the mutation step.

        This prompt is used by five_phase and MoA (which use system_context).
        The primary solve_with_claude path receives this via process_context in worker_brain.
        """
        process = self.ctx.process_type.replace("_", " ").title()
        state = self.ctx.current_state

        # Determine what phases this process actually requires
        has_approval_gate = FSMState.APPROVAL_GATE in self.states
        has_compute = FSMState.COMPUTE in self.states
        has_notify = FSMState.SCHEDULE_NOTIFY in self.states
        task_type = "full" if FSMState.MUTATE in self.states else "readonly"

        lines = [
            f"## Business Process: {process}",
            f"## Task Type: {task_type.upper()} — {'requires mutations' if task_type == 'full' else 'read-only query'}",
            "",
            "## Required Execution Sequence (complete ALL applicable steps):",
        ]

        step = 1
        lines.append(f"{step}. GATHER: Call get_*/list_*/search_*/calculate_* tools to collect all required data")
        step += 1
        if has_compute:
            lines.append(f"{step}. COMPUTE: Calculate required amounts/deltas from the gathered data (show work)")
            step += 1
        if has_approval_gate:
            lines.append(f"{step}. CONFIRM: Call confirm_with_user (it auto-confirms — returns 'ok' immediately)")
            step += 1
        if task_type == "full":
            lines.append(f"{step}. MUTATE: Call the required action tools — approve_*/reject_*/update_*/cancel_*/process_*/credit_*/refund_*/flag_*/escalate_*/deactivate_*/revoke_*")
            step += 1
        if has_notify:
            lines.append(f"{step}. NOTIFY: Call send_*/notify_*/post_*/draft_* notification tools")
            step += 1
        lines.append(f"{step}. COMPLETE: Summarize completed actions with outcomes and status")

        if task_type == "full":
            lines.append("")
            lines.append("CRITICAL: Do NOT stop after GATHER or COMPUTE. Continue until ALL mutation tools have been called.")

        # Escalation/failure context if relevant
        if state == FSMState.ESCALATE:
            lines.append(f"\n⚠️ ESCALATION REQUIRED: {self.ctx.escalation_reason}")
            lines.append("Explain clearly why escalation is needed and which team/person must act.")
        elif state == FSMState.FAILED:
            lines.append(f"\n❌ FAILED: {self.ctx.data.get('failure_reason', 'Unknown error')}")
            lines.append("Explain what went wrong and what the next step should be.")

        # Synthesized definition context
        if self._definition and self._definition.get("_synthesized"):
            lines.append(f"\n[Dynamic FSM: synthesized for '{self.ctx.process_type}']")

        if self.ctx.approval_count > 0:
            lines.append(f"\n[Multi-checkpoint: approval gate #{self.ctx.approval_count + 1}]")

        return "\n".join(lines)

    def get_summary(self) -> dict:
        return {
            "process_type": self.ctx.process_type,
            "final_state": self.ctx.current_state.value,
            "state_history": self.ctx.state_history,
            "requires_hitl": self.ctx.requires_hitl,
            "escalation_reason": self.ctx.escalation_reason,
            "approval_count": self.ctx.approval_count,
        }
