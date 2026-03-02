"""
sequence_enforcer.py
Tool-sequence hint injector with SequenceGraph caching.

Problem it solves:
  seq=0 in scoring — Claude skips required tool calls (approval gate bypass,
  incomplete gather phases, missing notification steps).

Approach:
  1. SequenceGraph: maps process_type → ordered step list, cached in sequence_graph.json.
  2. build_sequence_hint(): returns a structured prompt directive with ordered tool steps.
     Zero LLM for cached process types; one Haiku call for novel types (cached permanently).
  3. record_sequence_outcome(): updates graph confidence weights from RL quality scores.

Key design choices:
  - Every function is best-effort; exceptions return None/no-op (never block execution).
  - Approval gate detection is explicit: tasks requiring approval get an extra warning.
  - The directive is injected into process_context (not system_context) so it reaches
    both solve_with_claude and two_phase_execute paths.
  - SequenceGraph is seeded with high-confidence priors for the 15 FSM process types;
    actual task outcomes update confidence via exponential moving average (α=0.3).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

from src.config import ANTHROPIC_API_KEY

_SEQUENCE_MODEL = "claude-haiku-4-5-20251001"
_SEQUENCE_FILE = Path(os.environ.get("RL_CACHE_DIR", "/app")) / "sequence_graph.json"
_SYNTHESIS_TIMEOUT = 6.0  # seconds per Haiku call
_EMA_ALPHA = 0.3          # exponential moving average factor for confidence updates

# ── In-memory graph (process_type → sequence entry) ──────────────────────────
_graph: dict[str, dict] = {}
_loaded = False


# ── Seeded sequences for all 15 FSM process types ────────────────────────────
# Each step: step, description, tool_hints (prefixes/names), required, gate.
# gate="approval" means confirm_with_user MUST fire before any mutation.
# Seeded with confidence=0.85, hits=3 so real task outcomes update quickly.
_SEED_SEQUENCES: dict[str, dict] = {
    "expense_approval": {
        "steps": [
            {"step": 1, "description": "Gather expense record and employee data",
             "tool_hints": ["get_expense", "fetch_expense", "list_expense", "get_employee", "fetch_employee"],
             "required": True, "gate": None},
            {"step": 2, "description": "Compute totals, variance, and budget check",
             "tool_hints": ["calculate_", "compute_", "check_budget", "get_budget"],
             "required": True, "gate": None},
            {"step": 3, "description": "Request human approval (REQUIRED before any write)",
             "tool_hints": ["confirm_with_user"],
             "required": True, "gate": "approval"},
            {"step": 4, "description": "Execute approval or rejection decision",
             "tool_hints": ["approve_", "reject_", "approve_expense", "reject_expense",
                            "update_expense_status", "process_expense"],
             "required": True, "gate": None},
            {"step": 5, "description": "Log audit trail of the decision (REQUIRED)",
             "tool_hints": ["log_audit_trail", "log_", "create_audit_log", "audit_entry", "log_activity"],
             "required": True, "gate": None},
            {"step": 6, "description": "Update budget allocation to reflect approved/rejected spend",
             "tool_hints": ["update_budget_allocation", "update_budget", "deduct_budget", "adjust_budget"],
             "required": True, "gate": None},
            {"step": 7, "description": "Notify employee and manager of outcome",
             "tool_hints": ["notify_employee", "notify_manager", "send_notification", "send_email"],
             "required": False, "gate": None},
        ],
        "confidence": 0.85, "hits": 3, "haiku_generated": False,
    },
    "procurement": {
        "steps": [
            {"step": 1, "description": "Gather purchase order and vendor data",
             "tool_hints": ["get_purchase_order", "fetch_po", "get_vendor", "list_vendor"],
             "required": True, "gate": None},
            {"step": 2, "description": "Validate budget and compute spend",
             "tool_hints": ["calculate_", "check_budget", "get_budget"],
             "required": True, "gate": None},
            {"step": 3, "description": "Request approval for spend above threshold",
             "tool_hints": ["confirm_with_user"],
             "required": True, "gate": "approval"},
            {"step": 4, "description": "Process purchase order",
             "tool_hints": ["approve_purchase_order", "create_purchase_order", "update_purchase_order"],
             "required": True, "gate": None},
            {"step": 5, "description": "Notify vendor",
             "tool_hints": ["send_notification", "notify_vendor", "send_email"],
             "required": True, "gate": None},
        ],
        "confidence": 0.85, "hits": 3, "haiku_generated": False,
    },
    "hr_offboarding": {
        "steps": [
            {"step": 1, "description": "Fetch employee record and access list",
             "tool_hints": ["get_employee", "fetch_employee", "list_access", "get_access"],
             "required": True, "gate": None},
            {"step": 2, "description": "Check policy and unvested equity",
             "tool_hints": ["check_policy", "get_policy", "calculate_equity"],
             "required": True, "gate": None},
            {"step": 3, "description": "Revoke all system access",
             "tool_hints": ["revoke_access", "deactivate_user", "remove_access", "terminate_access"],
             "required": True, "gate": None},
            {"step": 4, "description": "Process final payroll and equipment return",
             "tool_hints": ["process_payroll", "update_employee", "close_employee"],
             "required": True, "gate": None},
            {"step": 5, "description": "Schedule exit interview and send notifications",
             "tool_hints": ["schedule_", "send_notification", "notify_"],
             "required": True, "gate": None},
        ],
        "confidence": 0.85, "hits": 3, "haiku_generated": False,
    },
    "incident_response": {
        "steps": [
            {"step": 1, "description": "Fetch incident details and affected systems",
             "tool_hints": ["get_incident", "fetch_incident", "list_incident", "get_system"],
             "required": True, "gate": None},
            {"step": 2, "description": "Compute SLA breach and severity",
             "tool_hints": ["calculate_sla", "compute_", "calculate_"],
             "required": True, "gate": None},
            {"step": 3, "description": "Request escalation approval",
             "tool_hints": ["confirm_with_user"],
             "required": True, "gate": "approval"},
            {"step": 4, "description": "Apply resolution or escalation",
             "tool_hints": ["update_incident", "resolve_incident", "escalate_incident"],
             "required": True, "gate": None},
            {"step": 5, "description": "Notify stakeholders",
             "tool_hints": ["send_notification", "notify_", "schedule_"],
             "required": True, "gate": None},
        ],
        "confidence": 0.85, "hits": 3, "haiku_generated": False,
    },
    "invoice_reconciliation": {
        "steps": [
            {"step": 1, "description": "Fetch invoice, PO, and goods receipt for 3-way match",
             "tool_hints": ["get_invoice", "fetch_invoice", "get_purchase_order",
                            "get_goods_receipt", "fetch_receipt", "get_receipt"],
             "required": True, "gate": None},
            {"step": 2, "description": "3-way match: verify PO qty = receipt qty = invoice qty",
             "tool_hints": ["match_goods_receipt", "verify_receipt", "check_receipt",
                            "calculate_variance", "calculate_", "compute_"],
             "required": True, "gate": None},
            {"step": 3, "description": "Check variance against policy approval threshold",
             "tool_hints": ["check_policy", "get_policy", "validate_"],
             "required": True, "gate": None},
            {"step": 4, "description": "Request approval for above-threshold variance",
             "tool_hints": ["confirm_with_user"],
             "required": True, "gate": "approval"},
            {"step": 5, "description": "Approve, flag, or reject invoice",
             "tool_hints": ["approve_invoice", "update_invoice", "flag_invoice",
                            "reject_invoice", "process_invoice"],
             "required": True, "gate": None},
        ],
        "confidence": 0.85, "hits": 3, "haiku_generated": False,
    },
    "customer_onboarding": {
        "steps": [
            {"step": 1, "description": "Fetch customer and subscription data",
             "tool_hints": ["get_customer", "fetch_customer", "list_customer", "get_subscription"],
             "required": True, "gate": None},
            {"step": 2, "description": "Validate KYC and compliance",
             "tool_hints": ["check_kyc", "verify_", "check_compliance"],
             "required": True, "gate": None},
            {"step": 3, "description": "Create customer account and assign plan",
             "tool_hints": ["create_customer", "create_account", "activate_subscription"],
             "required": True, "gate": None},
            {"step": 4, "description": "Send welcome notification",
             "tool_hints": ["send_notification", "send_email", "notify_"],
             "required": True, "gate": None},
        ],
        "confidence": 0.80, "hits": 3, "haiku_generated": False,
    },
    "compliance_audit": {
        "steps": [
            {"step": 1, "description": "Gather audit scope data",
             "tool_hints": ["get_audit", "fetch_audit", "list_control", "get_control"],
             "required": True, "gate": None},
            {"step": 2, "description": "Evaluate controls and compute risk scores",
             "tool_hints": ["calculate_risk", "compute_", "check_compliance"],
             "required": True, "gate": None},
            {"step": 3, "description": "Flag gaps and escalate critical findings",
             "tool_hints": ["flag_", "escalate_", "update_control"],
             "required": True, "gate": None},
            {"step": 4, "description": "Generate audit report",
             "tool_hints": ["create_report", "generate_report"],
             "required": True, "gate": None},
        ],
        "confidence": 0.80, "hits": 3, "haiku_generated": False,
    },
    "dispute_resolution": {
        "steps": [
            {"step": 1, "description": "Fetch dispute and transaction data",
             "tool_hints": ["get_dispute", "fetch_dispute", "get_chargeback", "get_transaction", "get_invoice"],
             "required": True, "gate": None},
            {"step": 2, "description": "Verify chargeback reason and get merchant response (BEFORE disputing)",
             "tool_hints": ["get_merchant_response", "get_chargeback_details", "verify_chargeback",
                            "check_merchant", "get_reason_code"],
             "required": True, "gate": None},
            {"step": 3, "description": "Compute credit or refund amount",
             "tool_hints": ["calculate_", "compute_refund", "calculate_credit"],
             "required": True, "gate": None},
            {"step": 4, "description": "Request approval for resolution",
             "tool_hints": ["confirm_with_user"],
             "required": True, "gate": "approval"},
            {"step": 5, "description": "Issue credit, dispute, or close",
             "tool_hints": ["credit_", "refund_", "dispute_chargeback", "close_dispute",
                            "resolve_dispute", "update_dispute", "apply_credit_memo"],
             "required": True, "gate": None},
            {"step": 6, "description": "Notify customer and card network",
             "tool_hints": ["send_notification", "notify_customer", "notify_card_network"],
             "required": True, "gate": None},
        ],
        "confidence": 0.85, "hits": 3, "haiku_generated": False,
    },
    "order_management": {
        "steps": [
            {"step": 1, "description": "Fetch order and inventory data",
             "tool_hints": ["get_order", "fetch_order", "list_order", "get_inventory"],
             "required": True, "gate": None},
            {"step": 2, "description": "Compute price delta and shipping cost",
             "tool_hints": ["calculate_", "compute_price"],
             "required": True, "gate": None},
            {"step": 3, "description": "Apply order modification",
             "tool_hints": ["update_order", "modify_order", "cancel_order", "process_order"],
             "required": True, "gate": None},
            {"step": 4, "description": "Notify customer of order update",
             "tool_hints": ["send_notification", "notify_customer"],
             "required": False, "gate": None},
        ],
        "confidence": 0.80, "hits": 3, "haiku_generated": False,
    },
    "sla_breach": {
        "steps": [
            {"step": 1, "description": "Fetch SLA contract and incident data",
             "tool_hints": ["get_sla", "fetch_sla", "get_incident", "get_contract"],
             "required": True, "gate": None},
            {"step": 2, "description": "Compute breach duration and credit amount",
             "tool_hints": ["calculate_sla_credit", "calculate_", "compute_"],
             "required": True, "gate": None},
            {"step": 3, "description": "Apply SLA credit to customer account",
             "tool_hints": ["credit_", "apply_credit", "update_contract", "issue_credit"],
             "required": True, "gate": None},
            {"step": 4, "description": "Notify customer and schedule escalation",
             "tool_hints": ["send_notification", "notify_", "schedule_"],
             "required": True, "gate": None},
        ],
        "confidence": 0.85, "hits": 3, "haiku_generated": False,
    },
    "month_end_close": {
        "steps": [
            {"step": 1, "description": "Fetch all period transactions and ledger data",
             "tool_hints": ["get_transaction", "list_transaction", "fetch_ledger", "get_ledger",
                            "get_journal", "list_entry"],
             "required": True, "gate": None},
            {"step": 2, "description": "Compute P&L and revenue recognition",
             "tool_hints": ["calculate_revenue", "calculate_", "compute_", "calculate_pnl"],
             "required": True, "gate": None},
            {"step": 3, "description": "Validate against budget and policy",
             "tool_hints": ["check_budget", "get_budget", "validate_", "check_policy"],
             "required": True, "gate": None},
            {"step": 4, "description": "Get finance manager approval (REQUIRED before posting)",
             "tool_hints": ["confirm_with_user", "get_manager_approval", "request_approval"],
             "required": True, "gate": "approval"},
            {"step": 5, "description": "Post journal entries and close period",
             "tool_hints": ["post_journal", "post_journal_entry", "close_period", "update_ledger"],
             "required": True, "gate": None},
        ],
        "confidence": 0.85, "hits": 3, "haiku_generated": False,
    },
    "ar_collections": {
        "steps": [
            {"step": 1, "description": "Fetch overdue invoices and customer records",
             "tool_hints": ["get_invoice", "list_invoice", "get_customer", "fetch_overdue"],
             "required": True, "gate": None},
            {"step": 2, "description": "Compute DSO and aging buckets",
             "tool_hints": ["calculate_dso", "calculate_aging", "calculate_"],
             "required": True, "gate": None},
            {"step": 3, "description": "Apply collection action (payment plan / flag)",
             "tool_hints": ["update_invoice", "flag_invoice", "create_payment_plan"],
             "required": True, "gate": None},
            {"step": 4, "description": "Send payment reminder to customer",
             "tool_hints": ["send_notification", "send_email", "notify_customer"],
             "required": True, "gate": None},
        ],
        "confidence": 0.80, "hits": 3, "haiku_generated": False,
    },
    "subscription_migration": {
        "steps": [
            {"step": 1, "description": "Fetch customer and current subscription",
             "tool_hints": ["get_customer", "get_subscription", "fetch_subscription"],
             "required": True, "gate": None},
            {"step": 2, "description": "Compute proration and fee delta",
             "tool_hints": ["calculate_proration", "calculate_", "compute_"],
             "required": True, "gate": None},
            {"step": 3, "description": "Request approval for plan change",
             "tool_hints": ["confirm_with_user"],
             "required": True, "gate": "approval"},
            {"step": 4, "description": "Apply subscription change",
             "tool_hints": ["update_subscription", "migrate_subscription", "activate_subscription"],
             "required": True, "gate": None},
            {"step": 5, "description": "Notify customer of plan change",
             "tool_hints": ["send_notification", "notify_customer"],
             "required": True, "gate": None},
        ],
        "confidence": 0.85, "hits": 3, "haiku_generated": False,
    },
    "payroll": {
        "steps": [
            {"step": 1, "description": "Fetch employee payroll data",
             "tool_hints": ["get_employee", "get_payroll", "fetch_payroll", "list_employee"],
             "required": True, "gate": None},
            {"step": 2, "description": "Compute gross pay, tax, and net pay",
             "tool_hints": ["calculate_payroll", "calculate_tax", "calculate_"],
             "required": True, "gate": None},
            {"step": 3, "description": "Validate payroll totals",
             "tool_hints": ["verify_payroll", "check_policy"],
             "required": True, "gate": None},
            {"step": 4, "description": "Process payroll disbursement",
             "tool_hints": ["process_payroll", "submit_payroll", "pay_"],
             "required": True, "gate": None},
            {"step": 5, "description": "Send pay stubs to employees",
             "tool_hints": ["send_notification", "notify_employee"],
             "required": True, "gate": None},
        ],
        "confidence": 0.85, "hits": 3, "haiku_generated": False,
    },
    # ── HR Leave / PTO Approval (explicit seed — covers task_26 pto_approval gap) ──
    "hr_leave": {
        "steps": [
            {"step": 1, "description": "Fetch employee record and current PTO/leave balance",
             "tool_hints": ["get_employee", "get_pto_balance", "get_leave_balance",
                            "fetch_leave_request", "get_leave_request", "get_leave"],
             "required": True, "gate": None},
            {"step": 2, "description": "Check leave policy and compute available days",
             "tool_hints": ["check_policy", "get_policy", "calculate_", "check_leave_policy",
                            "get_hr_policy", "get_leave_policy"],
             "required": True, "gate": None},
            {"step": 3, "description": "Request manager approval (REQUIRED before any write)",
             "tool_hints": ["confirm_with_user"],
             "required": True, "gate": "approval"},
            {"step": 4, "description": "Approve or reject the PTO/leave request (MUST call the action tool)",
             "tool_hints": ["approve_pto_request", "reject_pto_request", "approve_leave",
                            "reject_leave", "approve_leave_request", "reject_leave_request",
                            "update_leave_request", "approve_", "reject_"],
             "required": True, "gate": None},
            {"step": 5, "description": "Update employee leave/PTO balance",
             "tool_hints": ["update_leave_balance", "update_pto_balance", "deduct_leave",
                            "update_employee", "adjust_leave_balance"],
             "required": True, "gate": None},
            {"step": 6, "description": "Notify employee and manager of decision",
             "tool_hints": ["notify_employee", "notify_manager", "send_notification", "send_email"],
             "required": False, "gate": None},
        ],
        "confidence": 0.85, "hits": 3, "haiku_generated": False,
    },
    "pto_approval": {
        "steps": [
            {"step": 1, "description": "Fetch employee record and current PTO balance",
             "tool_hints": ["get_employee", "get_pto_balance", "get_leave_balance",
                            "fetch_leave_request", "get_leave_request"],
             "required": True, "gate": None},
            {"step": 2, "description": "Check policy and compute days available",
             "tool_hints": ["check_policy", "get_policy", "calculate_", "check_leave_policy"],
             "required": True, "gate": None},
            {"step": 3, "description": "Request manager approval (REQUIRED before any write)",
             "tool_hints": ["confirm_with_user"],
             "required": True, "gate": "approval"},
            {"step": 4, "description": "Approve or reject PTO request (MUST call action tool, not just analyze)",
             "tool_hints": ["approve_pto_request", "reject_pto_request", "approve_leave_request",
                            "reject_leave_request", "update_leave_request", "approve_", "reject_"],
             "required": True, "gate": None},
            {"step": 5, "description": "Update PTO balance after approval/rejection",
             "tool_hints": ["update_pto_balance", "update_leave_balance", "deduct_leave",
                            "update_employee", "adjust_pto"],
             "required": True, "gate": None},
            {"step": 6, "description": "Notify employee of outcome",
             "tool_hints": ["notify_employee", "send_notification", "send_email"],
             "required": False, "gate": None},
        ],
        "confidence": 0.85, "hits": 3, "haiku_generated": False,
    },
    "leave_approval": {
        "steps": [
            {"step": 1, "description": "Fetch employee and leave request data",
             "tool_hints": ["get_employee", "get_leave_request", "get_leave_balance",
                            "fetch_leave_request", "get_pto_balance"],
             "required": True, "gate": None},
            {"step": 2, "description": "Validate against leave policy",
             "tool_hints": ["check_policy", "get_policy", "check_leave_policy"],
             "required": True, "gate": None},
            {"step": 3, "description": "Request approval (REQUIRED before any write)",
             "tool_hints": ["confirm_with_user"],
             "required": True, "gate": "approval"},
            {"step": 4, "description": "Execute approve or reject action (MUST call tool, not just analyze)",
             "tool_hints": ["approve_leave", "reject_leave", "approve_leave_request",
                            "reject_leave_request", "approve_pto_request", "update_leave_request",
                            "approve_", "reject_"],
             "required": True, "gate": None},
            {"step": 5, "description": "Update leave balance",
             "tool_hints": ["update_leave_balance", "update_pto_balance", "deduct_leave"],
             "required": True, "gate": None},
            {"step": 6, "description": "Notify employee",
             "tool_hints": ["notify_employee", "send_notification"],
             "required": False, "gate": None},
        ],
        "confidence": 0.85, "hits": 3, "haiku_generated": False,
    },
    "leave_management": {
        "steps": [
            {"step": 1, "description": "Fetch employee and leave records",
             "tool_hints": ["get_employee", "get_leave_request", "get_pto_balance",
                            "get_leave_balance", "list_leave_requests"],
             "required": True, "gate": None},
            {"step": 2, "description": "Check policy and compute available leave",
             "tool_hints": ["check_policy", "get_policy", "calculate_"],
             "required": True, "gate": None},
            {"step": 3, "description": "Request approval (REQUIRED before any write)",
             "tool_hints": ["confirm_with_user"],
             "required": True, "gate": "approval"},
            {"step": 4, "description": "Apply leave action (approve/reject/adjust)",
             "tool_hints": ["approve_leave", "reject_leave", "approve_pto_request",
                            "update_leave_request", "adjust_leave", "approve_", "reject_"],
             "required": True, "gate": None},
            {"step": 5, "description": "Update leave balance",
             "tool_hints": ["update_leave_balance", "update_pto_balance"],
             "required": True, "gate": None},
            {"step": 6, "description": "Notify employee",
             "tool_hints": ["notify_employee", "send_notification"],
             "required": False, "gate": None},
        ],
        "confidence": 0.85, "hits": 3, "haiku_generated": False,
    },
    # ── Travel / Booking (explicit seed — eliminates Haiku synthesis variance) ──
    "airline_booking": {
        "steps": [
            {"step": 1, "description": "Search for available flights",
             "tool_hints": ["search_flights", "search_flight", "find_flights", "list_flights"],
             "required": True, "gate": None},
            {"step": 2, "description": "Select seat and get booking details",
             "tool_hints": ["select_seat", "get_seat", "choose_seat", "get_flight_details"],
             "required": True, "gate": None},
            {"step": 3, "description": "Book the flight",
             "tool_hints": ["book_flight", "create_booking", "reserve_flight", "book_"],
             "required": True, "gate": None},
            {"step": 4, "description": "Send booking confirmation to passenger",
             "tool_hints": ["send_confirmation", "send_notification", "confirm_booking",
                            "send_email", "notify_passenger"],
             "required": True, "gate": None},
        ],
        "output_schema": "flight_id, passenger_name, seat, confirmation_code, total_price",
        "confidence": 0.85, "hits": 3, "haiku_generated": False,
    },
    "flight_booking": {  # alias — smart_classifier may use either label
        "steps": [
            {"step": 1, "description": "Search for available flights",
             "tool_hints": ["search_flights", "search_flight", "find_flights", "list_flights"],
             "required": True, "gate": None},
            {"step": 2, "description": "Select seat and get booking details",
             "tool_hints": ["select_seat", "get_seat", "choose_seat", "get_flight_details"],
             "required": True, "gate": None},
            {"step": 3, "description": "Book the flight",
             "tool_hints": ["book_flight", "create_booking", "reserve_flight", "book_"],
             "required": True, "gate": None},
            {"step": 4, "description": "Send booking confirmation to passenger",
             "tool_hints": ["send_confirmation", "send_notification", "confirm_booking",
                            "send_email", "notify_passenger"],
             "required": True, "gate": None},
        ],
        "output_schema": "flight_id, passenger_name, seat, confirmation_code, total_price",
        "confidence": 0.85, "hits": 3, "haiku_generated": False,
    },
    "general": {
        "steps": [
            {"step": 1, "description": "Gather all required data",
             "tool_hints": ["get_", "fetch_", "list_", "search_"],
             "required": True, "gate": None},
            {"step": 2, "description": "Compute and validate",
             "tool_hints": ["calculate_", "compute_", "verify_"],
             "required": False, "gate": None},
            {"step": 3, "description": "Execute required action",
             "tool_hints": ["update_", "create_", "process_", "approve_", "submit_"],
             "required": True, "gate": None},
        ],
        "confidence": 0.70, "hits": 3, "haiku_generated": False,
    },
}

# Haiku synthesis prompt for novel process types
_SYNTHESIS_PROMPT = """You are a business process workflow expert.

Given a task description and available tools, output the required tool-call sequence as JSON.

Available tool names: {tool_names}
Task: {task_text}
Process type: {process_type}

Output JSON only:
{{
  "steps": [
    {{
      "step": 1,
      "description": "Brief phase description (6 words max)",
      "tool_hints": ["tool_name_or_prefix_1", "tool_name_or_prefix_2"],
      "required": true,
      "gate": null
    }}
  ],
  "approval_required": true_or_false
}}

Rules:
- Maximum 6 steps.
- Include "confirm_with_user" in tool_hints for any approval step (set "gate": "approval").
- tool_hints should be exact tool names from the available list OR common prefixes like "get_", "update_".
- Set approval_required: true if the task involves spend approval, cancellation, or irreversible actions.
- Set required: false only for optional notification steps."""


# ── Persistence ───────────────────────────────────────────────────────────────

def _load_graph() -> None:
    global _graph, _loaded
    if _loaded:
        return
    _loaded = True
    # Always start with seeds
    _graph = {k: dict(v) for k, v in _SEED_SEQUENCES.items()}
    try:
        if _SEQUENCE_FILE.exists():
            stored = json.loads(_SEQUENCE_FILE.read_text())
            # Merge stored entries, preferring stored if higher confidence
            for ptype, entry in stored.items():
                if ptype in _graph:
                    if entry.get("confidence", 0) > _graph[ptype]["confidence"]:
                        _graph[ptype] = entry
                else:
                    _graph[ptype] = entry
    except (json.JSONDecodeError, OSError):
        pass  # run with seeds if file corrupt


def _save_graph() -> None:
    try:
        _SEQUENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SEQUENCE_FILE.write_text(json.dumps(_graph, indent=2))
    except OSError:
        pass


# ── Sequence hint building ────────────────────────────────────────────────────

def _detect_approval_required(task_text: str, steps: list[dict]) -> bool:
    """Approval is required if any step has gate='approval', or task text signals it."""
    if any(s.get("gate") == "approval" for s in steps):
        return True
    _APPROVAL_SIGNALS = [
        r"\bapprove\b", r"\bapproval\b", r"\bapproval required\b",
        r"\bauthorize\b", r"\bsign.?off\b", r"\bhitl\b",
        r"\bhuman.in.the.loop\b", r"\bescalat",
    ]
    return any(re.search(p, task_text, re.I) for p in _APPROVAL_SIGNALS)


def _resolve_tool_hints(tool_hints: list[str], available_tools: set[str]) -> list[str]:
    """
    Resolve hint prefixes/names against the actually available tool set.
    If no tools match a hint, keep the hint as-is (useful when tool set is empty).
    """
    resolved = []
    for hint in tool_hints:
        if hint in available_tools:
            resolved.append(hint)
        else:
            # Match by prefix
            matched = [t for t in available_tools if t.startswith(hint)]
            if matched:
                resolved.extend(matched[:2])  # at most 2 per hint to keep directive short
            else:
                resolved.append(hint)  # keep original if nothing matched
    # Deduplicate preserving order
    seen: set[str] = set()
    return [x for x in resolved if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]


def _build_directive(
    process_type: str,
    steps: list[dict],
    available_tools: set[str],
    approval_required: bool,
) -> str:
    """Format the sequence directive for injection into the system prompt."""
    label = process_type.replace("_", " ").title()
    lines = [f"⚡ TOOL SEQUENCE: {label} — execute steps in order (skipping steps loses score):"]
    for s in steps:
        hints = _resolve_tool_hints(s.get("tool_hints", []), available_tools)
        tool_str = " | ".join(hints[:4]) if hints else "(any matching tool)"
        gate_tag = " ← MUST call before any mutation" if s.get("gate") == "approval" else ""
        req_tag = "" if s.get("required", True) else " [optional]"
        lines.append(f"  Step {s['step']}: {s['description']}{req_tag}")
        lines.append(f"           Tools → {tool_str}{gate_tag}")
    if approval_required:
        lines.append(
            "⛔ APPROVAL GATE: You MUST call confirm_with_user BEFORE any write/mutation tool. "
            "Skipping this gate is a policy violation."
        )
    return "\n".join(lines)


async def build_sequence_hint(
    task_text: str,
    process_type: str,
    policy_result: dict | None,
    tool_names: list[str],
) -> dict | None:
    """
    Return a sequence hint dict for injection into the system prompt, or None if unavailable.

    Returns:
        {
            "directive": str,            # formatted string to inject
            "approval_required": bool,   # True if HITL gate needed
            "steps": list[dict],         # raw step list
        }
    """
    try:
        _load_graph()
        available = set(tool_names)

        entry = _graph.get(process_type)

        # Synthesize for novel process types via Haiku (cached permanently)
        if entry is None or entry.get("confidence", 0) < 0.4:
            if ANTHROPIC_API_KEY and tool_names:
                entry = await _synthesize_sequence(task_text, process_type, tool_names)
            if entry is None:
                # Fall back to "general" template
                entry = _graph.get("general")
            if entry is None:
                return None

        steps = entry.get("steps", [])
        if not steps:
            return None

        approval_required = _detect_approval_required(task_text, steps)

        # Policy override: if policy says approval required, force it
        if policy_result and policy_result.get("requires_approval"):
            approval_required = True

        directive = _build_directive(process_type, steps, available, approval_required)
        output_schema = entry.get("output_schema", "")

        return {
            "directive": directive,
            "approval_required": approval_required,
            "steps": steps,
            "output_schema": output_schema,
        }

    except Exception:
        return None  # never block execution


async def _synthesize_sequence(
    task_text: str,
    process_type: str,
    tool_names: list[str],
) -> dict | None:
    """
    Call Haiku once to synthesize a sequence for a novel process type.
    Caches the result in _graph and persists to disk.
    """
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        tool_sample = ", ".join(tool_names[:30])  # cap to avoid prompt bloat
        prompt = _SYNTHESIS_PROMPT.format(
            tool_names=tool_sample,
            task_text=task_text[:800],
            process_type=process_type,
        )
        resp = await asyncio.wait_for(
            client.messages.create(
                model=_SEQUENCE_MODEL,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=_SYNTHESIS_TIMEOUT,
        )
        text = (resp.content[0].text if resp.content else "").strip()
        # Strip markdown fences
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
        parsed = json.loads(text)
        steps = parsed.get("steps", [])
        if not steps:
            return None
        entry = {
            "steps": steps,
            "confidence": 0.75,
            "hits": 1,
            "haiku_generated": True,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _graph[process_type] = entry
        _save_graph()
        return entry
    except (asyncio.TimeoutError, json.JSONDecodeError, Exception):
        return None


# ── Outcome recording ─────────────────────────────────────────────────────────

def record_sequence_outcome(
    process_type: str,
    quality: float,
    steps_observed: list[str] | None = None,
) -> None:
    """
    Update SequenceGraph confidence for a process type from a completed task outcome.

    Args:
        process_type: The FSM process type that ran.
        quality: RL quality score (0.0–1.0) from record_outcome().
        steps_observed: List of tool names actually called (for future coverage analysis).
    """
    try:
        _load_graph()
        if process_type not in _graph:
            return

        entry = _graph[process_type]
        old_conf = entry.get("confidence", 0.7)
        hits = entry.get("hits", 1)

        # Exponential moving average update: conf = α × quality + (1−α) × conf
        new_conf = _EMA_ALPHA * quality + (1 - _EMA_ALPHA) * old_conf
        entry["confidence"] = round(new_conf, 4)
        entry["hits"] = hits + 1
        entry["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        _save_graph()
    except Exception:
        pass  # never block for RL recording failure
