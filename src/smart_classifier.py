"""
smart_classifier.py
LLM-based process type classification — replaces keyword matching.
Ported from BrainOS llm-query-interpreter.ts.

A single Haiku call (~200ms) semantically classifies the task instead of
relying on surface keywords. Accurate routing = right FSM template =
right state instructions = better benchmark scores.

Example where keywords fail:
  "Process the Q3 vendor payment for the marketing team budget"
  Keywords → ambiguous (could be procurement OR expense_approval)
  Haiku    → "invoice_reconciliation" (payment processing = AP workflow)

Falls back to keyword matching if Haiku unavailable or times out.
"""
from __future__ import annotations
import asyncio
import json
import re
import time

from src.config import ANTHROPIC_API_KEY, FALLBACK_MODEL

CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"
CLASSIFIER_TIMEOUT = 5.0   # seconds — fall back to keywords if exceeded

# Used for documentation and _keyword_fallback only.
# The _call_classifier function trusts whatever Haiku returns without filtering.
VALID_PROCESS_TYPES = {
    "expense_approval", "procurement", "hr_offboarding", "incident_response",
    "invoice_reconciliation", "customer_onboarding", "compliance_audit",
    "dispute_resolution", "order_management", "sla_breach", "month_end_close",
    "ar_collections", "subscription_migration", "payroll", "general",
}

_CLASSIFIER_PROMPT = """You are a business process classifier. Given a task description, output the single best process type.

Process types and when to use them:
- expense_approval: employee expense claims, reimbursements, spend approvals
- procurement: vendor purchases, POs, supplier contracts, RFPs
- hr_offboarding: employee exit, access revocation, termination processing
- incident_response: service outages, P1/P2 incidents, production issues
- invoice_reconciliation: AP invoice matching, 3-way match, payment approval
- customer_onboarding: new client setup, account provisioning, welcome workflows
- compliance_audit: SOX/GDPR/PCI audits, regulatory reviews, control testing
- dispute_resolution: billing disputes, chargebacks, customer complaints
- order_management: sales orders, fulfillment, shipping, inventory
- sla_breach: SLA violations, uptime breaches, penalty credits
- month_end_close: period close, P&L finalization, accounting close
- ar_collections: overdue invoices, payment reminders, collections
- subscription_migration: plan changes, upgrades, downgrades, cancellations
- payroll: salary processing, pay runs, payroll adjustments
- general: anything that doesn't clearly fit the above

Respond with JSON only: {"process_type": "<type>", "confidence": 0.0-1.0, "reasoning": "<one sentence>"}"""


async def classify_process_type(task_text: str) -> tuple[str, float]:
    """
    Classify task into a process type using Haiku.
    Returns (process_type, confidence).
    Falls back to keyword matching on timeout, error, or missing API key.
    """
    if not ANTHROPIC_API_KEY:
        return _keyword_fallback(task_text), 0.5

    try:
        result = await asyncio.wait_for(
            _call_classifier(task_text),
            timeout=CLASSIFIER_TIMEOUT,
        )
        return result
    except asyncio.TimeoutError:
        return _keyword_fallback(task_text), 0.5
    except Exception:
        return _keyword_fallback(task_text), 0.5


async def _call_classifier(task_text: str) -> tuple[str, float]:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    resp = await client.messages.create(
        model=CLASSIFIER_MODEL,
        max_tokens=120,
        system=_CLASSIFIER_PROMPT,
        messages=[{"role": "user", "content": task_text[:1500]}],
    )
    text = resp.content[0].text if resp.content else ""
    # Strip markdown fences that Haiku sometimes prepends (```json ... ```)
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-z]*\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean).strip()
    parsed = json.loads(clean)
    ptype = parsed.get("process_type", "general")
    conf = float(parsed.get("confidence", 0.7))
    # Trust whatever Haiku returns — only reject if falsy/None/empty string.
    # Competition may introduce unknown types with valid process-specific FSM logic.
    if not ptype:
        return _keyword_fallback(task_text), 0.4
    return ptype, conf


def _keyword_fallback(task_text: str) -> str:
    """Original keyword-based detection — used as fallback and for VALID_PROCESS_TYPES reference."""
    KEYWORDS: dict[str, list[str]] = {
        "expense_approval":       ["expense", "reimbursement", "receipt", "spend", "claim"],
        "procurement":            ["vendor", "purchase order", "rfp", "supplier", "procurement"],
        "hr_offboarding":         ["offboarding", "termination", "access revocation", "exit", "last day"],
        "incident_response":      ["incident", "outage", "p1", "p2", "emergency", "sev"],
        "invoice_reconciliation": ["invoice", "reconcile", "3-way match", "accounts payable", "ap "],
        "customer_onboarding":    ["onboarding", "new customer", "new client", "provision"],
        "compliance_audit":       ["compliance", "audit", "kyc", "gdpr", "pci", "sox"],
        "dispute_resolution":     ["dispute", "chargeback", "complaint", "contested"],
        "order_management":       ["order", "fulfillment", "shipment", "delivery"],
        "sla_breach":             ["sla", "service level", "uptime breach", "penalty", "credit"],
        "month_end_close":        ["month-end", "month end", "close", "p&l", "financial close"],
        "ar_collections":         ["accounts receivable", "overdue", "collection", "aging"],
        "subscription_migration": ["migrate", "migration", "downgrade", "upgrade", "plan change",
                                   "plan migration", "saas migration", "subscription migration",
                                   "require_customer_signoff", "customer signoff", "enterprise migration"],
        "payroll":                ["payroll", "salary", "pay run", "wages"],
    }
    text = task_text.lower()
    best, best_score = "general", 0
    for ptype, kws in KEYWORDS.items():
        score = sum(1 for kw in kws if kw in text)
        if score > best_score:
            best_score, best = score, ptype
    return best
