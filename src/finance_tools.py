"""
finance_tools.py
Financial computation support — context injection only.

The amortization synthetic tool has been migrated to dynamic_tools.py
(seeded into tool_registry.json at startup). This file now contains ONLY the
context injection pattern — zero synthetic tool definitions.

Pattern A — Context injection (all standard calculations):
  build_finance_context() is called in PRIME phase.
  Checks context_rl confidence BEFORE injecting.
  Injects decision chains, not just numbers: "variance = 2.04% → APPROVE"
  When drift is detected: injects WARNING instead of value.
  Zero query budget cost. ~30-50 tokens vs 560 for old 7-tool approach.

Pattern B (tools) — Handled by dynamic_tools.py:
  finance_loan_amortization: seeded into tool_registry.json at startup
  All other math gaps: synthesized on-demand by dynamic_tools.synthesize_and_register()

RL integration (via context_rl.py):
  Every injection is confidence-gated:
    ≥75% accurate → inject with "trust this" annotation
    55–74%        → inject with "verify first" annotation
    <55%          → inject drift WARNING (tells Claude rules may have changed)
  This makes the computation layer adaptive: it learns within the benchmark run.
"""
from __future__ import annotations

import re as _re
from src.financial_calculator import (
    apply_variance_check,
    prorated_amount,
    prorated_for_period,
    apply_early_termination_fee,
    compute_sla_credit,
    amortize_loan,
    payment_plan_summary,
)
from src.context_rl import (
    should_inject,
    is_drift_detected,
    get_drift_warning,
    get_confidence_annotation,
)

# ── Patterns ────────────────────────────────────────────────────────────────────

_DOLLAR_PAT = _re.compile(r"\$\s?([\d,]+(?:\.\d{1,2})?)", _re.IGNORECASE)
_PCT_PAT    = _re.compile(r"(\d+(?:\.\d+)?)\s*%")

# Labeled extraction patterns (more reliable than positional)
_LABEL_INVOICE = _re.compile(
    r"(?:invoice[d]?|billed|charged|amount due)[^\d$]*\$\s?([\d,]+(?:\.\d{1,2})?)",
    _re.IGNORECASE,
)
_LABEL_PO = _re.compile(
    r"(?:po|purchase order|approved|contracted|budgeted)[^\d$]*\$\s?([\d,]+(?:\.\d{1,2})?)",
    _re.IGNORECASE,
)
_LABEL_THRESHOLD = _re.compile(
    r"(?:threshold|limit|tolerance|variance)[^\d%]*(\d+(?:\.\d+)?)\s*%",
    _re.IGNORECASE,
)


# ── Context injection — RL-gated ───────────────────────────────────────────────

def build_finance_context(task_text: str, process_type: str) -> str:
    """
    Pre-compute financial facts for COMPUTE state injection.

    RL-gated: checks context_rl confidence before each injection.
    Returns drift warnings when confidence drops below threshold.
    Adds decision chains: not just the number, but the recommended action.

    Zero API cost. ~30-50 tokens on hit.
    """
    lines: list[str] = []

    # ── Variance (invoice_reconciliation, procurement, expense_approval) ──────
    if process_type in ("invoice_reconciliation", "procurement", "expense_approval"):
        drift = is_drift_detected(process_type, "variance")
        if drift:
            lines.append(get_drift_warning("variance"))
        elif should_inject(process_type, "variance"):
            result = _compute_variance(task_text)
            if result:
                conf_note = get_confidence_annotation(process_type, "variance")
                lines.append(result + conf_note)

    # ── SLA credit (sla_breach) ───────────────────────────────────────────────
    if process_type == "sla_breach":
        drift = is_drift_detected(process_type, "sla_credit")
        if drift:
            lines.append(get_drift_warning("sla_credit"))
        elif should_inject(process_type, "sla_credit"):
            result = _compute_sla(task_text)
            if result:
                conf_note = get_confidence_annotation(process_type, "sla_credit")
                lines.append(result + conf_note)

    # ── Proration (subscription_migration, ar_collections, month_end_close) ───
    if process_type in ("subscription_migration", "ar_collections", "month_end_close"):
        drift = is_drift_detected(process_type, "proration")
        if drift:
            lines.append(get_drift_warning("proration"))
        elif should_inject(process_type, "proration"):
            result = _compute_proration(task_text)
            if result:
                conf_note = get_confidence_annotation(process_type, "proration")
                lines.append(result + conf_note)

    if not lines:
        return ""

    return (
        "## Pre-Computed Financial Facts (integer-cent precision)\n"
        "Use as ground truth in COMPUTE reasoning. "
        "⚠ warnings below mean standard formulas may not apply — retrieve fresh data.\n"
        + "\n".join(f"- {ln}" for ln in lines)
    )


# ── Computation helpers ─────────────────────────────────────────────────────────

def _compute_variance(task_text: str) -> str | None:
    """Extract amounts + threshold and compute variance with decision chain."""
    try:
        # Try labeled extraction first (more reliable)
        inv_m = _LABEL_INVOICE.search(task_text)
        po_m  = _LABEL_PO.search(task_text)
        thr_m = _LABEL_THRESHOLD.search(task_text)

        if inv_m and po_m:
            invoiced  = float(inv_m.group(1).replace(",", ""))
            po        = float(po_m.group(1).replace(",", ""))
        else:
            # Fallback: appearance order (first two dollar amounts in text).
            # Do NOT use sorted/largest-first: when PO > invoice (within-budget case),
            # largest-first assigns PO as "invoiced", inverting the variance decision.
            amounts_ordered = [float(m.replace(",", "")) for m in _DOLLAR_PAT.findall(task_text)]
            if len(amounts_ordered) < 2:
                return None
            invoiced, po = amounts_ordered[0], amounts_ordered[1]

        threshold = float(thr_m.group(1)) if thr_m else next(
            (p for p in _extract_pcts(task_text) if p < 20), 5.0
        )

        result = apply_variance_check(invoiced, po, threshold)
        action = "ESCALATE for approval" if result["exceeds"] else "APPROVE"
        return (
            f"Variance: ${invoiced:,.2f} vs ${po:,.2f} PO = {result['pct']:.4f}% "
            f"({'exceeds' if result['exceeds'] else 'within'} {threshold}% threshold) "
            f"→ Recommended action: {action}. "
            f"Variance amount: ${result['variance']:,.2f}."
        )
    except Exception:
        return None


def _compute_sla(task_text: str) -> str | None:
    """Compute SLA credit with correct sla_max_mins formula."""
    try:
        amounts = _extract_dollars(task_text)
        pcts    = _extract_pcts(task_text)
        down_m  = _re.search(r"(\d+)\s*(?:minute|min)", task_text, _re.IGNORECASE)
        down_h  = _re.search(r"(\d+(?:\.\d+)?)\s*(?:hour|hr)", task_text, _re.IGNORECASE)

        if not amounts or not (down_m or down_h):
            return None

        downtime_min = (
            float(down_m.group(1)) if down_m else float(down_h.group(1)) * 60
        )
        contract_val = amounts[0]
        sla_tgt      = next((p for p in pcts if p > 90), 99.9)
        credit_pct   = next((p for p in pcts if 1 <= p < 50), 10.0)
        cap          = next((p for p in pcts if 20 <= p <= 50), 30.0)

        # sla_max_mins = allowed downtime per 30-day month at target SLA
        sla_max_mins = 30 * 24 * 60 * (1.0 - sla_tgt / 100.0)
        credit = compute_sla_credit(downtime_min, sla_max_mins, contract_val, credit_pct, cap)

        breach = downtime_min > sla_max_mins
        return (
            f"SLA credit: {downtime_min:.0f} min downtime vs {sla_max_mins:.1f} min allowed "
            f"(SLA {sla_tgt}%) → {'BREACH' if breach else 'within SLA'}. "
            f"Credit: ${credit:,.2f} (cap {cap:.0f}% of ${contract_val:,.2f} contract)."
        )
    except Exception:
        return None


def _compute_proration(task_text: str) -> str | None:
    """Compute prorated remaining value from partial period usage."""
    try:
        amounts = _extract_dollars(task_text)
        months  = [int(m) for m in _re.findall(r"(\d+)\s*months?", task_text, _re.IGNORECASE)]
        days    = [int(d) for d in _re.findall(r"(\d+)\s*days?",   task_text, _re.IGNORECASE)]

        if not amounts:
            return None

        total_val = amounts[0]

        if months and len(months) >= 2:
            used_mo, total_mo = min(months), max(months)
            remaining = prorated_amount(total_val, used_mo * 30, total_mo * 30)
            return (
                f"Proration: ${total_val:,.2f} for {used_mo}/{total_mo} months "
                f"= ${remaining:,.2f} remaining (${total_val - remaining:,.2f} consumed)."
            )
        elif days and len(days) >= 2:
            used_d, total_d = min(days), max(days)
            remaining = prorated_amount(total_val, used_d, total_d)
            return (
                f"Proration: ${total_val:,.2f} for {used_d}/{total_d} days "
                f"= ${remaining:,.2f} remaining."
            )
        return None
    except Exception:
        return None


def _extract_dollars(text: str) -> list[float]:
    """Extract dollar amounts, largest first."""
    return sorted(
        [float(m.replace(",", "")) for m in _DOLLAR_PAT.findall(text)],
        reverse=True,
    )


def _extract_pcts(text: str) -> list[float]:
    return [float(m) for m in _PCT_PAT.findall(text)]
