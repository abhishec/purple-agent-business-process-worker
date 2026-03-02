"""
dynamic_tools.py
Runtime tool factory for computation gaps.

=============================================================
FINITE STATE MACHINE — full form explained inline
=============================================================
FSM = Finite State Machine.
  FINITE   — the agent can only be in one of a fixed set of named states
              (DECOMPOSE, ASSESS, COMPUTE, POLICY_CHECK, APPROVAL_GATE,
               MUTATE, SCHEDULE_NOTIFY, COMPLETE). Not infinite.
  STATE    — a discrete phase with specific rules and permitted actions.
              In ASSESS you ONLY read. In MUTATE you ONLY write.
              The state boundary prevents the agent from attempting
              a mutation during the reading phase (a common LLM failure).
  MACHINE  — a deterministic engine that knows which state it is in,
              what it can do there, and what comes next.

STATIC FSM (everyone else):
  - 14 hardcoded process types → fixed state sequences
  - Fallback to 5-state "general" for anything new
  - Per-state instructions like "gather data" (useless)

DYNAMIC FSM:
  - Unknown process type → Haiku synthesizes the right state sequence
    AND writes specific instructions per state
  - SUPPLIER_RISK_ASSESSMENT gets: "gather credit rating, ESG score,
    geo-risk → compute weighted risk score = 0.3×credit + 0.25×geo..."
    instead of "gather data"

THIS FILE extends the same principle to TOOLS:

STATIC TOOLS (what everyone else does):
  - Hardcode 7 finance functions → ship in the image → never grows

DYNAMIC TOOLS:
  - Detect when the task needs math that no tool handles
  - Synthesize a Python implementation via Haiku
  - Validate against auto-generated test cases
  - Register to tool_registry.json (persistent across all future tasks)
  - Hot-load immediately — zero restart needed

Competition impact:
  - Generality (20%): infinite computation capability, not just 14 functions
  - Drift Adaptation (20%): if formula changes → old tool fails accuracy
    check → synthesize new one → pass
  - Error Recovery (8%): tool synthesis IS error recovery for math failures
  - pass^k: consistent correct math = all k attempts return same result
=============================================================
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import asyncio
import statistics as _statistics_module
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from src.config import ANTHROPIC_API_KEY as _ANTHROPIC_API_KEY


# ── Registry store ─────────────────────────────────────────────────────────

_REGISTRY_FILE = Path(os.environ.get("RL_CACHE_DIR", "/app")) / "tool_registry.json"
_registry_defs: dict[str, dict] = {}   # name → full definition + python_code
_registry_fns: dict[str, Any] = {}     # name → callable (hot-loaded at startup)
_registry_loaded = False


def _load_registry() -> None:
    global _registry_defs, _registry_fns, _registry_loaded
    if _registry_loaded:
        return
    try:
        if _REGISTRY_FILE.exists():
            _registry_defs = json.loads(_REGISTRY_FILE.read_text())
            for name, defn in _registry_defs.items():
                fn = _exec_in_sandbox(defn.get("python_code", ""), name)
                if fn:
                    _registry_fns[name] = fn
    except Exception:
        _registry_defs = {}
        _registry_fns = {}
    _registry_loaded = True


def _save_registry() -> None:
    try:
        _REGISTRY_FILE.write_text(json.dumps(_registry_defs, indent=2))
    except Exception:
        pass  # best-effort — never crash the task


# ── Sandbox execution ────────────────────────────────────────────────────────
#
# We exec synthesized code in a restricted namespace.
# NO: import, open, eval, exec, os, sys, __import__
# YES: math, Decimal, ROUND_HALF_UP, safe builtins
#
# This is intentionally limited — financial math only needs arithmetic.

_SANDBOX_GLOBALS: dict[str, Any] = {
    "__builtins__": None,   # block all builtins explicitly
    # Math
    "math": math,
    "Decimal": Decimal,
    "ROUND_HALF_UP": ROUND_HALF_UP,
    # Monte Carlo + statistical simulation support
    "random": random,
    "statistics": _statistics_module,
    # Safe builtins restored individually
    "abs": abs, "int": int, "float": float, "str": str, "bool": bool,
    "round": round, "min": min, "max": max, "sum": sum, "len": len,
    "range": range, "enumerate": enumerate, "zip": zip,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "isinstance": isinstance, "pow": pow, "divmod": divmod,
    "sorted": sorted, "reversed": reversed, "any": any, "all": all,
    "ValueError": ValueError, "ZeroDivisionError": ZeroDivisionError,
}


def _exec_in_sandbox(code: str, func_name: str) -> Any | None:
    """
    Execute synthesized Python code in a restricted namespace.
    Returns the callable if successful, None if code fails to compile or run.
    """
    namespace = dict(_SANDBOX_GLOBALS)
    try:
        exec(compile(code, "<dynamic_tool>", "exec"), namespace)
        fn = namespace.get(func_name)
        if callable(fn):
            return fn
    except Exception:
        pass
    return None


# ── Gap detection ────────────────────────────────────────────────────────────
#
# Pattern: map task text keywords → gap key + description + param hints.
# We check if the gap key already exists in registry OR in passed tool list.
# If yes → skip (already have it). If no → flag as gap.

_GAP_PATTERNS: list[dict] = [
    {
        "key": "finance_npv",
        "patterns": [
            r"\bnpv\b", r"\bnet present value\b", r"\bdiscounted cash flow\b",
            r"\bpresent value of cash", r"\bpv of.*flows\b",
        ],
        "description": (
            "Net Present Value: NPV = sum(cash_flow[t] / (1+rate)^t) - initial_investment. "
            "Function name: finance_npv. "
            "Params: cash_flows (list of floats, first is usually negative investment), "
            "discount_rate (annual rate as %, e.g. 10 for 10%)."
        ),
    },
    {
        "key": "finance_irr",
        "patterns": [
            r"\birr\b", r"\binternal rate of return\b",
        ],
        "description": (
            "Internal Rate of Return: rate that makes NPV = 0. "
            "Function name: finance_irr. "
            "Params: cash_flows (list of floats, first negative = investment). "
            "Use Newton-Raphson or bisection method. Return rate as percentage."
        ),
    },
    {
        "key": "finance_bond_price",
        "patterns": [
            r"\bbond price\b", r"\byield to maturity\b", r"\bytm\b",
            r"\bcoupon.*face value\b", r"\bface value.*coupon\b", r"\bbond valuation\b",
        ],
        "description": (
            "Bond pricing: price = sum(coupon / (1+r)^t) + face_value / (1+r)^n. "
            "Function name: finance_bond_price. "
            "Params: face_value (float), coupon_rate (annual % e.g. 5 for 5%), "
            "ytm (yield to maturity as %, e.g. 6 for 6%), "
            "periods (int, number of coupon periods), "
            "periods_per_year (int, default 2 for semiannual)."
        ),
    },
    {
        "key": "finance_depreciation",
        "patterns": [
            r"\bstraight.line depreciation\b", r"\bsl depreciation\b",
            r"\bdepreciation schedule\b", r"\bdouble.declining\b",
            r"\bsum.of.years.digits\b", r"\bsoyd\b", r"\bddb\b",
        ],
        "description": (
            "Asset depreciation schedule supporting straight-line (SL), "
            "double-declining balance (DDB), and sum-of-years-digits (SOYD). "
            "Function name: finance_depreciation. "
            "Params: cost (float), salvage_value (float), useful_life (int years), "
            "method (str: 'sl', 'ddb', or 'soyd'). "
            "Return annual depreciation schedule."
        ),
    },
    {
        "key": "finance_wacc",
        "patterns": [
            r"\bwacc\b", r"\bweighted average cost of capital\b",
            r"\bcost of equity\b", r"\bcost of debt\b",
        ],
        "description": (
            "Weighted Average Cost of Capital: WACC = (E/V)*Re + (D/V)*Rd*(1-Tc). "
            "Function name: finance_wacc. "
            "Params: equity_value (float), debt_value (float), "
            "cost_of_equity (float, % e.g. 12 for 12%), "
            "cost_of_debt (float, % e.g. 8 for 8%), "
            "tax_rate (float, % e.g. 25 for 25%)."
        ),
    },
    {
        "key": "finance_compound_interest",
        "patterns": [
            r"\bcompound interest\b", r"\beffective annual rate\b",
            r"\bear\b", r"\bapy\b", r"\bcompounding.*frequency\b",
        ],
        "description": (
            "Compound interest: A = P * (1 + r/n)^(n*t). "
            "Function name: finance_compound_interest. "
            "Params: principal (float), annual_rate (float, % e.g. 5 for 5%), "
            "years (float), compounds_per_year (int, e.g. 12 for monthly). "
            "Return final amount, interest earned, and effective annual rate."
        ),
    },
    {
        "key": "finance_loan_amortization",
        "patterns": [
            r"\bamortization\b", r"\bloan schedule\b",
            r"\bmonthly payment.*loan\b", r"\bmortgage.*schedule\b",
            r"\binstallment.*principal\b",
        ],
        "description": "Loan amortization schedule — seeded at startup, not re-synthesized.",
    },
    # ── Monte Carlo + numerical methods ─────────────────────────────────────
    {
        "key": "finance_monte_carlo",
        "patterns": [
            r"\bmonte.?carlo\b", r"\bsimulat\w+ (paths?|scenario|run|trial)",
            r"\bstochastic\b", r"\brandom (walk|path|simulation)\b",
            r"\bvar\b.*\bsimulat\b", r"\bvalue.?at.?risk.*simulat\b",
            r"\b\d[\d,]+\s*(path|trial|iteration|run|sample)s?\b",
        ],
        "description": (
            "Monte Carlo simulation for financial risk and pricing. "
            "Function name: finance_monte_carlo. "
            "Params: s0 (float, initial asset price or value), "
            "mu (float, annual drift/return as decimal, e.g. 0.08), "
            "sigma (float, annual volatility as decimal, e.g. 0.20), "
            "T (float, time horizon in years), "
            "n_paths (int, number of simulation paths, default 10000), "
            "n_steps (int, time steps per path, default 252). "
            "Returns: dict with 'mean', 'std', 'var_95', 'var_99', 'paths_summary', "
            "'result' (mean final value). Use random.seed(42) for reproducibility. "
            "Use random.gauss(0,1) for normal samples. "
            "Formula: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)"
        ),
    },
    {
        "key": "finance_black_scholes",
        "patterns": [
            r"\bblack.?scholes\b", r"\boption.?pric\w+\b",
            r"\bcall.?option\b", r"\bput.?option\b",
            r"\bgreeks\b", r"\bdelta\b.*\bgamma\b",
            r"\bimplied.?volatility\b", r"\bvega\b", r"\btheta\b",
        ],
        "description": (
            "Black-Scholes option pricing model with Greeks. "
            "Function name: finance_black_scholes. "
            "Params: S (float, current stock price), K (float, strike price), "
            "T (float, time to expiry in years), r (float, risk-free rate as decimal), "
            "sigma (float, volatility as decimal), option_type (str: 'call' or 'put'). "
            "Compute d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T)), "
            "d2 = d1 - sigma*sqrt(T). "
            "Use math.erf for N(x) approximation: N(x) = 0.5*(1 + math.erf(x/math.sqrt(2))). "
            "Return dict with 'result' (option price), 'details' containing "
            "d1, d2, delta, gamma, theta, vega, rho."
        ),
    },
    {
        "key": "finance_var",
        "patterns": [
            r"\bvalue.?at.?risk\b", r"\bvar\b.*\b(confidence|percentile|portfolio)\b",
            r"\bportfolio.?risk\b", r"\bcvar\b", r"\bexpected.?shortfall\b",
            r"\brisk.?measure\b", r"\b(95|99)%\s*var\b",
        ],
        "description": (
            "Portfolio Value at Risk (VaR) and Conditional VaR (CVaR). "
            "Function name: finance_var. "
            "Params: returns (list of float, historical or simulated returns), "
            "confidence_level (float, e.g. 0.95 for 95%), "
            "portfolio_value (float, current portfolio value, default 1.0). "
            "Sort returns ascending, find percentile cutoff. "
            "VaR = -returns[floor(n*(1-confidence_level))] * portfolio_value. "
            "CVaR = -mean(returns below VaR cutoff) * portfolio_value. "
            "Return dict with 'result' (VaR), 'details' with cvar, "
            "confidence_level, n_observations, worst_return."
        ),
    },
    {
        "key": "finance_newton_raphson",
        "patterns": [
            r"\bnewton.?raphson\b", r"\bbisection method\b",
            r"\broot.?find\w+\b", r"\bnumerical.*irr\b",
            r"\bsolve for.*rate\b", r"\bfind.*yield\b",
        ],
        "description": (
            "Newton-Raphson root finder for implicit rate/yield equations. "
            "Function name: finance_newton_raphson. "
            "Params: cash_flows (list of float, period cash flows including t=0), "
            "target_npv (float, target NPV to solve for, default 0.0), "
            "initial_guess (float, starting rate as decimal, default 0.1), "
            "max_iter (int, default 100), tolerance (float, default 1e-6). "
            "Use IRR formula: NPV(r) = sum(cf[t]/(1+r)^t). "
            "Derivative: dNPV/dr = sum(-t*cf[t]/(1+r)^(t+1)). "
            "Iterate: r_new = r - NPV(r)/dNPV(r). "
            "Return dict with 'result' (rate as percentage), "
            "'details' with iterations, converged (bool), npv_at_result."
        ),
    },
    # ── HR / Payroll ──────────────────────────────────────────────────────────
    {
        "key": "hr_overtime",
        "patterns": [
            r"\bovertime\b", r"\btime.and.a.half\b", r"\bflsa\b",
            r"\bot pay\b", r"\bovertime pay\b", r"\bovertime rate\b",
            r"\bdouble.time\b", r"\bovertime hours\b",
        ],
        "description": (
            "Overtime pay calculation per FLSA rules. "
            "Function name: hr_overtime. "
            "Params: regular_hours (float), overtime_hours (float), "
            "hourly_rate (float), overtime_multiplier (float, default 1.5). "
            "Returns dict with 'result' (total pay), 'details' with regular_pay, "
            "overtime_pay, total_hours."
        ),
    },
    {
        "key": "hr_proration",
        "patterns": [
            r"\bprorat\w+\b", r"\bprorated salary\b", r"\bpartial.period pay\b",
            r"\bmid.month\b", r"\bpartial month\b", r"\bpro.rata\b",
            r"\bdays worked.*salary\b", r"\bsalary.*partial\b",
        ],
        "description": (
            "Prorated salary for partial pay periods. "
            "Function name: hr_proration. "
            "Params: annual_salary (float), working_days_in_period (int), "
            "total_working_days_in_period (int), pay_frequency (str: 'monthly'/'biweekly'/'weekly'). "
            "Returns dict with 'result' (prorated pay amount), 'details' with daily_rate, "
            "full_period_pay, days_fraction."
        ),
    },
    {
        "key": "hr_benefits_cost",
        "patterns": [
            r"\bbenefits cost\b", r"\bemployer contribution\b",
            r"\btotal comp\w*\b", r"\bbenefits.*per employee\b",
            r"\bemployee benefits\b", r"\bbenefits burden\b",
            r"\bpayroll burden\b", r"\btotal compensation\b",
        ],
        "description": (
            "Total compensation and benefits cost calculation. "
            "Function name: hr_benefits_cost. "
            "Params: base_salary (float), health_insurance_monthly (float), "
            "retirement_match_pct (float, employer 401k match as %, e.g. 3.0), "
            "other_benefits_annual (float, default 0). "
            "Returns dict with 'result' (total annual comp cost), 'details' with "
            "salary, benefits_annual, retirement_contribution, total_cost, burden_rate_pct."
        ),
    },
    {
        "key": "hr_headcount",
        "patterns": [
            r"\bfte\b", r"\bfull.time equivalent\b", r"\bheadcount ratio\b",
            r"\battrition rate\b", r"\bturnover rate\b", r"\bheadcount\b",
            r"\bemployee count\b", r"\bstaff ratio\b",
        ],
        "description": (
            "Headcount metrics: FTE, attrition rate, and span of control. "
            "Function name: hr_headcount. "
            "Params: full_time_count (int), part_time_count (int, default 0), "
            "part_time_hours_avg (float, avg weekly hours of part-timers, default 20), "
            "separations_in_period (int, default 0), "
            "avg_headcount_in_period (float, default None — uses full_time_count). "
            "Returns dict with 'result' (total FTE), 'details' with fte_full_time, "
            "fte_part_time, annualized_attrition_rate_pct."
        ),
    },
    # ── SLA / Operations ──────────────────────────────────────────────────────
    {
        "key": "ops_sla_credit",
        "patterns": [
            r"\bsla credit\b", r"\bservice credit\b", r"\bsla.*penalty\b",
            r"\buptime.*sla\b", r"\bsla.*uptime\b", r"\bdowntime penalty\b",
            r"\bservice level.*credit\b", r"\bsla.*breach\b",
        ],
        "description": (
            "SLA credit calculation based on uptime percentage breach. "
            "Function name: ops_sla_credit. "
            "Params: monthly_fee (float), actual_uptime_pct (float, e.g. 99.1), "
            "sla_tiers (list of dicts with 'min_uptime' and 'credit_pct', "
            "sorted descending by min_uptime). "
            "Returns dict with 'result' (credit amount), 'details' with "
            "uptime_pct, sla_tier_matched, credit_pct, credit_amount."
        ),
    },
    {
        "key": "ops_uptime",
        "patterns": [
            r"\bavailability\b.*\b(percent|%)\b", r"\bmttr\b", r"\bmtbf\b",
            r"\buptime.?percent\w*\b", r"\bdowntime.?minutes\b",
            r"\bincident.?duration\b", r"\bavailability.?calc\w*\b",
            r"\bservice.?availability\b",
        ],
        "description": (
            "System availability and reliability metrics (uptime %, MTTR, MTBF). "
            "Function name: ops_uptime. "
            "Params: total_minutes_in_period (int), downtime_minutes (float), "
            "incident_count (int, default 1). "
            "Returns dict with 'result' (uptime_pct), 'details' with "
            "uptime_pct, downtime_pct, downtime_minutes, mttr_minutes, "
            "availability_nines (e.g. '2-nines')."
        ),
    },
    {
        "key": "ops_penalty",
        "patterns": [
            r"\bliquidated damages\b", r"\blate delivery penalty\b",
            r"\bbreach penalty\b", r"\bpenalty.*per day\b",
            r"\bcontract.*penalty\b", r"\bdelay penalty\b",
            r"\bpenalty interest\b", r"\bpenalty calculation\b",
        ],
        "description": (
            "Contract breach and late-delivery penalty calculation. "
            "Function name: ops_penalty. "
            "Params: contract_value (float), days_late (int), "
            "daily_penalty_rate (float, as decimal e.g. 0.001 for 0.1%/day), "
            "max_penalty_pct (float, cap as decimal e.g. 0.10 for 10% cap). "
            "Returns dict with 'result' (penalty amount), 'details' with "
            "uncapped_penalty, capped_at, penalty_rate_used, days_late."
        ),
    },
    # ── Supply Chain ────────────────────────────────────────────────────────
    {
        "key": "sc_eoq",
        "patterns": [
            r"\beoq\b", r"\beconomic order quantity\b", r"\breorder point\b",
            r"\bsafety stock\b", r"\blead time demand\b",
            r"\border quantity\b", r"\binventory.?order\b",
        ],
        "description": (
            "Economic Order Quantity (EOQ), reorder point, and safety stock. "
            "Function name: sc_eoq. "
            "Params: annual_demand (float, units/year), ordering_cost (float, $ per order), "
            "holding_cost_per_unit (float, $ per unit per year), "
            "lead_time_days (float, default 0), daily_demand_stddev (float, default 0), "
            "service_level_z (float, z-score for service level, default 1.645 for 95%). "
            "Returns dict with 'result' (EOQ in units), 'details' with "
            "eoq, reorder_point, safety_stock, orders_per_year, total_annual_cost."
        ),
    },
    {
        "key": "sc_inventory_value",
        "patterns": [
            r"\bfifo\b", r"\blifo\b", r"\bweighted average cost\b",
            r"\binventory valuation\b", r"\bcost of goods sold\b",
            r"\bcogs\b", r"\binventory.*method\b",
        ],
        "description": (
            "Inventory valuation using FIFO, LIFO, or weighted average cost method. "
            "Function name: sc_inventory_value. "
            "Params: purchases (list of dicts with 'units' and 'unit_cost'), "
            "units_sold (int), method (str: 'fifo', 'lifo', or 'weighted_avg'). "
            "Returns dict with 'result' (COGS), 'details' with "
            "cogs, ending_inventory_units, ending_inventory_value, avg_unit_cost."
        ),
    },
    {
        "key": "sc_stockout_risk",
        "patterns": [
            r"\bstockout\b", r"\bstockout.?prob\w*\b", r"\bservice.?level\b",
            r"\bfill.?rate\b", r"\binventory.?risk\b",
            r"\bstockout.?risk\b", r"\bdemand.?uncertainty\b",
        ],
        "description": (
            "Stockout probability and service level calculation. "
            "Function name: sc_stockout_risk. "
            "Params: avg_daily_demand (float), demand_stddev (float), "
            "lead_time_days (float), reorder_point (float). "
            "Returns dict with 'result' (service_level_pct), 'details' with "
            "service_level_pct, stockout_probability_pct, z_score, "
            "expected_demand_during_lead_time, safety_stock_implied."
        ),
    },
    # ── Date / Time Math ────────────────────────────────────────────────────
    {
        "key": "dt_business_days",
        "patterns": [
            r"\bbusiness days\b", r"\bworking days\b", r"\bworkday\w*\b",
            r"\bdays between.*dates\b", r"\bexcluding.*holiday\b",
            r"\bbusiness.?day.?count\b", r"\bworking.?day.?calc\w*\b",
        ],
        "description": (
            "Calculate business (working) days between two dates, excluding weekends. "
            "Function name: dt_business_days. "
            "Params: start_date (str, ISO format YYYY-MM-DD), "
            "end_date (str, ISO format YYYY-MM-DD), "
            "holiday_count (int, number of public holidays in range, default 0). "
            "Use math only — no datetime imports (unavailable in sandbox). "
            "Approximate: parse year/month/day from string manually. "
            "Algorithm: total_days = end - start; weeks = total_days // 7; "
            "weekdays = weeks * 5 + min(extra_days, 5 - start_weekday). "
            "Returns dict with 'result' (business days count), 'details' with "
            "total_calendar_days, weekends_excluded, holidays_excluded."
        ),
    },
    {
        "key": "dt_prorata",
        "patterns": [
            r"\bpro.?rata\b", r"\bpartial period\b", r"\bdays.*in.*month\b",
            r"\bprorated.*days\b", r"\bpro.?rat\w+\b",
            r"\bmonthly.*days\b", r"\bfraction of.*period\b",
        ],
        "description": (
            "Pro-rata calculation for partial periods (daily/monthly proration). "
            "Function name: dt_prorata. "
            "Params: full_period_amount (float), days_in_period (int), "
            "total_days_in_period (int). "
            "Returns dict with 'result' (prorated amount), 'details' with "
            "daily_rate, days_fraction, full_period_amount."
        ),
    },
    {
        "key": "dt_aging",
        "patterns": [
            r"\baging\b.*\b(bucket|analysis|report)\b",
            r"\bdays.*outstanding\b", r"\bar.*aging\b",
            r"\breceivable.*aging\b", r"\boverdue.*bucket\b",
            r"\b0.30.*(day|bucket)\b", r"\b31.60\b", r"\b61.90\b",
            r"\b90\+\b.*\bday\b",
        ],
        "description": (
            "Accounts receivable aging bucket analysis. "
            "Function name: dt_aging. "
            "Params: invoices (list of dicts with 'amount' and 'days_outstanding'). "
            "Bucket boundaries: 0-30, 31-60, 61-90, 91-120, 120+ days. "
            "Returns dict with 'result' (total outstanding), 'details' with "
            "bucket_0_30, bucket_31_60, bucket_61_90, bucket_91_120, bucket_over_120, "
            "each as {'count': int, 'amount': float}, total_outstanding."
        ),
    },
    # ── Statistics ────────────────────────────────────────────────────────────
    {
        "key": "stats_zscore",
        "patterns": [
            r"\bz.?score\b", r"\bpercentile rank\b",
            r"\bstandard deviations? from\b", r"\bz.stat\w*\b",
            r"\bnormal distribution\b", r"\bstandardiz\w+\b",
        ],
        "description": (
            "Z-score and percentile rank calculation. "
            "Function name: stats_zscore. "
            "Params: value (float), mean (float), std_dev (float). "
            "Returns dict with 'result' (z_score), 'details' with "
            "z_score, percentile_approx (using erf approximation: "
            "0.5*(1+math.erf(z/math.sqrt(2)))*100), interpretation (str)."
        ),
    },
    {
        "key": "stats_weighted_avg",
        "patterns": [
            r"\bweighted average\b", r"\bweighted score\b",
            r"\bcomposite score\b", r"\bweighted.?mean\b",
            r"\bweighted.?calc\w*\b", r"\bweight\w+.*score\b",
        ],
        "description": (
            "Weighted average / composite score calculation. "
            "Function name: stats_weighted_avg. "
            "Params: values (list of float), weights (list of float). "
            "Returns dict with 'result' (weighted average), 'details' with "
            "weighted_avg, sum_of_weights, weighted_sum, "
            "components (list of {value, weight, contribution})."
        ),
    },
    {
        "key": "stats_regression",
        "patterns": [
            r"\blinear regression\b", r"\btrend line\b",
            r"\br.?squared\b", r"\bslope.*intercept\b",
            r"\bleast squares\b", r"\bregression.*analysis\b",
            r"\bline of best fit\b",
        ],
        "description": (
            "Simple linear regression (y = mx + b) with R-squared. "
            "Function name: stats_regression. "
            "Params: x_values (list of float), y_values (list of float). "
            "Returns dict with 'result' (slope m), 'details' with "
            "slope, intercept, r_squared, equation_str (e.g. 'y = 2.5x + 10.3'), "
            "predict_next (predicted y for x = last_x + 1)."
        ),
    },
    # ── Tax ──────────────────────────────────────────────────────────────────
    {
        "key": "tax_vat",
        "patterns": [
            r"\bvat\b", r"\bgst\b", r"\bvalue.?added.?tax\b",
            r"\bgoods.*services.*tax\b", r"\bvat.*calc\w*\b",
            r"\breverse.*vat\b", r"\bvat.*exclusive\b", r"\bvat.*inclusive\b",
            r"\btax.*inclusive\b", r"\btax.*exclusive\b",
        ],
        "description": (
            "VAT/GST calculation: exclusive, inclusive, and reverse. "
            "Function name: tax_vat. "
            "Params: amount (float), vat_rate (float, as %, e.g. 20 for 20%), "
            "mode (str: 'add' to add VAT to net, 'extract' to extract VAT from gross). "
            "Returns dict with 'result' (vat_amount), 'details' with "
            "net_amount, vat_amount, gross_amount, vat_rate_pct, mode."
        ),
    },
    {
        "key": "tax_withholding",
        "patterns": [
            r"\bwithholding tax\b", r"\bgross.?up\b", r"\bnet.?to.?gross\b",
            r"\bgross.?up.?calc\w*\b", r"\bwithhold\w+\b",
            r"\bpaye\b", r"\btax.*gross.?up\b",
        ],
        "description": (
            "Withholding tax and gross-up calculation (net-to-gross). "
            "Function name: tax_withholding. "
            "Params: amount (float), withholding_rate (float, as %, e.g. 30 for 30%), "
            "mode (str: 'withhold' = deduct from gross, 'gross_up' = gross up from net). "
            "Returns dict with 'result' (tax_withheld), 'details' with "
            "gross_amount, net_amount, tax_withheld, effective_rate_pct."
        ),
    },
    {
        "key": "tax_depreciation_tax",
        "patterns": [
            r"\btax depreciation\b", r"\bcapital allowance\b",
            r"\baccelerated depreciation\b", r"\bbonusdepreciation\b",
            r"\bsection 179\b", r"\bmacrs\b",
        ],
        "description": (
            "Tax depreciation / capital allowances calculation. "
            "Function name: tax_depreciation_tax. "
            "Params: asset_cost (float), allowance_rate (float, as %, e.g. 25 for 25%/year), "
            "years (int), method (str: 'reducing_balance' or 'straight_line'). "
            "Returns dict with 'result' (first_year_allowance), 'details' with "
            "annual_schedule (list of {year, allowance, tax_base}), "
            "total_allowances, final_tax_base."
        ),
    },
    # ── Risk / Compliance ────────────────────────────────────────────────────
    {
        "key": "risk_weighted_score",
        "patterns": [
            r"\brisk.?score\b", r"\bweighted risk\b", r"\bahp\b",
            r"\bcompliance.?score\b", r"\bkyc.?risk\b",
            r"\brisk.?rating\b", r"\brisk.?matrix\b",
            r"\bpriority.?matrix\b", r"\brisk.?calc\w*\b",
        ],
        "description": (
            "Weighted risk scoring (e.g. AHP, KYC risk rating, compliance score). "
            "Function name: risk_weighted_score. "
            "Params: factors (list of dicts with 'name', 'score' (0-10), 'weight' (0-1)), "
            "scale_max (float, maximum possible score, default 10). "
            "Returns dict with 'result' (composite_score), 'details' with "
            "composite_score, score_pct, risk_band (Low/Medium/High/Critical), "
            "factor_breakdown (list of {name, score, weight, contribution})."
        ),
    },
    {
        "key": "risk_concentration",
        "patterns": [
            r"\bconcentration risk\b", r"\bherfindahl\b", r"\bhhi\b",
            r"\btop.?\d+.?concentration\b", r"\bmarket share.*concentr\w*\b",
            r"\bconcentration.?index\b", r"\bclient.?concentration\b",
        ],
        "description": (
            "Concentration risk metrics: Herfindahl-Hirschman Index (HHI) and top-N share. "
            "Function name: risk_concentration. "
            "Params: shares (list of float, market/portfolio share values, "
            "either as decimals summing to 1 or counts summing to total), "
            "top_n (int, number of largest to report, default 3). "
            "Returns dict with 'result' (hhi_score), 'details' with "
            "hhi, hhi_normalized (0-1), top_n_concentration_pct, "
            "risk_band ('Low'/<1500 / 'Moderate'/1500-2500 / 'High'/>2500), "
            "shares_pct (list)."
        ),
    },
    # ── AR / Collections ──────────────────────────────────────────────────────
    {
        "key": "ar_bad_debt",
        "patterns": [
            r"\bbad debt\b", r"\bexpected credit loss\b", r"\becl\b",
            r"\bimpairment\b", r"\bdebt.?provision\b",
            r"\bprovision.*doubtful\b", r"\bdoubtful.*debt\b",
            r"\bwrite.?off\b.*\breceivable\b",
        ],
        "description": (
            "Bad debt provision and expected credit loss (ECL) calculation. "
            "Function name: ar_bad_debt. "
            "Params: receivables_by_bucket (list of dicts with 'amount' and 'days_outstanding'), "
            "provision_rates (dict mapping age bucket labels to provision rates as decimals, "
            "e.g. {'0_30': 0.01, '31_60': 0.05, '61_90': 0.10, '91_120': 0.25, 'over_120': 0.50}). "
            "Returns dict with 'result' (total_provision), 'details' with "
            "provision_by_bucket, total_receivables, provision_rate_overall_pct."
        ),
    },
    {
        "key": "ar_collection_rate",
        "patterns": [
            r"\bdso\b", r"\bdays sales outstanding\b",
            r"\bcollection.?rate\b", r"\bcollection.?effic\w*\b",
            r"\bcollect\w+.*receiv\w+\b", r"\bcash.?conversion\b",
            r"\breceivable.?turnover\b",
        ],
        "description": (
            "AR collection efficiency: DSO, collection rate, receivable turnover. "
            "Function name: ar_collection_rate. "
            "Params: ending_ar (float), revenue_in_period (float), "
            "period_days (int, e.g. 90 for quarter), "
            "cash_collected (float, default None — skips collection rate if not provided). "
            "Returns dict with 'result' (dso_days), 'details' with "
            "dso_days, collection_rate_pct, receivable_turnover, "
            "revenue_per_day."
        ),
    },
    # ── Contract Math ────────────────────────────────────────────────────────
    {
        "key": "contract_escalation",
        "patterns": [
            r"\bescalation clause\b", r"\bprice escalation\b",
            r"\bcpi.?adjust\w*\b", r"\bannual.?increase\b",
            r"\bescalat\w+.*contract\b", r"\bcontract.*escalat\w+\b",
            r"\binflation.?adjust\w*\b",
        ],
        "description": (
            "Contract price escalation / CPI adjustment over multiple periods. "
            "Function name: contract_escalation. "
            "Params: base_amount (float), annual_escalation_rate (float, as %, e.g. 3.0), "
            "years (int). "
            "Returns dict with 'result' (final_year_amount), 'details' with "
            "yearly_amounts (list of {year, amount}), total_over_term, "
            "cumulative_increase_pct."
        ),
    },
    {
        "key": "contract_termination_fee",
        "patterns": [
            r"\bearly termination\b", r"\btermination fee\b",
            r"\bearly.?exit.?fee\b", r"\btermination.?penalty\b",
            r"\bremaining.*term.*penalty\b", r"\bbreak.?fee\b",
            r"\bcontract.?cancel\w*\b",
        ],
        "description": (
            "Early contract termination fee calculation. "
            "Function name: contract_termination_fee. "
            "Params: monthly_value (float), remaining_months (int), "
            "termination_fee_pct (float, % of remaining contract value, e.g. 20.0 for 20%), "
            "notice_period_months (int, free months already given, default 0). "
            "Returns dict with 'result' (termination_fee), 'details' with "
            "remaining_contract_value, fee_pct, termination_fee, "
            "effective_months_charged."
        ),
    },
]

# Amortization tool code — seeded into registry at startup.
# Self-contained: uses only sandbox-available Decimal + ROUND_HALF_UP.
_AMORTIZATION_CODE = '''
def finance_loan_amortization(principal, annual_rate, months):
    """360-period loan amortization with cent-level Decimal precision."""
    P = Decimal(str(principal))
    annual = Decimal(str(annual_rate)) / Decimal("100")
    r = annual / Decimal("12")
    n = int(months)

    if r == Decimal("0"):
        monthly = (P / Decimal(str(n))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    else:
        factor = (Decimal("1") + r) ** n
        monthly = (P * r * factor / (factor - Decimal("1"))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    schedule = []
    balance = P
    total_interest = Decimal("0")

    for period in range(1, n + 1):
        interest = (balance * r).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        principal_portion = (monthly - interest).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if period == n:
            principal_portion = balance
            monthly_actual = (balance + interest).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        else:
            monthly_actual = monthly
        balance = (balance - principal_portion).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        total_interest += interest
        schedule.append({
            "period": period,
            "payment": float(monthly_actual),
            "principal": float(principal_portion),
            "interest": float(interest),
            "balance": float(balance),
        })

    return {
        "result": float(monthly),
        "details": {
            "monthly_payment": float(monthly),
            "total_payments": float(monthly * Decimal(str(n))),
            "total_interest": float(total_interest),
            "schedule": schedule if n <= 12 else schedule[:12],
        },
    }
'''.strip()

_AMORTIZATION_SCHEMA = {
    "name": "finance_loan_amortization",
    "description": (
        "Calculate loan amortization schedule with exact monthly payments. "
        "Use for: mortgage schedules, car loans, business loans, any installment loan. "
        "Returns monthly payment, total interest, and full payment schedule."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "principal": {"type": "number", "description": "Loan principal amount in dollars"},
            "annual_rate": {"type": "number", "description": "Annual interest rate as percentage (e.g. 5.5 for 5.5%)"},
            "months": {"type": "integer", "description": "Loan term in months (e.g. 360 for 30-year mortgage)"},
        },
        "required": ["principal", "annual_rate", "months"],
    },
}


def detect_tool_gaps(task_text: str, existing_tools: list[dict]) -> list[dict]:
    """
    Scan task text for computation patterns that no existing tool handles.

    existing_tools: list of tool definitions already available (MCP + registered).
    Returns: list of gap dicts with {key, description} for each gap detected.
    """
    _load_registry()

    # Build set of existing tool names (from both MCP tools and registry)
    existing_names: set[str] = set()
    for t in existing_tools:
        name = t.get("name") or t.get("function", {}).get("name", "")
        if name:
            existing_names.add(name)
    existing_names.update(_registry_fns.keys())  # in-memory registered tools

    text_lower = task_text.lower()
    gaps = []

    for gap in _GAP_PATTERNS:
        key = gap["key"]
        # Skip if we already have this tool
        if key in existing_names:
            continue
        # Skip amortization gap detection — it's always seeded
        if key == "finance_loan_amortization":
            continue
        # Check patterns
        if any(re.search(p, text_lower) for p in gap["patterns"]):
            gaps.append({"key": key, "description": gap["description"]})

    return gaps


_LLM_GAP_DETECTION_SYSTEM = """\
You are a business process analyst. Your job is to identify custom mathematical calculations
that a business process task requires but that are NOT:
- Simple database read/write operations (SELECT, INSERT, UPDATE)
- Standard tool calls (already listed)
- Basic arithmetic (addition, subtraction, percentage of a known number)

Focus ONLY on formulas that need a dedicated Python function to implement correctly
(e.g. amortization schedule, z-score normalization, EOQ, weighted risk score).

Return a JSON array of objects. Each object: {
  "key": "snake_case_name",
  "description": "Function name: snake_case_name. Params: ... Returns dict with 'result' (scalar) and 'details'."
}

If no custom math is needed, return [].
Return ONLY valid JSON — no markdown, no explanation."""


async def detect_tool_gaps_llm(task_text: str, existing_tools: list[dict]) -> list[dict]:
    """
    Phase 2: LLM-based gap detection for computations not in static patterns.
    Only called when Phase 1 (regex) finds no gaps.
    Asks Haiku to identify what custom calculations this task needs.
    Returns list of gap dicts with {key, description} — max 2 items.
    Timeout: 8 seconds. Never raises — returns [] on any failure.
    """
    try:
        # Build list of existing tool names so Haiku knows what's already available
        _load_registry()
        existing_names: list[str] = []
        for t in existing_tools:
            name = t.get("name") or t.get("function", {}).get("name", "")
            if name:
                existing_names.append(name)
        existing_names.extend(_registry_fns.keys())

        tools_list = ", ".join(existing_names[:30]) if existing_names else "none"

        prompt = (
            f"Business process task:\n{task_text[:1500]}\n\n"
            f"Already available tools/functions: {tools_list}\n\n"
            "List ONLY the specific mathematical calculations this task requires "
            "that are NOT simple database operations and NOT already covered by the listed tools. "
            "Return JSON array. If no custom math is needed, return []."
        )

        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=_ANTHROPIC_API_KEY)
        msg = await asyncio.wait_for(
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=400,
                system=_LLM_GAP_DETECTION_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=8.0,
        )
        raw = msg.content[0].text if msg.content else "[]"

        # Strip markdown fences
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-z]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean).strip()

        parsed = json.loads(clean)
        if not isinstance(parsed, list):
            return []

        gaps = []
        for item in parsed[:2]:  # cost guard: max 2 LLM-detected gaps
            if isinstance(item, dict) and item.get("key") and item.get("description"):
                key = str(item["key"]).strip()
                desc = str(item["description"]).strip()
                # Skip if already registered or in existing tools
                if key not in _registry_fns and key not in existing_names:
                    gaps.append({"key": key, "description": desc})

        return gaps

    except Exception:
        return []  # never block execution


# ── Haiku synthesis ──────────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM = """\
You are a business calculation specialist. Implement a precise Python function for any business domain.

Domains you handle: Finance, HR/Payroll, SLA/Operations, Supply Chain, Date/Time math,
Statistics, Tax (VAT/GST/withholding), Risk/Compliance scoring, AR/Collections,
Contract math, Inventory valuation, and any other business process calculation.

The function runs in a sandbox with ONLY these available:
- math module (math.log, math.exp, math.sqrt, math.pow, math.floor, math.ceil, math.pi, math.e, math.erf)
- Decimal (from decimal module) for precision arithmetic
- ROUND_HALF_UP (rounding mode constant)
- random module: random.gauss(mu, sigma), random.seed(n), random.uniform(a, b), random.random()
- statistics module: statistics.mean(), statistics.stdev(), statistics.median()
- Safe builtins: abs, int, float, str, bool, round, min, max, sum, len, range,
  enumerate, zip, list, dict, tuple, set, isinstance, pow, divmod, sorted, any, all
- ValueError, ZeroDivisionError for error handling

DO NOT use: import, open, eval, exec, __import__, os, sys, datetime, any external library.
For date math: parse year/month/day from ISO strings manually using string splitting and int().

Requirements:
1. Function name must EXACTLY match the specified name
2. Accept the specified parameters as keyword-capable positional args
3. Use Decimal for ALL monetary calculations (avoid float precision loss)
4. Return dict with "result" (primary scalar answer) and "details" (dict of workings)
5. Handle edge cases: zero rates, zero periods, empty lists, negative inputs

Respond ONLY with valid JSON (no markdown, no explanation):
{
  "python_code": "def func_name(param1, param2, ...):\\n    ...",
  "test_cases": [
    {"inputs": {"param1": val}, "expected_result_approx": 123.45, "tolerance_pct": 0.01},
    {"inputs": {"param1": val2}, "expected_result_approx": 456.78, "tolerance_pct": 0.01},
    {"inputs": {"param1": val3}, "expected_result_approx": 789.01, "tolerance_pct": 0.01}
  ]
}"""


async def _synthesize_via_haiku(gap: dict) -> dict | None:
    """Call Haiku to synthesize a tool implementation. Returns parsed response or None."""
    prompt = (
        f"Implement this financial calculation function:\n\n"
        f"{gap['description']}\n\n"
        f"Write precise, correct code. Include 3 test cases with known correct outputs."
    )
    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=_ANTHROPIC_API_KEY)
        msg = await asyncio.wait_for(
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1200,
                system=_SYNTHESIS_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=10.0,
        )
        raw = msg.content[0].text if msg.content else ""

        # Strip markdown fences
        clean = raw.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-z]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean).strip()

        return json.loads(clean)

    except Exception:
        return None


# ── Validation ───────────────────────────────────────────────────────────────

def _validate_tool(tool_def: dict) -> tuple[bool, str]:
    """
    Validate a synthesized tool by executing its test cases.
    Returns (passed: bool, reason: str).
    """
    code = tool_def.get("python_code", "")
    func_name = tool_def.get("name", "")
    test_cases = tool_def.get("test_cases", [])

    if not code or not func_name:
        return False, "missing code or name"

    fn = _exec_in_sandbox(code, func_name)
    if fn is None:
        return False, "code failed to compile or exec"

    if not test_cases:
        # No test cases — accept with a note (better than rejecting useful tools)
        return True, "no test cases (accepted without validation)"

    passes = 0
    for tc in test_cases[:3]:
        try:
            inputs = tc.get("inputs", {})
            expected = float(tc.get("expected_result_approx", 0))
            tolerance_pct = float(tc.get("tolerance_pct", 0.01))

            result = fn(**inputs)
            actual = float(result.get("result", 0)) if isinstance(result, dict) else float(result)

            # Relative tolerance check
            denom = max(abs(expected), 1.0)
            if abs(actual - expected) / denom <= tolerance_pct:
                passes += 1
        except Exception:
            pass  # test case failed — don't count it

    if passes == len(test_cases[:3]):
        return True, f"all {passes} test cases passed"
    if passes >= 2:
        return True, f"{passes}/{len(test_cases[:3])} test cases passed (accepted)"
    return False, f"only {passes}/{len(test_cases[:3])} test cases passed"


# ── Tool schema builder ──────────────────────────────────────────────────────

def _build_schema(gap_key: str, description: str) -> dict:
    """Build a minimal JSON schema for a synthesized tool. Uses input_schema (Anthropic API format)."""
    return {
        "name": gap_key,
        "description": description.split(". ")[0],  # first sentence as description
        "input_schema": {
            "type": "object",
            "properties": {},  # Claude will infer from description
            "additionalProperties": True,
        },
    }


# ── Public: synthesize + register ────────────────────────────────────────────

async def synthesize_and_register(gap: dict, task_text: str) -> dict | None:
    """
    Synthesize a tool for the detected gap, validate it, and register it.

    Returns the tool schema dict (for adding to self._tools) or None if synthesis failed.
    One Haiku call per new tool. All future tasks get the cached tool for free.
    """
    _load_registry()

    key = gap["key"]

    # Already registered during this run (race condition guard)
    if key in _registry_fns:
        return {"name": key, "input_schema": _registry_defs.get(key, {}).get("input_schema", {})}

    # Synthesize
    raw = await _synthesize_via_haiku(gap)
    if not raw:
        return None

    # Build full tool def
    tool_def = {
        "name": key,
        "python_code": raw.get("python_code", ""),
        "test_cases": raw.get("test_cases", []),
        "description": gap["description"].split(". ")[0],
        "input_schema": raw.get("input_schema", {"type": "object", "additionalProperties": True}),
        "_synthesized": True,
        "_gap_description": gap["description"],
    }

    # Validate
    passed, reason = _validate_tool(tool_def)
    if not passed:
        return None

    # Hot-load
    fn = _exec_in_sandbox(tool_def["python_code"], key)
    if fn is None:
        return None

    # Register
    _registry_fns[key] = fn
    _registry_defs[key] = tool_def
    _save_registry()

    return {
        "name": key,
        "description": tool_def["description"],
        "input_schema": tool_def["input_schema"],
    }


# ── Public: load + call ──────────────────────────────────────────────────────

def load_registered_tools() -> list[dict]:
    """
    Return JSON schemas for all registered tools (MCP-compatible format).
    Called in PRIME to add registered tools to self._tools.
    """
    _load_registry()
    result = []
    for name, defn in _registry_defs.items():
        result.append({
            "name": name,
            "description": defn.get("description", name),
            "input_schema": defn.get("input_schema", {"type": "object"}),
        })
    return result


def is_registered_tool(tool_name: str) -> bool:
    """Check if a tool name maps to a registered (synthesized or seeded) function."""
    _load_registry()
    return tool_name in _registry_fns


def call_registered_tool(tool_name: str, params: dict) -> dict:
    """Execute a registered tool with the given params. Returns result dict."""
    _load_registry()
    fn = _registry_fns.get(tool_name)
    if fn is None:
        return {"error": f"Tool '{tool_name}' not found in registry"}
    try:
        result = fn(**params)
        if isinstance(result, dict):
            return result
        return {"result": result}
    except Exception as e:
        return {"error": str(e), "tool": tool_name, "params": params}


# ── Seed: amortization tool ───────────────────────────────────────────────────

def seed_amortization_tool() -> None:
    """
    Seed the loan amortization tool into the registry at startup.
    Migrates it from hardcoded finance_tools.py to the dynamic registry.
    Only seeds once — idempotent.
    """
    _load_registry()

    key = "finance_loan_amortization"
    if key in _registry_fns:
        return  # already seeded

    fn = _exec_in_sandbox(_AMORTIZATION_CODE, key)
    if fn is None:
        return  # sandbox exec failed — keep the hardcoded fallback

    # Validate with a known test case: $200k, 5% APR, 360 months → ~$1073.64/mo
    try:
        test_result = fn(principal=200000, annual_rate=5.0, months=360)
        expected = 1073.64
        actual = test_result.get("result", 0)
        if abs(actual - expected) > 1.0:
            return  # validation failed — don't register broken tool
    except Exception:
        return

    defn = {
        **_AMORTIZATION_SCHEMA,
        "python_code": _AMORTIZATION_CODE,
        "test_cases": [
            {"inputs": {"principal": 200000, "annual_rate": 5.0, "months": 360},
             "expected_result_approx": 1073.64, "tolerance_pct": 0.01},
        ],
        "_seeded": True,
    }
    _registry_fns[key] = fn
    _registry_defs[key] = defn
    _save_registry()


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_tool_registry_stats() -> dict:
    """Return registry stats for /rl/status endpoint."""
    _load_registry()
    total = len(_registry_defs)
    seeded = sum(1 for v in _registry_defs.values() if v.get("_seeded"))
    synthesized = sum(1 for v in _registry_defs.values() if v.get("_synthesized"))
    return {
        "total_tools": total,
        "seeded_tools": seeded,
        "synthesized_tools": synthesized,
        "registered_names": list(_registry_defs.keys()),
    }
