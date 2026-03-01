"""
strategy_bandit.py  — UCB1 Strategy Bandit
UCB1 multi-armed bandit for FSM strategy selection.

Inspired by BrainOS packages/memory-stack/src/causality/causal-method-bandit.ts.

Problem: For each task type we have 3 strategies:
  "fsm"        — 8-state FSM (default, structured, best for chained-tool tasks)
  "five_phase" — Five-Phase Executor (for analysis/synthesis tasks where subtasks are independent)
  "moa"        — Mixture of Agents (for pure-reasoning / numeric tasks, zero tool calls)

Improvement: Pre-seeded with domain knowledge so bandit starts in exploitation mode
rather than perpetual exploration (3 strategies × 15 types = 45 cold-start pulls needed
before any exploitation — at 5-10 tasks/type/day, that's days of pure exploration).

Pre-seeding logic:
- Process types that need chained mutations (offboarding, payroll, procurement):
  fsm = 0.80 (best for sequential tool calls), five_phase = 0.45, moa = 0.40
- Process types with heavy analysis + reporting (compliance_audit, month_end_close):
  five_phase = 0.70 (parallel subtasks independent), fsm = 0.65, moa = 0.50
- Financial calculation types (expense_approval, invoice_reconciliation, sla_breach):
  fsm = 0.75, moa = 0.65 (numeric MoA helps), five_phase = 0.50
- General: fsm = 0.70, others = 0.50

Bandit still learns and updates — pre-seeding just provides a warm start.
Low n values (n=2 pre-seeds) mean actual task outcomes quickly update the Q values.

UCB1 score: Q(arm) + C * sqrt(ln(N) / n(arm))
  Q    = mean reward for this arm (quality 0–1)
  N    = total pulls across all arms for this process type
  n    = pulls for this arm
  C    = exploration constant (1.41 = sqrt(2))

Persisted to strategy_bandit.json (same dir as tool_registry.json).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

_BANDIT_FILE = Path(os.environ.get("RL_CACHE_DIR", "/app")) / "strategy_bandit.json"

# Available arms
STRATEGIES = ("fsm", "five_phase", "moa")

# UCB1 exploration constant — sqrt(2) is standard
_C = math.sqrt(2)

# In-memory state: {process_type: {strategy: {q, n}}}
_state: dict[str, dict[str, dict]] = {}
_loaded = False

# Pre-seeded quality estimates per process type — warm starts the bandit.
# n=2 per arm so actual task outcomes update these quickly (incremental mean).
# Strategy rationale:
#   fsm:        Best for sequential chained-tool tasks (mutations depend on prior reads)
#   five_phase: Best for parallel analysis tasks (subtasks are independent data fetches)
#   moa:        Best for pure reasoning / numeric verification (no or few tools)
_PRIORS: dict[str, dict[str, tuple[float, int]]] = {
    # (Q, n) per arm
    "hr_offboarding":         {"fsm": (0.80, 3), "five_phase": (0.45, 2), "moa": (0.40, 2)},
    "payroll":                {"fsm": (0.78, 3), "five_phase": (0.48, 2), "moa": (0.42, 2)},
    "procurement":            {"fsm": (0.78, 3), "five_phase": (0.50, 2), "moa": (0.40, 2)},
    "customer_onboarding":    {"fsm": (0.76, 3), "five_phase": (0.52, 2), "moa": (0.40, 2)},
    "subscription_migration": {"fsm": (0.75, 3), "five_phase": (0.52, 2), "moa": (0.42, 2)},
    "order_management":       {"fsm": (0.75, 3), "five_phase": (0.50, 2), "moa": (0.42, 2)},
    "compliance_audit":       {"fsm": (0.65, 2), "five_phase": (0.72, 3), "moa": (0.50, 2)},
    "month_end_close":        {"fsm": (0.68, 2), "five_phase": (0.70, 3), "moa": (0.52, 2)},
    "incident_response":      {"fsm": (0.74, 3), "five_phase": (0.60, 2), "moa": (0.44, 2)},
    "expense_approval":       {"fsm": (0.76, 3), "five_phase": (0.50, 2), "moa": (0.65, 2)},
    "invoice_reconciliation": {"fsm": (0.74, 3), "five_phase": (0.52, 2), "moa": (0.66, 2)},
    "sla_breach":             {"fsm": (0.73, 3), "five_phase": (0.52, 2), "moa": (0.64, 2)},
    "ar_collections":         {"fsm": (0.74, 3), "five_phase": (0.54, 2), "moa": (0.60, 2)},
    "dispute_resolution":     {"fsm": (0.77, 3), "five_phase": (0.52, 2), "moa": (0.46, 2)},
    "general":                {"fsm": (0.70, 2), "five_phase": (0.55, 2), "moa": (0.55, 2)},
}


# ── Persistence ───────────────────────────────────────────────────────────────

def _load() -> None:
    global _state, _loaded
    if _loaded:
        return
    try:
        if _BANDIT_FILE.exists():
            _state = json.loads(_BANDIT_FILE.read_text())
    except Exception:
        _state = {}
    _loaded = True


def _save() -> None:
    try:
        _BANDIT_FILE.write_text(json.dumps(_state, indent=2))
    except Exception:
        pass


def _arms(process_type: str) -> dict[str, dict]:
    """Get or initialise arms for a process type, with pre-seeded priors."""
    if process_type not in _state:
        priors = _PRIORS.get(process_type, _PRIORS["general"])
        _state[process_type] = {
            s: {"q": priors.get(s, (0.5, 1))[0], "n": priors.get(s, (0.5, 1))[1]}
            for s in STRATEGIES
        }
    return _state[process_type]


# ── UCB1 selection ────────────────────────────────────────────────────────────

# Process types that require sequential chained mutations — five_phase can't do this
# because its GATHER phase runs subtasks in parallel without passing prior results forward.
# Even with the new sequential GATHER, five_phase lacks a proper MUTATE phase.
_CHAINED_MUTATION_TYPES = frozenset({
    "hr_offboarding",        # revoke multiple systems sequentially
    "payroll",               # chained: validate → compute → approve → disburse
    "procurement",           # chained: verify → approve → create PO → notify vendor
    "customer_onboarding",   # chained: create account → provision → notify
    "subscription_migration",# chained: verify → confirm → update plan → notify
})


def _is_chained_mutation_task(task_text: str) -> bool:
    """Detect tasks where tool calls must be sequential (each step depends on prior result)."""
    text = task_text.lower()
    # Sequential dependency patterns: "then", "after that", "once X is done, do Y"
    _SEQUENTIAL_PATTERNS = [
        " then ", "and then", "after that", "once approved", "upon approval",
        "followed by", "before sending", "after updating", "following the",
        "next step", "subsequent", "sequentially", "in order",
    ]
    return any(p in text for p in _SEQUENTIAL_PATTERNS)


def select_strategy(process_type: str, task_text: str = "") -> str:
    """
    Return the UCB1-optimal strategy for this process type.
    Pre-seeded with domain priors for faster convergence.

    Guards:
    - five_phase blocked for chained-mutation types (sequential tool dependencies)
    - moa blocked when tools are expected (MoA is for pure reasoning only)

    task_text is used for heuristic guards.
    """
    _load()
    arms = _arms(process_type)

    # Build the eligible set (apply structural guards before UCB1 selection)
    eligible = list(STRATEGIES)

    # Guard 1: Never use five_phase for process types that need chained sequential mutations.
    # five_phase runs GATHER subtasks in parallel — it can't chain get_order → process_refund.
    if process_type in _CHAINED_MUTATION_TYPES or _is_chained_mutation_task(task_text):
        eligible = [s for s in eligible if s != "five_phase"]

    # Guard 2: Never use moa for process types that clearly require tool calls.
    # MoA works on pure reasoning; tool-dependent tasks need fsm or five_phase.
    _TOOL_HEAVY_TYPES = frozenset({
        "hr_offboarding", "payroll", "procurement", "order_management",
        "customer_onboarding", "subscription_migration", "ar_collections",
    })
    if process_type in _TOOL_HEAVY_TYPES:
        eligible = [s for s in eligible if s != "moa"]

    if not eligible:
        eligible = ["fsm"]  # ultimate fallback

    N_total = sum(d["n"] for d in arms.values())

    best_score = -1.0
    best_arm = "fsm"
    for strategy in eligible:
        data = arms[strategy]
        q = data["q"]
        n = data["n"]
        if n == 0:
            return strategy  # unvisited eligible arm → explore it first
        ucb1 = q + _C * math.sqrt(math.log(max(N_total, 1)) / n)
        if ucb1 > best_score:
            best_score = ucb1
            best_arm = strategy

    return best_arm


def record_outcome(process_type: str, strategy: str, quality: float) -> None:
    """
    Update the bandit with the observed quality reward (0–1).
    Uses incremental mean update: Q_new = Q_old + (reward - Q_old) / n
    """
    _load()
    arms = _arms(process_type)
    if strategy not in arms:
        arms[strategy] = {"q": 0.5, "n": 0}

    data = arms[strategy]
    data["n"] += 1
    # Incremental mean
    data["q"] = data["q"] + (quality - data["q"]) / data["n"]
    _save()


def get_stats() -> dict:
    """Return bandit stats for /health endpoint."""
    _load()
    total_pulls = sum(
        d["n"]
        for arms in _state.values()
        for d in arms.values()
    )
    process_types_learned = len(_state)
    best_arms = {
        pt: max(arms.items(), key=lambda x: x[1]["q"])[0]
        for pt, arms in _state.items()
        if any(d["n"] > 0 for d in arms.values())
    }
    return {
        "total_pulls": total_pulls,
        "process_types_learned": process_types_learned,
        "best_arms": best_arms,
    }
