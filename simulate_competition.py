"""
Competition simulation trace — 14 critical path checks.
Runs purely locally (no API key needed — tests logic, not LLM outputs).
"""
import asyncio
import json
import sys
import os

os.chdir("/tmp/purple-agent")
sys.path.insert(0, "/tmp/purple-agent")

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition, detail))
    print(f"{status}: {name}" + (f" — {detail}" if detail else ""))
    return condition

# ─── 1. Bracket format survives format_final_answer ─────────────────────────
from src.structured_output import format_final_answer, extract_ranked_items

def test_bracket_survives_format():
    ans = '["INV-001", "INV-002", "INV-003"]'
    out = format_final_answer(ans, policy_result={"passed": False, "summary": "Policy failed"})
    check("Bracket survives policy prefix in format_final_answer",
          out.strip() == ans.strip(),
          f"got: {repr(out[:80])}")

test_bracket_survives_format()

# ─── 2. Prose list → bracket extraction ─────────────────────────────────────
def test_prose_to_bracket():
    prose = "1. Alice\n2. Bob\n3. Charlie"
    items = extract_ranked_items(prose)
    bracket = json.dumps(items)
    check("Prose numbered list → bracket extraction",
          items == ["Alice", "Bob", "Charlie"],
          f"items={items}")

test_prose_to_bracket()

# ─── 3. Compute verifier bracket fast-path ───────────────────────────────────
from src.compute_verifier import verify_compute_output

async def test_compute_verifier_bracket():
    result = await verify_compute_output('[\"X\", \"Y\"]', '[\"X\", \"Y\"]', 'general')
    check("Compute verifier bracket fast-path",
          not result.has_errors and result.confidence >= 0.90,
          f"has_errors={result.has_errors} confidence={result.confidence}")

asyncio.run(test_compute_verifier_bracket())

# ─── 4. Self-reflection bracket fast-path ────────────────────────────────────
from src.self_reflection import reflect_on_answer, should_improve

async def test_reflection_bracket():
    r = await reflect_on_answer('["X", "Y"]', '["X", "Y"]', 'general', 0)
    check("Self-reflection bracket fast-path returns score=1.0",
          r['score'] == 1.0,
          f"score={r['score']}")
    check("Self-reflection bracket: should_improve=False",
          not should_improve(r),
          f"should_improve={should_improve(r)}")

asyncio.run(test_reflection_bracket())

# ─── 5. Output validator bracket fast-path ──────────────────────────────────
from src.output_validator import validate_output

def test_output_validator_bracket():
    r = validate_output('["Alice", "Bob"]', "expense_approval")
    check("Output validator bracket fast-path",
          r["valid"] and r["score"] == 1.0,
          f"valid={r['valid']} score={r['score']}")

test_output_validator_bracket()

# ─── 6. Numeric MoA bracket fast-path ───────────────────────────────────────
from src.self_moa import numeric_moa_synthesize

async def test_numeric_moa_bracket():
    initial = '["Option A", "Option B"]'
    result = await numeric_moa_synthesize("rank these options", initial)
    check("Numeric MoA bracket fast-path preserves format",
          result.strip().startswith('['),
          f"result starts: {repr(result[:40])}")

asyncio.run(test_numeric_moa_bracket())

# ─── 7. Variance boundary precision ─────────────────────────────────────────
from src.financial_calculator import apply_variance_check

def test_variance_boundary():
    # Exactly 5% variance — boundary case
    # invoice=1050, PO=1000 → variance=50 → 5.0% exactly
    r = apply_variance_check(1050.0, 1000.0, 5.0)
    check("Variance boundary: 5.0% with threshold=5.0% does NOT exceed",
          not r["exceeds"],
          f"pct={r['pct']} exceeds={r['exceeds']}")
    # 5.001% — just over threshold
    r2 = apply_variance_check(1050.01, 1000.0, 5.0)
    check("Variance boundary: 5.001% DOES exceed threshold=5.0%",
          r2["exceeds"],
          f"pct={r2['pct']} exceeds={r2['exceeds']}")

test_variance_boundary()

# ─── 8. SLA 100% uptime target guard (sla_max_mins=0) ───────────────────────
from src.financial_calculator import compute_sla_credit

def test_sla_100pct_target():
    # 100% SLA = sla_max_mins=0 means ANY downtime = breach
    # 30 mins downtime, 5% per breach, 25% cap, $10000 invoice
    credit = compute_sla_credit(30, 0, 10000.0, 5.0, 25.0)
    check("SLA 100% target: any downtime triggers credit",
          credit > 0,
          f"credit=${credit}")
    # Should be 1 breach_count * 5% = 5% of $10000 = $500
    check("SLA 100% target: credit = 5% of $10000 = $500",
          abs(credit - 500.0) < 0.01,
          f"credit={credit}")

test_sla_100pct_target()

# ─── 9. SLA credit within threshold (no breach) ─────────────────────────────
def test_sla_no_breach():
    credit = compute_sla_credit(30, 60, 10000.0, 5.0, 25.0)
    check("SLA: downtime <= threshold yields 0 credit",
          credit == 0.0,
          f"credit={credit}")

test_sla_no_breach()

# ─── 10. Policy checker OR/AND precedence ───────────────────────────────────
from src.policy_checker import _evaluate_condition

def test_policy_precedence():
    # "A || B && C" should be A OR (B AND C), not (A OR B) AND C
    # AND binds tighter: split OR first (outermost), handle AND within each clause
    ctx = {"A": False, "B": True, "C": True}  # false || (true && true) = true
    result = _evaluate_condition("A || B && C", ctx)
    check("Policy checker: A||B&&C = A OR (B AND C) = False OR True = True",
          result == True,
          f"result={result}")

    ctx2 = {"A": False, "B": True, "C": False}  # false || (true && false) = false
    result2 = _evaluate_condition("A || B && C", ctx2)
    check("Policy checker: A||B&&C = A OR (B AND C) = False OR False = False",
          result2 == False,
          f"result2={result2}")

test_policy_precedence()

# ─── 11. Schema adapter fires for "column not found" ────────────────────────
from src.schema_adapter import resilient_tool_call

async def test_schema_fix():
    """Verify has_error fires for non-'error'-containing error messages."""
    calls = []

    async def mock_tool(tool_name, params):
        calls.append((tool_name, params))
        if tool_name == "get_records" and calls.count(("get_records", params)) == 1:
            return {"error": "column 'foo' not found in table"}
        if tool_name in ("describe_table", "get_schema", "list_columns", "schema_introspect"):
            return {"columns": ["id", "name", "status", "amount", "created_at"]}
        # Retry with corrected params — succeed
        return {"data": [{"id": 1}]}

    schema_cache = {}
    result = await resilient_tool_call(
        "get_records",
        {"foo": "bar"},
        mock_tool,
        schema_cache,
    )
    # After fix: error message "column 'foo' not found" triggers correction path
    # Schema correction must have attempted → at least 2 tool calls
    correction_attempted = len(calls) >= 2
    check("Schema: 'column not found' triggers schema correction",
          correction_attempted,
          f"calls made: {len(calls)} — {[c[0] for c in calls]}")

asyncio.run(test_schema_fix())

# ─── 12. Proration integer-cent precision ───────────────────────────────────
from src.financial_calculator import prorated_amount

def test_proration_precision():
    # $1200 / 12 months, 7 months used → 5 months remaining
    # 1200 * (12-7)/12 = 1200 * 5/12 = 500.00
    result = prorated_amount(1200.0, 7, 12)
    check("Proration: $1200 for 12 months, 7 used → $500.00 remaining",
          abs(result - 500.0) < 0.01,
          f"result={result}")

    # Edge case: odd division $100 / 3 months, 1 used → 2 remaining
    # 100 * 2/3 = 66.666... → rounds to $66.67
    result2 = prorated_amount(100.0, 1, 3)
    check("Proration: $100 / 3 months, 1 used → $66.67 (integer-cent)",
          abs(result2 - 66.67) < 0.01,
          f"result={result2}")

test_proration_precision()

# ─── 13. UCB1 bandit: first visit always returns 'fsm' ──────────────────────
from src.strategy_bandit import select_strategy, record_outcome, _state, _load

def test_ucb1_first_visit():
    _load()
    # Use a unique process type to avoid state pollution
    pt = "test_ucb1_unique_99"
    if pt in _state:
        del _state[pt]
    first = select_strategy(pt, "")
    check("UCB1 bandit: first visit returns 'fsm'",
          first == "fsm",
          f"first={first}")

test_ucb1_first_visit()

# ─── 14. UCB1 bandit: record_outcome for 'moa' updates q correctly ───────────
def test_ucb1_moa_arm():
    pt = "test_ucb1_moa_arm_99"
    if pt in _state:
        del _state[pt]
    # Seed all 3 arms with 1 pull each
    for s in ("fsm", "five_phase", "moa"):
        record_outcome(pt, s, 0.5)
    # Now record a high-quality moa result
    record_outcome(pt, "moa", 1.0)
    moa_data = _state[pt]["moa"]
    # General prior seeds moa at (q=0.55, n=2). After 1 seed record (q=0.5)
    # and 1 high-quality record (q=1.0): n=4, q=0.650.
    # Verify n grew from prior + 2 explicit records, and q > prior (0.55) confirming
    # the high-quality outcome raised the arm's estimate.
    check("UCB1 bandit: moa arm records outcome correctly",
          moa_data["n"] == 4 and moa_data["q"] > 0.55,
          f"n={moa_data['n']} q={moa_data['q']:.3f}")

test_ucb1_moa_arm()

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"FINAL: {passed}/{total} checks passed")
if passed == total:
    print("ALL CHECKS PASS — competition simulation complete ✅")
else:
    print("FAILURES:")
    for name, ok, detail in results:
        if not ok:
            print(f"  ❌ {name}: {detail}")
