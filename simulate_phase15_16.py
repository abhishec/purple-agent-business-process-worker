"""
Phase 15 + 16 targeted tests.
Tests the three Phase 15 fixes and three Phase 16 upgrades without needing
a live LLM or MCP server — all logic-level checks.
"""
import asyncio, sys, os
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

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 15 — Seed fixes
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Phase 15: Seed correctness ──────────────────────────────────────────")
from src.sequence_enforcer import _SEED_SEQUENCES, _resolve_tool_hints, build_sequence_hint

# 1. expense_approval has 7 steps with log+budget steps required
ea = _SEED_SEQUENCES["expense_approval"]
check("expense_approval: 7 steps total", len(ea["steps"]) == 7,
      f"steps={len(ea['steps'])}")
step5 = ea["steps"][4]
step6 = ea["steps"][5]
check("expense_approval step5: log_audit_trail required",
      step5["required"] and "log_audit_trail" in step5["tool_hints"],
      f"step5={step5['description']}")
check("expense_approval step6: update_budget_allocation required",
      step6["required"] and "update_budget_allocation" in step6["tool_hints"],
      f"step6={step6['description']}")

# 2. pto_approval + hr_leave + leave_approval + leave_management seeds exist
for ptype in ["pto_approval", "hr_leave", "leave_approval", "leave_management"]:
    seed = _SEED_SEQUENCES.get(ptype)
    has_approve_step = seed and any(
        any("approve_" in h for h in s.get("tool_hints", []))
        for s in seed.get("steps", [])
    )
    has_update_balance = seed and any(
        any("update_" in h or "balance" in h for h in s.get("tool_hints", []))
        for s in seed.get("steps", [])
    )
    check(f"{ptype}: seed exists with approve + update_balance steps",
          bool(seed) and has_approve_step and has_update_balance,
          f"found={bool(seed)} approve={has_approve_step} balance={has_update_balance}")

# 3. invoice_reconciliation has 3-way match (goods receipt + match step)
ir = _SEED_SEQUENCES["invoice_reconciliation"]
check("invoice_reconciliation: 5 steps", len(ir["steps"]) == 5,
      f"steps={len(ir['steps'])}")
check("invoice_reconciliation step1: get_goods_receipt in hints",
      "get_goods_receipt" in ir["steps"][0]["tool_hints"],
      f"hints={ir['steps'][0]['tool_hints']}")
check("invoice_reconciliation step2: 3-way match description",
      "3-way match" in ir["steps"][1]["description"],
      f"desc={ir['steps'][1]['description']}")

# 4. _resolve_tool_hints dynamic resolution
available = {"log_decision", "log_event", "update_balance", "approve_expense"}
resolved = _resolve_tool_hints(["log_", "update_", "approve_expense"], available)
check("_resolve_tool_hints: 'log_' expands to actual log tools",
      "log_decision" in resolved or "log_event" in resolved,
      f"resolved={resolved}")
check("_resolve_tool_hints: exact name 'approve_expense' preserved",
      "approve_expense" in resolved,
      f"resolved={resolved}")

# 5. build_sequence_hint for pto_approval returns directive with action language
async def test_seq_hint():
    task = "Approve Jane's PTO request for next week"
    hint = await build_sequence_hint(task, "pto_approval", None, [
        "get_employee", "get_pto_balance", "check_policy", "confirm_with_user",
        "approve_pto_request", "update_pto_balance", "notify_employee"
    ])
    check("pto_approval: build_sequence_hint returns directive",
          hint is not None and "directive" in hint,
          f"hint_keys={list(hint.keys()) if hint else None}")
    if hint:
        check("pto_approval: directive mentions approve_pto_request",
              "approve_pto_request" in hint["directive"],
              f"directive_snippet={hint['directive'][:200]}")
        check("pto_approval: approval_required=True",
              hint.get("approval_required") is True,
              f"approval_required={hint.get('approval_required')}")

asyncio.run(test_seq_hint())

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 16 — Structural gate + dynamic L3b + dead code
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Phase 16: Structural approval gate ──────────────────────────────────")
import inspect
from src.worker_brain import MiniAIWorker
from src.hitl_guard import classify_tool

# 6. Universal Structural Approval Gate code is present
src = inspect.getsource(MiniAIWorker._execute)
check("_execute: UNIVERSAL STRUCTURAL APPROVAL GATE block present",
      "UNIVERSAL STRUCTURAL APPROVAL GATE" in src,
      "marker found" if "UNIVERSAL STRUCTURAL APPROVAL GATE" in src else "MISSING")
check("_execute: _approval_granted flag present",
      "_approval_granted = False" in src,
      "found" if "_approval_granted = False" in src else "MISSING")
check("_execute: POLICY_GATE error in _direct_call",
      "POLICY_GATE" in src,
      "found" if "POLICY_GATE" in src else "MISSING")

# 7. Gate logic: classify_tool correctly identifies mutation tools
check("hitl_guard: 'approve_expense' is mutate",
      classify_tool("approve_expense") == "mutate", f"got={classify_tool('approve_expense')}")
check("hitl_guard: 'log_audit_trail' is mutate",
      classify_tool("log_audit_trail") == "mutate", f"got={classify_tool('log_audit_trail')}")
check("hitl_guard: 'update_budget_allocation' is mutate",
      classify_tool("update_budget_allocation") == "mutate",
      f"got={classify_tool('update_budget_allocation')}")
check("hitl_guard: 'get_employee' is read",
      classify_tool("get_employee") == "read", f"got={classify_tool('get_employee')}")
check("hitl_guard: 'confirm_with_user' — gate tool check",
      classify_tool("confirm_with_user") in ("mutate", "read", "compute"),  # just exists
      f"got={classify_tool('confirm_with_user')}")

# 8. Gate is fail-open: if _has_confirm_tool is False, gate is disabled
check("_execute: gate conditioned on _has_confirm_tool",
      "_has_confirm_tool" in src,
      "found" if "_has_confirm_tool" in src else "MISSING — gate may permanently block")

# 9. Dead code removed: _has_analysis_lang gone
check("_execute: _has_analysis_lang dead code removed",
      "_has_analysis_lang" not in src,
      "cleaned" if "_has_analysis_lang" not in src else "STILL PRESENT — dead code")
check("_execute: _L2_ANALYSIS_MARKERS dead code removed",
      "_L2_ANALYSIS_MARKERS" not in src,
      "cleaned" if "_L2_ANALYSIS_MARKERS" not in src else "STILL PRESENT — dead code")

# 10. Dynamic L3b uses _resolve_tool_hints
check("_execute: L3b uses _resolve_tool_hints for dynamic resolution",
      "_resolve_tool_hints" in src and "L3b" in src,
      "found" if "_resolve_tool_hints" in src else "MISSING — L3b still uses exact match")
# Verify per-step coverage logic
check("_execute: L3b checks step_covered per step",
      "step_covered" in src,
      "found" if "step_covered" in src else "MISSING")

# 11. L3b simulation: prefix hint resolves correctly and catches miss
available_write = {"log_decision", "update_budget_allocation", "notify_manager", "approve_expense"}
seq_steps = [
    {"step": 5, "description": "Log audit", "tool_hints": ["log_audit_trail", "log_"],
     "required": True, "gate": None},
    {"step": 6, "description": "Update budget", "tool_hints": ["update_budget_allocation"],
     "required": True, "gate": None},
]
called = {"approve_expense"}  # only approve was called — log+budget missed

missed = []
for step in seq_steps:
    if not step.get("required") or step.get("gate") == "approval":
        continue
    resolved = _resolve_tool_hints(step.get("tool_hints", []), available_write)
    actual_tools = [t for t in resolved if not t.endswith("_")]
    step_covered = any(t in called for t in actual_tools)
    if not step_covered:
        miss = next((t for t in actual_tools if t not in called), None)
        if miss:
            missed.append(miss)

check("L3b simulation: 'log_' prefix resolves and detects miss",
      "log_decision" in missed or any("log" in m for m in missed),
      f"missed={missed}")
check("L3b simulation: 'update_budget_allocation' exact match detected",
      "update_budget_allocation" in missed,
      f"missed={missed}")
check("L3b simulation: 'approve_expense' NOT in missed (was called)",
      "approve_expense" not in missed,
      f"missed={missed}")

# 12. fsm_runner: magic number replaced with named constant
from src.fsm_runner import detect_task_complexity
fsm_src = open("src/fsm_runner.py").read()
check("fsm_runner: _READONLY_MAX_CHARS replaces magic number 120",
      "_READONLY_MAX_CHARS" in fsm_src,
      "found" if "_READONLY_MAX_CHARS" in fsm_src else "MISSING")
# Behavior preserved
check("fsm_runner: short pure-query returns 'readonly'",
      detect_task_complexity("what is the status?") == "readonly",
      f"got={detect_task_complexity('what is the status?')}")
check("fsm_runner: action task returns 'full'",
      detect_task_complexity("approve the PTO request for Jane Smith effective next Monday") == "full",
      f"got={detect_task_complexity('approve the PTO request')}")
check("fsm_runner: long task returns 'full' even with readonly signal",
      detect_task_complexity("Please generate a comprehensive analysis " * 5 + " and approve it") == "full",
      "full")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"FINAL: {passed}/{total} checks passed")
if passed == total:
    print("ALL PHASE 15+16 CHECKS PASS ✅")
else:
    print("FAILURES:")
    for name, ok, detail in results:
        if not ok:
            print(f"  ❌ {name}: {detail}")
