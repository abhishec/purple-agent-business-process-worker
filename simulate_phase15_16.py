"""
Phase 15 + 16 + 17 targeted tests.
Tests the three Phase 15 fixes, three Phase 16 upgrades, and Phase 17
scalable approval gate + SaaS migration fixes — all without a live LLM or MCP server.
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
# PHASE 17 — Scalable approval gate + SaaS migration (task_09 fix)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Phase 17: Scalable approval gate + SaaS migration ───────────────────")
from src.hitl_guard import is_approval_tool, find_approval_tool
from src.mutation_verifier import _is_write_tool

# 1. is_approval_tool patterns
check("is_approval_tool: confirm_with_user → True",
      is_approval_tool("confirm_with_user"), "got=True")
check("is_approval_tool: require_customer_signoff → True",
      is_approval_tool("require_customer_signoff"), "got=True")
check("is_approval_tool: request_manager_approval → True",
      is_approval_tool("request_manager_approval"), "got=True")
check("is_approval_tool: customer_signoff → True",
      is_approval_tool("customer_signoff"), "got=True")
check("is_approval_tool: proceed_migration → False (it's a mutation, not a gate)",
      not is_approval_tool("proceed_migration"), "got=False")
check("is_approval_tool: get_employee → False",
      not is_approval_tool("get_employee"), "got=False")
check("is_approval_tool: approve_leave → False (executes approval, not a gate)",
      not is_approval_tool("approve_leave"), "got=False")

# 2. find_approval_tool: dynamic lookup
tools_with_require = [{"name": "get_subscription"}, {"name": "require_customer_signoff"},
                      {"name": "proceed_migration"}]
tools_with_confirm = [{"name": "get_employee"}, {"name": "confirm_with_user"},
                      {"name": "update_leave"}]
tools_no_approval  = [{"name": "get_employee"}, {"name": "proceed_migration"}]

check("find_approval_tool: finds require_customer_signoff",
      find_approval_tool(tools_with_require) == "require_customer_signoff",
      f"got={find_approval_tool(tools_with_require)}")
check("find_approval_tool: prefers confirm_with_user (backward compat)",
      find_approval_tool(tools_with_confirm) == "confirm_with_user",
      f"got={find_approval_tool(tools_with_confirm)}")
check("find_approval_tool: returns None when no approval tool present",
      find_approval_tool(tools_no_approval) is None,
      f"got={find_approval_tool(tools_no_approval)}")

# 3. _is_write_tool: read-suffix detection + approval gate exclusion
check("_is_write_tool: run_integration_compatibility_test → False (ends with _test)",
      not _is_write_tool("run_integration_compatibility_test"),
      f"got={_is_write_tool('run_integration_compatibility_test')}")
check("_is_write_tool: generate_conflict_report → False (ends with _report)",
      not _is_write_tool("generate_conflict_report"),
      f"got={_is_write_tool('generate_conflict_report')}")
check("_is_write_tool: require_customer_signoff → False (approval gate, not a DB write)",
      not _is_write_tool("require_customer_signoff"),
      f"got={_is_write_tool('require_customer_signoff')}")
check("_is_write_tool: proceed_migration → True (still a real mutation)",
      _is_write_tool("proceed_migration"),
      f"got={_is_write_tool('proceed_migration')}")
check("_is_write_tool: initiate_data_export → True (mutation)",
      _is_write_tool("initiate_data_export"),
      f"got={_is_write_tool('initiate_data_export')}")
check("_is_write_tool: calculate_prorated_billing → False (calculate_ read prefix)",
      not _is_write_tool("calculate_prorated_billing"),
      f"got={_is_write_tool('calculate_prorated_billing')}")

# 4. subscription_migration seed: updated for task_09
sm = _SEED_SEQUENCES["subscription_migration"]
check("subscription_migration: 6 steps (updated from 5)",
      len(sm["steps"]) == 6,
      f"steps={len(sm['steps'])}")
step4_hints = sm["steps"][3]["tool_hints"]
check("subscription_migration step4: require_customer_signoff in hints",
      "require_customer_signoff" in step4_hints,
      f"hints={step4_hints}")
check("subscription_migration step4: gate=approval",
      sm["steps"][3].get("gate") == "approval",
      f"gate={sm['steps'][3].get('gate')}")
step2_hints = sm["steps"][1]["tool_hints"]
check("subscription_migration step2: run_integration_compatibility_test in hints",
      "run_integration_compatibility_test" in step2_hints,
      f"hints={step2_hints}")
step5_hints = sm["steps"][4]["tool_hints"]
check("subscription_migration step5: proceed_migration in hints",
      "proceed_migration" in step5_hints,
      f"hints={step5_hints}")

# 5. smart_classifier maps saas migration keywords
from src.smart_classifier import _keyword_fallback
check("smart_classifier: 'saas migration' → subscription_migration",
      _keyword_fallback("Process SaaS migration for enterprise customer") == "subscription_migration",
      f"got={_keyword_fallback('Process SaaS migration for enterprise customer')}")

# 6. worker_brain: _split_tools_for_phases puts require_customer_signoff in write_tools
from src.worker_brain import _split_tools_for_phases
task09_tools = [
    {"name": "get_subscription"}, {"name": "get_current_features"},
    {"name": "get_new_plan_features"}, {"name": "generate_conflict_report"},
    {"name": "initiate_data_export"}, {"name": "require_customer_signoff"},
    {"name": "proceed_migration"}, {"name": "pause_migration"},
    {"name": "calculate_export_files"}, {"name": "run_integration_compatibility_test"},
    {"name": "calculate_prorated_billing"}, {"name": "notify_enterprise_team"},
]
read_t, write_t = _split_tools_for_phases(task09_tools)
read_names  = {t["name"] for t in read_t}
write_names = {t["name"] for t in write_t}
check("_split_tools_for_phases: require_customer_signoff → write_tools (approval gate)",
      "require_customer_signoff" in write_names,
      f"write={sorted(write_names)}")
check("_split_tools_for_phases: run_integration_compatibility_test → read_tools (ends _test)",
      "run_integration_compatibility_test" in read_names,
      f"read has it={('run_integration_compatibility_test' in read_names)}")
check("_split_tools_for_phases: generate_conflict_report → read_tools (ends _report)",
      "generate_conflict_report" in read_names,
      f"read has it={('generate_conflict_report' in read_names)}")
check("_split_tools_for_phases: proceed_migration → write_tools (real mutation)",
      "proceed_migration" in write_names,
      f"write has it={('proceed_migration' in write_names)}")

# 7. worker_brain: dynamic _approval_tool_name logic (source-level)
wb_src = open("src/worker_brain.py").read()
check("worker_brain: _find_approval_tool used for gate (not hardcoded)",
      "_find_approval_tool" in wb_src,
      "found" if "_find_approval_tool" in wb_src else "MISSING — still hardcoded")
check("worker_brain: tool_name == _approval_tool_name (dynamic check)",
      "tool_name == _approval_tool_name" in wb_src,
      "found" if "tool_name == _approval_tool_name" in wb_src else "MISSING")
check("worker_brain: _has_confirm_tool uses _approval_tool_name",
      "_approval_tool_name is not None" in wb_src,
      "found" if "_approval_tool_name is not None" in wb_src else "MISSING")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"FINAL: {passed}/{total} checks passed")
if passed == total:
    print("ALL PHASE 15+16+17 CHECKS PASS ✅")
else:
    print("FAILURES:")
    for name, ok, detail in results:
        if not ok:
            print(f"  ❌ {name}: {detail}")
