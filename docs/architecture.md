# Architecture

## Cognitive Loop

Every request runs three phases:

```
PRIME → EXECUTE → REFLECT
```

**PRIME** assembles the system context before any LLM call. **EXECUTE** runs the FSM pipeline. **REFLECT** persists outcomes to all feedback channels.

---

## PRIME Phase

All steps run before Claude sees the task. Their outputs are joined into a single system prompt.

**Privacy guard** — `privacy_guard.check_privacy()` refuses immediately if the task contains PII, credentials, or SSNs. Zero API cost.

**RL primer** — `rl_loop.build_rl_primer()` loads `case_log.json`, finds the three most similar past tasks by keyword overlap, and injects learned patterns. Stale entries (>72 h), low-quality entries (score <0.35), and repeated-failure patterns (3+ failures, ≥50% keyword overlap) are pruned first.

**Session context** — `session_context.get_context_prompt()` loads Haiku-compressed history for multi-turn sessions.

**FSM classification** — `smart_classifier.classify_process_type()` identifies the process type (`invoice_reconciliation`, `hr_offboarding`, `sla_breach`, etc.) via a Haiku call. Checkpoint-restored sessions skip re-classification.

**Dynamic FSM synthesis** — For process types not in the 15 built-in templates, `dynamic_fsm.synthesize_if_needed()` calls Haiku once to produce a custom state sequence. The result is cached; subsequent tasks of the same type skip synthesis.

**Policy evaluation** — Parses the `policy_doc` JSON field and evaluates approval thresholds, spend limits, and escalation conditions. Zero LLM calls.

**Tool discovery** — `mcp_bridge.discover_tools()` fetches live tool schemas from the MCP endpoint. `load_registered_tools()` appends any tools synthesized in prior tasks.

**Dynamic tool gap detection** — Two-phase detection:

*Phase 1 (regex, zero API cost)* — 36 static patterns across 10 domains: finance (NPV, amortization, depreciation), HR/payroll (overtime, proration), SLA/operations (uptime, credits), supply chain (EOQ, safety stock), date/time (business days, AR aging), statistics (z-score, regression), tax (VAT, withholding), risk/compliance (weighted risk score), and AR/collections (DSO, bad debt).

*Phase 2 (LLM, conditional)* — Fires only if Phase 1 finds nothing and the task is ≥100 chars. Haiku identifies required calculations, synthesizes a Python implementation, validates it in a restricted sandbox (`math`, `Decimal`, `random`, `statistics`; `import`, `open`, `os`, `sys` blocked), and persists it to `tool_registry.json`.

**HITL gate** — If the FSM is at `APPROVAL_GATE`, mutation tools are listed in the system prompt as explicitly blocked.

**Knowledge base and entity memory** — `knowledge_extractor.get_relevant_knowledge()` retrieves domain facts from past tasks. `entity_extractor.get_entity_context()` injects cross-task entity history.

**Finance pre-computation** — Extracts numbers from the task text and computes variance percentages, SLA credits, and proration amounts at zero API cost.

---

## EXECUTE Phase

### Strategy Selection

`strategy_bandit.select_strategy()` uses the UCB1 algorithm to select among three execution strategies per process type:

- `fsm` — 8-state FSM (default for unvisited types)
- `five_phase` — PLAN → GATHER → SYNTHESIZE → ARTIFACT → INSIGHT
- `moa` — dual top-p Haiku calls with word-overlap consensus; Sonnet synthesizes if divergent

UCB1: `Q(arm) + √2 × √(ln(N) / n(arm))`

After sufficient tasks of the same type, the bandit converges to the best-performing strategy for that process class.

### Tool Call Stack

```
Claude calls a tool
        │
        ▼
MutationVerifier.call()       records write calls; reads back after each write (WAL flush)
        │
        ▼
wrap_with_recovery()          on error: synonym tool names, relaxed params
        │
        ▼
resilient_tool_call()         on column-not-found: fuzzy name match, retry
        │
        ▼
paginated_fetch()             cursor-loops bulk data tools to collect all records
        │
        ▼
direct call → MCP endpoint
```

### Post-Execution Passes

All passes are best-effort and non-blocking:

1. **Mutation verification log** — appends `## Mutation Verification Log` with per-write VERIFIED / FAILED / UNVERIFIABLE status for each write call
2. **COMPUTE math verification** — Haiku audits numeric answers for arithmetic errors; triggers a correction pass if errors found
3. **Numeric MoA** — for data-driven tasks: two parallel Haiku calls (verify + challenge) at different sampling temperatures; synthesizes the best result
4. **Approval brief** — if `APPROVAL_GATE` fired and the answer is thin (<200 chars), builds a formal approval document
5. **Output validation** — checks required fields for the process type; re-runs with a targeted improvement prompt if any are missing
6. **Self-reflection** — scores the answer on completeness and tool coverage; triggers an improvement pass if below threshold
7. **MoA for reasoning tasks** — for read-only tasks (tool_count == 0): dual top-p consensus check

---

## REFLECT Phase

```
answer finalized
        │
        ├─→ session history updated
        ├─→ FSM checkpoint saved (process_type, state_idx, state_history, requires_hitl)
        ├─→ session memory compressed async (Haiku; fires when turns > 20)
        ├─→ rl_loop.record_outcome()         → case_log.json (max 200 entries, FIFO)
        ├─→ strategy_bandit.record_outcome() → UCB1 arm updated: Q += (quality - Q) / n
        ├─→ context_rl.check_context_accuracy() → adjusts confidence for pre-computed facts
        ├─→ knowledge_extractor.extract_and_store() → domain facts if quality ≥ 0.5
        └─→ entity_extractor.record_task_entities() → vendors, amounts, people
```

---

## FSM Process Templates

15 built-in templates cover common enterprise process types:

| Process type | Key states | Notes |
|---|---|---|
| `expense_approval` | DECOMPOSE→ASSESS→COMPUTE→POLICY_CHECK→APPROVAL_GATE→MUTATE→COMPLETE | Budget threshold gate |
| `procurement` | + SCHEDULE_NOTIFY | Vendor notification after approval |
| `hr_offboarding` | DECOMPOSE→ASSESS→POLICY_CHECK→MUTATE→SCHEDULE_NOTIFY→COMPLETE | Sequenced access revocation |
| `incident_response` | DECOMPOSE→ASSESS→COMPUTE→APPROVAL_GATE→MUTATE→SCHEDULE_NOTIFY→COMPLETE | Two-person approval |
| `invoice_reconciliation` | DECOMPOSE→ASSESS→COMPUTE→POLICY_CHECK→MUTATE→COMPLETE | Variance checking |
| `compliance_audit` | Full 8 states | Gap tracking and escalation routing |
| `dispute_resolution` | + APPROVAL_GATE | Elevated review escalation |
| `order_management` | + COMPUTE | Price delta before modification |
| `sla_breach` | →SCHEDULE_NOTIFY→ESCALATE | Quiet-hours scheduling |
| `month_end_close` | Full 8 states | Bulk data and revenue recognition |
| `ar_collections` | DECOMPOSE→ASSESS→COMPUTE→POLICY_CHECK→MUTATE→SCHEDULE_NOTIFY→COMPLETE | Multi-customer routing |
| `subscription_migration` | Full 8 states + `reopen_approval_gate()` | Sequential multi-gate confirmation |
| `payroll` | Full 8 states | Multi-country tax and disbursement |

Unknown process types are handled by `dynamic_fsm.py`, which synthesizes a custom state sequence via Haiku and caches it permanently.

---

## Schema Drift Resilience

`schema_adapter.py` handles column-name drift across tool schema versions:

1. Extract the bad column name from the error message (regex patterns)
2. Check `KNOWN_COLUMN_ALIASES` (10 canonical columns → known variants)
3. Fuzzy match via `difflib.SequenceMatcher` (cutoff 0.6)
4. Levenshtein ratio fallback (threshold 0.7)
5. Retry once with corrected params
6. Cache the correction in session for subsequent calls

---

## Reinforcement Learning

**Quality score formula:**

```
quality = 0.35 × answer_score   (answer length relative to task complexity)
        + 0.35 × tool_score     (fewer tools = higher efficiency)
        + 0.30 × policy_score   (1.0 passed · 0.5 not applicable · 0.0 failed)
```

**Feedback channels (five, simultaneous):**

1. Case log — keyword-matched patterns injected into future similar tasks
2. UCB1 bandit — strategy win rates per process type
3. Knowledge base — extracted domain facts (vendor terms, thresholds, constraints)
4. Entity memory — vendor, person, and amount history across tasks
5. Context-RL — finance pre-computation accuracy tracking

---

## Financial Arithmetic

`financial_calculator.py` uses `decimal.Decimal` throughout. All monetary values are stored and computed in integer cents internally to eliminate floating-point rounding errors.

Available functions: prorated amounts, early termination fees, SLA credits, insurance sublimit calculation, gift card capacity, loan amortization, straight-line depreciation, revenue recognition, and order price delta.

---

## Policy Engine

`policy_checker.py` evaluates policy rules without any LLM calls.

**Condition syntax:**

```
amount > 5000                   threshold comparison
status === "active"             string equality
has_unvested_equity             boolean field check
amount > 1000 && !is_manager   AND
escalate || requires_board      OR
```

**Rule actions:** `require_approval`, `escalate`, `block`

**Escalation chain:** `manager` → `hr` → `finance` → `committee` → `legal` → `cfo` → `ciso`
