# Purple Agent

A business-process AI agent built on the **Reflexive Agent Architecture** — a dual-process cognitive design that separates deterministic task logic from LLM reasoning to achieve reliable, auditable enterprise workflow execution.

---

## The Problem with Pure LLM Agents

LLM agents fail at reliable execution in two distinct ways.

**Type 1 — execution failure:** The agent understands what to do but gets the *how* wrong — wrong API formats, wrong argument types, wrong tool call sequences. Prompt engineering addresses this at the surface; the failures shift rather than disappear.

**Type 2 — decision failure:** On adversarial or deceptive inputs, the agent picks the wrong action entirely — booking when it should compensate, mutating when it should read, acting when it should wait for approval.

Both failure types have the same root: *reasoning and execution are entangled*. The LLM is simultaneously deciding what to do and doing it, with no separation between the two.

---

## Reflexive Agent Architecture

Purple Agent resolves this by separating the cognitive loop into three structurally distinct layers:

```
  ┌──────────────────────────────────────────────────────────────┐
  │                     REFLEX LAYER                             │
  │  Fires BEFORE the LLM call. Deterministic, zero-latency.    │
  │  Pattern detection → FSM classification → policy evaluation  │
  │  → tool gap detection → sequence injection → pre-compute     │
  │  For known patterns: injects actions directly, bypasses LLM  │
  └───────────────────────────┬──────────────────────────────────┘
                              │  if no reflex match
  ┌───────────────────────────▼──────────────────────────────────┐
  │                     LLM CORTEX                               │
  │  Claude (Sonnet/Haiku). Handles novel, open-ended reasoning. │
  │  Operates inside structural constraints set by reflex layer: │
  │  read-only vs. mutate phase, approval gate, tool whitelist   │
  └───────────────────────────┬──────────────────────────────────┘
                              │
  ┌───────────────────────────▼──────────────────────────────────┐
  │                     VERIFICATION LAYER                       │
  │  Post-execution. Deterministic auditing, zero-LLM.          │
  │  Completion contract, tool coverage, math verification,      │
  │  output validation, RL outcome recording                     │
  └──────────────────────────────────────────────────────────────┘
```

The key design principle: **the LLM handles only what genuinely requires reasoning**. Everything that can be expressed as a rule, pattern match, or state machine is handled deterministically — before, during constraints, and after.

### Reflexive Injection

The most aggressive form of the reflex layer is **reflexive injection**: for known task archetypes, the agent inserts fully-formed tool-call messages directly into the conversation stream, bypassing the LLM for that turn entirely. The LLM never runs; it receives the tool result on the next turn and continues reasoning from the new state.

This enables multi-phase deterministic plans to execute inside a stochastic conversation:

```
detect(trigger condition)                  → inject Phase 1 tool call
detect(Phase 1 result in session history)  → inject Phase 2 tool call
detect(Phase 2 result in session history)  → inject Phase 3 tool call (or final action)
```

Each phase fires on a different conversation turn, gated on results from the previous. The FSM spans multiple turns without the LLM being in the loop for any of them.

### Why This Matters

| Approach | Known patterns | Novel patterns | Inspectable | Incrementally improvable |
|---|---|---|---|---|
| Pure LLM agent | Unreliable | ✓ | No | No (retraining) |
| Pure rule engine | ✓ | No | Yes | Yes |
| Reflexive Agent Architecture | ✓ (deterministic) | ✓ (LLM fallback) | Yes | Yes (add reflexes) |

---

## Cognitive Loop: PRIME → EXECUTE → REFLECT

```
POST /  (A2A JSON-RPC 2.0)
        │
        ▼
    PRIME  ← Reflex Layer
    ├── Privacy guard              (zero API cost)
    ├── RL primer                  (past-task patterns injected into context)
    ├── Session context            (Haiku-compressed history)
    ├── FSM classification         (Haiku process-type detection)
    ├── Dynamic FSM synthesis      (novel process types via Haiku)
    ├── Policy evaluation          (deterministic, zero LLM)
    ├── Tool discovery             (MCP + local registry)
    ├── Task mode detection        (read-only vs. full — strips mutation tools)
    ├── Tool gap detection         (36 regex patterns + LLM synthesis)
    ├── HITL gate check            (mutation blocking)
    ├── Sequence hint injection    (ordered step directive — seeded + Haiku)
    ├── Output format adapter      (schema detection → format directive)
    ├── Knowledge + entities       (domain facts, cross-task memory)
    └── Finance pre-compute        (variance, SLA, proration, dispute credit)
        │
        ▼
    EXECUTE  ← LLM Cortex (inside structural constraints)
    ├── UCB1 bandit selects strategy: fsm | five_phase | moa
    ├── Two-phase state-gated execution:
    │     Phase A — GATHER (read-only tools; Claude cannot write)
    │     Phase B — MUTATE (write tools; structural approval gate enforced)
    ├── Structural Approval Gate   (mutations blocked until confirm_with_user fires)
    ├── Anti-refusal retry         (safety-refusal override for B2B tasks)
    ├── L2 Completion Contract     (mutation_count==0 on full task → retry)
    ├── L3 Tool Coverage Check     (task-text-mentioned tools not called → retry)
    │     L3 Negation Guard        (tools mentioned as "do NOT call X" excluded)
    ├── L3b Sequence Coverage      (seq_hint required steps missed → retry)
    └── Post-execution: math verification · numeric MoA · output validation · self-reflection
        │
        ▼
    REFLECT  ← Verification Layer
    ├── FSM checkpoint saved
    ├── Session memory compressed  (async, Haiku)
    ├── RL outcome recorded        (case_log.json + quality score)
    ├── UCB1 bandit updated
    ├── Sequence graph updated     (EMA confidence on process-type sequences)
    ├── Knowledge extracted        (quality ≥ 0.5)
    └── Entity memory updated
```

---

## 8-State FSM

The FSM enforces execution order structurally — data collection before computation, computation before policy check, policy check before approval, approval before mutation. This eliminates the common failure modes of agentic loops: premature mutations, bypassed approval gates, and incomplete downstream chains.

| State | Mutations | Purpose |
|---|---|---|
| `DECOMPOSE` | No | Parse task, identify entities and data needs |
| `ASSESS` | No | Read-only data gathering via tools |
| `COMPUTE` | No | Arithmetic from collected data — no side effects |
| `POLICY_CHECK` | No | Evaluate policy rules against computed values |
| `APPROVAL_GATE` | **Blocked** | Present approval request; await human-in-the-loop |
| `MUTATE` | Yes | Execute all state changes; each write followed by read-back |
| `SCHEDULE_NOTIFY` | Yes | Notifications, scheduling, audit entries |
| `COMPLETE` | No | Final summary |

Read-only tasks (no action verbs, ≤120 chars) collapse to a 3-state path: `DECOMPOSE → ASSESS → COMPLETE`. Task mode classifier removes mutation tools entirely for confirmed read-only tasks.

---

## Execution Safety Net

Five independent layers. Each fires independently — the structural gate prevents the problem; L2–L3b catch it if it slips through.

```
PRIMARY:  Two-phase GATHER → MUTATE (structural separation)
GATE:     Universal Approval Gate in _direct_call() (mechanical, not prompt-based)
L2:       Completion Contract  — mutation_count==0 on full task → write-tools-only retry
L3:       Tool Coverage Check  — tools named in task_text not called → targeted retry
          L3 Negation Guard    — tools mentioned as "do NOT call X" excluded from required set
L3b:      Sequence Coverage    — seq_hint required steps not called → retry (prefix-resolved)
COMPUTE:  Math Reflection      — Haiku critiques numeric answers before MUTATE
```

**L3 Negation Guard:** L3's tool coverage check extracts tool names from task text and retries if any required tool was not called. The negation guard prevents false positives: if the task says "do NOT call X", "avoid X", or "skip X", that tool is excluded from the required set. This handles tasks where the correct behavior is to explicitly *not* execute a particular mutation — e.g., subscription migrations where `proceed_migration` must be held until export completes.

**Dynamic Approval Gate:** `hitl_guard.is_approval_tool()` detects any HITL gate tool by regex — covering `confirm_with_user`, `require_customer_signoff`, `request_manager_approval`, and all similar conventions. No hardcoded tool names anywhere in the gate logic.

---

## Sequence Enforcer

`sequence_enforcer.py` injects an ordered tool-call directive into every task's system prompt before the LLM runs. Seeds cover 22 process types; Haiku synthesis handles novel types. The `SequenceGraph` caches sequences and updates confidence via EMA (α=0.3) from RL outcomes.

Each seed describes tool intent via **prefixes** (e.g. `"log_"`, `"update_budget_"`). `_resolve_tool_hints()` expands prefixes against the actual available tool set at runtime — no hardcoded tool names in the detection logic.

---

## Component Reference

| Module | Role |
|---|---|
| `server.py` | FastAPI application; A2A JSON-RPC 2.0 handler; reflexive injection layer |
| `worker_brain.py` | Core cognitive loop: PRIME / EXECUTE / REFLECT |
| `fsm_runner.py` | 8-state FSM engine; 15 built-in process templates |
| `dynamic_fsm.py` | Haiku-based FSM synthesizer for unknown process types |
| `claude_executor.py` | Agentic Claude execution loop (max 20 iterations, 25 tool calls) |
| `five_phase_executor.py` | PLAN → GATHER → SYNTHESIZE → ARTIFACT → INSIGHT executor |
| `self_moa.py` | Dual top-p mixture-of-agents with word-overlap consensus |
| `strategy_bandit.py` | UCB1 multi-armed bandit; learns win rate per strategy per process type |
| `token_budget.py` | 10K token budget; state-aware model selection |
| `mutation_verifier.py` | Write-tool tracking; WAL flush via read-back |
| `compute_verifier.py` | COMPUTE-state arithmetic audit; correction pass before MUTATE |
| `self_reflection.py` | Answer completeness scoring; improvement pass if below threshold |
| `hitl_guard.py` | Tool classification (read / compute / mutate); blocks mutations at APPROVAL_GATE |
| `sequence_enforcer.py` | Ordered tool-call directives; 22 seeded process types; Haiku synthesis fallback |
| `task_mode_classifier.py` | Deterministic read-only detection; strips mutation tools from read-only tasks |
| `smart_classifier.py` | Haiku process-type classification with keyword fallback |
| `policy_checker.py` | Deterministic policy rule evaluation; supports `&&`, `\|\|`, `!`, comparisons |
| `schema_adapter.py` | Schema drift resilience; 5-tier fuzzy column matching |
| `dynamic_tools.py` | Two-phase tool gap detection (36 regex patterns + LLM); sandboxed synthesis |
| `rl_loop.py` | Case log persistence; quality scoring; RL primer construction |
| `context_pruner.py` | Case log quality filtering; stale and repeated-failure entry removal |
| `knowledge_extractor.py` | Post-task domain fact extraction; keyword-keyed retrieval |
| `entity_extractor.py` | Zero-cost regex entity tracking across tasks |
| `financial_calculator.py` | Exact `Decimal` arithmetic for financial calculations |
| `finance_tools.py` | Finance context builder; proration, dispute credit, SLA pre-compute |
| `mcp_bridge.py` | MCP tool bridge; pre-flight parameter validation; schema patching |
| `session_context.py` | Multi-turn session state; FSM checkpoints; Haiku memory compression |
| `recovery_agent.py` | Tool failure recovery: synonym → decompose → Haiku advice → degrade |
| `structured_output.py` | Final answer formatting; exact-match bracket format enforcement |
| `output_validator.py` | Output format validation; required-field checking per process type |
| `privacy_guard.py` | PII and credential detection before any API call |
| `paginated_tools.py` | Cursor-loop bulk data fetching across all pagination styles |

---

## Requirements

Python 3.11+

```
fastapi>=0.115
uvicorn[standard]>=0.30
anthropic>=0.34
httpx>=0.27
pydantic>=2.0
```

---

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key |
| `GREEN_AGENT_MCP_URL` | Yes | — | MCP tool server base URL |
| `FALLBACK_MODEL` | No | `claude-sonnet-4-6` | Default execution model |
| `TOOL_TIMEOUT` | No | `10` | Seconds per tool call |
| `TASK_TIMEOUT` | No | `120` | Seconds per task |
| `RL_CACHE_DIR` | No | `/app` | Directory for JSON state files |

---

## Running

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
export GREEN_AGENT_MCP_URL=http://localhost:9009
python main.py --host 0.0.0.0 --port 9010
```

---

## Testing

```bash
# Core logic checks (no API key required — ~3 seconds)
python simulate_competition.py        # 19 checks: math, FSM, bandit, schema, brackets

# Phase 15-18 targeted checks (no API key required — ~5 seconds)
python simulate_phase15_16.py         # 82 checks: seeds, gate, L3b, dynamic approval, SaaS migration, L3 negation
```

Both suites test deterministic logic only — no live LLM or MCP server required.

---

## API

All requests use A2A JSON-RPC 2.0.

| Endpoint | Method | Description |
|---|---|---|
| `/` | POST | `tasks/send` — submit a business process task |
| `/.well-known/agent-card.json` | GET | Agent capability declaration |
| `/health` | GET | Health check |
| `/rl/status` | GET | Case log, bandit state, tool registry, FSM cache |

```json
{
  "jsonrpc": "2.0",
  "method": "tasks/send",
  "id": "task-001",
  "params": {
    "id": "task-001",
    "message": {
      "role": "user",
      "parts": [{ "text": "Process vendor invoice INV-2024-447 for Acme Corp..." }]
    },
    "metadata": {
      "policy_doc": "{ \"rules\": [...] }",
      "tools_endpoint": "https://mcp.example.com",
      "session_id": "worker-abc"
    }
  }
}
```

---

## Tech Stack

- **Runtime:** Python 3.11, FastAPI, uvicorn
- **LLM:** Anthropic Claude — Haiku for classification, synthesis, and audit; Sonnet for COMPUTE and MUTATE
- **FSM:** Custom 8-state engine with dynamic synthesis for novel process types
- **Tool bridge:** MCP HTTP with pre-flight validation and schema drift correction
- **Numerics:** `decimal.Decimal` in sandboxed tool execution
- **RL:** UCB1 bandit + case log + quality scoring + knowledge extraction + sequence graph
- **Storage:** Local JSON (tool registry, bandit state, sequence graph, entity memory, knowledge base, case log)

---

## License

Apache 2.0
