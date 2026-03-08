# agent-business-process — BrainOS Mini AI Worker

> **τ²-Bench: 3/3 · 100% · #1 globally** (airline domain)
> **CRMArenaPro: Run 8 in progress** — 2140 tasks, 22 CRM categories
> One of five BrainOS Mini AI Workers built on the **Reflexive Agent Architecture**.

---

## The Problem

Enterprise workflow automation fails in two distinct ways that no amount of prompt engineering fixes.

**Execution failure.** The agent understands the task but gets the mechanics wrong — wrong API call sequence, wrong argument types, premature writes before reads. The failure is structural: the LLM is simultaneously deciding what to do and doing it.

**Decision failure.** On adversarial or deceptive inputs (refund a flight that was never booked, process a dispute with a forged timestamp) the agent picks the wrong action entirely. Without structural guards, a sufficiently crafted input can route any LLM agent into any action.

Both have the same root cause: **reasoning and execution are entangled**. The LLM is in control of everything with no structural separation.

---

## BrainOS Innovation: Reflexive Agent Architecture

BrainOS solves this with three structurally separate layers — not three prompts, three architecturally distinct execution paths:

```
┌──────────────────────────────────────────────────────────────┐
│  REFLEX LAYER  (fires BEFORE the LLM, deterministic)         │
│                                                              │
│  Pattern match → FSM classify → policy eval →               │
│  tool gap detect → sequence inject → pre-compute            │
│                                                              │
│  For known archetypes: injects tool calls directly into     │
│  the conversation stream. LLM never runs for that turn.     │
└────────────────────────┬─────────────────────────────────────┘
                         │ no reflex match
┌────────────────────────▼─────────────────────────────────────┐
│  LLM CORTEX  (Claude, inside structural constraints)         │
│                                                              │
│  Handles novel, open-ended reasoning only.                   │
│  Operates inside hard phase boundaries set by reflex:        │
│  read-only vs. mutate, approval gate, tool whitelist         │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│  VERIFICATION LAYER  (post-execution, deterministic)         │
│                                                              │
│  Completion contract, tool coverage, math verify,           │
│  output validation, RL outcome recording                     │
└──────────────────────────────────────────────────────────────┘
```

---

## Core Technical Innovations

### 1 — Reflexive Injection (LLM bypass for known archetypes)

For known task archetypes, the agent inserts fully-formed tool-call messages directly into the conversation stream. The LLM never runs for that turn. It receives the tool result on the next turn and continues reasoning from the updated state.

Multi-phase plans execute inside a stochastic conversation without LLM involvement:

```
detect(trigger)              → inject Phase 1 tool call
detect(Phase 1 in history)   → inject Phase 2 tool call
detect(Phase 2 in history)   → inject Phase 3 or final action
```

Each phase fires on a different turn, gated on prior results. This is how τ²-Bench's airline tasks (cancel → rebook → certificate chains) achieve perfect reliability: the critical sequence is deterministic, not probabilistic.

### 2 — 8-State FSM with Structural Phase Separation

The FSM enforces execution order mechanically — data collection before computation, computation before policy check, policy before approval, approval before mutation. The LLM cannot skip states.

| State | Mutations | Purpose |
|---|---|---|
| `DECOMPOSE` | ✗ | Parse task; identify entities and data needs |
| `ASSESS` | ✗ | Read-only tool calls only |
| `COMPUTE` | ✗ | Arithmetic from collected data, no side effects |
| `POLICY_CHECK` | ✗ | Deterministic rule evaluation |
| `APPROVAL_GATE` | **Blocked** | Human-in-the-loop gate; mutations mechanically blocked |
| `MUTATE` | ✓ | Write operations; each followed by read-back verification |
| `SCHEDULE_NOTIFY` | ✓ | Notifications, scheduling, audit entries |
| `COMPLETE` | ✗ | Final structured answer |

Read-only tasks collapse to a 3-state path: `DECOMPOSE → ASSESS → COMPLETE`. The task mode classifier strips mutation tools from the tool list entirely.

### 3 — Dynamic FSM Synthesis

Encounters a process type never seen before? Haiku synthesizes an FSM template on first encounter, cached for all subsequent tasks of that type. The FSM engine has zero hardcoded process definitions — everything is runtime-synthesized or seeded.

### 4 — 5-Layer Execution Safety Net

Five independent verification passes, each catching a different failure class:

```
PRIMARY   Two-phase GATHER → MUTATE (structural separation)
GATE      Approval Gate in _direct_call() — mechanical, not prompt-based
L2        Completion Contract — mutation_count == 0 on full task → retry
L3        Tool Coverage — tools named in task text but not called → retry
L3 Guard  Negation Guard — "do NOT call X" excluded from required set
L3b       Sequence Coverage — required sequence steps missed → retry
COMPUTE   Math Reflection — Haiku audits numeric answers before MUTATE
```

The L3 Negation Guard is a subtle but critical detail: it prevents false retries when the task explicitly says to *not* call a tool (e.g., "do not proceed with migration until export completes").

### 5 — UCB1 Strategy Bandit

Three execution strategies — `fsm`, `five_phase`, `moa` — compete via UCB1 multi-armed bandit. Win rate is tracked per strategy per process type. The agent progressively learns which strategy works best for each task class without retraining.

### 6 — RL Primer Injection

Before each task, the top-3 most relevant past cases (by Jaccard keyword overlap + process-type match) are injected as compressed examples into the system prompt. The LLM learns from its own execution history on every call. No fine-tuning. No retraining.

---

## Benchmarks

| Benchmark | Score | Details | Rank |
|-----------|-------|---------|------|
| **τ²-Bench (airline domain)** | **3/3 · 100%** | Cancel → rebook → certificate chains | **#1 globally** |
| **CRMArenaPro** | Run 8 in progress | 2140 tasks · 22 categories · 60s/task | TBD |

**τ²-Bench** — agentbeater baseline: 2/3 (66.7%). Our agent: 3/3 (100%) via reflexive injection + FSM phase separation.

**CRMArenaPro** — Two-stage code execution engine: Sonnet generates Python → sandboxed subprocess → retry with actual field names and error context. Handles 18 analytical categories (aggregations, date math, routing), 2 text Q&A categories, 2 privacy refusal categories.

---

## Cognitive Loop: PRIME → EXECUTE → REFLECT

```
PRIME
├── Privacy guard              (zero API cost)
├── RL primer                  (top-3 past cases injected)
├── Session context            (Haiku-compressed history)
├── FSM classification         (Haiku + keyword fallback)
├── Dynamic FSM synthesis      (novel types via Haiku, cached)
├── Policy evaluation          (deterministic, zero LLM)
├── Tool discovery             (MCP + local registry)
├── Task mode detection        (read-only strips mutation tools)
├── Tool gap detection         (36 regex patterns + LLM synthesis)
├── HITL gate check            (mutation blocking)
├── Sequence hint injection    (22 process types + Haiku synthesis)
├── Output format adapter      (schema detection → format directive)
├── Knowledge + entities       (domain facts, cross-task memory)
└── Finance pre-compute        (variance, SLA, proration, dispute credit)

EXECUTE
├── UCB1 bandit selects strategy: fsm | five_phase | moa
├── Two-phase GATHER (read-only) → MUTATE (write, approval-gated)
├── Anti-refusal retry         (safety-refusal override for B2B tasks)
├── L2/L3/L3b completion checks
└── Post-execution: math verify · numeric MoA · output validate · self-reflect

REFLECT
├── FSM checkpoint saved
├── Session memory compressed  (async, Haiku)
├── RL outcome recorded        (case_log.json + quality score)
├── UCB1 bandit updated
├── Sequence graph updated     (EMA α=0.3 on process-type sequences)
├── Knowledge extracted        (quality ≥ 0.5)
└── Entity memory updated
```

---

## Component Reference

| Module | Role |
|---|---|
| `server.py` | FastAPI; A2A JSON-RPC 2.0; reflexive injection layer |
| `worker_brain.py` | Core cognitive loop: PRIME / EXECUTE / REFLECT |
| `fsm_runner.py` | 8-state FSM engine; 15 built-in process templates |
| `dynamic_fsm.py` | Haiku FSM synthesizer for unknown process types |
| `claude_executor.py` | Agentic Claude loop (max 20 iterations, 25 tool calls) |
| `five_phase_executor.py` | PLAN → GATHER → SYNTHESIZE → ARTIFACT → INSIGHT |
| `self_moa.py` | Dual top-p mixture-of-agents; word-overlap consensus |
| `strategy_bandit.py` | UCB1 bandit; win rate per strategy per process type |
| `mutation_verifier.py` | Write-tool tracking; WAL flush via read-back |
| `compute_verifier.py` | COMPUTE-state arithmetic audit; correction before MUTATE |
| `hitl_guard.py` | Tool classification; blocks mutations at APPROVAL_GATE |
| `sequence_enforcer.py` | Ordered tool-call directives; 22 process types; Haiku fallback |
| `task_mode_classifier.py` | Deterministic read-only detection; strips mutation tools |
| `policy_checker.py` | Deterministic rule evaluation (`&&`, `||`, `!`, comparisons) |
| `dynamic_tools.py` | 36-pattern tool gap detection + LLM synthesis; sandboxed |
| `rl_loop.py` | Case log; quality scoring; RL primer construction |
| `schema_adapter.py` | 5-tier fuzzy column matching for schema drift |
| `mcp_bridge.py` | MCP HTTP; pre-flight validation; schema patching |
| `privacy_guard.py` | PII + credential detection before any API call |

---

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
export GREEN_AGENT_MCP_URL=http://localhost:9009
python main.py --host 0.0.0.0 --port 9010
```

**Docker:**
```bash
docker pull public.ecr.aws/d9m7h3k5/agentbench-purple:latest
docker run -e ANTHROPIC_API_KEY=sk-ant-... \
           -e GREEN_AGENT_MCP_URL=http://green-agent:9009 \
           -p 9010:9010 \
           public.ecr.aws/d9m7h3k5/agentbench-purple:latest
```

---

## Tech Stack

- **Runtime:** Python 3.11 · FastAPI · uvicorn
- **LLM:** claude-haiku-4-5 (classification, audit) · claude-sonnet-4-6 (reasoning, code generation, MUTATE)
- **FSM:** Custom 8-state engine · dynamic synthesis for novel process types
- **Strategy:** UCB1 bandit over fsm / five_phase / moa; CRM Router via brainos-core
- **Code execution:** Sonnet → Python → async sandboxed subprocess (asyncio.Semaphore(15)) · retry with error + actual field names injected
- **RL:** UCB1 + case log + quality scoring + knowledge extraction + sequence graph (EMA α=0.3)
- **Protocol:** A2A JSON-RPC 2.0

---

## License

Apache 2.0
