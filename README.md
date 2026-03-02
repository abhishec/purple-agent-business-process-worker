# Purple Agent

An autonomous business-process AI worker that executes enterprise workflows end-to-end via a structured finite state machine, deterministic policy enforcement, and a compounding reinforcement-learning loop.

## Overview

Purple Agent connects to an MCP tool server and operates an 8-state finite state machine that enforces execution order structurally — data collection before computation, computation before policy check, policy check before mutation. This eliminates the common failure mode of agentic loops that attempt mutations before gathering required data or satisfying policy gates.

## Architecture

```
POST /  (A2A JSON-RPC 2.0)
        │
        ▼
    PRIME
    ├── Privacy guard          (zero API cost)
    ├── RL primer              (past-task patterns)
    ├── Session context        (Haiku-compressed history)
    ├── FSM classification     (process type detection)
    ├── Dynamic FSM synthesis  (novel process types)
    ├── Policy evaluation      (deterministic, zero LLM)
    ├── Tool discovery         (MCP + local registry)
    ├── Tool gap detection     (36 regex patterns + LLM)
    ├── HITL gate check        (mutation blocking)
    ├── Knowledge + entities   (domain facts, cross-task)
    └── Finance pre-compute    (variance, SLA, proration)
        │
        ▼
    EXECUTE
    ├── UCB1 bandit selects strategy: fsm | five_phase | moa
    ├── 8-state FSM: DECOMPOSE → ASSESS → COMPUTE → POLICY_CHECK
    │                → APPROVAL_GATE → MUTATE → SCHEDULE_NOTIFY → COMPLETE
    └── Post-execution: mutation log · math verification · numeric MoA
                        approval brief · output validation · self-reflection
        │
        ▼
    REFLECT
    ├── FSM checkpoint saved
    ├── Session memory compressed (async, Haiku)
    ├── RL outcome recorded → case_log.json
    ├── UCB1 bandit updated
    ├── Knowledge extracted (quality ≥ 0.5)
    └── Entity memory updated
```

## 8-State FSM

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

Read-only tasks (no action verbs detected) collapse to a 3-state path: `DECOMPOSE → ASSESS → COMPLETE`.

## Component Reference

| Module | Role |
|---|---|
| `server.py` | FastAPI application; A2A JSON-RPC 2.0 handler |
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
| `policy_checker.py` | Deterministic policy rule evaluation; supports `&&`, `\|\|`, `!`, comparisons |
| `schema_adapter.py` | Schema drift resilience; 5-tier fuzzy column matching |
| `smart_classifier.py` | Haiku process-type classification with keyword fallback |
| `dynamic_tools.py` | Two-phase tool gap detection (36 regex patterns + LLM); sandboxed synthesis |
| `rl_loop.py` | Case log persistence; quality scoring; RL primer construction |
| `context_pruner.py` | Case log quality filtering; stale and repeated-failure entry removal |
| `knowledge_extractor.py` | Post-task domain fact extraction; keyword-keyed retrieval |
| `entity_extractor.py` | Zero-cost regex entity tracking across tasks |
| `financial_calculator.py` | Exact `Decimal` arithmetic for financial calculations |
| `mcp_bridge.py` | MCP tool bridge; pre-flight parameter validation; schema patching |
| `session_context.py` | Multi-turn session state; FSM checkpoints; Haiku memory compression |
| `recovery_agent.py` | Tool failure recovery: synonym → decompose → Haiku advice → degrade |
| `structured_output.py` | Final answer formatting; exact-match bracket format enforcement |
| `output_validator.py` | Output format validation; required-field checking per process type |
| `privacy_guard.py` | PII and credential detection before any API call |
| `paginated_tools.py` | Cursor-loop bulk data fetching across all pagination styles |

## Requirements

Python 3.11+

```
fastapi>=0.115
uvicorn[standard]>=0.30
anthropic>=0.34
httpx>=0.27
pydantic>=2.0
```

## Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API key |
| `GREEN_AGENT_MCP_URL` | Yes | — | MCP tool server base URL |
| `FALLBACK_MODEL` | No | `claude-sonnet-4-6` | Default execution model |
| `TOOL_TIMEOUT` | No | `10` | Seconds per tool call |
| `TASK_TIMEOUT` | No | `120` | Seconds per task |
| `RL_CACHE_DIR` | No | `/app` | Directory for JSON state files |

## Running

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
export GREEN_AGENT_MCP_URL=http://localhost:9009
python main.py --host 0.0.0.0 --port 9010
```

## API

All requests use A2A JSON-RPC 2.0.

**Endpoints**

| Endpoint | Method | Description |
|---|---|---|
| `/` | POST | `tasks/send` — submit a business process task |
| `/.well-known/agent-card.json` | GET | Agent capability declaration |
| `/health` | GET | Health check |
| `/rl/status` | GET | Case log, bandit state, tool registry, FSM cache |

**Request format**

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

## Tech Stack

- **Runtime:** Python 3.11, FastAPI, uvicorn
- **LLM:** Anthropic Claude — Haiku for classification, synthesis, and audit; Sonnet for COMPUTE and MUTATE
- **FSM:** Custom 8-state engine with dynamic synthesis for novel process types
- **Tool bridge:** MCP HTTP with pre-flight validation and schema drift correction
- **Numerics:** `decimal.Decimal` in sandboxed tool execution
- **RL:** UCB1 bandit + case log + quality scoring + knowledge extraction
- **Storage:** Local JSON (tool registry, bandit state, entity memory, knowledge base, case log)

## License

Apache 2.0
