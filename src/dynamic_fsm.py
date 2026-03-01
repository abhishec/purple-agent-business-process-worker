"""
dynamic_fsm.py
Runtime FSM synthesizer for novel business process types.

When the competition sends a process type we've never seen, this module calls
Haiku once to synthesize an optimal state sequence + per-state instructions.
Results are cached to synthesized_definitions.json — one Haiku call per new type.
All future tasks of the same type get the cached definition for free.

Industry backing:
  - StateFlow (Wu et al., 2024): explicit FSM modeling achieves +13-28% accuracy
    vs ReAct at 3-5x lower cost (NeurIPS 2024)
  - MetaAgent (ICML 2025): dynamic FSM synthesis + traceback = +58.8% on complex tasks
  - Adaptive Case Management (IBM/ServiceNow): workflow synthesized at runtime, not hardcoded

RL integration:
  - After synthesis, enriches DECOMPOSE instructions with past RL patterns
    from knowledge_extractor for this process type (if any exist)
  - Over time: same novel types get better instructions as RL accumulates evidence
  - get_synthesis_stats() feeds into /rl/status endpoint for transparency
"""
from __future__ import annotations

import json
import os
import re
import asyncio
from pathlib import Path

from src.process_definitions import PROCESS_DEFINITIONS

# ── Cache ──────────────────────────────────────────────────────────────────

_CACHE_FILE = Path(os.environ.get("RL_CACHE_DIR", "/app")) / "synthesized_definitions.json"
_cache: dict[str, dict] = {}
_cache_loaded = False


def _load_cache() -> None:
    global _cache, _cache_loaded
    if _cache_loaded:
        return
    try:
        if _CACHE_FILE.exists():
            _cache = json.loads(_CACHE_FILE.read_text())
    except Exception:
        _cache = {}
    _cache_loaded = True


def _save_cache() -> None:
    try:
        _CACHE_FILE.write_text(json.dumps(_cache, indent=2))
    except Exception:
        pass  # best-effort — never crash the task


def is_known_type(process_type: str) -> bool:
    """Return True if this type has a built-in process definition (no synthesis needed)."""
    return process_type in PROCESS_DEFINITIONS


# ── Synthesis prompt ────────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM = """\
You are a business process analyst specializing in workflow automation.

Given a process type name and task description, synthesize an optimal FSM workflow.

Available FSM states (choose the right subset for this process):
- DECOMPOSE: Break task into sub-tasks, identify required entities and data
- ASSESS: Gather all required data via read-only tools (no write actions yet)
- COMPUTE: Run calculations using gathered data (no tools — pure math and scoring)
- POLICY_CHECK: Verify business rules, thresholds, and compliance constraints
- APPROVAL_GATE: Human-in-the-loop approval required before any mutations
- MUTATE: Execute state changes via write tools (only after all prior phases complete)
- SCHEDULE_NOTIFY: Send notifications, schedule follow-up actions
- COMPLETE: Summarize all outcomes concisely

Design rules:
1. ALWAYS include DECOMPOSE (first) and COMPLETE (last)
2. Include ASSESS if data needs to be gathered from external sources
3. Include COMPUTE only if calculations are needed (financial math, risk scoring, metrics)
4. Include POLICY_CHECK if business rules or approval thresholds must be verified
5. Include APPROVAL_GATE only for high-risk processes requiring explicit human sign-off
6. Include MUTATE if any state changes or write operations are required
7. Include SCHEDULE_NOTIFY only if notifications or scheduling are part of the outcome
8. Write specific, actionable state_instructions for EVERY state you include

Risk levels:
- low: informational queries or low-impact reversible changes
- medium: financial changes under $10K or access changes that are reversible
- high: financial changes over $10K, irreversible actions, or compliance-critical

Respond ONLY with valid JSON. No explanation. No markdown fences."""

_SYNTHESIS_PROMPT = """\
Process type: {process_type}
Task description: {task_text}

Synthesize the optimal FSM workflow for this process type.
Return JSON with exactly this schema:
{{
  "states": ["DECOMPOSE", "ASSESS", ...],
  "hitl_required": false,
  "risk_level": "low",
  "connector_hints": ["tool-prefix-1", "tool-prefix-2"],
  "state_instructions": {{
    "DECOMPOSE": "Specific instruction for decompose phase...",
    "ASSESS": "Specific instruction for assess phase..."
  }}
}}"""

def _get_valid_states() -> frozenset:
    """Generate _VALID_STATES from FSMState enum — stays in sync automatically when new states are added."""
    from src.fsm_runner import FSMState
    _TERMINAL = {"ESCALATE", "FAILED"}
    return frozenset(s.value for s in FSMState if s.value not in _TERMINAL)

# Called once at module load, cached
_VALID_STATES = _get_valid_states()


# ── Response parsing ────────────────────────────────────────────────────────

def _parse_synthesis_response(text: str) -> dict | None:
    """Extract and validate JSON from Haiku synthesis response."""
    clean = text.strip()
    # Strip markdown fences if present
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-z]*\n?", "", clean)
        clean = re.sub(r"\n?```$", "", clean).strip()

    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None

    if not isinstance(data, dict):
        return None

    # Filter states to valid values only
    raw_states = data.get("states", [])
    if not isinstance(raw_states, list):
        return None
    valid_states = [s for s in raw_states if s in _VALID_STATES]
    if not valid_states:
        return None

    # Enforce mandatory bookends
    if valid_states[0] != "DECOMPOSE":
        valid_states.insert(0, "DECOMPOSE")
    if valid_states[-1] != "COMPLETE":
        valid_states.append("COMPLETE")
    data["states"] = valid_states

    # Validate and default other fields
    if data.get("risk_level") not in ("low", "medium", "high"):
        data["risk_level"] = "medium"
    if not isinstance(data.get("hitl_required"), bool):
        data["hitl_required"] = data.get("risk_level") == "high"
    if not isinstance(data.get("connector_hints"), list):
        data["connector_hints"] = []
    if not isinstance(data.get("state_instructions"), dict):
        data["state_instructions"] = {}

    return data


def _build_fallback_definition(process_type: str) -> dict:
    """Minimal fallback when Haiku synthesis fails — better than generic 'general' template."""
    label = process_type.replace("_", " ")
    return {
        "states": ["DECOMPOSE", "ASSESS", "COMPUTE", "POLICY_CHECK", "MUTATE", "COMPLETE"],
        "hitl_required": False,
        "risk_level": "medium",
        "connector_hints": [],
        "state_instructions": {
            "DECOMPOSE": (
                f"Break the {label} task into sub-tasks. "
                "Identify all entities, IDs, amounts, and parties involved. "
                "List what data you need to collect before taking any action."
            ),
            "ASSESS": (
                "Using the read-only tools available for this workspace, gather all required data. "
                "Do NOT take any write actions yet. Retrieve records and check statuses."
            ),
            "COMPUTE": (
                f"Run all calculations required for {label}. "
                "Use only data already collected in ASSESS — do not call additional tools."
            ),
            "POLICY_CHECK": (
                "Verify all business rules, thresholds, and constraints "
                "before executing any changes."
            ),
            "MUTATE": (
                "Execute all required state changes via the write tools available. "
                "Log each action with its outcome."
            ),
            "COMPLETE": (
                "Summarize all completed actions and their outcomes. "
                "Include amounts, entity IDs, and any deferred items."
            ),
        },
        "_synthesized": False,
        "_fallback": True,
    }


# ── Core synthesis ──────────────────────────────────────────────────────────

async def synthesize_if_needed(process_type: str, task_text: str) -> dict | None:
    """
    Synthesize an FSM definition for a novel process type.

    Returns:
        dict with states, state_instructions, hitl_required, risk_level, connector_hints
        None if process_type is already known (use process_definitions.py instead)

    Cost: one Haiku call per new type, then cached for all future tasks.
    Total cost per unknown type: ~$0.0001 (negligible vs the competition win value).
    """
    if is_known_type(process_type):
        return None  # Already have a hardcoded definition — no synthesis needed

    _load_cache()

    # Cache hit: return existing synthesis (enriched with latest RL patterns)
    if process_type in _cache:
        return _enrich_with_rl(_cache[process_type], task_text, process_type)

    # Cache miss: synthesize via Haiku
    definition = await _call_haiku_synthesizer(process_type, task_text)

    # Persist to cache
    _cache[process_type] = definition
    _save_cache()

    return _enrich_with_rl(definition, task_text, process_type)


async def _call_haiku_synthesizer(process_type: str, task_text: str) -> dict:
    """Call Haiku to synthesize an FSM definition. Returns fallback on any error."""
    prompt = _SYNTHESIS_PROMPT.format(
        process_type=process_type,
        task_text=task_text[:800],  # cap task text to save tokens
    )

    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic()

        msg = await asyncio.wait_for(
            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=700,
                system=_SYNTHESIS_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=8.0,  # hard timeout — never block execution path
        )

        raw = msg.content[0].text if msg.content else ""
        parsed = _parse_synthesis_response(raw)

        if parsed:
            parsed["_synthesized"] = True
            parsed["_process_type"] = process_type
            return parsed

    except Exception:
        pass  # fall through to fallback

    return _build_fallback_definition(process_type)


# ── RL enrichment ───────────────────────────────────────────────────────────

def _enrich_with_rl(definition: dict, task_text: str, process_type: str) -> dict:
    """
    Enrich synthesized instructions with RL-discovered patterns.

    Reads past successes from knowledge_extractor for this process type.
    Appends "RL Patterns:" to DECOMPOSE instruction so the agent benefits from
    past task experience even on the first turn of a novel process.

    This is the compounding loop: novel types synthesize once, then RL improves
    every future task of that type without additional API calls.
    """
    try:
        from src.knowledge_extractor import get_relevant_knowledge
        rl_patterns = get_relevant_knowledge(task_text, process_type)
        if not rl_patterns or len(rl_patterns) < 20:
            return definition

        enriched = dict(definition)
        instructions = dict(enriched.get("state_instructions", {}))
        decompose_instr = instructions.get("DECOMPOSE", "")
        instructions["DECOMPOSE"] = (
            decompose_instr
            + f"\n\n[RL patterns for {process_type} from past tasks]\n"
            + rl_patterns[:600]
        )
        enriched["state_instructions"] = instructions
        return enriched

    except Exception:
        pass

    return definition


# ── Read-only accessors ─────────────────────────────────────────────────────

def get_synthesized(process_type: str) -> dict | None:
    """Get cached synthesis for a process type (read-only, no API call)."""
    _load_cache()
    return _cache.get(process_type)


def get_synthesis_stats() -> dict:
    """Return synthesis cache stats for /rl/status endpoint."""
    _load_cache()
    total = len(_cache)
    synthesized = sum(1 for v in _cache.values() if v.get("_synthesized"))
    fallback = sum(1 for v in _cache.values() if v.get("_fallback"))
    return {
        "total_novel_types": total,
        "haiku_synthesized": synthesized,
        "fallback_definitions": fallback,
        "cached_types": list(_cache.keys()),
    }
