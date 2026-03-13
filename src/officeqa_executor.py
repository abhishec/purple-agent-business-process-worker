"""
officeqa_executor.py — OfficeQA Task Handler for Purple Agent
==============================================================
Self-contained OfficeQA evaluator module. All OfficeQA-specific logic
lives here — zero changes to any existing Purple Agent module.

This module handles tasks sent by the AgentBeats OfficeQA green agent:
  Green Agent (OfficeQA evaluator, ID: 019ca14f-cbd5-7a71-9b2b-d43db92a09e1)
      ↓  A2A task: { question, uid, source_files, difficulty }
  Purple Agent server.py
      ↓  is_officeqa_task(text) == True
  _handle_officeqa_turn(task_text, session_id)
      ↓  fetch Treasury Bulletin TXT files (officeqa_tools.py)
      ↓  call Claude with bulletin context + question
      →  FINAL ANSWER: [value]

Integration with server.py (3-line addition, following tau2/crm pattern):
    from src.officeqa_executor import is_officeqa_task, handle_officeqa_turn
    ...
    elif is_officeqa_task(task_text):
        answer = await handle_officeqa_turn(task_text, session_id=context_id)

Architecture alignment:
  • Mirrors _handle_crm_turn() and _handle_tau2_turn() in structure
  • Uses FAST_MODEL (Haiku) for keyword extraction, FALLBACK_MODEL (Sonnet) for reasoning
  • Answer extraction follows FINAL ANSWER: convention used across all evaluators
  • No UCB1 bandit modification — OfficeQA results are logged via existing rl_loop
"""

from __future__ import annotations

import json
import re
import time
import asyncio
from typing import Optional

import anthropic

from src.config import ANTHROPIC_API_KEY, FAST_MODEL, FALLBACK_MODEL
from src.officeqa_tools import (
    build_bulletin_context,
    extract_search_keywords,
    parse_source_files,
)

# ── Client ────────────────────────────────────────────────────────────────────

_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

# ── Task detection ─────────────────────────────────────────────────────────────

# Signals that identify an OfficeQA task from the green agent
_OFFICEQA_SIGNALS = (
    '"source_files"',           # JSON field from OfficeQA CSV
    '"uid"',                    # OfficeQA question UID field
    "treasury bulletin",        # plain-text question reference
    "treasury_bulletin_",       # filename reference
    "federal reserve",          # corpus source
)

_OFFICEQA_NEGATIVE = (
    '"task_category"',          # CRMArena
    '"crm_task"',               # CRMArena
    '"flight_number"',          # tau2-bench
    '"reservation"',            # tau2-bench
)


def is_officeqa_task(text: str) -> bool:
    """
    Detect if this A2A message is an OfficeQA task.
    Checks for OfficeQA-specific JSON fields or question content.
    Returns False immediately if it looks like CRM or tau2.
    """
    if not text:
        return False

    text_lower = text[:500].lower()

    # Negative signals: bail immediately
    for neg in _OFFICEQA_NEGATIVE:
        if neg in text_lower:
            return False

    # Positive signals
    for sig in _OFFICEQA_SIGNALS:
        if sig in text_lower:
            return True

    return False


# ── Task parsing ───────────────────────────────────────────────────────────────

def _parse_officeqa_task(task_text: str) -> dict:
    """
    Parse an OfficeQA task message. Supports two formats:
      1. JSON: { "question": "...", "uid": "...", "source_files": "...", "difficulty": "easy" }
      2. Plain text: just the question string
    Returns normalized dict with keys: question, uid, source_files, difficulty.
    """
    task_text = task_text.strip()

    # Try JSON parse
    try:
        data = json.loads(task_text)
        if isinstance(data, dict):
            return {
                "question":     data.get("question", task_text),
                "uid":          data.get("uid", ""),
                "source_files": data.get("source_files", ""),
                "difficulty":   data.get("difficulty", "easy"),
            }
    except (json.JSONDecodeError, ValueError):
        pass

    # Plain text — the full text is the question
    return {
        "question":     task_text,
        "uid":          "",
        "source_files": "",
        "difficulty":   "easy",
    }


# ── System prompt ──────────────────────────────────────────────────────────────

_OFFICEQA_SYSTEM = """You are a financial document analysis expert specializing in U.S. Treasury Bulletins.

You will be given:
1. The full text content of one or more Treasury Bulletin documents
2. A specific question about data in those documents

Your task:
- Read the bulletin documents carefully
- Find the exact table, chart, or paragraph that contains the answer
- Extract the precise value requested

Answer format rules (CRITICAL):
- End EVERY response with: FINAL ANSWER: [value]
- For numbers: give just the number (e.g. "543" not "five hundred forty-three")
- For amounts: use numeric form without commas (e.g. "1234.5" not "$1,234.50")
- For percentages: give the number WITHOUT the % symbol unless explicitly asked
- For dates: use "Month YYYY" format (e.g. "June 1962")
- For text: give the exact phrase from the document
- No markdown formatting after FINAL ANSWER
- No explanation after FINAL ANSWER

If you cannot find the answer in the documents, state what you searched for, then give your best estimate with FINAL ANSWER: [estimate]."""

_OFFICEQA_SYSTEM_NO_DOCS = """You are a financial expert specializing in U.S. Treasury financial history and policy (1939–2025).

Answer questions about Treasury Bulletins, interest rates, public debt, fiscal data, and related financial statistics.

Answer format rules (CRITICAL):
- End EVERY response with: FINAL ANSWER: [value]
- For numbers: give just the number without units unless the question asks for units
- For percentages: give the number without % unless asked
- For dates: use "Month YYYY" format
- No explanation after FINAL ANSWER"""


# ── Core handler ───────────────────────────────────────────────────────────────

async def handle_officeqa_turn(
    task_text: str,
    session_id: str = "",
) -> str:
    """
    Handle an OfficeQA task end-to-end:
      1. Parse task JSON
      2. Fetch relevant Treasury Bulletin(s)
      3. Call Claude Sonnet with bulletin context + question
      4. Extract and return FINAL ANSWER

    This is the single entry point called by server.py.
    """
    task_start = time.monotonic()
    task = _parse_officeqa_task(task_text)
    question     = task["question"]
    source_files = task["source_files"]
    difficulty   = task["difficulty"]
    uid          = task.get("uid", "")[:12] or "anon"

    print(
        f"[officeqa] uid={uid} difficulty={difficulty} "
        f"src_files={source_files[:60] if source_files else '(none)'} "
        f"q={question[:80].replace(chr(10), ' ')}",
        flush=True,
    )

    # ── Step 1: Fetch Treasury Bulletin documents ────────────────────────────
    search_kws = extract_search_keywords(question)
    bulletin_ctx = ""
    try:
        bulletin_ctx = await asyncio.wait_for(
            build_bulletin_context(source_files, question, search_kws),
            timeout=45.0,
        )
    except asyncio.TimeoutError:
        print(f"[officeqa] bulletin fetch timeout uid={uid}", flush=True)
    except Exception as exc:
        print(f"[officeqa] bulletin fetch error uid={uid}: {exc}", flush=True)

    has_docs = bool(bulletin_ctx)
    print(
        f"[officeqa] bulletin_ctx={'yes' if has_docs else 'no'} "
        f"chars={len(bulletin_ctx):,} "
        f"fetch_elapsed={time.monotonic() - task_start:.1f}s",
        flush=True,
    )

    # ── Step 2: Build prompt ─────────────────────────────────────────────────
    system_prompt = _OFFICEQA_SYSTEM if has_docs else _OFFICEQA_SYSTEM_NO_DOCS

    if has_docs:
        user_message = (
            f"{bulletin_ctx}\n\n"
            "---\n\n"
            f"Question: {question}\n\n"
            "Based on the Treasury Bulletin documents above, answer the question precisely.\n"
            "Remember: end with FINAL ANSWER: [value]"
        )
    else:
        user_message = (
            f"Question: {question}\n\n"
            "Answer based on your knowledge of U.S. Treasury Bulletins and financial history.\n"
            "Remember: end with FINAL ANSWER: [value]"
        )

    # ── Step 3: Call Claude ───────────────────────────────────────────────────
    model = FALLBACK_MODEL  # Sonnet — document comprehension needs it
    timeout_secs = 90.0 if difficulty == "hard" else 60.0

    try:
        response = await asyncio.wait_for(
            _client.messages.create(
                model=model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ),
            timeout=timeout_secs,
        )

        raw_answer = response.content[0].text if response.content else ""

    except asyncio.TimeoutError:
        print(f"[officeqa] Claude timeout uid={uid}", flush=True)
        raw_answer = "FINAL ANSWER: (timeout)"
    except Exception as exc:
        print(f"[officeqa] Claude error uid={uid}: {exc}", flush=True)
        raw_answer = f"FINAL ANSWER: (error: {str(exc)[:60]})"

    # ── Step 4: Extract and return answer ─────────────────────────────────────
    final_answer = _extract_final_answer(raw_answer)
    elapsed = time.monotonic() - task_start

    print(
        f"[officeqa] uid={uid} answer={final_answer[:60]!r} "
        f"model={model} elapsed={elapsed:.1f}s",
        flush=True,
    )

    return final_answer


def _extract_final_answer(text: str) -> str:
    """
    Extract the answer after 'FINAL ANSWER:' marker.
    Falls back to last non-empty line if no marker found.
    """
    # Match FINAL ANSWER: [value]
    m = re.search(r"FINAL ANSWER[:\s—\-]+(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        answer = m.group(1).strip()
        # Strip wrapping quotes/backticks
        answer = re.sub(r"^[`\"']+|[`\"']+$", "", answer).strip()
        return answer

    # No marker — return last meaningful line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return ""
    # Skip lines that are preamble
    for line in reversed(lines):
        if not re.match(r"^(based on|according to|the answer|looking at)", line, re.IGNORECASE):
            return line
    return lines[-1]
