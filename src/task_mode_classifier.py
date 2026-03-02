"""
task_mode_classifier.py
Deterministic task execution mode detection — zero LLM, zero latency.

Two modes:
  read_only    → task asks for analysis/reporting only; mutation tools removed from tool set
  full_execute → task requires writes; full tool set used (default)

Borrowed concept: ACE's "Observer" pattern applied at PRIME phase instead of during execution.
Key insight: if we remove mutation tools entirely for read-only tasks, Claude physically
cannot call them — no amount of prompt engineering can fix what the tool set prevents.
"""
from __future__ import annotations
import re


# ── Read-only signal patterns ─────────────────────────────────────────────────
# These patterns indicate the task is asking for reporting/analysis, never mutations.
# Conservative: only trigger when the signal is strong and unambiguous.
_READ_ONLY_PATTERNS: list[re.Pattern] = [
    # "generate/create/write/produce/prepare/draft a report/summary/analysis/..."
    re.compile(
        r'\b(generate|create|write|produce|prepare|draft|build)\s+(a\s+|an\s+|the\s+)?'
        r'(report|summary|analysis|review|document|presentation|slide|overview|deck|qbr|assessment)\b',
        re.I,
    ),
    # QBR / quarterly business review
    re.compile(r'\b(qbr|quarterly\s+business\s+review)\b', re.I),
    # Explicit read-only instructions
    re.compile(r'\bdo\s+not\s+(modify|change|update|alter|delete|cancel|create)\b', re.I),
    re.compile(r'\bread[\s-]only\b', re.I),
    re.compile(r'\bfor\s+(review|analysis|reporting)\s+only\b', re.I),
    re.compile(r'\bno\s+(changes?|mutations?|updates?|modifications?)\s+(should|are|to)\s+be\b', re.I),
    # "analyze and report", "review and summarize" — analysis-only verbs with no action verb
    re.compile(r'\b(analyze|assess|review|evaluate|summarize|audit)\s+.*\band\s+(report|summarize|present|document)\b', re.I),
]

# ── Action signal patterns ─────────────────────────────────────────────────────
# Any of these override read-only signals — the task requires mutation actions.
_ACTION_PATTERNS: list[re.Pattern] = [
    re.compile(
        r'\b(approve|reject|cancel|update|modify|post|process|apply|book|schedule|'
        r'assign|revoke|terminate|submit|create|send|escalate|flag|credit|refund|'
        r'dispute|issue|close|open|activate|deactivate|transfer|pay|charge)\b',
        re.I,
    ),
]

# ── Mutation tool name patterns ───────────────────────────────────────────────
# Tools whose names match these prefixes/patterns are treated as write tools.
# Applied only when mode is "read_only" to filter the tool set.
_MUTATION_TOOL_PREFIXES = (
    "update_", "modify_", "cancel_", "delete_", "remove_", "post_",
    "apply_", "approve_", "reject_", "create_", "submit_", "send_",
    "process_", "pay_", "charge_", "credit_", "refund_", "issue_",
    "close_", "open_", "activate_", "deactivate_", "transfer_", "flag_",
    "assign_", "revoke_", "terminate_", "book_", "schedule_", "escalate_",
    "dispute_", "log_", "notify_", "record_",
)

# Tools that are always safe even in read-only mode
_ALWAYS_ALLOW = (
    "confirm_with_user",  # auto-confirms in benchmark, never mutates
    "get_", "fetch_", "list_", "search_", "check_", "calculate_",
    "compute_", "estimate_", "verify_", "look_", "find_", "read_",
    "describe_", "show_", "view_", "query_",
)


def classify_task_mode(task_text: str) -> str:
    """
    Classify the task's execution mode without any LLM call.

    Returns:
        "read_only"    — task is analysis/reporting only; remove mutation tools
        "full_execute" — task requires write actions (default)
    """
    has_read_only_signal = any(p.search(task_text) for p in _READ_ONLY_PATTERNS)
    if not has_read_only_signal:
        return "full_execute"

    # Even with a read-only signal, if there's a strong action verb, it's full_execute
    has_action = any(p.search(task_text) for p in _ACTION_PATTERNS)
    if has_action:
        return "full_execute"

    return "read_only"


def filter_tools_for_mode(tools: list[dict], mode: str) -> list[dict]:
    """
    Filter the tool list based on execution mode.
    In read_only mode, removes mutation tools so Claude physically cannot call them.
    """
    if mode != "read_only":
        return tools

    filtered = []
    for tool in tools:
        name = tool.get("name", "").lower()
        # Keep tools that match always-allow prefixes
        if any(name.startswith(p) for p in _ALWAYS_ALLOW):
            filtered.append(tool)
            continue
        # Remove mutation tools
        if any(name.startswith(p) for p in _MUTATION_TOOL_PREFIXES):
            continue
        # Default: keep (non-matching tools are kept to avoid over-filtering)
        filtered.append(tool)

    return filtered


def build_mode_directive(mode: str) -> str:
    """
    Returns a system prompt directive based on task mode.
    Empty string if mode is full_execute (no constraint needed).
    """
    if mode != "read_only":
        return ""
    return (
        "\n⚠️  TASK MODE: READ-ONLY / REPORT GENERATION\n"
        "This task requires ANALYSIS and REPORTING ONLY.\n"
        "Do NOT call any mutation tools (modify_*, update_*, cancel_*, create_*, post_*, etc.).\n"
        "Use only read/query/calculate tools. Your output is a report/analysis document.\n"
    )
