"""
officeqa_tools.py — Treasury Bulletin Retrieval Tools for OfficeQA
===================================================================
Self-contained module: all OfficeQA-specific corpus access lives here.
Zero imports from or modifications to any existing Purple Agent module.

The OfficeQA corpus is the databricks/officeqa Treasury Bulletins dataset:
  • 697 parsed TXT files, 1939–2025, ~200MB total
  • File naming: treasury_bulletin_{YEAR}_{MONTH_NUM}.txt
  • RAW URL: https://raw.githubusercontent.com/databricks/officeqa/main/
             treasury_bulletins_parsed/{filename}

Tools exported (used by officeqa_executor.py):
  fetch_bulletin(year, month_num)     → full bulletin text
  search_bulletin(content, keywords)  → relevant excerpts
  parse_source_files(source_files)    → list of filenames from task field
  build_bulletin_context(filenames)   → combined context for Claude
"""

from __future__ import annotations

import asyncio
import httpx
import re
from functools import lru_cache
from typing import Optional

# ── Corpus constants ──────────────────────────────────────────────────────────

CORPUS_BASE = (
    "https://raw.githubusercontent.com/databricks/officeqa/main"
    "/treasury_bulletins_parsed"
)

# Month name → zero-padded number (for question parsing)
MONTH_NAMES: dict[str, str] = {
    "january": "01",  "jan": "01",
    "february": "02", "feb": "02",
    "march": "03",    "mar": "03",
    "april": "04",    "apr": "04",
    "may": "05",
    "june": "06",     "jun": "06",
    "july": "07",     "jul": "07",
    "august": "08",   "aug": "08",
    "september": "09","sep": "09", "sept": "09",
    "october": "10",  "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}

# Max chars per bulletin to inject (keep LLM context manageable)
MAX_BULLETIN_CHARS = 50_000

# Max total context across all source bulletins for one question
MAX_TOTAL_CHARS = 100_000

# ── In-process cache (avoids re-downloading same bulletin) ────────────────────

_bulletin_cache: dict[str, str] = {}


async def fetch_bulletin(year: int, month_num: str) -> str:
    """
    Fetch a Treasury Bulletin by year and month number (e.g. '06' for June).
    Returns full text content, truncated to MAX_BULLETIN_CHARS.
    Returns empty string on fetch failure.
    """
    filename = f"treasury_bulletin_{year}_{month_num.zfill(2)}.txt"
    cache_key = filename

    if cache_key in _bulletin_cache:
        return _bulletin_cache[cache_key]

    url = f"{CORPUS_BASE}/{filename}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                text = resp.text
                if len(text) > MAX_BULLETIN_CHARS:
                    text = text[:MAX_BULLETIN_CHARS]
                _bulletin_cache[cache_key] = text
                print(f"[officeqa-tools] fetched {filename} ({len(text):,} chars)", flush=True)
                return text
            else:
                print(
                    f"[officeqa-tools] fetch failed {filename}: HTTP {resp.status_code}",
                    flush=True,
                )
                return ""
    except Exception as exc:
        print(f"[officeqa-tools] fetch error {filename}: {exc}", flush=True)
        return ""


def parse_source_files(source_files: str) -> list[str]:
    """
    Parse the source_files field from the OfficeQA CSV.
    Returns list of bulletin filenames (e.g. ['treasury_bulletin_1962_06.txt']).
    Handles comma-separated or space-separated values.
    """
    if not source_files or not source_files.strip():
        return []
    # Split on commas, semicolons, or whitespace
    parts = re.split(r"[,;\s]+", source_files.strip())
    return [p.strip() for p in parts if p.strip().endswith(".txt")]


def parse_filename_to_year_month(filename: str) -> tuple[int, str] | None:
    """
    Parse 'treasury_bulletin_1962_06.txt' → (1962, '06').
    Returns None if filename doesn't match expected pattern.
    """
    m = re.match(r"treasury_bulletin_(\d{4})_(\d{2})\.txt", filename)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def extract_year_month_from_question(question: str) -> list[tuple[int, str]]:
    """
    Heuristic: extract (year, month_num) pairs from question text.
    Examples:
      "In the June 1962 bulletin..." → [(1962, '06')]
      "...the March 1977 Treasury Bulletin..." → [(1977, '03')]
      "...in fiscal year 1965..." → [(1965, '01')] with fallback month

    Used as fallback when source_files is not provided.
    """
    results: list[tuple[int, str]] = []

    # Pattern: "Month YYYY" or "YYYY Month"
    month_year_pattern = re.compile(
        r"\b(" + "|".join(MONTH_NAMES.keys()) + r")\s+(\d{4})\b",
        re.IGNORECASE,
    )
    year_month_pattern = re.compile(
        r"\b(\d{4})\s+(" + "|".join(MONTH_NAMES.keys()) + r")\b",
        re.IGNORECASE,
    )

    for m in month_year_pattern.finditer(question):
        month_num = MONTH_NAMES[m.group(1).lower()]
        year = int(m.group(2))
        if 1939 <= year <= 2025:
            results.append((year, month_num))

    for m in year_month_pattern.finditer(question):
        year = int(m.group(1))
        month_num = MONTH_NAMES[m.group(2).lower()]
        if 1939 <= year <= 2025:
            results.append((year, month_num))

    # Dedup preserving order
    seen: set[tuple[int, str]] = set()
    unique: list[tuple[int, str]] = []
    for item in results:
        if item not in seen:
            seen.add(item)
            unique.append(item)

    return unique


def search_bulletin(content: str, keywords: list[str], context_lines: int = 10) -> str:
    """
    Search bulletin content for relevant sections containing the given keywords.
    Returns up to (context_lines * 2) lines around each match, deduped.

    Used when the full bulletin is too large and we want to extract
    only the relevant table/paragraph.
    """
    if not content or not keywords:
        return content[:10_000] if content else ""  # fallback: first 10k chars

    lines = content.split("\n")
    total = len(lines)
    relevant_line_sets: set[int] = set()

    for kw in keywords:
        kw_lower = kw.lower()
        for idx, line in enumerate(lines):
            if kw_lower in line.lower():
                # Include surrounding context
                start = max(0, idx - context_lines)
                end   = min(total, idx + context_lines + 1)
                for li in range(start, end):
                    relevant_line_sets.add(li)

    if not relevant_line_sets:
        # No keyword match — return first portion as fallback
        return "\n".join(lines[:200])

    # Sort and collect, adding "..." separators for gaps
    sorted_idxs = sorted(relevant_line_sets)
    chunks: list[str] = []
    prev_end = -1

    for idx in sorted_idxs:
        if prev_end >= 0 and idx > prev_end + 1:
            chunks.append("...")
        chunks.append(lines[idx])
        prev_end = idx

    result = "\n".join(chunks)
    # Cap at MAX_BULLETIN_CHARS
    if len(result) > MAX_BULLETIN_CHARS:
        result = result[:MAX_BULLETIN_CHARS]
    return result


async def build_bulletin_context(
    source_files: str,
    question: str = "",
    search_keywords: list[str] | None = None,
) -> str:
    """
    Build the full bulletin context string for injection into the prompt.

    Strategy:
      1. Parse source_files field → list of filenames
      2. If empty, extract (year, month) from question text
      3. Fetch each bulletin async
      4. If search_keywords provided, extract relevant sections only
      5. Return combined context string

    Returns empty string if no bulletins found.
    """
    # Determine which bulletins to fetch
    filenames = parse_source_files(source_files)

    if not filenames:
        # Fallback: extract from question
        year_months = extract_year_month_from_question(question)
        filenames = [
            f"treasury_bulletin_{y}_{m}.txt"
            for y, m in year_months
        ]

    if not filenames:
        return ""  # no bulletin hint — let Claude answer from its training

    # Fetch concurrently
    parsed = [parse_filename_to_year_month(f) for f in filenames]
    fetch_tasks = [
        fetch_bulletin(ym[0], ym[1])
        for ym in parsed if ym is not None
    ]

    if not fetch_tasks:
        return ""

    contents = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    # Build context
    parts: list[str] = []
    total_chars = 0

    for i, (filename, content) in enumerate(zip(filenames, contents)):
        if isinstance(content, Exception) or not content:
            continue
        if total_chars >= MAX_TOTAL_CHARS:
            break

        allowed = min(len(content), MAX_TOTAL_CHARS - total_chars)
        if search_keywords:
            section = search_bulletin(str(content), search_keywords)
            allowed = min(len(section), MAX_TOTAL_CHARS - total_chars)
            text = section[:allowed]
        else:
            text = str(content)[:allowed]

        parts.append(f"## Source Document: {filename}\n\n{text}")
        total_chars += len(text)

    if not parts:
        return ""

    return (
        "# Treasury Bulletin Documents\n\n"
        + "\n\n---\n\n".join(parts)
    )


# ── Keyword extraction from question ─────────────────────────────────────────

def extract_search_keywords(question: str) -> list[str]:
    """
    Extract meaningful search keywords from the question to focus bulletin search.
    Filters out common words, keeps financial terms, table names, and values.
    """
    # Financial / Treasury-specific terms to always include
    financial_terms = {
        "yield", "rate", "interest", "maturity", "coupon", "price",
        "outstanding", "million", "billion", "public", "debt", "treasury",
        "bills", "bonds", "notes", "securities", "redemption", "issue",
        "total", "amount", "receipts", "expenditures", "surplus", "deficit",
        "balance", "deposit", "savings", "certificates", "series",
    }

    stopwords = {
        "the", "a", "an", "in", "on", "at", "to", "of", "and", "or", "for",
        "with", "what", "which", "who", "how", "when", "where", "was", "were",
        "is", "are", "be", "by", "from", "as", "that", "this", "into", "its",
        "their", "they", "did", "does", "do", "much", "many", "total", "value",
        "bulletin", "according", "per", "table", "figure", "fiscal", "year",
    }

    words = re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())
    keywords = []
    for w in words:
        if w in stopwords:
            continue
        if w in financial_terms or len(w) >= 6:
            keywords.append(w)

    # Also extract any explicit dollar amounts or percentages
    amounts = re.findall(r"\$[\d,\.]+|\d+\.?\d*\s*(?:million|billion|percent|%)", question)
    keywords.extend(amounts)

    # Keep unique, preserve order
    seen: set[str] = set()
    result: list[str] = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            result.append(k)

    return result[:15]  # max 15 keywords
