# ArtAgent/core/dataset/tokens.py
"""
Token counts for captions against each target model's text-encoder limits.

CLIP (SD 1.5 / SDXL / Pony / Illustrious / NoobAI) is counted exactly and offline.
T5 and LLM encoders (FLUX, Wan, Qwen-Image...) are *estimated*: their limits
(256-512 tokens) are rarely reached by captions, so an estimate is enough to warn.
"""
from __future__ import annotations

import math
import re

from .clip_tokenizer import clip_token_count

_WORDS = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def estimate_tokens(text: str, tokenizer: str) -> int:
    """Rough token estimate for sentencepiece (t5) and LLM BPE tokenizers."""
    if not text:
        return 0
    words = len(_WORDS.findall(text))
    punct = len(_PUNCT.findall(text))
    per_word = 1.35 if tokenizer == "t5" else 1.3
    return int(math.ceil(words * per_word + punct * 0.6))


def count_tokens(text: str, tokenizer: str) -> tuple[int, bool]:
    """(count, exact). tokenizer: clip | t5 | llm."""
    if tokenizer == "clip":
        return clip_token_count(text), True
    return estimate_tokens(text, tokenizer), False


def budget_checks(text: str, budgets: list[dict]) -> list[dict]:
    """
    One entry per budget: {label, count, exact, budget, max, status}
    status: ok | over_budget (past the soft budget, e.g. CLIP's first 75-token chunk)
            | over_max (past the hard maximum, will be truncated) | unknown (no limit known)
    """
    results = []
    for budget in budgets or []:
        count, exact = count_tokens(text, budget.get("tokenizer", "llm"))
        soft, hard = budget.get("budget"), budget.get("max")
        if soft is None:
            status = "unknown"
        elif hard and count > hard:
            status = "over_max"
        elif count > soft:
            status = "over_max" if not hard else "over_budget"
        else:
            status = "ok"
        results.append({"label": budget.get("label", budget.get("tokenizer", "")), "count": count, "exact": exact,
                        "budget": soft, "max": hard, "status": status})
    return results


def budget_markdown(text: str, budgets: list[dict]) -> str:
    """Compact status line for the caption editor, e.g. 'CLIP 62/75 ✅ · T5 ≈140/512 ✅'."""
    if not text:
        return ""
    parts = []
    for check in budget_checks(text, budgets):
        approx = "" if check["exact"] else "≈"
        if check["status"] == "unknown":
            parts.append(f"{check['label']}: {approx}{check['count']} tokens")
            continue
        icon = {"ok": "✅", "over_budget": "⚠️", "over_max": "⛔"}[check["status"]]
        limit = f"{check['budget']}" + (f" (max {check['max']})" if check["max"] else "")
        note = ""
        if check["status"] == "over_budget":
            note = " uses more than one CLIP chunk"
        elif check["status"] == "over_max":
            note = " will be truncated"
        parts.append(f"{check['label']}: {approx}{check['count']}/{limit} {icon}{note}")
    return " · ".join(parts)
