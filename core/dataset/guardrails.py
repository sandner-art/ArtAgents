# ArtAgent/core/dataset/guardrails.py
"""
Cleaning and checking AI-written captions before they are saved.

Vision models occasionally loop, stop mid-sentence, refuse, or open with filler
("This image shows..."); JoyCaption's author reports a 1.5-3% glitch rate. These
checks catch the common cases so the caption can be retried automatically.
"""
from __future__ import annotations

import re

from .caption_io import single_line

ISSUE_EMPTY = "empty"
ISSUE_REFUSAL = "refusal"
ISSUE_META_OPENER = "meta opener"
ISSUE_REPETITION = "repetition loop"
ISSUE_TRUNCATED = "truncated"
ISSUE_OMITTED_TERM = "omitted term used"
ISSUE_NOT_TAGS = "sentence instead of tags"

RETRYABLE = {ISSUE_EMPTY, ISSUE_REFUSAL, ISSUE_META_OPENER, ISSUE_REPETITION, ISSUE_TRUNCATED, ISSUE_NOT_TAGS}

_META_OPENER = re.compile(
    r"^\s*(?:"
    r"(?:this|the)\s+(?:image|picture|photo(?:graph)?|illustration|artwork|drawing|painting|render(?:ing)?|scene)\s+"
    r"(?:shows|depicts|features|displays|presents|captures|portrays|contains|is\s+of|is\s+an?)"
    r"|in\s+this\s+(?:image|picture|photo(?:graph)?|illustration|artwork|scene)"
    r"|here\s+is\s+(?:a|the)\s+(?:caption|description)"
    r"|caption\s*:"
    r")",
    re.IGNORECASE,
)
_REFUSAL = re.compile(
    r"\b(?:i'?m\s+sorry|i\s+cannot|i\s+can'?t\s+(?:help|assist|describe|provide)|as\s+an\s+ai\b|i\s+am\s+unable|i'?m\s+unable)",
    re.IGNORECASE,
)
_LABEL_PREFIX = re.compile(r"^\s*(?:caption|description|tags)\s*:\s*", re.IGNORECASE)
_WORD = re.compile(r"[\w'-]+", re.UNICODE)


def strip_meta_opener(text: str) -> str:
    """Removes a leading 'Caption:' label and wrapping quotes."""
    text = _LABEL_PREFIX.sub("", text or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        text = text[1:-1].strip()
    return text


def clean_caption(raw: str, style: str) -> str:
    """One clean caption line. style: 'tags' joins lines with ', ', anything else with ' '."""
    joiner = ", " if style == "tags" else " "
    text = strip_meta_opener(single_line(raw, joiner))
    if style == "tags":
        tags = [t.strip().strip(".") for t in re.split(r",", text)]
        seen, unique = set(), []
        for tag in tags:
            key = normalize_tag(tag)
            if tag and key not in seen:
                seen.add(key)
                unique.append(tag)
        text = ", ".join(unique)
    return text


def normalize_tag(tag: str) -> str:
    """Comparable form of a booru tag: lowercase, underscores as spaces, single spaces."""
    return re.sub(r"\s+", " ", (tag or "").replace("_", " ")).strip().lower()


def has_repetition(text: str, style: str) -> bool:
    if style == "tags":
        tags = [normalize_tag(t) for t in text.split(",") if t.strip()]
        return len(tags) >= 6 and len(set(tags)) / len(tags) < 0.5
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < 12:
        return False
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - 3):
        gram = tuple(words[i:i + 4])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= 3:
            return True
    return False


def looks_like_sentence(text: str) -> bool:
    """A tag caption that is really prose: few commas, long comma-free stretches."""
    pieces = [p.strip() for p in text.split(",") if p.strip()]
    longest = max((len(_WORD.findall(p)) for p in pieces), default=0)
    return longest >= 8 or (len(pieces) == 1 and len(_WORD.findall(text)) >= 6)


def looks_truncated(text: str, style: str, max_tokens: int | None) -> bool:
    """Prose that stops without final punctuation after using (almost) the whole token budget."""
    if style == "tags" or not text or not max_tokens:
        return False
    approx_tokens = len(_WORD.findall(text)) * 1.3
    return approx_tokens >= 0.85 * max_tokens and not re.search(r"[.!?\"')\]]\s*$", text)


def find_terms(text: str, terms: list[str], style: str) -> list[str]:
    """Which of `terms` occur in the caption (as tags, or as whole words/phrases in prose)."""
    found = []
    if style == "tags":
        present = {normalize_tag(t) for t in text.split(",")}
        found = [t for t in terms if normalize_tag(t) in present]
    else:
        for term in terms:
            pattern = r"\b" + re.escape(term.strip()).replace(r"\ ", r"[\s_-]+") + r"\b"
            if term.strip() and re.search(pattern, text, re.IGNORECASE):
                found.append(term)
    return found


def remove_tags(text: str, terms: list[str]) -> str:
    """Drops blacklisted tags from a comma-separated tag caption."""
    blocked = {normalize_tag(t) for t in terms if t.strip()}
    kept = [t.strip() for t in text.split(",") if t.strip() and normalize_tag(t) not in blocked]
    return ", ".join(kept)


def check_caption(text: str, style: str, max_tokens: int | None = None, omit_terms: list[str] | None = None) -> list[str]:
    """Returns the list of issues found (empty list = looks fine)."""
    issues = []
    if not text or not text.strip():
        return [ISSUE_EMPTY]
    if _REFUSAL.search(text[:200]):
        issues.append(ISSUE_REFUSAL)
    if _META_OPENER.search(text):
        issues.append(ISSUE_META_OPENER)
    if style == "tags" and looks_like_sentence(text):
        issues.append(ISSUE_NOT_TAGS)
    if has_repetition(text, style):
        issues.append(ISSUE_REPETITION)
    if looks_truncated(text, style, max_tokens):
        issues.append(ISSUE_TRUNCATED)
    if omit_terms and find_terms(text, omit_terms, style):
        issues.append(ISSUE_OMITTED_TERM)
    return issues


def apply_prefix_suffix(text: str, prefix: str, suffix: str) -> str:
    """Adds trigger prefix/suffix verbatim, without doubling a trigger the model already wrote."""
    prefix = prefix or ""
    suffix = suffix or ""
    core = prefix.strip().rstrip(",.").strip()
    if core and text.lower().startswith(core.lower()):
        text = text[len(core):].lstrip(" ,.")
    if suffix.strip() and text.lower().endswith(suffix.strip().lower()):
        suffix = ""
    return f"{prefix}{text}{suffix}"
