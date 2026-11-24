# ArtAgent/core/dataset/terms.py
"""
Dataset-wide term statistics and bulk edits for captions.

Tags mode splits comma-separated tags; words / phrases modes count content words
or two-word phrases in natural-language captions. Frequencies are per image
(how many captions contain the term), which is what matters for binding: an
uncaptioned element present in most images is learned as part of the trigger,
and a captioned one present in most images gets tied to its word.
"""
from __future__ import annotations

import re

from .caption_io import split_variants
from .guardrails import normalize_tag

MODE_TAGS = "Tags"
MODE_WORDS = "Words"
MODE_PHRASES = "Phrases (2 words)"
TERM_MODES = [MODE_TAGS, MODE_WORDS, MODE_PHRASES]

BINDING_RISK_SHARE = 0.5  # Heuristic, not a measured threshold

_STOPWORDS = set("""
a an the and or but if then than of in on at to from by for with without into onto over under above below between
behind beside near through across along around about against among is are was were be been being has have had
do does did this that these those it its it's there their they them he she his her him its we our you your i my me
as while which who whom whose what where when how all any both each few more most other some such no nor not only
own same so too very can will just also into up down out off again further once here why one two three very
image picture photo shows showing depicts depicting features featuring appears visible seen
""".split())
_WORD_RE = re.compile(r"[^\W\d_][\w'-]*", re.UNICODE)
_CLAUSE_SPLIT_RE = re.compile(r"[,;:!?\n]+|\.(?=\s|$)")


def caption_terms(caption: str, mode: str) -> set[str]:
    """Distinct terms in one caption file (all variants)."""
    terms: set[str] = set()
    for line in split_variants(caption):
        if mode == MODE_TAGS:
            terms.update(normalize_tag(t) for t in line.split(",") if t.strip())
        else:
            words = [w.lower().strip("'-") for w in _WORD_RE.findall(line)]
            content = [w for w in words if w and w not in _STOPWORDS and len(w) > 2]
            if mode == MODE_WORDS:
                terms.update(content)
            else:
                for chunk in _CLAUSE_SPLIT_RE.split(line):  # Phrases never span , ; : . ! ?
                    chunk_words = [w for w in (w.lower().strip("'-") for w in _WORD_RE.findall(chunk)) if w]
                    for a, b in zip(chunk_words, chunk_words[1:]):
                        if a not in _STOPWORDS and b not in _STOPWORDS and len(a) > 2 and len(b) > 2:
                            terms.add(f"{a} {b}")
    terms.discard("")
    return terms


def term_table(captions: dict[str, str], mode: str, protected: list[str] | None = None, limit: int = 500) -> list[list]:
    """
    Rows [term, images, percent, note] sorted by frequency.

    `protected` terms (trigger, class noun) are never flagged as binding risks.
    """
    total = len(captions)
    if not total:
        return []
    counts: dict[str, int] = {}
    for caption in captions.values():
        for term in caption_terms(caption or "", mode):
            counts[term] = counts.get(term, 0) + 1
    protected_set = {normalize_tag(p) for p in protected or [] if p.strip()}
    display = _tag_spellings(captions) if mode == MODE_TAGS else {}
    rows = []
    for term, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]:
        share = n / total
        note = ""
        if term in protected_set:
            note = "trigger/class"
        elif share >= BINDING_RISK_SHARE and total >= 4:
            note = "⚠ over-represented: may bind to the trigger"
        elif n == 1 and total >= 10:
            note = "singleton"
        rows.append([display.get(term, term), n, round(100 * share, 1), note])
    return rows


def _tag_spellings(captions: dict[str, str]) -> dict[str, str]:
    """Most common original spelling of each tag (e.g. 'red_hair' rather than the normalized 'red hair')."""
    spellings: dict[str, dict[str, int]] = {}
    for caption in captions.values():
        for line in split_variants(caption or ""):
            for raw in line.split(","):
                raw = raw.strip()
                if raw:
                    forms = spellings.setdefault(normalize_tag(raw), {})
                    forms[raw] = forms.get(raw, 0) + 1
    return {key: max(forms.items(), key=lambda kv: kv[1])[0] for key, forms in spellings.items()}


def files_with_term(captions: dict[str, str], term: str, mode: str) -> list[str]:
    key = normalize_tag(term) if mode == MODE_TAGS else term.lower().strip()
    return sorted(name for name, caption in captions.items() if key in caption_terms(caption or "", mode))


def common_terms(captions: dict[str, str], filenames: list[str], mode: str) -> list[str]:
    """Terms shared by every caption in `filenames`."""
    sets = [caption_terms(captions.get(name, "") or "", mode) for name in filenames]
    if not sets:
        return []
    shared = set.intersection(*sets)
    return sorted(shared)


def _tag_line_edit(line: str, edit) -> str:
    tags = [t.strip() for t in line.split(",")]
    out, seen = [], set()
    for tag in tags:
        if not tag:
            continue
        new_tag = edit(tag)
        if new_tag is None:
            continue
        key = normalize_tag(new_tag)
        if key in seen:
            continue
        seen.add(key)
        out.append(new_tag)
    return ", ".join(out)


def rename_tag(caption: str, old: str, new: str) -> str:
    """Renames a tag on every variant line (keeps order, removes resulting duplicates)."""
    old_key = normalize_tag(old)
    return "\n".join(_tag_line_edit(line, lambda t: new.strip() if normalize_tag(t) == old_key else t)
                     for line in split_variants(caption))


def delete_tag(caption: str, tag: str) -> str:
    key = normalize_tag(tag)
    lines = (_tag_line_edit(line, lambda t: None if normalize_tag(t) == key else t) for line in split_variants(caption))
    return "\n".join(line for line in lines if line.strip())  # An empty first line would be an empty caption


def replace_text(caption: str, find: str, replace: str, use_regex: bool, whole_word: bool = True,
                 case_sensitive: bool = False) -> str:
    """Find & replace in prose or tags (each variant line separately). Raises re.error for bad patterns."""
    if not find:
        return caption
    pattern = find if use_regex else re.escape(find)
    if whole_word and not use_regex:
        pattern = rf"(?<![\w]){pattern}(?![\w])"
    flags = 0 if case_sensitive else re.IGNORECASE
    regex = re.compile(pattern, flags)
    lines = []
    for line in split_variants(caption):
        new_line = regex.sub(replace, line)
        new_line = re.sub(r"\s+,", ",", re.sub(r",\s*,", ",", new_line))   # tidy empty tags
        new_line = re.sub(r"\s{2,}", " ", new_line).strip().strip(",").strip()
        lines.append(new_line)
    return "\n".join(l for l in lines if l)
