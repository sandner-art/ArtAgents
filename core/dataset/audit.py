# ArtAgent/core/dataset/audit.py
"""
Caption consistency audit (deterministic) and an experimental cross-model verifier.

Audit: contradictory terms, trigger coverage and position, several triggers in one
caption, words the dataset type says to leave out, caption length outliers, and
disagreements with the booru tagger's stored probabilities.

Verifier: a second vision model (ideally from a different family than the
captioner) answers yes / no / unsure for each statement in a caption. Two
statements about objects that are not mentioned are mixed in as controls: a
"yes" to a control means the verifier is agreeing too easily on that image.
Results are triage hints stored in <dataset>/.artagents/verify/<stem>.json;
captions are never changed automatically.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from statistics import median

import yaml

from ..utils import get_absolute_path
from . import caption_io, tagger
from .guardrails import find_terms, normalize_tag
from .snapshots import META_DIR
from .vision import VisionError, VisionTarget, ask, parse_json_object

CONTRADICTIONS_FILE = os.path.join("dataset_profiles", "contradictions.yaml")
DISAGREE_BELOW = 0.05      # Caption tag the tagger scored below this
MISSING_ABOVE = 0.85       # Tagger tag at or above this that the caption lacks
CONTROL_OBJECTS = ["a giraffe", "a bicycle", "a red umbrella", "a clock tower", "a sailboat", "a pineapple",
                   "a fire hydrant", "a violin", "a snowman", "a hot air balloon", "a lighthouse", "a cactus"]
MAX_CLAIMS = 20

CHECK_CONTRADICTION = "contradictory terms"
CHECK_NO_TRIGGER = "trigger missing"
CHECK_TRIGGER_TWICE = "trigger repeated"
CHECK_TRIGGER_POSITION = "trigger not first"
CHECK_SEVERAL_TRIGGERS = "several triggers"
CHECK_OMITTED = "should be left out"
CHECK_LONG = "much longer than usual"
CHECK_SHORT = "much shorter than usual"
CHECK_TAGGER_DISAGREES = "tagger disagrees"
CHECK_TAGGER_MISSING = "tagger sees more"
CHECK_COUNT = "person count differs"


def load_contradictions() -> list[list[str]]:
    try:
        with open(get_absolute_path(CONTRADICTIONS_FILE), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        print(f"Warning: could not read {CONTRADICTIONS_FILE}: {e}")
        return []
    groups = data.get("groups", []) if isinstance(data, dict) else []
    return [[str(t) for t in g] for g in groups if isinstance(g, list) and len(g) > 1]


def _present(text: str, terms: list[str], style: str) -> list[str]:
    found = find_terms(text, terms, style)
    if style != "tags":  # In prose 'very long hair' also contains 'long hair'
        found = [t for t in found if not any(t != u and t.lower() in u.lower() for u in found)]
    return found


def _word_count(text: str, style: str) -> int:
    if style == "tags":
        return len([t for t in text.split(",") if t.strip()])
    return len(re.findall(r"\w+", text))


def audit(captions: dict[str, str], image_paths: dict[str, str], style: str = "prose", triggers: list[str] | None = None,
          omit_terms: list[str] | None = None, tagger_label: str | None = None) -> dict:
    """Returns {"rows": [[image, check, detail]], "summary": markdown}."""
    rows: list[list[str]] = []
    triggers = [t.strip() for t in triggers or [] if t.strip()]
    groups = load_contradictions()
    lengths = {}
    trigger_hits = {t: 0 for t in triggers}
    tcfg = tagger.get_tagger(tagger_label) if tagger_label else None
    vocab = tagger.vocabulary(tcfg) if tcfg else None
    compared = 0

    for name in sorted(image_paths):
        text = (captions.get(name) or "").strip()
        if not text:
            continue
        lines = caption_io.split_variants(text)
        first = lines[0] if lines else text
        lengths[name] = _word_count(first, style)

        for group in groups:
            found = _present(first, group, style)
            if len(found) > 1:
                rows.append([name, CHECK_CONTRADICTION, " + ".join(found)])

        found_triggers = []
        for trig in triggers:
            n = len(re.findall(r"(?<!\w)" + re.escape(trig) + r"(?!\w)", first, re.IGNORECASE))
            if n:
                found_triggers.append(trig)
                trigger_hits[trig] += 1
                if n > 1:
                    rows.append([name, CHECK_TRIGGER_TWICE, f"'{trig}' ×{n}"])
                if not first.lower().startswith(trig.lower()):
                    rows.append([name, CHECK_TRIGGER_POSITION, f"'{trig}' is not at the start (keep_tokens / shuffling expect it first)"])
        if triggers and not found_triggers:
            rows.append([name, CHECK_NO_TRIGGER, ", ".join(triggers)])
        if len(found_triggers) > 1:
            rows.append([name, CHECK_SEVERAL_TRIGGERS, " + ".join(found_triggers) + " (fine for group images; check for leakage)"])

        if omit_terms:
            bad = _present(first, omit_terms, style)
            if bad:
                rows.append([name, CHECK_OMITTED, ", ".join(bad)])

        if tcfg and style == "tags" and name in image_paths:
            data = tagger.load_sidecar(image_paths[name])
            if data.get("model") == tcfg.get("id") and data.get("tags"):
                compared += 1
                rows += _tagger_rows(name, first, tagger.sidecar_tags(data), vocab, triggers)

    if len(lengths) >= 5:
        mid = median(lengths.values())
        for name, n in sorted(lengths.items()):
            if mid and n > 2 * mid:
                rows.append([name, CHECK_LONG, f"{n} vs median {mid:g}"])
            elif mid and n < 0.5 * mid:
                rows.append([name, CHECK_SHORT, f"{n} vs median {mid:g}"])

    counts: dict[str, int] = {}
    for _, check, _ in rows:
        counts[check] = counts.get(check, 0) + 1
    captioned = len(lengths)
    lines = [f"**Audited {captioned} caption(s); {len({r[0] for r in rows})} with something to check.**"]
    for trig, hits in trigger_hits.items():
        lines.append(f"- Trigger '{trig}': in {hits}/{captioned} caption(s)")
    lines += [f"- {check}: {n}" for check, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    if tcfg and style == "tags":
        if compared:
            lines.append(f"\nCompared with {tcfg['label']} for {compared} image(s)"
                         + ("" if vocab else " (tagger vocabulary not on disk: only stored probabilities were used)") + ".")
        else:
            lines.append(f"\nNo tagger results for {tcfg['label']} yet: run the tagger to compare captions with it.")
    lines.append("\n_Contradictions and length outliers are hints: multi-person images and deliberate caption "
                 "variety can be correct._")
    return {"rows": rows, "summary": "\n".join(lines)}


def _tagger_rows(name: str, caption: str, probs: dict, vocab: dict | None, triggers: list[str]) -> list[list[str]]:
    rows = []
    by_norm = {normalize_tag(t): p for t, (p, _) in probs.items()}
    cats = {normalize_tag(t): c for t, (_, c) in probs.items()}
    caption_tags = [normalize_tag(t) for t in caption.split(",") if t.strip()]
    skip = {normalize_tag(t) for t in triggers}
    disagree = []
    for tag in caption_tags:
        if tag in skip:
            continue
        known = tag in by_norm or (vocab is not None and tag in vocab)
        p = by_norm.get(tag, 0.0)
        if known and p < DISAGREE_BELOW and cats.get(tag, (vocab or {}).get(tag)) != tagger.CATEGORY_RATING:
            disagree.append(f"{tag} ({p:.2f})" if tag in by_norm else f"{tag} (<{tagger.SIDECAR_FLOOR})")
    if disagree:
        rows.append([name, CHECK_TAGGER_DISAGREES, ", ".join(disagree[:8])])
    present = set(caption_tags)
    counts_caption = [t for t in caption_tags if tagger._is_count(t)]
    counts_tagger = sorted(((t, p) for t, p in by_norm.items() if tagger._is_count(t) and p >= 0.5), key=lambda x: -x[1])
    if counts_caption and counts_tagger and counts_tagger[0][0] not in present:
        rows.append([name, CHECK_COUNT, f"caption: {', '.join(counts_caption)}; tagger: {counts_tagger[0][0]} ({counts_tagger[0][1]:.2f})"])
    missing = [f"{t} ({p:.2f})" for t, p in sorted(by_norm.items(), key=lambda x: -x[1])
               if p >= MISSING_ABOVE and t not in present and cats.get(t) not in (tagger.CATEGORY_RATING,)
               and not tagger._is_count(t)]
    if missing:
        rows.append([name, CHECK_TAGGER_MISSING, ", ".join(missing[:6])])
    return rows


# ---------- verifier ----------
def claims_from_caption(caption: str, style: str, skip_terms: list[str] | None = None) -> list[str]:
    """Atomic statements to check: each tag, or each clause of a prose caption."""
    first = caption_io.split_variants(caption)[0] if caption.strip() else ""
    skip = {normalize_tag(t) for t in skip_terms or [] if t.strip()}
    if style == "tags":
        parts = [t.strip() for t in first.split(",")]
        claims = [t for t in parts if t and normalize_tag(t) not in skip and not normalize_tag(t).startswith(("score ", "source ", "rating "))]
    else:
        clauses = re.split(r"(?<=[.!?;])\s+|,\s+(?=(?:and |with |wearing |holding |in |on |under |behind |near )?\w)", first)
        claims = []
        for c in clauses:
            c = c.strip(" ,.;")
            for s in sorted(skip, key=lambda t: (-len(t), t)):  # Longest first: 'txcl painting' before 'txcl'
                c = re.sub(r"(?i)^" + re.escape(s) + r"[\s,]*", "", c).strip()
            if len(c.split()) >= 2:
                claims.append(c)
    seen, out = set(), []
    for c in claims:
        if c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out[:MAX_CLAIMS]


def control_claims(caption: str, key: str, n: int = 2) -> list[str]:
    """Absent-object statements, chosen deterministically per image, that the caption does not mention."""
    lower = caption.lower()
    pool = [o for o in CONTROL_OBJECTS if o.split()[-1] not in lower]
    if not pool:
        return []
    start = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % len(pool)
    return [pool[(start + i) % len(pool)] for i in range(min(n, len(pool)))]


ANSWER_SCHEMA = {"type": "object", "properties": {"answers": {"type": "array", "items": {
    "type": "string", "enum": ["yes", "no", "unsure"]}}}, "required": ["answers"]}


def verify_prompt(statements: list[str]) -> str:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, start=1))
    return ("Look carefully at the image. For each numbered statement, answer \"yes\" if it is clearly true for this "
            "image, \"no\" if it is false, or \"unsure\" if you cannot tell.\n\n"
            f"Statements (each describes something that may or may not be in the image):\n{numbered}\n\n"
            'Answer with JSON only: {"answers": [...]} with exactly one answer per statement, in order.')


def verify(image_path: str, caption: str, img, target: VisionTarget, settings: dict, style: str = "prose",
           skip_terms: list[str] | None = None, seed: int | None = 7) -> dict:
    claims = claims_from_caption(caption, style, skip_terms)
    if not claims:
        raise VisionError("Nothing to verify: the caption is empty or only contains the trigger.")
    controls = control_claims(caption, os.path.basename(image_path))
    statements = [f"The image shows {c}." if style == "tags" else c for c in claims]
    statements += [f"The image shows {c}." for c in controls]
    order = list(range(len(statements)))
    # Interleave controls so they are not always last
    if controls:
        rng = int(hashlib.sha256(image_path.encode("utf-8")).hexdigest()[:8], 16)
        for i, ctrl in enumerate(range(len(claims), len(statements))):
            pos = (rng >> (i * 4)) % (len(claims) + 1)
            order.remove(ctrl)
            order.insert(min(pos, len(order)), ctrl)
    shown = [statements[i] for i in order]
    raw = ask(target, settings, verify_prompt(shown), [img], temperature=0.0, seed=seed,
              max_tokens=40 + 8 * len(shown), json_schema=ANSWER_SCHEMA)
    try:
        answers = [str(a).strip().lower() for a in parse_json_object(raw).get("answers", [])]
    except ValueError as e:
        raise VisionError(f"The verifier did not return valid JSON: {e}") from e
    if len(answers) != len(shown):
        raise VisionError(f"The verifier answered {len(answers)} of {len(shown)} statements.")
    by_index = {idx: answers[pos] for pos, idx in enumerate(order)}
    claim_answers = [{"claim": claims[i], "answer": by_index[i]} for i in range(len(claims))]
    control_answers = [{"claim": controls[j], "answer": by_index[len(claims) + j]} for j in range(len(controls))]
    result = {
        "model": target.label, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "caption_hash": hashlib.sha256(caption.strip().encode("utf-8")).hexdigest()[:16],
        "claims": claim_answers, "controls": control_answers,
        "disputed": [c["claim"] for c in claim_answers if c["answer"] == "no"],
        "unsure": [c["claim"] for c in claim_answers if c["answer"] not in ("yes", "no")],
        "control_failures": sum(1 for c in control_answers if c["answer"] == "yes"),
    }
    save_verification(image_path, result)
    return result


def verification_path(image_path: str) -> str:
    folder, filename = os.path.split(image_path)
    return os.path.join(folder, META_DIR, "verify", os.path.splitext(filename)[0] + ".json")


def save_verification(image_path: str, result: dict) -> None:
    path = verification_path(image_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def load_verification(image_path: str) -> dict:
    try:
        with open(verification_path(image_path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def verification_rows(image_paths: dict[str, str], captions: dict[str, str]) -> list[list]:
    """[image, disputed, unsure, controls failed, verifier, note] for every image with a stored result."""
    rows = []
    for name in sorted(image_paths):
        data = load_verification(image_paths[name])
        if not data:
            continue
        current = hashlib.sha256((captions.get(name) or "").strip().encode("utf-8")).hexdigest()[:16]
        notes = []
        if data.get("control_failures"):
            notes.append("⚠ said yes to an absent object: treat this result with caution")
        if data.get("caption_hash") and data["caption_hash"] != current:
            notes.append("caption changed since")
        rows.append([name, "; ".join(data.get("disputed", [])) or "—", len(data.get("unsure", [])),
                     data.get("control_failures", 0), data.get("model", ""), "; ".join(notes)])
    return rows
