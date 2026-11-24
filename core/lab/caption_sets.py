# ArtAgent/core/lab/caption_sets.py
"""
Named caption sets: alternative captions for the same images, for caption A/B tests.

`img.txt` holds the active captions (what trainers read by default) and
`img.<set>.txt` holds each alternate set. kohya-ss and musubi-tuner can train on
a set directly with caption_extension = ".<set>.txt". A manifest in
<dataset>/.artagents/caption_sets.json records how each set was made.

Strategies for a Caption Sweep cover open questions from the research report:
trigger-only, tags, short vs long prose, structured assembly, a random-length mix
(Brack et al. 2025), long captions plus trigger-only variants (fal), a 95/5 model /
human blend (DALL-E 3), and quality tags on vs off.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import time
from statistics import mean

from ..dataset import caption_io, snapshots, terms
from ..dataset.audit import load_verification
from ..dataset.snapshots import META_DIR
from ..dataset.tokens import budget_checks, count_tokens

ACTIVE = "(active captions)"
SET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
RESERVED = {"txt", "caption", "captions"}

# id: (label, kind) — kind "model" calls the captioner; "derived" is built from other captions
STRATEGIES: dict[str, tuple[str, str]] = {
    "trigger_only": ("Trigger only (no description)", "derived"),
    "tags": ("Tags (tagger or vision model)", "model"),
    "short_prose": ("Short prose (~15 words)", "model"),
    "long_prose": ("Long prose (~60 words)", "model"),
    "structured": ("Structured fields → caption", "model"),
    "random_length": ("Random-length mix of short / long / trigger only (Brack et al.)", "derived"),
    "long_plus_trigger": ("Long prose + trigger-only line as a variant (fal)", "derived"),
    "blend_95_5": ("95% model captions / 5% current captions (DALL-E 3)", "derived"),
    "quality_on": ("Current captions + quality tags (Pony / Illustrious)", "derived"),
}
STRATEGY_LABELS = {v[0]: k for k, v in STRATEGIES.items()}
WORD_TARGETS = {"short_prose": 15, "long_prose": 60}


class CaptionSetError(Exception):
    pass


def extension(name: str) -> str:
    return f".{name}.txt"


def validate_name(name: str) -> str:
    name = (name or "").strip().lower()
    if not SET_NAME_RE.match(name) or name in RESERVED:
        raise CaptionSetError("Set names use lowercase letters, digits, '-' and '_' (max 40), e.g. 'short-prose'.")
    return name


# ---------- manifest ----------
def _manifest_path(folder: str) -> str:
    return os.path.join(folder, META_DIR, "caption_sets.json")


def load_manifest(folder: str) -> dict:
    try:
        with open(_manifest_path(folder), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {"sets": {}}
    except (OSError, ValueError):
        return {"sets": {}}


def _save_manifest(folder: str, data: dict) -> None:
    path = _manifest_path(folder)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    caption_io.write_text_atomic(path, json.dumps(data, indent=2, ensure_ascii=False))


def record_set(folder: str, name: str, settings: dict, description: str = "") -> None:
    data = load_manifest(folder)
    sets = data.setdefault("sets", {})
    entry = sets.get(name, {"created": time.strftime("%Y-%m-%d %H:%M:%S")})
    entry.update({"updated": time.strftime("%Y-%m-%d %H:%M:%S"), "settings": settings})
    if description:
        entry["description"] = description
    sets[name] = entry
    _save_manifest(folder, data)


def discover_sets(image_paths: dict[str, str]) -> list[str]:
    """Set names present on disk (img.<set>.txt next to an image) or in the manifest."""
    if not image_paths:
        return []
    folder = os.path.dirname(next(iter(image_paths.values())))
    stems = {os.path.splitext(os.path.basename(p))[0] for p in image_paths.values()}
    found = set(load_manifest(folder).get("sets", {}))
    try:
        for f in os.listdir(folder):
            if not f.endswith(".txt") or f.count(".") < 2:
                continue
            stem, name = f[:-4].rsplit(".", 1)
            if stem in stems and SET_NAME_RE.match(name) and f[:-4] not in stems:  # 'a.b.txt' of image 'a.b.png' is no set
                found.add(name)
    except OSError:
        pass
    return sorted(found)


def _image_stems(image_paths: dict[str, str]) -> set[str]:
    return {os.path.splitext(os.path.basename(p))[0] for p in image_paths.values()}


def collisions(image_paths: dict[str, str], name: str) -> list[str]:
    """Images whose set file img.<set>.txt would be another image's active caption (e.g. 'a.png' + set 'b' vs 'a.b.png')."""
    stems = _image_stems(image_paths)
    return sorted(f for f, p in image_paths.items() if f"{os.path.splitext(os.path.basename(p))[0]}.{name}" in stems)


def read_set(image_paths: dict[str, str], name: str) -> dict[str, str]:
    """{filename: caption} for images that have a caption in the set (ACTIVE = the .txt files)."""
    ext = ".txt" if name in (None, "", ACTIVE) else extension(name)
    blocked = set(collisions(image_paths, name)) if ext != ".txt" else set()
    out = {}
    for fname, path in image_paths.items():
        if fname not in blocked and os.path.exists(caption_io.caption_path(path, ext)):
            out[fname] = caption_io.read_caption(path, ext)
    return out


def write_set(image_paths: dict[str, str], name: str, captions: dict[str, str], settings: dict,
              description: str = "") -> int:
    name = validate_name(name)
    clash = collisions(image_paths, name)
    if clash:
        raise CaptionSetError(f"Set name '{name}' clashes with image file names ({', '.join(clash[:3])}): "
                              "its files would overwrite another image's caption. Choose another name.")
    n = 0
    for fname, text in captions.items():
        if fname in image_paths and text is not None:
            caption_io.write_caption(image_paths[fname], text, extension(name))
            n += 1
    record_set(os.path.dirname(next(iter(image_paths.values()))), name, settings, description)
    return n


def save_active_as_set(image_paths: dict[str, str], captions: dict[str, str], name: str, description: str = "") -> int:
    current = {f: t for f, t in (captions or {}).items() if (t or "").strip()}
    if not current:
        raise CaptionSetError("There are no captions to save.")
    return write_set(image_paths, name, current, {"strategy": "copy of the active captions"}, description)


def activate_set(image_paths: dict[str, str], name: str) -> tuple[dict[str, str], str, int]:
    """
    Makes a set the active captions (img.<set>.txt → img.txt), snapshot first.
    Returns (new active captions for the changed images, snapshot id, images without a caption in the set).
    """
    name = validate_name(name)
    folder = os.path.dirname(next(iter(image_paths.values())))
    content = read_set(image_paths, name)
    if not content:
        raise CaptionSetError(f"Set '{name}' has no caption files.")
    snap = snapshots.create_snapshot(folder, image_paths, f"before activating set {name}")
    for fname, text in content.items():
        caption_io.write_caption(image_paths[fname], text)
    data = load_manifest(folder)
    data["active_from"] = {"set": name, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save_manifest(folder, data)
    return content, snap, len(image_paths) - len(content)


def delete_set(image_paths: dict[str, str], name: str) -> str:
    """Moves the set's files to .artagents/deleted_sets/<name>-<time>/ (not deleted). Returns that folder."""
    name = validate_name(name)
    folder = os.path.dirname(next(iter(image_paths.values())))
    base = os.path.join(folder, META_DIR, "deleted_sets", f"{name}-{time.strftime('%Y%m%d-%H%M%S')}")
    target, n = base, 2
    while os.path.exists(target):  # Never mix two deletions in one folder
        target, n = f"{base}-{n}", n + 1
    os.makedirs(target)
    for path in image_paths.values():
        src = caption_io.caption_path(path, extension(name))
        if os.path.exists(src):
            shutil.move(src, os.path.join(target, os.path.basename(src)))
    data = load_manifest(folder)
    entry = data.get("sets", {}).pop(name, None)
    if entry:
        with open(os.path.join(target, "manifest_entry.json"), "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)
    _save_manifest(folder, data)
    return target


# ---------- statistics ----------
STAT_HEADERS = ["Set", "Captions", "Avg tokens", "Max tokens", "Over budget %", "Avg words", "Trigger %",
                "Distinct terms", "Several lines", "Verifier disputed"]


def set_stats(image_paths: dict[str, str], name: str, budgets: list | None = None, trigger: str = "",
              term_mode: str = terms.MODE_WORDS) -> list:
    caps = read_set(image_paths, name)
    if not caps:
        return [name or ACTIVE, 0, "", "", "", "", "", "", "", ""]
    firsts = {f: (caption_io.split_variants(t) or [""])[0] for f, t in caps.items()}
    tokenizer = (budgets or [{}])[0].get("tokenizer", "clip") if budgets else "clip"
    tokens = [count_tokens(t, tokenizer)[0] for t in firsts.values()]
    over = sum(1 for t in firsts.values() for c in budget_checks(t, budgets or [])[:1]
               if c["status"] in ("over_budget", "over_max"))
    words = [len(re.findall(r"\w+", t)) for t in firsts.values()]
    trig = (trigger or "").strip().lower()
    trigger_pct = round(100 * sum(1 for t in caps.values() if trig and trig in t.lower()) / len(caps), 1) if trig else ""
    distinct = len(terms.term_table(firsts, term_mode, [trigger]) if firsts else [])
    multi = sum(1 for t in caps.values() if caption_io.variant_count(t) > 1)
    checked = disputed = 0
    for f, text in caps.items():
        data = load_verification(image_paths[f])
        if data and data.get("caption_hash") == hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]:
            checked += 1
            disputed += int(bool(data.get("disputed")))
    return [name or ACTIVE, len(caps), round(mean(tokens), 1), max(tokens),
            round(100 * over / len(caps), 1) if budgets else "", round(mean(words), 1), trigger_pct, distinct, multi,
            f"{disputed}/{checked}" if checked else "—"]


def length_histogram(image_paths: dict[str, str], name: str, bins: tuple[int, ...] = (10, 20, 40, 75, 150, 225)) -> list[list]:
    """Rows [token range, captions] for the set (CLIP tokens of the first line)."""
    caps = read_set(image_paths, name)
    counts = [0] * (len(bins) + 1)
    for text in caps.values():
        n = count_tokens((caption_io.split_variants(text) or [""])[0], "clip")[0]
        idx = next((i for i, b in enumerate(bins) if n <= b), len(bins))
        counts[idx] += 1
    labels = [f"≤{bins[0]}"] + [f"{bins[i - 1] + 1}–{bins[i]}" for i in range(1, len(bins))] + [f">{bins[-1]}"]
    return [[label, c] for label, c in zip(labels, counts)]


# ---------- derived strategies ----------
def derive(strategy: str, image_paths: dict[str, str], active: dict[str, str], sets: dict[str, dict[str, str]],
           prefix: str, quality_prefix: str = "", seed: int = 42) -> dict[str, str]:
    """Builds a derived set from the active captions and already generated sets (see STRATEGIES)."""
    rng = random.Random(seed)
    trigger_line = (prefix or "").strip().rstrip(",").strip()
    names = sorted(image_paths)
    if strategy == "trigger_only":
        if not trigger_line:
            raise CaptionSetError("Trigger-only needs a trigger (prefix) in AI Captioning.")
        return {n: trigger_line for n in names}
    if strategy == "random_length":
        short, long_ = sets.get("short_prose") or {}, sets.get("long_prose") or {}
        if not short or not long_:
            raise CaptionSetError("Random-length mix needs the short and long prose sets (select them too).")
        out = {}
        for n in names:
            choices = [c for c in (short.get(n), long_.get(n), trigger_line or None) if c]
            if choices:
                out[n] = rng.choice(choices)
        return out
    if strategy == "long_plus_trigger":
        long_ = sets.get("long_prose") or {}
        if not long_ or not trigger_line:
            raise CaptionSetError("This strategy needs the long prose set and a trigger.")
        return {n: f"{(caption_io.split_variants(t) or [''])[0]}\n{trigger_line}" for n, t in long_.items()}
    if strategy == "blend_95_5":
        model_set = sets.get("long_prose") or sets.get("short_prose") or sets.get("tags") or {}
        if not model_set:
            raise CaptionSetError("The 95/5 blend needs a model-written set (tags, short or long prose).")
        keep = set(rng.sample(names, max(1, round(0.05 * len(names))))) if names else set()
        return {n: (active.get(n) if n in keep and (active.get(n) or "").strip() else model_set.get(n, ""))
                for n in names if model_set.get(n) or (n in keep and active.get(n))}
    if strategy == "quality_on":
        if not quality_prefix:
            raise CaptionSetError("The target model has no quality tags (Pony / Illustrious / NoobAI have).")
        out = {}
        for n, text in active.items():
            if not text.strip():
                continue
            first, *rest = caption_io.split_variants(text) or [""]
            if quality_prefix.lower() in first.lower():
                out[n] = text
                continue
            if trigger_line and first.lower().startswith(trigger_line.lower()):
                body = first[len(trigger_line):].lstrip(" ,")
                first = f"{trigger_line}, {quality_prefix}" + (f", {body}" if body else "")
            else:
                first = f"{quality_prefix}, {first}"
            out[n] = "\n".join([first] + rest)
        return out
    raise CaptionSetError(f"'{strategy}' is not a derived strategy.")


# ---------- export: materialize variants per trainer ----------
VARIANT_MODES = {
    "lines": "One caption per line (SimpleTuner, OneTrainer)",
    "kohya_multiline": "kohya: ---===--- separators + enable_multiline_caption",
    "duplicate": "Duplicate images, one caption each (kohya with cached text encoder, other trainers)",
}


def materialize(image_paths: dict[str, str], name: str, mode: str) -> tuple[str, int]:
    """
    Writes a training copy of the set into <dataset>/.artagents/export/<set>-<mode>/ (images hard-linked
    when possible, else copied). Returns (folder, files written).
    """
    caps = read_set(image_paths, name)
    if not caps:
        raise CaptionSetError("This set has no captions.")
    folder = os.path.dirname(next(iter(image_paths.values())))
    label = name if name not in (None, "", ACTIVE) else "active"
    out = os.path.join(folder, META_DIR, "export", f"{label}-{mode}")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)

    def place(src: str, dst: str) -> None:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    written = 0
    for fname, text in sorted(caps.items()):
        src = image_paths[fname]
        stem, ext = os.path.splitext(os.path.basename(src))
        variants = caption_io.split_variants(text) or [""]
        if mode == "duplicate":
            for i, v in enumerate(variants):
                base = stem if i == 0 else f"{stem}__v{i + 1}"
                place(src, os.path.join(out, base + ext))
                caption_io.write_text_atomic(os.path.join(out, base + ".txt"), v)
                written += 1
            continue
        place(src, os.path.join(out, stem + ext))
        body = "\n---===---\n".join(variants) if mode == "kohya_multiline" else "\n".join(variants)
        caption_io.write_text_atomic(os.path.join(out, stem + ".txt"), body)
        written += 1
    return out, written
