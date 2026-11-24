# ArtAgent/core/lab/validation.py
"""
Versioned validation prompt sets for testing LoRAs.

Four buckets, following DreamBooth-style evaluation and the research report's protocol:
  training-like  prompts like the training captions (fidelity)
  held-out       the subject/style in new contexts (generalization)
  trigger-less   the held-out prompts without the trigger (leakage: does the LoRA act without it?)
  unrelated      prompts unrelated to the dataset (prior preservation)

Prompts use a [trigger] placeholder. Per-prompt flags follow the kohya / FluxGym sample
prompt syntax (--w --h --d --l --s --n), so an exported file pastes into trainer configs.
Sets are saved as validation_sets/<name>.v<N>.json; saving never overwrites a version.
"""
from __future__ import annotations

import json
import os
import re
import time

from ..utils import get_absolute_path

SETS_DIR = "validation_sets"
BUCKETS = ["training-like", "held-out", "trigger-less", "unrelated"]
TRIGGER = "[trigger]"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,59}$")
HEADERS = ["Bucket", "Prompt", "Negative", "Flags"]

HELD_OUT_TEMPLATES = [
    "{subject} in a snowy forest", "{subject} on a city street at night, neon lights", "{subject} on a beach at sunset",
    "{subject} in a cozy kitchen", "close-up portrait of {subject}", "{subject} on the surface of the moon",
    "{subject} in a rainy alley", "{subject} in a field of sunflowers",
]
STYLE_HELD_OUT = [
    "a lighthouse on a cliff, {subject}", "a portrait of an old fisherman, {subject}", "a bowl of fruit on a table, {subject}",
    "a busy market street, {subject}", "a fox in the snow, {subject}", "a mountain lake at dawn, {subject}",
]
UNRELATED = ["a red sports car on a mountain road", "a bowl of ramen on a wooden table, top view",
             "an astronaut riding a horse in a desert", "a modern living room with large windows",
             "a golden retriever puppy in the grass", "a medieval castle at sunrise"]


class ValidationError(Exception):
    pass


def _dir() -> str:
    path = get_absolute_path(SETS_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def list_sets() -> list[str]:
    names = set()
    for f in os.listdir(_dir()):
        m = re.match(r"^(.*)\.v(\d+)\.json$", f)
        if m:
            names.add(m.group(1))
    return sorted(names)


def list_versions(name: str) -> list[int]:
    versions = []
    for f in os.listdir(_dir()):
        m = re.match(r"^(.*)\.v(\d+)\.json$", f)
        if m and m.group(1) == name:
            versions.append(int(m.group(2)))
    return sorted(versions)


def load(name: str, version: int | None = None) -> dict:
    versions = list_versions(name)
    if not versions:
        raise ValidationError(f"No validation set named '{name}'.")
    v = int(version) if version else versions[-1]
    with open(os.path.join(_dir(), f"{name}.v{v}.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def parse_seeds(text: str) -> list[int]:
    seeds = []
    for part in re.split(r"[,\s]+", text or ""):
        if part.strip():
            try:
                seeds.append(int(part))
            except ValueError as e:
                raise ValidationError(f"Not a seed: {part!r}") from e
    return seeds


def clean_rows(rows: list[list]) -> list[dict]:
    prompts = []
    for row in rows or []:
        cells = [("" if c is None else str(c)).strip() for c in list(row) + ["", "", "", ""]][:4]
        bucket, prompt, negative, flags = cells
        if not prompt:
            continue
        if bucket not in BUCKETS:
            bucket = "held-out"
        prompts.append({"bucket": bucket, "prompt": prompt, "negative": negative, "flags": flags})
    return prompts


def save_new_version(name: str, rows: list[list], seeds_text: str, trigger: str, defaults: dict, notes: str = "") -> int:
    name = (name or "").strip()
    if not NAME_RE.match(name):
        raise ValidationError("Give the set a name (letters, digits, spaces, '-' or '_').")
    prompts = clean_rows(rows)
    if not prompts:
        raise ValidationError("The set has no prompts.")
    seeds = parse_seeds(seeds_text)
    if not seeds:
        raise ValidationError("Add at least one seed.")
    version = (list_versions(name) or [0])[-1] + 1
    data = {"name": name, "version": version, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "trigger": trigger or "",
            "seeds": seeds, "defaults": defaults, "prompts": prompts, "notes": notes}
    path = os.path.join(_dir(), f"{name}.v{version}.json")
    with open(path, "x", encoding="utf-8") as f:  # Never overwrite a version
        json.dump(data, f, indent=2, ensure_ascii=False)
    return version


def to_rows(data: dict) -> list[list[str]]:
    return [[p.get("bucket", ""), p.get("prompt", ""), p.get("negative", ""), p.get("flags", "")] for p in data.get("prompts", [])]


def render(prompt: str, trigger: str) -> str:
    """Replaces [trigger]; tidies the separators left when the trigger is empty."""
    text = prompt.replace(TRIGGER, (trigger or "").strip())
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+,", ",", text)
    return text.strip(" ,")


def default_rows(trigger: str, class_noun: str, captions: dict[str, str] | None, style_lora: bool = False,
                 per_bucket: int = 6) -> list[list[str]]:
    """Starter prompts for all four buckets (training-like ones come from the dataset's captions)."""
    trig = (trigger or "").strip()
    cls = (class_noun or "").strip()
    rows: list[list[str]] = []
    for caption in sorted((captions or {}).values())[:per_bucket]:
        first = caption.splitlines()[0].strip() if caption.strip() else ""
        if not first:
            continue
        trig_re = r"(?<!\w)" + re.escape(trig) + r"(?!\w)" if trig else None  # Whole word: 'art' not in 'smart'
        if trig_re and re.search(trig_re, first, re.IGNORECASE):
            first = re.sub(trig_re, TRIGGER, first, count=1, flags=re.IGNORECASE)
        else:
            first = f"{TRIGGER}, {first}"
        rows.append(["training-like", first, "", ""])
    subject = f"{TRIGGER} {cls}".strip() if cls else TRIGGER
    plain = cls or "a person"
    templates = STYLE_HELD_OUT if style_lora else HELD_OUT_TEMPLATES
    for t in templates[:per_bucket]:
        rows.append(["held-out", t.format(subject=f"{subject} style" if style_lora else subject), "", ""])
    for t in templates[:per_bucket]:
        rows.append(["trigger-less", t.format(subject=f"{cls} style" if style_lora and cls else ("" if style_lora else plain)).strip(" ,"), "", ""])
    for p in UNRELATED[:per_bucket]:
        rows.append(["unrelated", p, "", ""])
    return rows


DRAFT_SYSTEM = ("You write evaluation prompts for testing a LoRA (a fine-tuned image model). Each prompt places the "
                "trained subject or style in a new context that is NOT in the training captions. Use the literal "
                "token [trigger] where the trigger word goes. One prompt per line, no numbering, no commentary.")


def draft_prompt(captions: dict[str, str], class_noun: str, count: int, style_lora: bool) -> str:
    sample = "\n".join(f"- {c.splitlines()[0][:200]}" for c in sorted(captions.values())[:12] if c.strip())
    kind = "style" if style_lora else "subject"
    return (f"The LoRA teaches a {kind}" + (f" (class: {class_noun})" if class_noun else "") + ". Training captions:\n"
            f"{sample}\n\nWrite {count} new prompts with [trigger] that test the {kind} in varied, new settings, "
            "lighting and compositions (DreamBooth-style recontextualization).")


def parse_draft(text: str, limit: int) -> list[str]:
    prompts = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip().strip('"')
        if not line or line.endswith(":") or TRIGGER not in line:
            continue
        if line not in prompts:
            prompts.append(line)
    return prompts[:limit]


def flags_for(defaults: dict, seed: int | None = None, negative: str = "", extra: str = "") -> str:
    parts = []
    if negative:
        parts.append(f"--n {negative}")
    for flag, key in (("--w", "width"), ("--h", "height"), ("--l", "cfg"), ("--s", "steps")):
        if defaults.get(key):
            parts.append(f"{flag} {defaults[key]}")
    if seed is not None:
        parts.append(f"--d {seed}")
    if extra:
        parts.append(extra)
    return " ".join(parts)


def kohya_sample_prompts(data: dict, buckets: list[str] | None = None) -> str:
    """kohya / FluxGym sample prompt file: one line per prompt and seed."""
    lines = [f"# {data.get('name')} v{data.get('version')} (ArtAgents) – trigger: {data.get('trigger') or '(none)'}"]
    defaults = data.get("defaults", {})
    for p in data.get("prompts", []):
        if buckets and p["bucket"] not in buckets:
            continue
        text = render(p["prompt"], data.get("trigger", ""))
        negative = p.get("negative") or defaults.get("negative", "")
        own_seed = "seed" in parse_flags(p.get("flags", ""))
        for seed in ([None] if own_seed else data.get("seeds", [])):  # A prompt's own --d wins (as in the test sweep)
            lines.append(f"{text} {flags_for(defaults, seed, negative, p.get('flags', ''))}".strip())
    return "\n".join(lines) + "\n"


def parse_flags(flags: str) -> dict:
    """'--w 832 --h 1216 --l 4.5 --s 28 --d 7' → {'width': 832, ...} (unknown flags ignored)."""
    out = {}
    for flag, value in re.findall(r"--([whlsdn])\s+([^\s-][^\s]*)", flags or ""):
        key = {"w": "width", "h": "height", "l": "cfg", "s": "steps", "d": "seed"}.get(flag)
        if not key:
            continue
        try:
            out[key] = float(value) if key == "cfg" else int(value)
        except ValueError:
            continue
    return out


def image_count(n_prompts: int, n_seeds: int, n_loras: int, n_strengths: int, baseline: bool = True) -> int:
    return n_prompts * n_seeds * (max(0, n_loras) * max(0, n_strengths) + (1 if baseline else 0))
