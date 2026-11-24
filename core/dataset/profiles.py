# ArtAgent/core/dataset/profiles.py
"""
Target-model caption profiles and dataset-type presets (dataset_profiles/*.yaml),
and the captioning instruction / trigger prefix they compose.
"""
from __future__ import annotations

import os

import yaml

from ..utils import get_absolute_path

PROFILES_FILE = os.path.join("dataset_profiles", "models.yaml")
PRESETS_FILE = os.path.join("dataset_profiles", "presets.yaml")

STYLE_TAGS = "tags"
STYLE_PROSE = "prose"
STYLE_LABELS = {STYLE_PROSE: "Natural language", STYLE_TAGS: "Tags"}

EVIDENCE_BADGES = {
    "measured": "🟢 Measured",
    "docs": "🔵 Docs",
    "visual-test": "🟡 Visual test",
    "anecdotal": "🟠 Anecdotal",
    "inference": "⚪ Inference",
}

# Captioning checkboxes (JoyCaption-style extra options). id -> (label, instruction sentence)
CAPTION_OPTIONS = {
    "visible": ("Only clearly visible content", "Only mention things that are clearly visible; do not guess."),
    "no_meta": ("No 'This image shows…' openers", "Do not start with phrases like 'This image shows' or 'In this picture'."),
    "quote_text": ("Quote visible text exactly", "If there is legible text, quote it exactly in double quotes."),
    "no_style": ("Don't mention art style or medium", "Do not mention the art style, medium, technique, brushwork, color palette or artist."),
    "no_judgement": ("No quality judgements", "Do not use subjective words such as beautiful, stunning or high quality."),
    "camera": ("Include camera angle & framing", "Mention the camera angle and framing (e.g. close-up, from below)."),
    "lighting": ("Include lighting", "Mention the lighting."),
}
OPTION_LABEL_TO_ID = {label: oid for oid, (label, _) in CAPTION_OPTIONS.items()}

_DEFAULT_PROFILE = {
    "id": "sd_sdxl", "label": "SD 1.5 / SDXL", "style": STYLE_PROSE, "hybrid": True,
    "budgets": [{"tokenizer": "clip", "budget": 75, "max": 225, "label": "CLIP"}],
    "word_target": 30, "max_output_tokens": 150, "notes": [],
}
_DEFAULT_PRESET = {"id": "general", "label": "General / no preset", "prefix": "{trigger}", "instruction": "",
                   "omit_terms": [], "options": ["visible", "no_meta"], "guidance": []}


def _load_yaml_list(relative_path: str, key: str) -> list[dict]:
    path = get_absolute_path(relative_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Warning: {relative_path} not found; using built-in defaults.")
        return []
    except yaml.YAMLError as e:
        print(f"Warning: could not parse {relative_path}: {e}")
        return []
    items = data.get(key, []) if isinstance(data, dict) else []
    return [item for item in items if isinstance(item, dict) and item.get("id") and item.get("label")]


def load_profiles() -> list[dict]:
    return _load_yaml_list(PROFILES_FILE, "profiles") or [dict(_DEFAULT_PROFILE)]


def load_presets() -> list[dict]:
    return _load_yaml_list(PRESETS_FILE, "presets") or [dict(_DEFAULT_PRESET)]


def find_by_label(items: list[dict], label: str | None, fallback: dict) -> dict:
    for item in items:
        if item.get("label") == label or item.get("id") == label:
            return item
    return items[0] if items else fallback


def get_profile(label_or_id: str | None) -> dict:
    return find_by_label(load_profiles(), label_or_id, _DEFAULT_PROFILE)


def get_preset(label_or_id: str | None) -> dict:
    return find_by_label(load_presets(), label_or_id, _DEFAULT_PRESET)


def style_of(profile: dict, chosen_label: str | None = None) -> str:
    """The caption format in use: the user's choice (label) or the profile default."""
    for style, label in STYLE_LABELS.items():
        if chosen_label == label:
            return style
    return profile.get("style", STYLE_PROSE)


def style_mismatch_warning(profile: dict, style: str) -> str:
    """Civitai-style warning when the caption format doesn't match how the model was trained."""
    if profile.get("hybrid") or style == profile.get("style"):
        return ""
    names = {STYLE_TAGS: "tag", STYLE_PROSE: "natural-language"}
    return (f"⚠️ {profile['label']} was trained on {names.get(profile.get('style'), 'other')} captions; "
            f"{names.get(style, style)} captions may train worse.")


def default_options(profile: dict, preset: dict) -> list[str]:
    """Checkbox labels switched on for this profile + preset."""
    ids = list(preset.get("options", []))
    if profile.get("quote_text") and "quote_text" not in ids:
        ids.append("quote_text")
    if profile.get("include_camera") and "camera" not in ids:
        ids.append("camera")
    return [CAPTION_OPTIONS[i][0] for i in ids if i in CAPTION_OPTIONS]


def compose_prefix(preset: dict, trigger: str, class_noun: str, style: str) -> str:
    """Trigger prefix for captions, e.g. 'txcl painting, ' (tags) or 'txcl painting, ' (prose)."""
    trigger, class_noun = (trigger or "").strip(), (class_noun or "").strip()
    if not trigger and not class_noun:
        return ""
    template = preset.get("prefix") or "{trigger}"
    text = template.replace("{trigger}", trigger).replace("{class}", class_noun if "{class}" in template else "")
    text = " ".join(text.split())
    return f"{text}, " if text else ""


def build_instruction(
    profile: dict,
    preset: dict,
    style: str,
    option_labels: list[str],
    word_target: int | None,
    omit_terms: list[str],
    class_noun: str = "",
    extra: str = "",
) -> str:
    """Composes the instruction sent with every image to the vision model."""
    class_noun = (class_noun or "").strip() or "subject"
    lines = []
    if style == STYLE_TAGS:
        lines.append("Write tags for this image as a single comma-separated line, most important first.")
        if profile.get("tag_hint"):
            lines.append(profile["tag_hint"])
        if profile.get("tag_order"):
            lines.append(f"Order the tags as: {profile['tag_order']}.")
        if profile.get("underscores") is False:
            lines.append("Use spaces instead of underscores inside tags.")
        if word_target:
            lines.append(f"Use about {int(word_target)} tags.")
        lines.append("Output only the tags.")
    else:
        lines.append("Write one caption for this image as a single paragraph of plain natural language.")
        if word_target:
            lines.append(f"Use about {int(word_target)} words.")
        lines.append("Output only the caption.")
    preset_text = (preset.get("instruction") or "").replace("{class}", class_noun)
    if preset_text:
        lines.append(preset_text)
    for label in option_labels or []:
        option_id = OPTION_LABEL_TO_ID.get(label)
        if option_id:
            lines.append(CAPTION_OPTIONS[option_id][1])
    terms = [t.strip() for t in omit_terms or [] if t.strip()]
    if terms:
        lines.append("Do not use these words: " + ", ".join(terms) + ".")
    if extra and extra.strip():
        lines.append(extra.strip())
    return " ".join(lines)


def parse_terms(text: str) -> list[str]:
    """Comma- or newline-separated list -> clean list."""
    if not text:
        return []
    parts = [p.strip() for chunk in text.split("\n") for p in chunk.split(",")]
    return [p for p in parts if p]


def guidance_markdown(profile: dict, preset: dict) -> str:
    """Guidance panel: profile notes and preset rules, each with an evidence badge."""
    def bullets(items):
        out = []
        for item in items or []:
            badge = EVIDENCE_BADGES.get(item.get("evidence", ""), "")
            out.append(f"- {item.get('text', '')} <small>{badge}</small>")
        return "\n".join(out)

    parts = [f"**Target model: {profile['label']}**", bullets(profile.get("notes"))]
    parts.append(f"**Dataset type: {preset['label']}**")
    if preset.get("images"):
        parts.append(f"- Images: {preset['images']}")
    if preset.get("class_examples"):
        parts.append(f"- Class noun examples: {preset['class_examples']}")
    parts.append(bullets(preset.get("guidance")))
    parts.append("<small>🟢 measured in a study · 🔵 official docs/model card · 🟡 side-by-side image test · "
                 "🟠 practitioner advice · ⚪ reasoning from the evidence. Most LoRA caption advice is not measured: "
                 "test it with your own data.</small>")
    return "\n\n".join(p for p in parts if p)
