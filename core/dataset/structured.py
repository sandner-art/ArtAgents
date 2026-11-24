# ArtAgent/core/dataset/structured.py
"""
Structured captions: the vision model fills named fields (subject, clothing, pose, …)
as JSON; captions are then assembled from the fields with a template.

Fields are stored per image in <dataset>/.artagents/structured/<stem>.json, so
captions can be rebuilt with different fields or templates without calling the
model again. The attribute sheet shows every image's fields side by side and
flags values that are the same in most images (constant values are absorbed by
the trigger unless they are captioned, and captioning them makes them changeable).
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import time
from collections import Counter

from .snapshots import META_DIR

# id: (column label, what the model is asked for)
FIELDS: dict[str, tuple[str, str]] = {
    "subject": ("Subject", "the main subject in a few words (who or what, with count if more than one)"),
    "action": ("Action / pose", "what the subject is doing, body pose"),
    "expression": ("Expression", "facial expression, or empty"),
    "hair": ("Hair", "hair colour, length and style, or empty"),
    "clothing": ("Clothing", "clothing and accessories, or empty"),
    "objects": ("Objects", "other notable objects, comma-separated, or empty"),
    "setting": ("Setting", "location and background"),
    "lighting": ("Lighting", "lighting and time of day"),
    "camera": ("Camera / framing", "shot type and camera angle, e.g. close-up from below"),
    "text": ("Visible text", "legible text quoted exactly, or empty"),
    "style": ("Style / medium", "art style or medium, e.g. watercolor painting, photo"),
}
DEFAULT_FIELDS = ["subject", "action", "expression", "hair", "clothing", "objects", "setting", "lighting", "camera"]
CONSTANT_SHARE = 0.6  # A value in at least this share of images is flagged as constant

# How each field reads inside a prose caption (tags captions use the value as is).
_PROSE_PATTERNS = {"clothing": "wearing {}", "hair": "with {}", "text": 'with the text {}'}


def field_label(field_id: str) -> str:
    return FIELDS.get(field_id, (field_id.title(), ""))[0]


def label_to_id(label: str) -> str:
    for fid, (lab, _) in FIELDS.items():
        if label in (fid, lab):
            return fid
    return label


def json_schema(field_ids: list[str]) -> dict:
    ids = [f for f in field_ids if f in FIELDS] or DEFAULT_FIELDS
    return {"type": "object", "properties": {f: {"type": "string"} for f in ids}, "required": ids}


def instruction(field_ids: list[str], style: str = "prose", extra: str = "") -> str:
    ids = [f for f in field_ids if f in FIELDS] or DEFAULT_FIELDS
    form = "short comma-separated booru-style tags" if style == "tags" else "short plain phrases"
    lines = [f"Describe this image as a JSON object with exactly these keys. Write each value as {form}. "
             "Use an empty string when a key does not apply. Only describe what is clearly visible; do not guess.", ""]
    lines += [f'- "{f}": {FIELDS[f][1]}' for f in ids]
    if (extra or "").strip():
        lines += ["", extra.strip()]
    lines += ["", "Answer with the JSON object only."]
    return "\n".join(lines)


def normalize_fields(data: dict, field_ids: list[str] | None = None) -> dict[str, str]:
    """Coerces model output to {field: str}; lists are joined with ', '."""
    out = {}
    for key, value in (data or {}).items():
        key = str(key).strip().lower()
        if field_ids and key not in field_ids:
            continue
        if isinstance(value, list):
            value = ", ".join(str(v).strip() for v in value if str(v).strip())
        elif value is None:
            value = ""
        elif not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if key != "text":  # Visible text keeps its quotes ("quote text exactly")
            value = value.strip('"').strip()
        if value.lower() in ("none", "n/a", "null", "empty", "-", "not visible", "unknown"):
            value = ""
        out[key] = value
    return out


def default_template(field_ids: list[str], style: str = "tags") -> str:
    """'{subject}, {clothing}, …' for tags; prose joins subject and action into one phrase."""
    ids = list(field_ids)
    if style != "tags" and "subject" in ids and "action" in ids:
        ids.remove("action")
        return ", ".join("{subject} {action}" if f == "subject" else "{" + f + "}" for f in ids)
    return ", ".join("{" + f + "}" for f in ids)


def render(fields: dict[str, str], template: str, style: str = "prose", include: list[str] | None = None) -> str:
    """
    Fills a template such as '{subject}, {clothing}, {setting}'. Fields that are empty or not in
    `include` are dropped together with their separator. In prose, some fields get a short lead-in
    ('wearing …'); in tags, values are split into tags and de-duplicated.
    """
    allowed = set(include) if include is not None else None

    def value(match: re.Match) -> str:
        fid = match.group(1)
        if allowed is not None and fid not in allowed:
            return ""
        v = (fields.get(fid) or "").strip()
        if not v:
            return ""
        if style != "tags" and fid in _PROSE_PATTERNS:
            v = _PROSE_PATTERNS[fid].format(v)
        return v

    text = re.sub(r"\{([a-z_]+)\}", value, template or "")
    parts = [p.strip() for p in re.split(r"\s*,\s*", text)]
    parts = [p for p in parts if p]
    if style == "tags":
        seen, tags = set(), []
        for p in parts:
            if p.lower() not in seen:
                seen.add(p.lower())
                tags.append(p)
        return ", ".join(tags)
    sentence = ", ".join(parts)
    sentence = re.sub(r"\s{2,}", " ", sentence).strip(" ,")
    if sentence and style != "tags":
        sentence = sentence[0].upper() + sentence[1:]
        if not re.search(r"[.!?][\"']?$", sentence):  # 'the text "OPEN".' but not 'it says "Hi."'.
            sentence += "."
    return sentence


# ---------- storage ----------
def fields_path(image_path: str) -> str:
    folder, filename = os.path.split(image_path)
    return os.path.join(folder, META_DIR, "structured", os.path.splitext(filename)[0] + ".json")


def save_fields(image_path: str, fields: dict[str, str], model: str = "", source: str = "model") -> None:
    path = fields_path(image_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"image": os.path.basename(image_path), "fields": fields, "model": model, "source": source,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_record(image_path: str) -> dict:
    try:
        with open(fields_path(image_path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load_fields(image_path: str) -> dict[str, str]:
    try:
        with open(fields_path(image_path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return normalize_fields(data.get("fields") or {}) if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


# ---------- attribute sheet ----------
def sheet_columns(image_paths: dict[str, str]) -> list[str]:
    """Field ids present in any stored record, in FIELDS order."""
    present = set()
    for path in image_paths.values():
        present.update(k for k, v in load_fields(path).items())
    ordered = [f for f in FIELDS if f in present]
    return ordered + sorted(present - set(ordered))


def sheet_rows(image_paths: dict[str, str], columns: list[str]) -> list[list[str]]:
    rows = []
    for name in sorted(image_paths):
        fields = load_fields(image_paths[name])
        if fields:
            rows.append([name] + [fields.get(c, "") for c in columns])
    return rows


def save_sheet(image_paths: dict[str, str], columns: list[str], rows: list[list]) -> int:
    """Writes edited sheet rows back to the per-image JSON files. Returns the number changed."""
    changed = 0
    for row in rows:
        if not row:
            continue
        name = str(row[0])
        if name not in image_paths:
            continue
        old_record = load_record(image_paths[name])
        old = normalize_fields(old_record.get("fields") or {})
        new = {c: ("" if v is None else str(v)).strip() for c, v in zip(columns, row[1:])}
        new = {c: v for c, v in new.items() if c in old or v}  # Empty cells for fields an image never had: no change
        merged = dict(old, **new)
        if merged != old:
            save_fields(image_paths[name], merged, model=old_record.get("model", ""), source="edited")
            changed += 1
    return changed


def field_summary(columns: list[str], rows: list[list]) -> list[list]:
    """[field, filled images, distinct values, most common value, share, note]."""
    out = []
    n = len(rows)
    for i, col in enumerate(columns, start=1):
        values = [str(r[i]).strip().lower() for r in rows if i < len(r) and str(r[i]).strip()]
        if not values:
            out.append([field_label(col), 0, 0, "", 0.0, "empty"])
            continue
        counts = Counter(values)
        top, top_n = counts.most_common(1)[0]
        share = top_n / n if n else 0.0
        note = ""
        if n >= 4 and share >= CONSTANT_SHARE:
            note = "⚠ mostly the same: caption it to keep it changeable, or leave it out so the trigger learns it"
        elif len(counts) == len(values) and len(values) > 3:
            note = "varies in every image"
        out.append([field_label(col), len(values), len(counts), top, round(100 * share, 1), note])
    return out


def sheet_csv(columns: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["image"] + [field_label(c) for c in columns])
    writer.writerows(rows)
    return buf.getvalue()
