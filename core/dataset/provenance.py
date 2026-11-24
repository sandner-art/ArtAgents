# ArtAgent/core/dataset/provenance.py
"""
Caption provenance: which model, prompt and settings produced each caption.

Stored in <dataset>/.artagents/provenance/<image stem>.json (the latest record plus
a short history), outside the caption files so trainers never read it.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

from .snapshots import META_DIR

HISTORY_LENGTH = 10


def provenance_path(image_path: str) -> str:
    folder, filename = os.path.split(image_path)
    return os.path.join(folder, META_DIR, "provenance", os.path.splitext(filename)[0] + ".json")


def prompt_hash(*parts: str) -> str:
    digest = hashlib.sha256("\n\x1e\n".join(p or "" for p in parts).encode("utf-8")).hexdigest()
    return digest[:16]


def text_hash(text: str) -> str:
    """Hash of a caption file's text (whitespace-trimmed), to tell whether it changed since."""
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()[:16]


def write_provenance(image_path: str, record: dict) -> str:
    path = provenance_path(image_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = dict(record)
    record.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    existing = read_provenance(image_path)
    history = ([existing["latest"]] if existing.get("latest") else []) + existing.get("history", [])
    data = {"image": os.path.basename(image_path), "latest": record, "history": history[:HISTORY_LENGTH]}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def read_provenance(image_path: str) -> dict:
    path = provenance_path(image_path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def summarize(image_path: str, current_text: str | None = None) -> str:
    """
    One-line summary for the editor, e.g. 'gemma4:e4b · Direct · 2026-11-24 12:00 · 1 retry'.
    With `current_text`, says when the caption file has changed since the model wrote it
    (hand edits, batch edits or a snapshot restore).
    """
    latest = read_provenance(image_path).get("latest")
    if not latest:
        return ""
    if current_text is not None and not (current_text or "").strip():
        return ""  # No caption now (e.g. restored to before the AI run): the record no longer applies.
    bits = [latest.get("model") or "?", latest.get("agent") or "", latest.get("timestamp", "")]
    if latest.get("retries"):
        bits.append(f"{latest['retries']} retr{'y' if latest['retries'] == 1 else 'ies'}")
    if latest.get("issues"):
        bits.append("issues: " + ", ".join(latest["issues"]))
    if current_text is not None and latest.get("caption_hash") and latest["caption_hash"] != text_hash(current_text):
        bits.append("edited since")
    return " · ".join(b for b in bits if b)
