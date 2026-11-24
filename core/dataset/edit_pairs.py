# ArtAgent/core/dataset/edit_pairs.py
"""
Edit-pair datasets (FLUX.1 Kontext, FLUX.2 edit, Qwen-Image-Edit).

A target folder holds the edited results, one or more control folders hold the
originals with the same file names (any image extension). The caption of each
target is the edit instruction; control images get no captions.

Conventions (ai-toolkit Kontext/Qwen-Edit examples, musubi-tuner dataset docs,
BFL's FLUX.2 klein LoRA guide): match by file stem, up to 3 control images for
Qwen-Image-Edit 2509/2511, controls around 1 MP, 5-10 phrasings of the instruction
help generalization.
"""
from __future__ import annotations

import json
import os
import zipfile

from . import caption_io
from .vision import VisionTarget, ask

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
MAX_CONTROLS = 3
ASPECT_TOLERANCE = 0.05

ISSUE_NO_CONTROL = "no matching control image"
ISSUE_ASPECT = "aspect ratio differs from control"
ISSUE_CONTROL_SIZE = "control image size"
ISSUE_NO_INSTRUCTION = "no instruction yet"


def _images_by_stem(folder: str) -> dict[str, str]:
    out = {}
    if not folder or not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(name)
        if ext.lower() in IMAGE_EXTS and os.path.isfile(os.path.join(folder, name)):
            out.setdefault(stem.lower(), os.path.join(folder, name))
    return out


def parse_control_dirs(text: str) -> list[str]:
    return [line.strip().strip('"') for line in (text or "").replace(";", "\n").splitlines() if line.strip()]


def find_pairs(target_dir: str, control_dirs: list[str]) -> dict:
    """
    Returns {"pairs": [{"stem", "target", "controls": [paths or None]}], "orphans": [control files without a target],
             "errors": [str]}.
    """
    errors = []
    if not target_dir or not os.path.isdir(target_dir):
        errors.append(f"Target folder not found: {target_dir}")
    controls = []
    for d in control_dirs:
        if not os.path.isdir(d):
            errors.append(f"Control folder not found: {d}")
        controls.append(_images_by_stem(d))
    if len(control_dirs) > MAX_CONTROLS:
        errors.append(f"{len(control_dirs)} control folders: Qwen-Image-Edit 2509/2511 uses up to {MAX_CONTROLS}.")
    targets = _images_by_stem(target_dir)
    pairs = [{"stem": os.path.splitext(os.path.basename(path))[0], "target": path,
              "controls": [c.get(key) for c in controls]} for key, path in targets.items()]
    orphans = sorted({os.path.basename(p) for c in controls for k, p in c.items() if k not in targets})
    return {"pairs": pairs, "orphans": orphans, "errors": errors}


def _size(path: str) -> tuple[int, int] | None:
    from ..image_utils import open_image
    try:
        with open_image(path) as img:
            return img.size
    except Exception:
        return None


def validate(pairs: list[dict]) -> list[list[str]]:
    """Rows [pair, issue, detail]."""
    rows = []
    for pair in pairs:
        stem = pair["stem"]
        t_size = _size(pair["target"])
        for i, control in enumerate(pair["controls"]):
            label = f"control {i + 1}" if len(pair["controls"]) > 1 else "control"
            if not control:
                rows.append([stem, ISSUE_NO_CONTROL, label])
                continue
            c_size = _size(control)
            if not c_size or not t_size:
                rows.append([stem, "cannot be opened", os.path.basename(control if not c_size else pair["target"])])
                continue
            t_ar, c_ar = t_size[0] / t_size[1], c_size[0] / c_size[1]
            if abs(t_ar - c_ar) / t_ar > ASPECT_TOLERANCE:
                rows.append([stem, ISSUE_ASPECT, f"target {t_size[0]}×{t_size[1]}, {label} {c_size[0]}×{c_size[1]}"])
            mp = c_size[0] * c_size[1] / 1e6
            if mp < 0.5 or mp > 2.5:
                rows.append([stem, ISSUE_CONTROL_SIZE, f"{label} is {mp:.2f} MP (about 1 MP is typical)"])
        if not caption_io.read_caption(pair["target"]).strip():
            rows.append([stem, ISSUE_NO_INSTRUCTION, ""])
    return rows


def instruction_prompt(n_controls: int, paraphrases: int = 0, trigger: str = "", extra: str = "") -> str:
    first = "The first image is the original" if n_controls == 1 else f"The first {n_controls} images are the inputs"
    lines = [f"{first}; the last image is the edited result.",
             "Write an instruction that turns the input into the result, as someone would ask an image editor: "
             "imperative, one sentence, describing only the change (for example: 'Make the car red' or "
             "'Turn the photo into a watercolor painting'). Do not describe the unchanged parts."]
    if paraphrases > 0:
        lines.append(f"Then write {paraphrases} more phrasings of the same instruction with different wording. "
                     "Put each phrasing on its own line, with no numbering.")
    else:
        lines.append("Answer with the instruction only.")
    if trigger:
        lines.append(f"Do not write the word '{trigger}'; it is added automatically.")
    if (extra or "").strip():
        lines.append(extra.strip())
    return "\n".join(lines)


def clean_instruction_lines(text: str, limit: int) -> list[str]:
    import re
    lines = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip()
        line = re.sub(r"(?i)^(?:instruction|phrasing|paraphrase|alternative|version)\s*\d*\s*[:\-–]\s*", "", line)
        line = line.strip().strip('"').strip()
        if not line or line.lower().startswith("here are") or line.endswith(":"):
            continue
        if line.lower() not in (l.lower() for l in lines):
            lines.append(line)
    return lines[:max(1, limit)]


def generate_instruction(pair: dict, target: VisionTarget, settings: dict, paraphrases: int = 0, trigger_prefix: str = "",
                         extra: str = "", temperature: float | None = 0.0, seed: int | None = 42,
                         max_side: int = 1024) -> list[str]:
    """Returns the instruction lines (first = main instruction), with the trigger prefix applied."""
    from ..image_utils import open_image
    images = []
    try:
        for path in [c for c in pair["controls"] if c] + [pair["target"]]:
            img = open_image(path)
            img.load()
            if max_side and max(img.size) > max_side:
                small = img.copy()
                small.thumbnail((max_side, max_side))
                img.close()
                img = small
            images.append(img)
        n_controls = len(images) - 1
        if n_controls < 1:
            raise ValueError("This pair has no control image.")
        trigger = (trigger_prefix or "").strip().strip(",").strip()
        raw = ask(target, settings, instruction_prompt(n_controls, paraphrases, trigger, extra), images,
                  temperature=temperature, seed=seed, max_tokens=60 + 40 * max(0, paraphrases))
    finally:
        for img in images:
            img.close()
    lines = clean_instruction_lines(raw, 1 + max(0, paraphrases))
    if not lines:
        raise ValueError("The model returned no instruction.")
    if trigger_prefix.strip():
        prefix = trigger_prefix.strip()
        sep = "" if prefix.endswith((",", ":")) else ","
        lines = [l if l.lower().startswith(prefix.lower().rstrip(",: ")) else f"{prefix}{sep} {l[0].lower() + l[1:]}"
                 for l in lines]
    return lines


# ---------- export ----------
def metadata_jsonl(pairs: list[dict]) -> str:
    """musubi-tuner JSONL: image_path, control_path (or control_path_0..N), caption (first line)."""
    out = []
    for pair in pairs:
        caption = caption_io.split_variants(caption_io.read_caption(pair["target"]))
        if not caption:
            continue
        controls = [c for c in pair["controls"] if c]
        if not controls:  # Without an input it would train as a plain image
            continue
        row = {"image_path": pair["target"]}
        if len(controls) == 1:
            row["control_path"] = controls[0]
        else:
            row.update({f"control_path_{i}": c for i, c in enumerate(controls)})
        row["caption"] = caption[0]
        out.append(json.dumps(row, ensure_ascii=False))
    return "\n".join(out) + ("\n" if out else "")


def fal_zip(pairs: list[dict], zip_path: str) -> int:
    """fal Qwen-Image-Edit trainer layout: X_start.ext, X_end.ext, X.txt. Returns pairs written."""
    written = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, pair in enumerate(pairs, start=1):
            control = next((c for c in pair["controls"] if c), None)
            if not control:
                continue
            key = f"{i:04d}"
            zf.write(control, f"{key}_start{os.path.splitext(control)[1].lower()}")
            zf.write(pair["target"], f"{key}_end{os.path.splitext(pair['target'])[1].lower()}")
            lines = caption_io.split_variants(caption_io.read_caption(pair["target"]))
            if lines:
                zf.writestr(f"{key}.txt", lines[0])
            written += 1
    return written
