# ArtAgent/core/dataset/tagger.py
"""
Booru tagger (WD v3 family and other ONNX taggers) with profile-aware output.

- Models come from Hugging Face (downloaded once into the HF cache) or a local folder.
- Raw probabilities are saved per image in <dataset>/.artagents/tagger/<stem>.json, so
  thresholds can be changed later without running the model again, and the
  consistency audit can compare captions with the tagger.
- Output follows the target profile: tag order, rating tag names, underscores.

onnxruntime is optional: the rest of ArtAgents works without it.
"""
from __future__ import annotations

import csv
import json
import os
import re
import time

import numpy as np
from PIL import Image

from .profiles import _load_yaml_list
from .snapshots import META_DIR

TAGGERS_FILE = "dataset_profiles/taggers.yaml"
SIDECAR_FLOOR = 0.02  # Probabilities below this are not stored
CATEGORY_GENERAL, CATEGORY_CHARACTER, CATEGORY_RATING = 0, 4, 9
RATING_NAMES = ("general", "sensitive", "questionable", "explicit")

DEFAULTS = {"general_threshold": 0.35, "character_threshold": 0.85, "general_mcut": False, "character_mcut": False,
            "hint_threshold": 0.5, "add_rating": False, "split_series": False, "blacklist": []}

# From SmilingWolf's wd-tagger reference app: these keep their underscores.
KAOMOJIS = {"0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<", "3_3", "6_9", ">_o",
            "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||"}

_COUNT_RE = re.compile(r"^(\d+\+?|multiple|no)[ _]?(girls?|boys?|others?|humans)$")
_SERIES_RE = re.compile(r"^(.+?)[ _]\(([^()]+)\)$")

_BUILTIN = [{"id": "wd-swinv2-v3", "label": "WD SwinV2 v3 (general use)", "repo": "SmilingWolf/wd-swinv2-tagger-v3",
             "model_file": "model.onnx", "labels_file": "selected_tags.csv", "license": "Apache-2.0"}]


class TaggerError(Exception):
    """Tagger unavailable or failed; the message is shown in the UI."""


# ---------- registry ----------
def load_taggers() -> list[dict]:
    return _load_yaml_list(TAGGERS_FILE, "taggers") or [dict(t) for t in _BUILTIN]


def tagger_labels() -> list[str]:
    return [t["label"] for t in load_taggers()]


def get_tagger(label_or_id: str | None) -> dict:
    taggers = load_taggers()
    for t in taggers:
        if label_or_id in (t.get("label"), t.get("id")):
            return t
    return taggers[0]


def onnxruntime_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("onnxruntime") is not None


def tagger_info_markdown(label: str) -> str:
    t = get_tagger(label)
    source = t.get("folder") or f"huggingface.co/{t.get('repo', '?')}"
    bits = [f"**{t['label']}** · {source} · license: {t.get('license', 'see model card')}"]
    if t.get("note"):
        bits.append(t["note"])
    if not onnxruntime_available():
        bits.append("⚠️ The tagger needs **onnxruntime**: `pip install onnxruntime` "
                    "(or `onnxruntime-gpu` / `onnxruntime-directml`), then restart ArtAgents.")
    return " · ".join(bits)


# ---------- model files ----------
def resolve_files(cfg: dict) -> tuple[str, str]:
    """Returns (model path, labels path), downloading from Hugging Face if needed."""
    model_file, labels_file = cfg.get("model_file", "model.onnx"), cfg.get("labels_file", "selected_tags.csv")
    if cfg.get("folder"):
        folder = os.path.expanduser(cfg["folder"])
        paths = os.path.join(folder, model_file), os.path.join(folder, labels_file)
        for p in paths:
            if not os.path.isfile(p):
                raise TaggerError(f"Tagger file not found: {p}")
        return paths
    repo = cfg.get("repo")
    if not repo:
        raise TaggerError(f"Tagger '{cfg.get('label')}' has neither 'repo' nor 'folder' in {TAGGERS_FILE}.")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise TaggerError("huggingface_hub is missing (it normally comes with Gradio): pip install huggingface_hub") from e
    try:
        return (hf_hub_download(repo, model_file, revision=cfg.get("revision")),
                hf_hub_download(repo, labels_file, revision=cfg.get("revision")))
    except Exception as e:  # Network, auth (gated repos) or missing file
        raise TaggerError(f"Could not download {repo} from Hugging Face: {e.__class__.__name__}: {str(e)[:200]}. "
                          f"Check your internet connection, or download {model_file} and {labels_file} into a folder "
                          f"and add it as a local tagger in {TAGGERS_FILE}.") from e


def load_labels(path: str) -> tuple[list[str], np.ndarray]:
    names, categories = [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "name" not in reader.fieldnames:
            raise TaggerError(f"{path} has no 'name' column.")
        for row in reader:
            names.append(row["name"])
            try:
                categories.append(int(row.get("category") or 0))
            except ValueError:
                categories.append(0)
    return names, np.array(categories, dtype=np.int32)


_SESSIONS: dict[str, tuple] = {}


def _providers(ort) -> list[str]:
    available = ort.get_available_providers()
    preferred = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in available]
    return preferred + ["CPUExecutionProvider"]


def load_model(cfg: dict):
    """Cached (session, input name, input size, labels, categories)."""
    model_path, labels_path = resolve_files(cfg)
    key = f"{model_path}|{labels_path}"
    if key not in _SESSIONS:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise TaggerError("The tagger needs onnxruntime: pip install onnxruntime (or onnxruntime-gpu / "
                              "onnxruntime-directml), then restart ArtAgents.") from e
        try:
            session = ort.InferenceSession(model_path, providers=_providers(ort))
        except Exception as e:
            raise TaggerError(f"Could not load {model_path}: {e}") from e
        inp = session.get_inputs()[0]
        layout = (cfg.get("layout") or "NHWC").upper()
        shape = inp.shape
        size = cfg.get("size") or (shape[1] if layout == "NHWC" else shape[2])
        if not isinstance(size, int):
            size = 448
        names, categories = load_labels(labels_path)
        _SESSIONS[key] = (session, inp.name, int(size), names, categories)
    return _SESSIONS[key]


# ---------- inference ----------
def preprocess(img: Image.Image, size: int, cfg: dict | None = None) -> np.ndarray:
    """WD reference preprocessing by default: alpha onto white, pad to square, resize, BGR, 0..255 floats."""
    cfg = cfg or {}
    pad = tuple(cfg.get("pad") or (255, 255, 255))
    image = img.convert("RGBA")
    canvas = Image.new("RGBA", image.size, pad + (255,))
    canvas.alpha_composite(image)
    image = canvas.convert("RGB")
    side = max(image.size)
    padded = Image.new("RGB", (side, side), pad)
    padded.paste(image, ((side - image.size[0]) // 2, (side - image.size[1]) // 2))
    if side != size:
        padded = padded.resize((size, size), Image.BICUBIC)
    arr = np.asarray(padded, dtype=np.float32)  # RGB
    norm = (cfg.get("normalize") or "none").lower()
    if norm in ("unit", "imagenet"):
        arr = arr / 255.0
    if norm == "imagenet":  # Mean/std are in RGB order, so normalize before any channel flip
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    if (cfg.get("channels") or "BGR").upper() == "BGR":
        arr = arr[:, :, ::-1]
    if (cfg.get("layout") or "NHWC").upper() == "NCHW":
        arr = arr.transpose(2, 0, 1)
    return np.ascontiguousarray(np.expand_dims(arr, 0), dtype=np.float32)


def predict(cfg: dict, img: Image.Image) -> dict[str, tuple[float, int]]:
    """{tag name: (probability, category)} for every tag at or above SIDECAR_FLOOR (ratings always)."""
    session, input_name, size, names, categories = load_model(cfg)
    try:
        output = session.run(None, {input_name: preprocess(img, size, cfg)})[0][0]
    except Exception as e:
        raise TaggerError(f"Tagger inference failed: {e}") from e
    probs = np.asarray(output, dtype=np.float64)
    if cfg.get("sigmoid"):
        probs = 1.0 / (1.0 + np.exp(-probs))
    if len(probs) != len(names):
        raise TaggerError(f"The model returned {len(probs)} scores but the labels file has {len(names)} tags.")
    keep = (probs >= SIDECAR_FLOOR) | (categories == CATEGORY_RATING)
    return {names[i]: (round(float(probs[i]), 4), int(categories[i])) for i in np.nonzero(keep)[0]}


def mcut_threshold(probs: list[float]) -> float:
    """Maximum Cut Thresholding (largest gap between sorted probabilities), as in the WD reference app."""
    ordered = sorted(probs, reverse=True)
    if len(ordered) < 2:
        return 0.0
    gaps = [ordered[i] - ordered[i + 1] for i in range(len(ordered) - 1)]
    t = int(np.argmax(gaps))
    return (ordered[t] + ordered[t + 1]) / 2


# ---------- sidecar ----------
def sidecar_path(image_path: str) -> str:
    folder, filename = os.path.split(image_path)
    return os.path.join(folder, META_DIR, "tagger", os.path.splitext(filename)[0] + ".json")


def save_sidecar(image_path: str, cfg: dict, tags: dict[str, tuple[float, int]]) -> None:
    path = sidecar_path(image_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"model": cfg.get("id"), "image_mtime": os.path.getmtime(image_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "floor": SIDECAR_FLOOR,
            "tags": {name: [p, c] for name, (p, c) in sorted(tags.items(), key=lambda kv: -kv[1][0])}}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def load_sidecar(image_path: str) -> dict:
    try:
        with open(sidecar_path(image_path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def sidecar_tags(data: dict) -> dict[str, tuple[float, int]]:
    return {name: (float(v[0]), int(v[1])) for name, v in (data.get("tags") or {}).items()
            if isinstance(v, (list, tuple)) and len(v) == 2}


def tag_probabilities(cfg: dict, image_path: str, img: Image.Image | None = None, reuse: bool = True):
    """Returns ({tag: (p, category)}, from_cache). Re-runs the model if the image or model changed."""
    data = load_sidecar(image_path) if reuse else {}
    if data.get("model") == cfg.get("id") and abs(float(data.get("image_mtime", -1)) - os.path.getmtime(image_path)) < 1e-6:
        return sidecar_tags(data), True
    from ..image_utils import open_image
    own = img is None
    image = open_image(image_path) if own else img
    try:
        tags = predict(cfg, image)
    finally:
        if own:
            image.close()
    try:
        save_sidecar(image_path, cfg, tags)
    except OSError as e:
        print(f"    Warning: could not save tagger results: {e}")
    return tags, False


# ---------- selection & formatting ----------
def select(tags: dict[str, tuple[float, int]], opts: dict | None = None) -> dict:
    """Applies thresholds (or MCut). Returns {"rating": (name, p) or None, "characters": [...], "general": [...]}."""
    o = dict(DEFAULTS, **(opts or {}))
    ratings = sorted(((n, p) for n, (p, c) in tags.items() if c == CATEGORY_RATING), key=lambda x: -x[1])
    general = [(n, p) for n, (p, c) in tags.items() if c not in (CATEGORY_RATING, CATEGORY_CHARACTER)]
    characters = [(n, p) for n, (p, c) in tags.items() if c == CATEGORY_CHARACTER]
    g_thr = mcut_threshold([p for _, p in general]) if o["general_mcut"] else float(o["general_threshold"])
    c_thr = max(0.15, mcut_threshold([p for _, p in characters])) if o["character_mcut"] else float(o["character_threshold"])
    return {
        "rating": ratings[0] if ratings else None,
        "characters": sorted(((n, p) for n, p in characters if p > c_thr), key=lambda x: -x[1]),
        "general": sorted(((n, p) for n, p in general if p > g_thr), key=lambda x: -x[1]),
        "thresholds": (round(g_thr, 3), round(c_thr, 3)),
    }


def display_tag(name: str, keep_underscores: bool = False) -> str:
    return name if keep_underscores or name in KAOMOJIS else name.replace("_", " ")


def _is_count(name: str) -> bool:
    return bool(_COUNT_RE.match(name.replace(" ", "_")))


def format_tags(selection: dict, profile: dict | None = None, opts: dict | None = None) -> list[str]:
    """Orders and renames selected tags for the target profile (see the `tagger:` block in models.yaml)."""
    o = dict(DEFAULTS, **(opts or {}))
    profile = profile or {}
    tcfg = profile.get("tagger") or {}
    keep_us = bool(profile.get("underscores"))
    blacklist = {display_tag(b.strip().lower()) for b in o.get("blacklist") or [] if b.strip()}

    def ok(tag: str) -> bool:
        return display_tag(tag.lower()) not in blacklist

    groups: dict[str, list[str]] = {"count": [], "character": [], "rating": [], "general": []}
    for name, _ in selection.get("general", []):
        if ok(name):
            groups["count" if _is_count(name) else "general"].append(display_tag(name, keep_us))
    for name, _ in selection.get("characters", []):
        if not ok(name):
            continue
        match = _SERIES_RE.match(name) if o.get("split_series") else None
        if match:
            groups["character"] += [display_tag(match.group(1), keep_us), display_tag(match.group(2), keep_us)]
        else:
            groups["character"].append(display_tag(name, keep_us))
    rating = selection.get("rating")
    if o.get("add_rating") and rating:
        mapped = (tcfg.get("rating_map") or {}).get(rating[0], rating[0])
        if mapped and ok(mapped):
            groups["rating"].append(mapped)
    order = tcfg.get("order") or ["count", "character", "rating", "general"]
    out, seen = [], set()
    for group in order + [g for g in groups if g not in order]:
        for tag in groups.get(group, []):
            if tag.lower() not in seen:
                seen.add(tag.lower())
                out.append(tag)
    return out


def hint_text(selection: dict, hint_threshold: float = 0.5, profile: dict | None = None) -> str:
    """Tagger tags for the vision model's instruction ('hints; mention only if clearly visible')."""
    keep_us = bool((profile or {}).get("underscores"))
    chars = [display_tag(n, keep_us) for n, _ in selection.get("characters", [])]
    general = [display_tag(n, keep_us) for n, p in selection.get("general", []) if p >= hint_threshold]
    if not chars and not general:
        return ""
    lines = ["Tags from an automatic image tagger (they can be wrong; mention only what is clearly visible):"]
    if chars:
        lines.append("Characters: " + ", ".join(chars))
    if general:
        lines.append("Tags: " + ", ".join(general))
    return "\n".join(lines)


def summarize_selection(selection: dict) -> str:
    rating = selection.get("rating")
    g_thr, c_thr = selection.get("thresholds", (None, None))
    return (f"{len(selection.get('general', []))} general (> {g_thr}), {len(selection.get('characters', []))} character "
            f"(> {c_thr})" + (f", rating {rating[0]} {rating[1]:.2f}" if rating else ""))


_VOCAB: dict[str, dict[str, int]] = {}


def vocabulary(cfg: dict) -> dict[str, int] | None:
    """{normalized tag: category} for a tagger whose files are already on disk; None otherwise (never downloads)."""
    key = cfg.get("id") or cfg.get("label") or ""
    if key in _VOCAB:
        return _VOCAB[key]
    labels_file = cfg.get("labels_file", "selected_tags.csv")
    path = None
    if cfg.get("folder"):
        candidate = os.path.join(os.path.expanduser(cfg["folder"]), labels_file)
        path = candidate if os.path.isfile(candidate) else None
    elif cfg.get("repo"):
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(cfg["repo"], labels_file, revision=cfg.get("revision"), local_files_only=True)
        except Exception:
            path = None
    if not path:
        return None
    try:
        names, categories = load_labels(path)
    except (OSError, TaggerError):
        return None
    _VOCAB[key] = {display_tag(n).lower(): int(c) for n, c in zip(names, categories)}
    return _VOCAB[key]
