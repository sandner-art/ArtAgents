# ArtAgent/core/dataset/health.py
"""
Dataset Health scan (CPU only; Pillow + numpy).

Per image: size, aspect, colour mode / transparency, exact duplicates (SHA-256),
near duplicates (64-bit perceptual hash, Hamming ≤ 10 as in imagededup), sharpness
(variance of the Laplacian), brightness and contrast. Per caption: missing, empty,
several lines, over the token budget, identical to another caption, trigger missing.

Flags are relative to the dataset (e.g. sharpness far below the dataset median)
rather than fixed cut-offs, and they are suggestions for review: nothing is deleted.
Results are cached per file in <dataset>/.artagents/cache/health.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image

from . import caption_io
from .snapshots import META_DIR
from .tokens import budget_checks

CACHE_VERSION = 1
NEAR_DUP_HAMMING = 10       # imagededup's default for 64-bit hashes
BLUR_RELATIVE = 0.25        # sharpness below 25% of the dataset median
SMALL_SIDE = 512            # short side below this is always flagged
QUARANTINE_DIR = "quarantine"

ISSUE_SMALL = "low resolution"
ISSUE_UPSCALE = "below training resolution"
ISSUE_ASPECT = "extreme aspect ratio"
ISSUE_MODE = "unusual colour mode"
ISSUE_ALPHA = "transparency"
ISSUE_BLUR = "blurry (relative)"
ISSUE_DARK = "very dark"
ISSUE_BRIGHT = "very bright"
ISSUE_LOW_CONTRAST = "low contrast"
ISSUE_DUP = "exact duplicate"
ISSUE_NEAR_DUP = "near duplicate"
ISSUE_NO_CAPTION = "no caption file"
ISSUE_EMPTY = "empty caption"
ISSUE_MULTILINE = "several caption lines"
ISSUE_BUDGET = "over token budget"
ISSUE_SAME_CAPTION = "same caption as another image"
ISSUE_NO_TRIGGER = "trigger missing"
ISSUE_UNREADABLE = "cannot be opened"


# ---------- per-image measurements ----------
def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)
    m = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2.0)
    return m


_DCT32 = _dct_matrix(32)


def phash(gray: Image.Image) -> int:
    """64-bit DCT perceptual hash (same idea as imagehash.phash)."""
    small = np.asarray(gray.resize((32, 32), Image.LANCZOS), dtype=np.float64)
    dct = _DCT32 @ small @ _DCT32.T
    low = dct[:8, :8]
    bits = (low > np.median(low)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def laplacian_variance(gray: Image.Image, max_side: int = 512) -> float:
    img = gray
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float64)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    lap = a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1]
    return float(lap.var())


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def measure(path: str) -> dict:
    """Pixel statistics for one image (no caption checks)."""
    from ..image_utils import open_image
    stat = os.stat(path)
    result = {"mtime": stat.st_mtime, "bytes": stat.st_size, "sha256": file_sha256(path)}
    with open_image(path) as img:
        result.update({"width": img.size[0], "height": img.size[1], "mode": img.mode, "format": img.format or ""})
        has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
        uses_alpha = False
        if has_alpha:
            alpha = np.asarray(img.convert("RGBA").getchannel("A"))
            uses_alpha = bool((alpha < 250).mean() > 0.001)
        result["alpha"] = uses_alpha
        gray = img.convert("L")
        a = np.asarray(gray, dtype=np.float64)
        result["brightness"] = round(float(a.mean()), 2)
        result["contrast"] = round(float(a.std()), 2)
        result["sharpness"] = round(laplacian_variance(gray), 2)
        result["phash"] = f"{phash(gray):016x}"
    return result


# ---------- cache ----------
def _cache_path(folder: str) -> str:
    return os.path.join(folder, META_DIR, "cache", "health.json")


def _load_cache(folder: str) -> dict:
    try:
        with open(_cache_path(folder), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("images", {}) if data.get("version") == CACHE_VERSION else {}
    except (OSError, ValueError, AttributeError):
        return {}


def _save_cache(folder: str, images: dict) -> None:
    path = _cache_path(folder)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": CACHE_VERSION, "images": images}, f)
        os.replace(tmp, path)
    except OSError as e:
        print(f"Warning: could not save the health cache: {e}")


def measure_all(image_paths: dict[str, str], progress=None) -> dict[str, dict]:
    """Measurements for every image, reusing the cache for unchanged files."""
    if not image_paths:
        return {}
    folder = os.path.dirname(next(iter(image_paths.values())))
    cache = _load_cache(folder)
    out, total = {}, len(image_paths)
    for i, name in enumerate(sorted(image_paths)):
        path = image_paths[name]
        if progress:
            progress((i, total), desc=f"Scanning {name}")
        try:
            stat = os.stat(path)
            cached = cache.get(name)
            if cached and cached.get("mtime") == stat.st_mtime and cached.get("bytes") == stat.st_size:
                out[name] = cached
            else:
                out[name] = measure(path)
        except Exception as e:  # Unreadable or unsupported file
            out[name] = {"error": f"{e.__class__.__name__}: {e}"}
    _save_cache(folder, {k: v for k, v in out.items() if "error" not in v})
    return out


# ---------- dataset-level analysis ----------
def near_duplicate_groups(stats: dict[str, dict], max_distance: int = NEAR_DUP_HAMMING) -> list[list[str]]:
    names = [n for n in sorted(stats) if stats[n].get("phash")]
    hashes = {n: int(stats[n]["phash"], 16) for n in names}
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if hamming(hashes[a], hashes[b]) <= max_distance:
                parent[find(a)] = find(b)
    groups = defaultdict(list)
    for n in names:
        groups[find(n)].append(n)
    return sorted((sorted(g) for g in groups.values() if len(g) > 1), key=lambda g: g[0])


def scan(image_paths: dict[str, str], captions: dict[str, str], resolution: int = 1024, budgets: list | None = None,
         trigger: str = "", progress=None) -> dict:
    """
    Returns {"issues": [[image, issue, detail]], "summary": markdown, "flagged": [names], "near_dupes": groups,
             "stats": per-image measurements}.
    """
    stats = measure_all(image_paths, progress)
    issues: list[list[str]] = []

    def add(name, issue, detail=""):
        issues.append([name, issue, detail])

    ok = {n: s for n, s in stats.items() if "error" not in s}
    for name, s in stats.items():
        if "error" in s:
            add(name, ISSUE_UNREADABLE, s["error"])

    sharp = [s["sharpness"] for s in ok.values()]
    median_sharp = float(np.median(sharp)) if sharp else 0.0
    target_area = resolution * resolution
    for name, s in sorted(ok.items()):
        w, h = s["width"], s["height"]
        if min(w, h) < SMALL_SIDE:
            add(name, ISSUE_SMALL, f"{w}×{h}")
        elif w * h < 0.75 * target_area:
            add(name, ISSUE_UPSCALE, f"{w}×{h} = {w * h / 1e6:.2f} MP; training at {resolution}px is {target_area / 1e6:.2f} MP")
        ar = max(w, h) / max(1, min(w, h))
        if ar > 2.5:
            add(name, ISSUE_ASPECT, f"{ar:.2f}:1")
        if s["mode"] in ("CMYK", "I", "I;16", "F", "1", "YCbCr", "LAB", "HSV"):
            add(name, ISSUE_MODE, s["mode"])
        if s.get("alpha"):
            add(name, ISSUE_ALPHA, "trainers usually composite transparent areas onto black or white")
        if len(ok) >= 5 and median_sharp > 0 and s["sharpness"] < BLUR_RELATIVE * median_sharp:
            add(name, ISSUE_BLUR, f"sharpness {s['sharpness']:.0f} vs dataset median {median_sharp:.0f}")
        if s["brightness"] < 25:
            add(name, ISSUE_DARK, f"mean {s['brightness']:.0f}/255")
        elif s["brightness"] > 235:
            add(name, ISSUE_BRIGHT, f"mean {s['brightness']:.0f}/255")
        if s["contrast"] < 12:
            add(name, ISSUE_LOW_CONTRAST, f"std {s['contrast']:.1f}")

    by_hash = defaultdict(list)
    for name, s in ok.items():
        by_hash[s["sha256"]].append(name)
    exact_groups = [sorted(g) for g in by_hash.values() if len(g) > 1]
    for group in exact_groups:
        for name in group[1:]:
            add(name, ISSUE_DUP, f"same file as {group[0]}")
    exact_members = {n for g in exact_groups for n in g}
    near = [g for g in near_duplicate_groups(ok) if not set(g) <= exact_members]
    for group in near:
        for name in group:
            add(name, ISSUE_NEAR_DUP, "similar to " + ", ".join(x for x in group if x != name))

    by_caption = defaultdict(list)
    trigger_l = (trigger or "").strip().lower()
    for name, path in sorted(image_paths.items()):
        text = captions.get(name)
        if not os.path.exists(caption_io.caption_path(path)):
            add(name, ISSUE_NO_CAPTION)
            continue
        text = text or ""
        if not text.strip():
            add(name, ISSUE_EMPTY)
            continue
        n_lines = caption_io.variant_count(text)
        if n_lines > 1:
            add(name, ISSUE_MULTILINE, f"{n_lines} lines (variants)")
        first = caption_io.split_variants(text)[0] if n_lines else text
        for check in budget_checks(first, budgets or []):
            if check["status"] in ("over_budget", "over_max"):
                add(name, ISSUE_BUDGET, f"{check['label']}: {check['count']}/{check['budget']}")
        by_caption[first.strip().lower()].append(name)
        if trigger_l and trigger_l not in text.lower():
            add(name, ISSUE_NO_TRIGGER, trigger)
    for text, names in by_caption.items():
        if len(names) > 1:
            for name in names:
                add(name, ISSUE_SAME_CAPTION, ", ".join(x for x in names if x != name)[:200])

    counts = defaultdict(int)
    for _, issue, _ in issues:
        counts[issue] += 1
    flagged = sorted({i[0] for i in issues})
    lines = [f"**{len(image_paths)} image(s) scanned, {len(flagged)} with something to check.**"]
    if ok:
        mps = [s["width"] * s["height"] / 1e6 for s in ok.values()]
        lines.append(f"Resolution: {min(mps):.2f}–{max(mps):.2f} MP (median {float(np.median(mps)):.2f}). "
                     f"Sharpness median {median_sharp:.0f}.")
    if counts:
        lines.append("")
        lines += [f"- {issue}: {n}" for issue, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    if near:
        lines.append(f"\nNear-duplicate groups (perceptual hash distance ≤ {NEAR_DUP_HAMMING}): "
                     + "; ".join(" + ".join(g) for g in near[:10]))
    lines.append("\n_Flags are suggestions relative to this dataset. Review them; nothing is deleted._")
    return {"issues": issues, "summary": "\n".join(lines), "flagged": flagged, "near_dupes": near,
            "exact_dupes": exact_groups, "stats": stats}


def sizes_from_stats(stats: dict[str, dict]) -> dict[str, tuple[int, int]]:
    return {n: (s["width"], s["height"]) for n, s in stats.items() if "width" in s}


# ---------- quarantine ----------
def quarantine_dir(folder: str) -> str:
    return os.path.join(folder, META_DIR, QUARANTINE_DIR)


def quarantine(image_path: str) -> str:
    """Moves an image (and its caption) into <dataset>/.artagents/quarantine/. Returns the new image path."""
    folder = os.path.dirname(image_path)
    target = quarantine_dir(folder)
    os.makedirs(target, exist_ok=True)
    moved = os.path.join(target, os.path.basename(image_path))
    if os.path.exists(moved):
        raise FileExistsError(f"{os.path.basename(image_path)} is already in quarantine.")
    shutil.move(image_path, moved)
    caption = caption_io.caption_path(image_path)
    if os.path.exists(caption):
        shutil.move(caption, os.path.join(target, os.path.basename(caption)))
    return moved


def restore_quarantine(folder: str) -> list[str]:
    """Moves every quarantined file back into the dataset (never overwrites). Returns restored file names."""
    source = quarantine_dir(folder)
    if not os.path.isdir(source):
        return []
    restored = []
    for name in sorted(os.listdir(source)):
        src, dst = os.path.join(source, name), os.path.join(folder, name)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.move(src, dst)
            restored.append(name)
    return restored
