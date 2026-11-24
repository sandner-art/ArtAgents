# ArtAgent/core/dataset/snapshots.py
"""
Automatic caption snapshots ("undo" for batch operations).

Before any batch write (AI batch captioning, batch append/prepend, find & replace,
normalizing, restoring) the caption files of the loaded images are copied to
<dataset>/.artagents/snapshots/<timestamp>-<operation>/ with a manifest that also
records which images had NO caption, so a restore is exact.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import time

from .caption_io import caption_path, read_caption, write_text_atomic

META_DIR = ".artagents"
SNAPSHOT_DIR = "snapshots"
MAX_SNAPSHOTS = 50


def meta_dir(folder: str) -> str:
    return os.path.join(folder, META_DIR)


def snapshots_root(folder: str) -> str:
    return os.path.join(folder, META_DIR, SNAPSHOT_DIR)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "snapshot").lower()).strip("-")[:40] or "snapshot"


def create_snapshot(folder: str, image_paths: dict[str, str], operation: str) -> str:
    """
    Copies the current caption of every image in `image_paths` ({filename: path}).

    Returns the snapshot id (folder name).
    """
    root = snapshots_root(folder)
    os.makedirs(root, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    snap_id = f"{stamp}-{_slug(operation)}"
    n = 2
    while os.path.exists(os.path.join(root, snap_id)):
        snap_id = f"{stamp}-{_slug(operation)}-{n}"
        n += 1
    snap_dir = os.path.join(root, snap_id)
    os.makedirs(snap_dir)

    files = {}
    for filename, image_path in sorted((image_paths or {}).items()):
        src = caption_path(image_path)
        stem_txt = os.path.basename(src)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(snap_dir, stem_txt))
            files[filename] = stem_txt
        else:
            files[filename] = None  # No caption at snapshot time
    manifest = {"id": snap_id, "operation": operation, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "created_ns": time.time_ns(), "files": files}
    with open(os.path.join(snap_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    prune_snapshots(folder)
    return snap_id


def list_snapshots(folder: str) -> list[dict]:
    """Newest first: [{"id", "operation", "created", "count"}]."""
    root = snapshots_root(folder)
    if not os.path.isdir(root):
        return []
    entries = []
    for name in os.listdir(root):
        manifest_path = os.path.join(root, name, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, ValueError):
            continue
        # Folder names start with the second; created_ns orders snapshots made in the same second.
        order = (name[:15], manifest.get("created_ns", 0), name)
        entries.append((order, {
            "id": name,
            "operation": manifest.get("operation", ""),
            "created": manifest.get("created", ""),
            "count": sum(1 for v in manifest.get("files", {}).values() if v),
        }))
    return [entry for _, entry in sorted(entries, key=lambda e: e[0], reverse=True)]


def prune_snapshots(folder: str, keep: int = MAX_SNAPSHOTS) -> None:
    for old in list_snapshots(folder)[keep:]:
        shutil.rmtree(os.path.join(snapshots_root(folder), old["id"]), ignore_errors=True)


def _load_manifest(folder: str, snap_id: str) -> tuple[str, dict]:
    if not snap_id or os.sep in snap_id or "/" in snap_id or snap_id.startswith("."):
        raise ValueError(f"Invalid snapshot id: {snap_id!r}")
    snap_dir = os.path.join(snapshots_root(folder), snap_id)
    with open(os.path.join(snap_dir, "manifest.json"), "r", encoding="utf-8") as f:
        return snap_dir, json.load(f)


def snapshot_captions(folder: str, snap_id: str) -> dict[str, str | None]:
    """{filename: caption text, or None if the image had no caption file}."""
    snap_dir, manifest = _load_manifest(folder, snap_id)
    captions = {}
    for filename, stored in manifest.get("files", {}).items():
        if stored is None:
            captions[filename] = None
        else:
            with open(os.path.join(snap_dir, os.path.basename(stored)), "r", encoding="utf-8", errors="replace") as f:
                captions[filename] = f.read()
    return captions


def restore_snapshot(folder: str, snap_id: str, image_paths: dict[str, str]) -> tuple[str, int]:
    """
    Restores captions from a snapshot (after snapshotting the current state first).

    Only images that are currently loaded are touched. Returns (safety_snapshot_id, files_changed).
    """
    stored = snapshot_captions(folder, snap_id)
    safety_id = create_snapshot(folder, image_paths, f"before restore of {snap_id}")
    changed = 0
    for filename, image_path in image_paths.items():
        if filename not in stored:
            continue
        target = caption_path(image_path)
        old_text = stored[filename]
        if old_text is None:
            if os.path.exists(target):
                os.remove(target)  # Caption did not exist at snapshot time
                changed += 1
        elif not os.path.exists(target) or read_caption(image_path) != old_text:
            write_text_atomic(target, old_text)
            changed += 1
    return safety_id, changed


def diff_snapshot(folder: str, snap_id: str, current: dict[str, str], max_files: int = 30) -> tuple[int, str]:
    """
    Compares a snapshot with the current captions. Returns (files_changed, markdown_diff).
    """
    stored = snapshot_captions(folder, snap_id)
    chunks, changed = [], 0
    for filename in sorted(set(stored) | set(current)):
        before = stored.get(filename) or ""
        after = current.get(filename, "") or ""
        if before == after:
            continue
        changed += 1
        if len(chunks) < max_files:
            diff = "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), "snapshot", "current", lineterm=""))
            chunks.append(f"**{filename}**\n```diff\n{diff or '(whitespace only)'}\n```")
    if not changed:
        return 0, "No differences: the current captions match this snapshot."
    more = f"\n\n…and {changed - max_files} more file(s)." if changed > max_files else ""
    return changed, f"{changed} caption file(s) differ.\n\n" + "\n\n".join(chunks) + more
