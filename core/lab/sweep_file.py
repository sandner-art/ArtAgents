# ArtAgent/core/lab/sweep_file.py
"""
LoRA test sweeps as files, for a renderer ArtAgents can't reach (GRADIENT on another
machine, or an overnight render): ArtAgents exports a sweep file, the renderer writes a
results folder, and ArtAgents imports that folder as an ordinary LoRA Lab run, so grids,
votes, metrics and experiment records work unchanged.

Sweep file (<run>.lora-sweep.json, format "artagents.lora-sweep", version 1):
  run     name, validation set and version, trigger, ArtAgents run folder name
  model   model_id, scheduler_id, quantization, vae_id, guidance (CFG override or schedule)
  loras   [{key, name, sha256}]   name = what the user picked (a file name or a full path);
                                  sha256 when ArtAgents could read the file, so the renderer
                                  can find a renamed copy and refuse a different file
  jobs    [{index, prompt_index, bucket, prompt, negative, seed, width, height, steps, cfg,
            lora (a key or null for the no-LoRA baseline), strength, condition}]

Results folder: the images plus artagents-results.json (format
"artagents.lora-sweep-results", version 1): sweep_sha256 (of the exact sweep file bytes),
renderer {app, version, ...}, results [{index, file, seed, output_id, error, loras_skipped}].
Results match jobs by index, never by file name; files are relative to the folder.

The results folder comes from another program (maybe another machine), so it is read
as untrusted: size limits, file names must stay inside the folder, images are opened as
PNG/JPEG/WEBP only and saved again as PNG.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict

from PIL import Image

from . import lora_sweep
from .lora_sweep import Job

FORMAT, RESULTS_FORMAT, VERSION = "artagents.lora-sweep", "artagents.lora-sweep-results", 1
SWEEP_FILE = "sweep.lora-sweep.json"   # Kept in the run folder: the exact bytes that were exported
RESULTS_FILE = "artagents-results.json"
MAX_RESULTS_BYTES = 20 * 1024 * 1024
MAX_IMAGE_BYTES = 200 * 1024 * 1024


class SweepFileError(ValueError):
    pass


def file_sha256(path: str) -> str | None:
    """sha256 of a local file, or None when it isn't a readable file here."""
    if not path or not os.path.isabs(path) or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def lora_keys(loras: list[str]) -> dict[str, str]:
    """name → short unique key (file stem: 'style_e04', 'style_e04-2' on a clash)."""
    keys: dict[str, str] = {}
    used: set[str] = set()
    for name in loras:
        if name in keys:
            continue
        base = re.sub(r"[^A-Za-z0-9._@-]+", "_", lora_sweep.lora_label(name)).strip("_") or "lora"
        key, n = base, 2
        while key in used:
            key, n = f"{base}-{n}", n + 1
        keys[name] = key
        used.add(key)
    return keys


def build_sweep(jobs: list[Job], meta: dict, run: str, model: str, options: dict | None = None) -> dict:
    options = {k: v for k, v in (options or {}).items() if v not in (None, "")}
    names = [j.lora for j in jobs if j.lora]
    keys = lora_keys(names)
    return {
        "format": FORMAT, "version": VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run": {"name": meta.get("name") or run, "folder": run, "validation_set": meta.get("validation_set"),
                "validation_version": meta.get("validation_version"), "trigger": meta.get("trigger", "")},
        "model": {"model_id": model, "scheduler_id": options.get("scheduler") or None,
                  "quantization": options.get("quantization") or None, "vae_id": options.get("vae_id") or None,
                  "guidance": options.get("guidance") or None},
        "loras": [{"key": key, "name": name, "sha256": file_sha256(name)} for name, key in keys.items()],
        "jobs": [{"index": j.index, "prompt_index": j.prompt_index, "bucket": j.bucket, "prompt": j.prompt,
                  "negative": j.negative, "seed": j.seed, "width": j.width, "height": j.height, "steps": j.steps,
                  "cfg": j.cfg, "lora": keys[j.lora] if j.lora else None, "strength": j.strength,
                  "condition": j.condition} for j in jobs],
    }


def sweep_bytes(sweep: dict) -> bytes:
    return (json.dumps(sweep, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def export_sweep(run_dir: str, jobs: list[Job], meta: dict, model: str, options: dict | None = None) -> tuple[str, str]:
    """Writes the sweep file into a new run folder (run.json waits for results). Returns (path, sha256)."""
    if not model:
        raise SweepFileError("Pick or type the GRADIENT model (Checkpoint) for the sweep.")
    if not jobs:
        raise SweepFileError("The sweep has no images.")
    run = os.path.basename(os.path.normpath(run_dir))
    data = sweep_bytes(build_sweep(jobs, meta, run, model, options))
    sha = hashlib.sha256(data).hexdigest()
    path = os.path.join(run_dir, SWEEP_FILE)
    with open(path, "wb") as f:
        f.write(data)
    lora_sweep.save_run(run_dir, dict(meta, jobs=[asdict(j) for j in jobs], results=[], backend="GRADIENT (sweep file)",
                                      model=model, workflow=None, sampler=(options or {}).get("scheduler") or None,
                                      options={k: v for k, v in (options or {}).items() if k != "scheduler"},
                                      started=time.strftime("%Y-%m-%d %H:%M:%S"), status="exported",
                                      sweep_file=SWEEP_FILE, sweep_sha256=sha))
    return path, sha


def find_export(root: str, sha: str) -> str | None:
    """The run folder whose exported sweep file has this sha256."""
    for run in lora_sweep.list_runs(root):
        run_dir = os.path.join(root, run)
        try:
            if lora_sweep.load_run(run_dir).get("sweep_sha256") == sha:
                return run_dir
        except (OSError, ValueError):
            continue
    return None


def _read_results(folder: str) -> dict:
    path = os.path.join(folder, RESULTS_FILE)
    if not os.path.isfile(path):
        raise SweepFileError(f"No {RESULTS_FILE} in {folder}: pick the folder GRADIENT wrote.")
    if os.path.getsize(path) > MAX_RESULTS_BYTES:
        raise SweepFileError(f"{RESULTS_FILE} is too large.")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise SweepFileError(f"{RESULTS_FILE} could not be read: {e}") from e
    if not isinstance(data, dict) or data.get("format") != RESULTS_FORMAT:
        raise SweepFileError(f"{RESULTS_FILE} is not an ArtAgents sweep results file.")
    if data.get("version") != VERSION:
        raise SweepFileError(f"{RESULTS_FILE} has version {data.get('version')!r}; this ArtAgents reads version {VERSION}.")
    if not isinstance(data.get("results"), list):
        raise SweepFileError(f"{RESULTS_FILE} has no results list.")
    return data


def _inside(folder: str, name) -> str | None:
    """The image path for a results entry, only if it stays inside the folder."""
    if not isinstance(name, str) or not name or os.path.isabs(name) or "\x00" in name:
        return None
    root = os.path.realpath(folder)
    path = os.path.realpath(os.path.join(root, name))
    return path if os.path.commonpath([root, path]) == root and os.path.isfile(path) else None


def import_results(root: str, folder: str) -> tuple[str, dict]:
    """Imports a results folder into the run that exported its sweep. Returns (run_dir, counts)."""
    folder = (folder or "").strip().strip('"')
    if not folder or not os.path.isdir(folder):
        raise SweepFileError(f"Not a folder: {folder or '(empty)'}")
    results = _read_results(folder)
    sha = str(results.get("sweep_sha256") or "")
    run_dir = find_export(root, sha) if re.fullmatch(r"[0-9a-f]{64}", sha) else None
    if run_dir is None:
        raise SweepFileError("These results are for a sweep file this ArtAgents didn't export (sweep_sha256 "
                             f"{sha[:12] or '?'}…). Import them where the sweep was exported.")
    with open(os.path.join(run_dir, SWEEP_FILE), "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != sha:
            raise SweepFileError(f"The exported sweep file in {run_dir} was changed; it no longer matches the results.")
    data = lora_sweep.load_run(run_dir)
    jobs = {j["index"]: Job(**j) for j in data.get("jobs", [])}
    entries: dict[int, dict] = {}
    for item in results["results"]:
        if isinstance(item, dict) and isinstance(item.get("index"), int) and item["index"] in jobs:
            entries[item["index"]] = item
    counts = {"images": 0, "errors": 0, "missing": 0}
    out = []
    for index, job in sorted(jobs.items()):
        item = entries.get(index)
        entry: dict = {"job": index, "path": None, "error": None}
        if item is None:
            entry["error"] = "Not in the results (not rendered)."
            counts["missing"] += 1
        else:
            remote = {k: item[k] for k in ("output_id", "run_uuid") if item.get(k) is not None}
            if remote:
                entry["remote"] = remote
            skipped = [str(s) for s in item.get("loras_skipped") or []]
            if item.get("error"):
                entry["error"] = str(item["error"])[:500]
            elif skipped:
                entry["error"] = ("LoRA not applied by the renderer (no matching layers: wrong model family?): "
                                  + ", ".join(skipped))
            else:
                src = _inside(folder, item.get("file"))
                if src is None:
                    entry["error"] = f"Image file missing or outside the results folder: {item.get('file')!r}"
                elif os.path.getsize(src) > MAX_IMAGE_BYTES:
                    entry["error"] = "Image file too large."
                else:
                    try:
                        with Image.open(src, formats=["PNG", "JPEG", "WEBP"]) as img:
                            img.load()
                            name = lora_sweep._file_name(job)
                            img.save(os.path.join(run_dir, name), "PNG")
                        entry["path"] = name
                    except Exception as e:  # Untrusted file: any decoder error is just a failed image
                        entry["error"] = f"Image could not be read: {e}"
            counts["images" if entry["path"] else "errors"] += 1
        out.append(entry)
    renderer = results.get("renderer") if isinstance(results.get("renderer"), dict) else {}
    data.update(results=out, status="imported", imported=time.strftime("%Y-%m-%d %H:%M:%S"),
                finished=time.strftime("%Y-%m-%d %H:%M:%S"), results_folder=os.path.abspath(folder),
                renderer={k: str(v)[:200] for k, v in renderer.items() if isinstance(v, (str, int, float))})
    lora_sweep.save_run(run_dir, data)
    lora_sweep.make_grids(run_dir)
    return run_dir, counts
