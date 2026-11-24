# ArtAgent/core/lab/lora_sweep.py
"""
LoRA Test Sweep: LoRA files × strengths × validation prompts × seeds, always with a
no-LoRA baseline, rendered through ComfyUI (a workflow with %lora_name% / %lora_strength%)
or Forge / A1111 / SD.Next (<lora:name:strength> added to the prompt).

Every run gets a folder outputs/lora_tests/<time>-<name>/ with the images, run.json
(jobs, settings, results) and one labelled grid per prompt.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass

from PIL import Image, ImageDraw, ImageFont, ImageOps

from ..image_gen import ImageGenError, ImageGenRequest
from ..image_gen.base import GenResult
from . import validation

BASELINE = "baseline (no LoRA)"
QUEUE_AHEAD = 8  # Jobs kept queued ahead in backends with their own queue (GRADIENT)
RUNS_DIR = os.path.join("outputs", "lora_tests")


@dataclass
class Job:
    index: int
    prompt_index: int
    bucket: str
    prompt: str
    negative: str
    seed: int
    lora: str | None      # None = baseline
    strength: float
    width: int
    height: int
    steps: int | None
    cfg: float | None

    @property
    def condition(self) -> str:
        return BASELINE if self.lora is None else f"{lora_label(self.lora)} @ {self.strength:g}"


def lora_label(name: str) -> str:
    base = os.path.basename(str(name))
    return base.rsplit(".", 1)[0] if base.lower().endswith((".safetensors", ".pt", ".ckpt")) else base


def parse_strengths(text: str) -> list[float]:
    out = []
    for part in re.split(r"[,\s]+", text or ""):
        if part.strip():
            try:
                out.append(float(part))
            except ValueError as e:
                raise ValueError(f"Not a strength: {part!r}") from e
    return out


def build_jobs(vset: dict, loras: list[str], strengths: list[float], include_baseline: bool = True,
               buckets: list[str] | None = None, max_prompts: int | None = None) -> list[Job]:
    defaults = vset.get("defaults", {})
    prompts = [p for p in vset.get("prompts", []) if not buckets or p["bucket"] in buckets]
    if max_prompts:
        prompts = prompts[:int(max_prompts)]
    conditions: list[tuple[str | None, float]] = [(None, 0.0)] if include_baseline else []
    conditions += [(l, float(s)) for l in loras for s in strengths]
    jobs = []
    for pi, p in enumerate(prompts):
        flags = validation.parse_flags(p.get("flags", ""))
        for seed in ([flags["seed"]] if "seed" in flags else vset.get("seeds", [])):
            for lora, strength in conditions:
                jobs.append(Job(len(jobs), pi, p["bucket"], validation.render(p["prompt"], vset.get("trigger", "")),
                                p.get("negative") or defaults.get("negative", ""), int(seed), lora, strength,
                                int(flags.get("width") or defaults.get("width") or 1024),
                                int(flags.get("height") or defaults.get("height") or 1024),
                                flags.get("steps") or defaults.get("steps"), flags.get("cfg") or defaults.get("cfg")))
    return jobs


def request_for(job: Job, backend_label: str, model: str | None, workflow: str | None, sampler: str | None,
                baseline_lora: str | None, extra: dict | None = None) -> ImageGenRequest:
    """ComfyUI baselines reuse the LoRA slot at strength 0 (ComfyUI then skips the LoRA); Forge and GRADIENT use none."""
    if job.lora is not None:
        loras = [(job.lora, job.strength)]
    elif backend_label == "ComfyUI" and baseline_lora:
        loras = [(baseline_lora, 0.0)]
    else:
        loras = []
    return ImageGenRequest(prompt=job.prompt, negative_prompt=job.negative, width=job.width, height=job.height,
                           steps=job.steps, cfg_scale=job.cfg, seed=job.seed, batch_size=1, model=model or None,
                           sampler=sampler or None, workflow=workflow or None, loras=loras, extra=dict(extra or {}))


def gradient_extra(job: Job, run_dir: str, meta: dict, options: dict | None = None) -> dict:
    """GRADIENT job options: render options plus how the job shows in GRADIENT's queue and gallery."""
    run = os.path.basename(os.path.normpath(run_dir))
    name = meta.get("name") or run
    extra = {k: v for k, v in (options or {}).items() if v not in (None, "")}
    extra.update({
        "label": f"ArtAgents · LoRA test · {name} · {job.condition} · seed {job.seed}",
        "group": f"ArtAgents · LoRA test · {run}",  # One gallery run in GRADIENT (0.9.1+)
        "tags": [f"ArtAgents/LoRA test/{name}"],
        "origin": {"app": "ArtAgents", "run": run, "job_index": job.index, "prompt_index": job.prompt_index,
                   "bucket": job.bucket, "condition": job.condition},
    })
    return extra


def new_run_dir(root: str, name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", (name or "test")).strip("-")[:40] or "test"
    base = os.path.join(root, f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}")
    path, n = base, 2
    while os.path.exists(path):  # Two runs with the same name in one second
        path, n = f"{base}-{n}", n + 1
    os.makedirs(path)
    return path


def save_run(run_dir: str, data: dict) -> None:
    tmp = os.path.join(run_dir, "run.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, os.path.join(run_dir, "run.json"))


def load_run(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "run.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def list_runs(root: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    return sorted((d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "run.json"))), reverse=True)


def _file_name(job: Job) -> str:
    return f"{job.index:04d}_p{job.prompt_index:02d}_s{job.seed}_{re.sub(r'[^A-Za-z0-9.@_-]+', '_', job.condition)}.png"


def run_iter(backend, backend_label: str, jobs: list[Job], run_dir: str, meta: dict, model: str | None = None,
             workflow: str | None = None, sampler: str | None = None, options: dict | None = None):
    """
    Generator: renders every job, saving images and run.json after each one.
    Yields (done, total, message, last image path or None). Stopping keeps finished images.

    Backends with their own queue (GRADIENT: `submit` + `wait_iter`) get QUEUE_AHEAD jobs queued ahead,
    ordered by condition so LoRAs switch as rarely as possible; closing the generator cancels what is
    still queued.
    """
    loras = [j.lora for j in jobs if j.lora]
    baseline_lora = loras[0] if loras else None
    data = dict(meta, jobs=[asdict(j) for j in jobs], results=[], backend=backend_label, model=model,
                workflow=workflow, sampler=sampler, options=dict(options or {}),
                started=time.strftime("%Y-%m-%d %H:%M:%S"))
    save_run(run_dir, data)
    total = len(jobs)

    def extra_for(job):
        return gradient_extra(job, run_dir, meta, options) if backend_label == "GRADIENT" else dict(options or {})

    def record(job, images, error):
        path, remote = None, {}
        if images and not error:
            info = images[0].info or {}
            remote = {k: info[k] for k in ("output_id", "run_uuid", "gradient_job") if info.get(k) is not None}
            if info.get("loras_skipped"):
                error = (f"LoRA not applied by {backend_label} (no matching layers: wrong model family?): "
                         + ", ".join(info["loras_skipped"]))
            else:
                path = os.path.join(run_dir, _file_name(job))
                images[0].image.save(path)
        entry = {"job": job.index, "path": os.path.basename(path) if path else None, "error": error}
        if remote:
            entry["remote"] = remote
        data["results"].append(entry)
        save_run(run_dir, data)
        return path, error

    if hasattr(backend, "submit") and hasattr(backend, "wait_iter"):
        # Queue a small window ahead (GRADIENT never idles, Stop has little to cancel, the user's own
        # queue isn't flooded), ordered by condition so LoRAs switch as rarely as possible.
        window = max(1, int(getattr(backend, "queue_ahead", QUEUE_AHEAD)))
        order = sorted(jobs, key=lambda j: (j.lora is not None, str(j.lora or ""), j.strength, j.index))
        submitted: dict[int, str] = {}
        failed: dict[int, str] = {}
        state = {"next": 0}

        def top_up(upto: int) -> None:
            while state["next"] < min(len(order), upto):
                pos = state["next"]
                state["next"] += 1
                job = order[pos]
                try:
                    submitted[pos] = backend.submit(request_for(job, backend_label, model, workflow, sampler,
                                                                baseline_lora, extra_for(job)))
                except ImageGenError as e:
                    failed[pos] = str(e)

        recorded = 0
        try:
            for pos, job in enumerate(order):
                top_up(pos + window)
                images, error = [], failed.get(pos)
                if pos in submitted:
                    try:
                        for event in backend.wait_iter(submitted[pos]):
                            if isinstance(event, GenResult):
                                images = event.images
                    except ImageGenError as e:
                        error = str(e)
                path, error = record(job, images, error)
                recorded = done = pos + 1
                yield done, total, f"{done}/{total}: {job.condition}, seed {job.seed}" + (f" ⚠ {error}" if error else ""), path
        finally:
            pending = [job_id for pos, job_id in submitted.items() if pos >= recorded]
            if pending:  # Stopped: cancel what is still queued in the backend
                if hasattr(backend, "cancel_many"):
                    backend.cancel_many(pending)
                else:
                    for job_id in pending:
                        backend.cancel(job_id)
    else:
        for done, job in enumerate(jobs, start=1):
            req = request_for(job, backend_label, model, workflow, sampler, baseline_lora, extra_for(job))
            images, error = [], None
            try:
                for event in backend.generate_iter(req):
                    if isinstance(event, GenResult):
                        images = event.images
                if not images:
                    raise ImageGenError("The backend returned no image.")
            except ImageGenError as e:
                error = str(e)
            path, error = record(job, images, error)
            yield done, total, f"{done}/{total}: {job.condition}, seed {job.seed}" + (f" ⚠ {error}" if error else ""), path
    data["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_run(run_dir, data)


# ---------- grids ----------
def _font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(draw, text: str, font, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for w in words:
        test = f"{line} {w}".strip()
        if draw.textlength(test, font=font) <= width or not line:
            line = test
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


def make_grids(run_dir: str, cell: int = 256) -> list[str]:
    """One PNG per prompt: rows = conditions (baseline first), columns = seeds. Returns grid paths."""
    data = load_run(run_dir)
    jobs = {j["index"]: j for j in data.get("jobs", [])}
    done = {r["job"]: r["path"] for r in data.get("results", []) if r.get("path")}
    by_prompt: dict[int, list[dict]] = {}
    for idx, job in jobs.items():
        by_prompt.setdefault(job["prompt_index"], []).append(job)
    font, small = _font(16), _font(13)
    paths = []
    for pi, pjobs in sorted(by_prompt.items()):
        conds = []
        for j in pjobs:
            c = BASELINE if j["lora"] is None else f"{lora_label(j['lora'])} @ {j['strength']:g}"
            if c not in conds:
                conds.append(c)
        conds.sort(key=lambda c: (c != BASELINE, c))
        seeds = sorted({j["seed"] for j in pjobs})
        label_w, header_h = 190, 78
        grid = Image.new("RGB", (label_w + cell * len(seeds), header_h + cell * len(conds)), "white")
        draw = ImageDraw.Draw(grid)
        title = f"[{pjobs[0]['bucket']}] {pjobs[0]['prompt']}"
        for li, line in enumerate(_wrap(draw, title, font, grid.width - 20)[:2]):
            draw.text((10, 6 + li * 20), line, fill="black", font=font)
        for si, seed in enumerate(seeds):
            draw.text((label_w + si * cell + 6, header_h - 22), f"seed {seed}", fill="#444", font=small)
        for ci, cond in enumerate(conds):
            y = header_h + ci * cell
            for li, line in enumerate(_wrap(draw, cond, small, label_w - 12)[:4]):
                draw.text((8, y + 8 + li * 16), line, fill="black", font=small)
            for si, seed in enumerate(seeds):
                job = next((j for j in pjobs if j["seed"] == seed and
                            (BASELINE if j["lora"] is None else f"{lora_label(j['lora'])} @ {j['strength']:g}") == cond), None)
                x = label_w + si * cell
                fname = done.get(job["index"]) if job else None
                if fname and os.path.exists(os.path.join(run_dir, fname)):
                    with Image.open(os.path.join(run_dir, fname)) as im:
                        tile = ImageOps.contain(im.convert("RGB"), (cell - 4, cell - 4))  # Fit the cell, up or down
                        grid.paste(tile, (x + (cell - tile.width) // 2, y + (cell - tile.height) // 2))
                else:
                    draw.rectangle([x + 2, y + 2, x + cell - 3, y + cell - 3], outline="#bbb")
                    draw.text((x + 10, y + cell // 2 - 8), "missing", fill="#999", font=small)
        path = os.path.join(run_dir, f"grid_p{pi:02d}.png")
        grid.save(path)
        paths.append(path)
    return paths


def copy_lora(src: str, comfy_dir: str) -> str:
    """Copies a LoRA file into <ComfyUI>/models/loras/ (never overwrites). Returns the new path."""
    if not os.path.isfile(src):
        raise FileNotFoundError(f"LoRA file not found: {src}")
    if not src.lower().endswith((".safetensors", ".pt", ".ckpt")):
        raise ValueError("Expected a .safetensors (or .pt / .ckpt) LoRA file.")
    target_dir = os.path.join(comfy_dir, "models", "loras") if os.path.isdir(os.path.join(comfy_dir, "models")) else comfy_dir
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Not a ComfyUI folder or loras folder: {comfy_dir}")
    dst = os.path.join(target_dir, os.path.basename(src))
    if os.path.exists(dst):
        raise FileExistsError(f"{os.path.basename(src)} already exists in {target_dir}")
    shutil.copy2(src, dst)
    return dst
