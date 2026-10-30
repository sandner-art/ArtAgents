# ArtAgent/core/image_gen/storage.py
"""Saves generated images as PNGs with prompt metadata (A1111-style) and optional .txt sidecars."""
from __future__ import annotations

import json
import os
from datetime import datetime

from PIL.PngImagePlugin import PngInfo

from .base import GeneratedImage, ImageGenRequest


def build_parameters_text(req: ImageGenRequest, image: GeneratedImage, backend_label: str) -> str:
    """A1111-style 'parameters' text, readable by most image viewers and prompt tools."""
    width, height = image.image.size
    lines = [req.prompt or ""]
    if req.negative_prompt:
        lines.append(f"Negative prompt: {req.negative_prompt}")
    fields = [
        ("Steps", req.steps if req.steps is not None else image.info.get("steps")),
        ("Sampler", req.sampler or image.info.get("sampler_name")),
        ("CFG scale", req.cfg_scale if req.cfg_scale is not None else image.info.get("cfg_scale")),
        ("Seed", image.seed),
        ("Size", f"{width}x{height}"),
        ("Model", req.model or image.info.get("sd_model_name") or image.info.get("model")),
        ("Workflow", req.workflow),
        ("Backend", backend_label),
    ]
    lines.append(", ".join(f"{k}: {v}" for k, v in fields if v not in (None, "")))
    return "\n".join(lines)


def _unique_base(directory: str, base: str) -> str:
    candidate, counter = base, 2
    while os.path.exists(os.path.join(directory, candidate + ".png")) or os.path.exists(os.path.join(directory, candidate + ".txt")):
        candidate = f"{base}-{counter}"
        counter += 1
    return candidate


def save_images(images: list[GeneratedImage], req: ImageGenRequest, backend_label: str, out_root: str,
                save_prompt_txt: bool = True) -> list[str]:
    """
    Saves images to out_root/YYYY-MM-DD/<timestamp>_seed<seed>_<n>.png and returns absolute paths.

    Filenames are always generated here (never taken from the server). Each PNG
    carries a 'parameters' text chunk (A1111 format) and an 'artagents' JSON chunk;
    with save_prompt_txt a same-named .txt holds just the prompt (caption-dataset ready).
    """
    now = datetime.now()
    directory = os.path.join(os.path.abspath(out_root), now.strftime("%Y-%m-%d"))
    os.makedirs(directory, exist_ok=True)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    paths = []
    for index, generated in enumerate(images, 1):
        seed_part = f"_seed{int(generated.seed)}" if isinstance(generated.seed, int) else ""
        base = _unique_base(directory, f"{stamp}{seed_part}_{index}")
        png_path = os.path.join(directory, base + ".png")

        metadata = PngInfo()
        metadata.add_text("parameters", build_parameters_text(req, generated, backend_label))
        metadata.add_text("artagents", json.dumps({
            "prompt": req.prompt, "negative_prompt": req.negative_prompt, "seed": generated.seed,
            "width": generated.image.size[0], "height": generated.image.size[1], "steps": req.steps,
            "cfg_scale": req.cfg_scale, "sampler": req.sampler, "model": req.model, "workflow": req.workflow,
            "backend": backend_label, "created": now.isoformat(timespec="seconds"),
        }, ensure_ascii=False))

        image = generated.image
        if image.mode not in ("RGB", "RGBA", "L", "LA", "P", "I", "I;16", "1"):
            image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
        image.save(png_path, format="PNG", pnginfo=metadata)
        if save_prompt_txt:
            with open(os.path.join(directory, base + ".txt"), "w", encoding="utf-8") as f:
                f.write(req.prompt or "")
        paths.append(png_path)
    return paths
