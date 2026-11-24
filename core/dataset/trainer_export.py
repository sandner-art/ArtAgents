# ArtAgent/core/dataset/trainer_export.py
"""
Trainer-aware export: dataset config files for common LoRA trainers, plus a repeats
calculator. ArtAgents never renames or moves your folders; repeats go into the config.

Formats were checked against each project's source/docs (Nov 2026):
kohya-ss sd-scripts (docs/config_README-en.md), musubi-tuner (docs/dataset_config.md),
ai-toolkit (config/examples), OneTrainer (modules/util/config/ConceptConfig.py),
SimpleTuner (documentation/DATALOADER.md: repeats 0 = no repeats), diffusion-pipe.
"""
from __future__ import annotations

import json
import math
import os

import yaml

from ..version import APP_VERSION
from .snapshots import META_DIR

TRAINERS = {
    "kohya": ("kohya-ss sd-scripts / kohya_ss GUI (dataset.toml)", "dataset_kohya.toml"),
    "musubi": ("musubi-tuner: FLUX, Qwen-Image, Wan… (dataset.toml)", "dataset_musubi.toml"),
    "ai_toolkit": ("ai-toolkit (datasets YAML block)", "ai_toolkit_datasets.yaml"),
    "onetrainer": ("OneTrainer (concepts JSON)", "onetrainer_concepts.json"),
    "simpletuner": ("SimpleTuner (multidatabackend.json)", "simpletuner_multidatabackend.json"),
    "diffusion_pipe": ("diffusion-pipe (dataset.toml)", "dataset_diffusion_pipe.toml"),
}
TRAINER_LABELS = {v[0]: k for k, v in TRAINERS.items()}


def repeats_for(num_images: int, target_per_epoch: int) -> int:
    """kohya's documented rule: max(1, round(target / images))."""
    if num_images <= 0:
        return 1
    return max(1, round(target_per_epoch / num_images))


def steps_per_epoch(num_images: int, repeats: int, batch_size: int) -> int:
    return math.ceil(max(0, num_images) * max(1, repeats) / max(1, batch_size))


def repeats_markdown(concepts: dict[str, int], target_per_epoch: int, epochs: int, batch_size: int) -> str:
    """concepts: {folder or concept name: image count}."""
    if not concepts:
        return "Load a folder first."
    lines = ["| Concept | Images | Repeats | Images per epoch | Steps per epoch |", "|---|---|---|---|---|"]
    total_steps = 0
    for name, n in concepts.items():
        r = repeats_for(n, target_per_epoch)
        s = steps_per_epoch(n, r, batch_size)
        total_steps += s
        lines.append(f"| {name} | {n} | {r} | {n * r} | {s} |")
    lines.append(f"\n**About {total_steps * max(1, epochs)} steps** for {epochs} epoch(s) at batch size {batch_size} "
                 f"(bucketing can add a few steps). Repeats = max(1, round({target_per_epoch} / images)), as in kohya's docs.")
    return "\n".join(lines)


def keep_tokens_for(prefix: str) -> int:
    """Comma-separated chunks in the trigger prefix ('txcl painting, ' -> 1), for keep_tokens."""
    return len([p for p in (prefix or "").split(",") if p.strip()])


def _toml_str(value: str) -> str:
    """TOML literal string (no escaping needed for Windows paths) when possible."""
    if "'" not in value and "\n" not in value:
        return f"'{value}'"
    return json.dumps(value)


def _header(settings: dict, comment: str = "#") -> list[str]:
    lines = [f"{comment} Written by ArtAgents {APP_VERSION} for {settings['folder']} ({settings['num_images']} images)."]
    if settings.get("variant_files"):
        lines.append(f"{comment} {settings['variant_files']} caption file(s) have several lines (caption variants).")
    if settings.get("webp_files") and settings.get("trainer") == "ai_toolkit":
        lines.append(f"{comment} ⚠ {settings['webp_files']} WEBP image(s): ai-toolkit warns that WEBP currently has issues; convert them to PNG.")
    return lines


def default_settings(folder: str, num_images: int) -> dict:
    return {"folder": folder, "num_images": num_images, "name": os.path.basename(os.path.normpath(folder)) or "dataset",
            "resolution": 1024, "repeats": 1, "batch_size": 1, "keep_tokens": 1, "shuffle": False,
            "caption_dropout": 0.05, "min_bucket": 256, "max_bucket": 2048, "variant_files": 0, "webp_files": 0,
            "trigger": "", "control_dirs": [], "trainer": "kohya"}


def kohya_toml(s: dict) -> str:
    lines = _header(s)
    lines += ["# With --cache_text_encoder_outputs (usual for FLUX / SD3), shuffle_caption, caption_dropout_rate,",
              "# enable_wildcard and multi-line captions have no effect.", "",
              "[general]", f"shuffle_caption = {str(bool(s['shuffle'])).lower()}", 'caption_extension = ".txt"',
              f"keep_tokens = {int(s['keep_tokens'])}", "",
              "[[datasets]]", f"resolution = {int(s['resolution'])}", f"batch_size = {int(s['batch_size'])}",
              "enable_bucket = true", f"min_bucket_reso = {int(s['min_bucket'])}", f"max_bucket_reso = {int(s['max_bucket'])}",
              "bucket_reso_steps = 64", "", "  [[datasets.subsets]]", f"  image_dir = {_toml_str(s['folder'])}",
              f"  num_repeats = {int(s['repeats'])}"]
    if s.get("caption_dropout"):
        lines.append(f"  caption_dropout_rate = {float(s['caption_dropout'])}")
    if s.get("variant_files"):
        lines.append("  enable_wildcard = true  # pick one caption line at random (then lines are not shuffled)")
    return "\n".join(lines) + "\n"


def musubi_toml(s: dict) -> str:
    folder = s["folder"]
    cache = os.path.join(os.path.dirname(os.path.normpath(folder)), s["name"] + "_cache")
    lines = _header(s)
    if s.get("variant_files"):
        lines.append("# How musubi-tuner treats extra caption lines was not verified: normalize them first to be safe.")
    lines += ["", "[general]", f"resolution = [{int(s['resolution'])}, {int(s['resolution'])}]", 'caption_extension = ".txt"',
              f"batch_size = {int(s['batch_size'])}", "enable_bucket = true", "bucket_no_upscale = false", "",
              "[[datasets]]", f"image_directory = {_toml_str(folder)}", f"cache_directory = {_toml_str(cache)}",
              f"num_repeats = {int(s['repeats'])}"]
    controls = s.get("control_dirs") or []
    if len(controls) == 1:
        lines.append(f"control_directory = {_toml_str(controls[0])}")
    elif len(controls) > 1:
        lines.append("# musubi expects several control images in ONE folder named <stem>_0, <stem>_1, …;")
        lines.append("# use the metadata JSONL export (control_path_0, control_path_1, …) instead.")
    return "\n".join(lines) + "\n"


def ai_toolkit_yaml(s: dict) -> str:
    res = [512, 768, 1024] if int(s["resolution"]) >= 1024 else [512]
    entry = {"folder_path": s["folder"], "caption_ext": "txt", "caption_dropout_rate": float(s.get("caption_dropout") or 0),
             "shuffle_tokens": bool(s["shuffle"]), "cache_latents_to_disk": True, "resolution": res}
    controls = s.get("control_dirs") or []
    if len(controls) == 1:
        entry["control_path"] = controls[0]
    elif controls:
        entry["control_path"] = controls[:3]
    lines = _header(s) + ["# Paste into config.process[0].datasets of your ai-toolkit config."]
    if s.get("trigger"):
        lines.append(f'# Also set config.process[0].trigger_word: "{s["trigger"]}" (captions may contain [trigger]).')
    if s.get("variant_files"):
        lines.append("# How ai-toolkit treats extra caption lines was not verified: normalize them first to be safe.")
    lines.append("# ai-toolkit trains for a number of steps; repeats are not needed.")
    return "\n".join(lines) + "\n" + yaml.safe_dump({"datasets": [entry]}, sort_keys=False, allow_unicode=True)


def onetrainer_json(s: dict) -> str:
    concept = {
        "__version": 2, "name": s["name"], "path": s["folder"], "enabled": True, "type": "STANDARD",
        "include_subdirectories": False, "image_variations": 1, "text_variations": 1,
        "balancing": float(s["repeats"]), "balancing_strategy": "REPEATS", "loss_weight": 1.0,
        "text": {"prompt_source": "sample", "prompt_path": "", "enable_tag_shuffling": bool(s["shuffle"]),
                 "tag_delimiter": ",", "keep_tags_count": int(s["keep_tokens"])},
    }
    return json.dumps([concept], indent=2, ensure_ascii=False) + "\n"


def simpletuner_json(s: dict) -> str:
    sid = s["name"].replace(" ", "-")
    backend = {
        "id": sid, "type": "local", "instance_data_dir": s["folder"], "caption_strategy": "textfile",
        "resolution": int(s["resolution"]), "resolution_type": "pixel_area", "minimum_image_size": int(s["min_bucket"]),
        "crop": False, "cache_dir_vae": f"cache/vae/{sid}",
        "repeats": max(0, int(s["repeats"]) - 1),  # SimpleTuner: 0 = no repeats (kohya's 1)
    }
    if s.get("variant_files"):
        backend["disable_multiline_split"] = False  # Each line is used as a separate caption
    text = {"id": f"{sid}-text-embeds", "type": "local", "dataset_type": "text_embeds", "default": True,
            "cache_dir": f"cache/text/{sid}"}
    return json.dumps([backend, text], indent=2, ensure_ascii=False) + "\n"


def diffusion_pipe_toml(s: dict) -> str:
    lines = _header(s)
    if s.get("variant_files"):
        lines.append("# How diffusion-pipe treats extra caption lines was not verified: normalize them first to be safe.")
    lines += ["", f"resolutions = [{int(s['resolution'])}]", "enable_ar_bucket = true", "min_ar = 0.5", "max_ar = 2.0",
              "num_ar_buckets = 7", "", "[[directory]]", f"path = {_toml_str(s['folder'])}", f"num_repeats = {int(s['repeats'])}"]
    return "\n".join(lines) + "\n"


_WRITERS = {"kohya": kohya_toml, "musubi": musubi_toml, "ai_toolkit": ai_toolkit_yaml, "onetrainer": onetrainer_json,
            "simpletuner": simpletuner_json, "diffusion_pipe": diffusion_pipe_toml}


def render_config(trainer: str, settings: dict) -> str:
    s = dict(settings, trainer=trainer)
    return _WRITERS[trainer](s)


def write_config(trainer: str, settings: dict) -> tuple[str, str]:
    """Writes the config into <dataset>/.artagents/export/. Returns (path, text)."""
    text = render_config(trainer, settings)
    out_dir = os.path.join(settings["folder"], META_DIR, "export")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, TRAINERS[trainer][1])
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path, text


def code_language(trainer: str) -> str | None:
    return {"ai_toolkit": "yaml", "onetrainer": "json", "simpletuner": "json"}.get(trainer)  # TOML: plain text
