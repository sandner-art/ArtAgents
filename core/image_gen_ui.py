# ArtAgent/core/image_gen_ui.py
"""Callbacks for the Image Generation tab (backends live in core.image_gen)."""
import os
import random
import time

import gradio as gr

from .image_gen import BACKENDS, get_backend
from .image_gen.base import GenProgress, GenResult, ImageGenError, ImageGenRequest
from .image_gen.comfyui import list_workflows
from .image_gen.storage import save_images
from .image_gen_settings import backend_url, image_gen_settings, resolve_output_dir

MAX_GALLERY_ITEMS = 48
COMFYUI_LABEL = "ComfyUI"


def backend_labels() -> list[str]:
    return list(BACKENDS.keys())


def default_backend(settings: dict) -> str:
    label = image_gen_settings(settings).get("backend")
    return label if label in BACKENDS else backend_labels()[0]


def refresh_backend_options(backend_label: str, settings: dict):
    """
    Called when the backend changes or 🔄 is pressed.

    Returns updates for: model dropdown, workflow dropdown, sampler dropdown,
    negative prompt, steps, cfg, status text.
    """
    backend_cls = BACKENDS.get(backend_label)
    if backend_cls is None:
        return (gr.update(), gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update(),
                f"Unknown backend '{backend_label}'.")
    url = backend_url(settings, backend_label)
    backend = get_backend(backend_label, url)
    ok, message = backend.check()
    models, samplers = [], []
    if ok:
        try:
            models = backend.list_models()
        except ImageGenError as e:
            message += f" (could not list models: {e})"
        try:
            samplers = backend.list_samplers()
        except ImageGenError:
            samplers = []
    is_comfy = backend_label == COMFYUI_LABEL
    workflows = list_workflows() if is_comfy else []
    status = f"{'✅' if ok else '⚠️'} {backend_label} at {url}: {message}"
    if ok and not models:
        status += " No models found."
    return (
        gr.update(choices=models, value=models[0] if models else None),
        gr.update(visible=is_comfy, choices=workflows, value=workflows[0] if workflows else None),
        gr.update(choices=[""] + samplers, value="", interactive=bool(getattr(backend_cls, "supports_sampler", True))),
        gr.update(interactive=bool(getattr(backend_cls, "supports_negative_prompt", True))),
        gr.update(interactive=bool(getattr(backend_cls, "supports_steps", True))),
        gr.update(interactive=bool(getattr(backend_cls, "supports_cfg", True))),
        status,
    )


def clean_prompt_text(text: str) -> str:
    """Strips progress headers ('⏳ ...') that may be in the chat box while streaming."""
    if not isinstance(text, str):
        return ""
    lines = [line for line in text.splitlines() if not line.startswith("⏳")]
    return "\n".join(lines).strip()


def send_response_to_image_tab(response_text: str):
    """Chat → Image Generation: copies the response into the prompt and switches tabs."""
    return clean_prompt_text(response_text), gr.Tabs(selected="image_gen")


def iter_generate_images(
    prompt: str,
    negative_prompt: str,
    backend_label: str,
    model: str | None,
    workflow: str | None,
    sampler: str | None,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
    seed: int,
    batch_size: int,
    settings: dict,
    gallery_items: list | None,
):
    """
    Generates images, streaming progress. Yields (status_text, gallery_value, gallery_items_state).
    Stopping the event closes this generator, which cancels the job on the server.
    """
    gallery_items = list(gallery_items or [])
    prompt = (prompt or "").strip()
    if not prompt:
        yield "Enter a prompt first (or use 'Use last LLM response').", gallery_items, gallery_items
        return
    if backend_label not in BACKENDS:
        yield f"Unknown backend '{backend_label}'.", gallery_items, gallery_items
        return

    seed = int(seed) if seed is not None else -1
    if seed < 0:
        seed = random.randint(0, 2**31 - 1) # Pick it here so it can be shown and saved
    request = ImageGenRequest(
        prompt=prompt,
        negative_prompt=(negative_prompt or "").strip(),
        width=int(width), height=int(height),
        steps=int(steps) if steps and int(steps) > 0 else None,
        cfg_scale=float(cfg_scale) if cfg_scale and float(cfg_scale) > 0 else None,
        seed=seed,
        batch_size=max(1, int(batch_size or 1)),
        model=model or None,
        sampler=sampler or None,
        workflow=workflow or None,
    )
    url = backend_url(settings, backend_label)
    backend = get_backend(backend_label, url)
    started = time.time()
    status_prefix = f"{backend_label} · seed {seed} · {request.width}x{request.height}"
    yield f"⏳ {status_prefix}: sending job...", gallery_items, gallery_items

    images = []
    try:
        for event in backend.generate_iter(request):
            if isinstance(event, GenProgress):
                pct = f" {event.fraction * 100:.0f}%" if event.fraction is not None else ""
                yield f"⏳ {status_prefix}:{pct} {event.message} ({time.time() - started:.0f}s)", gallery_items, gallery_items
            elif isinstance(event, GenResult):
                images = event.images
    except ImageGenError as e:
        yield f"⚠️ {e}", gallery_items, gallery_items
        return
    except Exception as e: # Unexpected backend bug; keep the UI alive
        yield f"⚠️ Unexpected error from {backend_label}: {e}", gallery_items, gallery_items
        return

    if not images:
        yield f"⚠️ {backend_label} finished but returned no images.", gallery_items, gallery_items
        return

    img_settings = image_gen_settings(settings)
    try:
        saved_paths = save_images(images, request, backend_label, resolve_output_dir(settings),
                                  save_prompt_txt=bool(img_settings.get("save_prompt_txt", True)))
    except OSError as e:
        saved_paths = []
        save_note = f" Could not save files: {e}"
    else:
        save_note = f" Saved to {os.path.dirname(saved_paths[0])}" if saved_paths else ""

    new_items = [(img.image, f"seed {img.seed}" if img.seed is not None else backend_label) for img in images]
    gallery_items = (new_items + gallery_items)[:MAX_GALLERY_ITEMS]
    elapsed = time.time() - started
    yield (f"✅ {len(images)} image(s) in {elapsed:.1f}s · {status_prefix}.{save_note}",
           gallery_items, gallery_items)


def clear_gallery():
    return [], [], "Gallery cleared (saved files are kept)."
