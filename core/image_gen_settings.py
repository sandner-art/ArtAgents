# ArtAgent/core/image_gen_settings.py
"""Defaults, validation and saving for connection / image-generation settings."""
import os
from urllib.parse import urlparse

from .utils import get_absolute_path, load_json, save_json
from .ollama_client import base_url_from_settings

SETTINGS_FILE = "settings.json"

BACKEND_URL_KEYS = {
    "ComfyUI": "comfyui_url",
    "Forge / A1111 / SD.Next": "sdwebui_url",
    "Ollama (experimental)": "ollama_images_url",
}

IMAGE_GEN_DEFAULTS = {
    "backend": "ComfyUI",
    "comfyui_url": "http://127.0.0.1:8188",
    "sdwebui_url": "http://127.0.0.1:7860",
    "ollama_images_url": "http://localhost:11434",
    "output_dir": "outputs/images",
    "save_prompt_txt": True,
}


def image_gen_settings(settings: dict | None) -> dict:
    """settings["image_generation"] merged over the defaults."""
    merged = dict(IMAGE_GEN_DEFAULTS)
    stored = (settings or {}).get("image_generation")
    if isinstance(stored, dict):
        merged.update({k: v for k, v in stored.items() if v not in (None, "")})
    return merged


def backend_url(settings: dict | None, backend_label: str) -> str:
    key = BACKEND_URL_KEYS.get(backend_label, "comfyui_url")
    return image_gen_settings(settings)[key]


def resolve_output_dir(settings: dict | None) -> str:
    """Absolute output folder; relative settings are inside the project folder."""
    out = image_gen_settings(settings)["output_dir"]
    return out if os.path.isabs(out) else get_absolute_path(out)


def is_valid_http_url(url: str) -> bool:
    parsed = urlparse((url or "").strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def save_connection_settings_callback(
    previous_status: str,
    ollama_url: str,
    think_mode: str,
    comfyui_url: str,
    sdwebui_url: str,
    ollama_images_url: str,
    output_dir: str,
    save_prompt_txt: bool,
):
    """
    Saves Ollama connection and image-server settings (chained after the main Save button).

    Returns (status_text, reloaded_settings) so the running app uses the new
    values immediately, without a restart.
    """
    settings = load_json(SETTINGS_FILE, is_relative=True)
    if not isinstance(settings, dict):
        settings = {}
    problems = []
    for label, url in (("Ollama", ollama_url), ("ComfyUI", comfyui_url),
                       ("Forge/A1111", sdwebui_url), ("Ollama images", ollama_images_url)):
        if url and not is_valid_http_url(url):
            problems.append(f"{label} URL must start with http:// or https:// ('{url}')")
    if problems:
        return f"{previous_status}\nNot saved: " + "; ".join(problems), settings

    if ollama_url:
        base = base_url_from_settings({"ollama_base_url": ollama_url})
        settings["ollama_base_url"] = base
        settings["ollama_url"] = base # Legacy key, kept in sync
    settings["ollama_think"] = think_mode if think_mode in ("off", "auto", "on") else "off"
    image_settings = image_gen_settings(settings)
    image_settings.update({
        "comfyui_url": (comfyui_url or IMAGE_GEN_DEFAULTS["comfyui_url"]).strip(),
        "sdwebui_url": (sdwebui_url or IMAGE_GEN_DEFAULTS["sdwebui_url"]).strip(),
        "ollama_images_url": (ollama_images_url or IMAGE_GEN_DEFAULTS["ollama_images_url"]).strip(),
        "output_dir": (output_dir or IMAGE_GEN_DEFAULTS["output_dir"]).strip(),
        "save_prompt_txt": bool(save_prompt_txt),
    })
    settings["image_generation"] = image_settings
    if not save_json(SETTINGS_FILE, settings, is_relative=True):
        return f"{previous_status}\nError: could not write {SETTINGS_FILE}.", settings
    reloaded = load_json(SETTINGS_FILE, is_relative=True)
    return f"{previous_status}\nConnection settings saved and applied.", reloaded if isinstance(reloaded, dict) else settings
