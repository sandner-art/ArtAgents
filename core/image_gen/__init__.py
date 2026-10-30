# ArtAgent/core/image_gen/__init__.py
"""Image generation backends: ComfyUI, Forge/A1111/SD.Next and Ollama (experimental)."""
from .base import (
    GeneratedImage, GenProgress, GenResult, ImageBackend, ImageGenError, ImageGenRequest, resolve_seed,
)
from .comfyui import ComfyUIBackend, list_workflows
from .ollama_images import OllamaImageBackend
from .sdwebui import SDWebUIBackend
from .storage import save_images

BACKENDS: dict[str, type[ImageBackend]] = {
    "ComfyUI": ComfyUIBackend,
    "Forge / A1111 / SD.Next": SDWebUIBackend,
    "Ollama (experimental)": OllamaImageBackend,
}

DEFAULT_URLS = {
    "ComfyUI": "http://127.0.0.1:8188",
    "Forge / A1111 / SD.Next": "http://127.0.0.1:7860",
    "Ollama (experimental)": "http://localhost:11434",
}


def get_backend(label: str, base_url: str | None = None) -> ImageBackend:
    """Creates a backend by its UI label; uses the default URL when base_url is empty."""
    if label not in BACKENDS:
        raise ImageGenError(f"Unknown image backend: {label!r}")
    return BACKENDS[label](base_url or DEFAULT_URLS[label])


__all__ = [
    "BACKENDS", "DEFAULT_URLS", "get_backend", "list_workflows", "save_images",
    "ImageBackend", "ImageGenRequest", "GeneratedImage", "GenProgress", "GenResult", "ImageGenError",
    "ComfyUIBackend", "SDWebUIBackend", "OllamaImageBackend", "resolve_seed",
]
