# ArtAgent/core/image_gen/ollama_images.py
"""
Ollama (experimental) image backend, via the OpenAI-compatible
POST /v1/images/generations endpoint.

Ollama added experimental image generation (x/z-image-turbo, x/flux2-klein) in
early 2026 and temporarily removed it again in v0.32.6: current versions answer
HTTP 400 "image generation models are not currently supported". This backend
reports that clearly, works again if Ollama restores the feature, and also
works with other OpenAI-compatible image servers.
"""
from __future__ import annotations

from typing import Iterator

import requests

from .base import (
    GeneratedImage, GenProgress, GenResult, ImageBackend, ImageGenError, ImageGenRequest,
    response_error_detail,
)
from .sdwebui import decode_b64_image

OLLAMA_IMAGE_GEN_REMOVED = (
    "This Ollama version has image generation turned off: Ollama removed its experimental "
    "image generation in v0.32.6 (\"image generation models are not currently supported\"). "
    "Use the ComfyUI or Forge / A1111 / SD.Next backend instead."
)
IMAGE_MODEL_PREFIXES = ("x/z-image", "x/flux")


class OllamaImageBackend(ImageBackend):
    label = "Ollama (experimental)"
    supports_negative_prompt = False
    supports_steps = False
    supports_cfg = False
    supports_sampler = False
    supports_seed = False
    url_suffixes = ("/api/generate", "/api/chat", "/api", "/v1")

    def __init__(self, base_url: str, timeout: float = 30.0, generation_timeout: float = 600.0):
        super().__init__(base_url, timeout)
        self.generation_timeout = generation_timeout

    def _get(self, path: str, **kwargs) -> requests.Response:
        return self._call(requests.get, self.base_url + path, **kwargs)

    def check(self) -> tuple[bool, str]:
        try:
            resp = self._get("/api/version")
            if resp.status_code == 200:
                version = (resp.json() or {}).get("version", "?")
                return True, f"Connected to Ollama {version} (image generation is experimental and may be disabled)"
            resp = self._get("/v1/models")
            if resp.status_code == 200:
                return True, f"Connected to an OpenAI-compatible server at {self.base_url}"
            return False, f"{self.base_url} answered HTTP {resp.status_code}."
        except (ImageGenError, ValueError) as e:
            return False, str(e) or f"Unexpected response from {self.base_url}."

    def list_models(self) -> list[str]:
        resp = self._get("/api/tags")
        if resp.status_code == 404:  # Not Ollama: OpenAI-compatible server
            resp = self._get("/v1/models")
            if resp.status_code != 200:
                raise ImageGenError(f"Could not list models (HTTP {resp.status_code}).")
            try:
                return [m["id"] for m in resp.json().get("data", []) if isinstance(m, dict) and m.get("id")]
            except (ValueError, AttributeError):
                raise ImageGenError("Server returned an invalid model list.") from None
        if resp.status_code != 200:
            raise ImageGenError(f"Ollama model list failed (HTTP {resp.status_code}).")
        try:
            models = resp.json().get("models", [])
        except (ValueError, AttributeError):
            raise ImageGenError("Ollama returned an invalid model list.") from None
        if any(isinstance(m, dict) and "capabilities" in m for m in models):
            return [m["name"] for m in models if "image" in (m.get("capabilities") or [])]
        return [m["name"] for m in models if str(m.get("name", "")).startswith(IMAGE_MODEL_PREFIXES)]

    def generate_iter(self, req: ImageGenRequest) -> Iterator[GenProgress | GenResult]:
        if not req.model:
            raise ImageGenError("Pick an image model (e.g. x/z-image-turbo) in the Model dropdown.")
        count = max(1, int(req.batch_size or 1))
        images: list[GeneratedImage] = []
        for index in range(count):
            yield GenProgress(index / count, f"Generating image {index + 1}/{count} with {req.model}… (no live progress)")
            resp = self._call(
                requests.post, self.base_url + "/v1/images/generations",
                json={"model": req.model, "prompt": req.prompt or "", "size": f"{int(req.width)}x{int(req.height)}",
                      "response_format": "b64_json"},
                timeout=self.generation_timeout,
            )
            if resp.status_code != 200:
                detail = response_error_detail(resp)
                if resp.status_code == 400 and "not currently supported" in detail:
                    raise ImageGenError(OLLAMA_IMAGE_GEN_REMOVED)
                if resp.status_code == 404:
                    raise ImageGenError(f"{self.base_url} has no /v1/images/generations endpoint "
                                        f"(model '{req.model}' not found or server too old): {detail}")
                raise ImageGenError(f"Image generation failed (HTTP {resp.status_code}): {detail}")
            try:
                data = resp.json().get("data") or []
            except (ValueError, AttributeError):
                raise ImageGenError("Server returned invalid JSON for the image.") from None
            encoded = [d.get("b64_json") for d in data if isinstance(d, dict) and d.get("b64_json")]
            if not encoded:
                raise ImageGenError("Server returned no base64 image (b64_json).")
            for item in encoded:
                images.append(GeneratedImage(image=decode_b64_image(item), seed=None, info={
                    "backend": self.label, "model": req.model, "batch_index": len(images),
                }))
        yield GenResult(images=images)
