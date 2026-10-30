# ArtAgent/core/image_gen/base.py
"""
Shared types and the abstract backend for image generation.

Every backend talks to a *local* server the user runs themselves (ComfyUI,
Forge/A1111/SD.Next, Ollama). Generation is exposed as an iterator so the UI can
show progress and stop a job: closing the iterator early cancels the job on the
server.
"""
from __future__ import annotations

import abc
import random
from dataclasses import dataclass, field
from typing import Callable, Iterator

import requests
from PIL import Image

# Largest seed accepted by all supported backends (A1111 uses 32-bit seeds).
MAX_SEED = 2**32 - 1


@dataclass
class ImageGenRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int | None = None          # None = backend/workflow default
    cfg_scale: float | None = None    # None = backend/workflow default
    seed: int = -1                    # -1 = random
    batch_size: int = 1
    model: str | None = None          # A1111 checkpoint title / ComfyUI ckpt_name / Ollama model name
    sampler: str | None = None
    workflow: str | None = None       # ComfyUI workflow filename (image_workflows/comfyui/*.json)


@dataclass
class GeneratedImage:
    image: Image.Image
    seed: int | None
    info: dict = field(default_factory=dict)


@dataclass
class GenProgress:
    fraction: float | None  # 0..1, or None when the backend can't tell
    message: str


@dataclass
class GenResult:
    images: list[GeneratedImage]


class ImageGenError(Exception):
    """Error with a user-friendly message (shown in the UI as-is)."""


def resolve_seed(seed: int | None) -> int:
    """Returns a concrete seed: a random one for None/-1, otherwise the given value."""
    if seed is None or int(seed) < 0:
        return random.randint(0, MAX_SEED)
    return int(seed)


def normalize_base_url(url: str, strip_suffixes: tuple[str, ...] = ()) -> str:
    """'127.0.0.1:8188/' -> 'http://127.0.0.1:8188'; also strips known API suffixes."""
    url = (url or "").strip()
    if not url:
        raise ImageGenError("No server URL configured.")
    if "://" not in url:
        url = "http://" + url
    url = url.rstrip("/")
    for suffix in strip_suffixes:
        if url.lower().endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
    return url


def response_error_detail(resp: requests.Response, limit: int = 300) -> str:
    """Best-effort short error text from a JSON or plain-text error response."""
    try:
        data = resp.json()
    except ValueError:
        return (resp.text or "").strip()[:limit]
    if isinstance(data, dict):
        for key in ("error", "detail", "errors", "message"):
            if data.get(key):
                value = data[key]
                if isinstance(value, dict):
                    value = value.get("message") or value
                return str(value)[:limit]
    return str(data)[:limit]


class ImageBackend(abc.ABC):
    """Base class for an image generation server connection."""

    label: str = "Image backend"
    supports_negative_prompt: bool = True
    supports_steps: bool = True
    supports_cfg: bool = True
    supports_sampler: bool = True
    supports_seed: bool = True
    # Suffixes users commonly paste along with the server URL.
    url_suffixes: tuple[str, ...] = ()

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = normalize_base_url(base_url, self.url_suffixes)
        self.timeout = timeout
        self.poll_interval = 0.5

    # --- HTTP helper -------------------------------------------------------
    def _call(self, fn: Callable[..., requests.Response], url: str, **kwargs) -> requests.Response:
        """Runs one HTTP call, turning connection problems into friendly ImageGenErrors."""
        kwargs.setdefault("timeout", self.timeout)
        try:
            return fn(url, **kwargs)
        except requests.exceptions.ConnectionError:
            raise ImageGenError(
                f"Could not connect to {self.label} at {self.base_url}. Is it running, and is the URL right?"
            ) from None
        except requests.exceptions.Timeout:
            raise ImageGenError(f"{self.label} at {self.base_url} did not respond in time.") from None
        except requests.exceptions.RequestException as e:
            raise ImageGenError(f"Request to {self.label} failed: {e}") from None

    # --- Interface ---------------------------------------------------------
    @abc.abstractmethod
    def check(self) -> tuple[bool, str]:
        """Returns (ok, human-readable status)."""

    @abc.abstractmethod
    def list_models(self) -> list[str]:
        """Model/checkpoint names the user can pick."""

    def list_samplers(self) -> list[str]:
        return []

    @abc.abstractmethod
    def generate_iter(self, req: ImageGenRequest) -> Iterator[GenProgress | GenResult]:
        """Yields GenProgress updates and finally exactly one GenResult; raises ImageGenError."""

    def generate(self, req: ImageGenRequest) -> list[GeneratedImage]:
        """Blocking convenience wrapper around generate_iter()."""
        result = None
        for item in self.generate_iter(req):
            if isinstance(item, GenResult):
                result = item
        if result is None:
            raise ImageGenError(f"{self.label} finished without returning images.")
        return result.images

    def interrupt(self) -> None:
        """Asks the server to stop the current job (no-op if unsupported)."""
