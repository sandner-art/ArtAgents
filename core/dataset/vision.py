# ArtAgent/core/dataset/vision.py
"""
Direct vision-model calls for dataset tools (no agent persona, no history).

One function, `ask`, works with Ollama and with OpenAI-compatible servers, so
structured captions, the verifier, edit-pair instructions and the captioner can
all use whichever backend the user picked. JSON-schema output is requested
where the backend supports it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..image_utils import encode_image_for_ollama
from ..ollama_client import OllamaClient, OllamaError, build_messages as ollama_messages, filter_options
from ..openai_compat import OpenAICompatClient, OpenAICompatError, build_messages as openai_messages

BACKEND_OLLAMA = "ollama"
BACKEND_OPENAI = "openai"
BACKEND_LABELS = {BACKEND_OLLAMA: "Ollama", BACKEND_OPENAI: "OpenAI-compatible server"}


class VisionError(Exception):
    """Any backend failure, with a message suitable for the UI."""


@dataclass
class VisionTarget:
    backend: str = BACKEND_OLLAMA
    model: str = ""
    base_url: str = ""   # OpenAI-compatible only (Ollama uses the App Settings URL)
    api_key: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.backend == BACKEND_OPENAI:
            return f"{self.model} @ {self.base_url}"
        return self.model


def clean_model_name(display_name: str) -> str:
    return (display_name or "").replace(" (VISION)", "").strip()


def ask(target: VisionTarget, settings: dict, prompt: str, images: list | None = None, system: str | None = None,
        temperature: float | None = None, seed: int | None = None, max_tokens: int | None = None,
        json_schema: dict | None = None) -> str:
    """Sends one prompt with images and returns the answer text. Raises VisionError."""
    model = clean_model_name(target.model)
    if not model:
        raise VisionError("Select a vision model first.")
    if target.backend == BACKEND_OPENAI:
        try:
            client = OpenAICompatClient(target.base_url, target.api_key)
            return client.chat(model, openai_messages(system, prompt, images), temperature=temperature, seed=seed,
                               max_tokens=max_tokens, json_schema=json_schema)
        except OpenAICompatError as e:
            raise VisionError(str(e)) from e

    options: dict = {}
    if temperature is not None:
        options["temperature"] = float(temperature)
    if seed is not None:
        options["seed"] = int(seed)
    if max_tokens:
        options["num_predict"] = int(max_tokens)
    try:
        client = OllamaClient.from_settings(settings or {})
        images_b64 = [encode_image_for_ollama(img) for img in images or []]
        think = client.think_value(model, (settings or {}).get("ollama_think", "off"))
        return client.chat(model, ollama_messages(system, prompt, images_b64), options=filter_options(options),
                           think=think, format=json_schema)
    except OllamaError as e:
        raise VisionError(str(e)) from e


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(text: str) -> dict:
    """Lenient JSON-object parsing: strips code fences and surrounding prose. Raises ValueError."""
    raw = _FENCE_RE.sub("", (text or "").strip())
    try:
        data = json.loads(raw)
    except ValueError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The model did not return a JSON object.")
        data = json.loads(raw[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("The model returned JSON, but not an object.")
    return data
