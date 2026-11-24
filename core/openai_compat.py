# ArtAgent/core/openai_compat.py
"""
Minimal client for OpenAI-compatible chat servers: LM Studio, vLLM, llama.cpp
(llama-server), text-generation-webui, KoboldCpp and similar.

Used for vision captioning with models that are easier to run outside Ollama
(e.g. JoyCaption Beta One or ToriiGate on vLLM). Only /models and
/chat/completions are used. Images are sent as base64 data URLs.
"""
from __future__ import annotations

import base64
import io
import os
from urllib.parse import urlparse

import requests

from .ollama_client import strip_think_blocks

API_KEY_ENV = "ARTAGENTS_VISION_API_KEY"  # Optional; never falls back to OPENAI_API_KEY (could leak to a local server)


class OpenAICompatError(Exception):
    """Connection or server error from an OpenAI-compatible endpoint."""


def normalize_base_url(url: str) -> str:
    """'http://localhost:1234' -> 'http://localhost:1234/v1'; URLs with a path are kept as given."""
    url = (url or "").strip().rstrip("/")
    if not url:
        raise OpenAICompatError("Enter the server URL, e.g. http://localhost:1234/v1 (LM Studio) or http://localhost:8000/v1 (vLLM).")
    if "://" not in url:
        url = "http://" + url
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise OpenAICompatError(f"Not a valid server URL: {url}")
    if parsed.path in ("", "/"):
        url += "/v1"
    return url


def image_part(img, max_side: int | None = None) -> dict:
    """PIL image -> OpenAI 'image_url' content part with a PNG (or JPEG for photos) data URL."""
    image = img
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    fmt, mime = ("JPEG", "image/jpeg") if max(image.size) > 768 else ("PNG", "image/png")
    image.save(buf, format=fmt, **({"quality": 92} if fmt == "JPEG" else {}))
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def build_messages(system: str | None, text: str, images: list | None = None) -> list[dict]:
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    if images:
        content = [image_part(img) for img in images] + [{"type": "text", "text": text}]
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": text})
    return messages


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 10.0, generation_timeout: float = 600.0):
        self.base_url = normalize_base_url(base_url)
        self.api_key = (api_key or os.environ.get(API_KEY_ENV, "")).strip()
        self.timeout = timeout
        self.generation_timeout = generation_timeout

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _error_text(resp: requests.Response) -> str:
        try:
            data = resp.json()
            err = data.get("error") if isinstance(data, dict) else None
            if isinstance(err, dict):
                return err.get("message") or str(err)
            if err:
                return str(err)
            if isinstance(data, dict) and data.get("detail"):
                return str(data["detail"])
        except ValueError:
            pass
        return (resp.text or resp.reason or "")[:300]

    def _post(self, path: str, payload: dict) -> requests.Response:
        try:
            return requests.post(self.base_url + path, json=payload, headers=self._headers(),
                                 timeout=(self.timeout, self.generation_timeout))
        except requests.exceptions.ConnectionError as e:
            raise OpenAICompatError(f"Cannot connect to {self.base_url}. Is the server running? ({e.__class__.__name__})") from e
        except requests.exceptions.Timeout as e:
            raise OpenAICompatError(f"The server at {self.base_url} did not answer in time.") from e
        except requests.exceptions.RequestException as e:
            raise OpenAICompatError(f"Request to {self.base_url} failed: {e}") from e

    def list_models(self) -> list[str]:
        try:
            resp = requests.get(self.base_url + "/models", headers=self._headers(), timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            raise OpenAICompatError(f"Cannot connect to {self.base_url}: {e.__class__.__name__}") from e
        if resp.status_code != 200:
            raise OpenAICompatError(f"{self.base_url}/models returned HTTP {resp.status_code}: {self._error_text(resp)}")
        try:
            data = resp.json().get("data", [])
        except (ValueError, AttributeError) as e:
            raise OpenAICompatError(f"{self.base_url}/models did not return JSON.") from e
        return sorted(str(m.get("id")) for m in data if isinstance(m, dict) and m.get("id"))

    def chat(self, model: str, messages: list[dict], temperature: float | None = None, seed: int | None = None,
             max_tokens: int | None = None, json_schema: dict | None = None) -> str:
        """Returns the answer text (reasoning blocks removed). Raises OpenAICompatError."""
        if not (model or "").strip():
            raise OpenAICompatError("Enter the model name served by the OpenAI-compatible server.")
        payload: dict = {"model": model.strip(), "messages": messages, "stream": False}
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if seed is not None:
            payload["seed"] = int(seed)
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        if json_schema:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "artagents_fields", "schema": json_schema}}
        resp = self._post("/chat/completions", payload)
        if resp.status_code == 400 and json_schema:
            # Some servers do not support structured output: retry once without it (the prompt still asks for JSON).
            payload.pop("response_format", None)
            resp = self._post("/chat/completions", payload)
        if resp.status_code != 200:
            raise OpenAICompatError(f"HTTP {resp.status_code} from {self.base_url}: {self._error_text(resp)}")
        try:
            choice = resp.json()["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content")
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise OpenAICompatError(f"Unexpected answer from {self.base_url}: {resp.text[:200]}") from e
        if isinstance(content, list):  # Some servers return content parts
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return strip_think_blocks(content or "")
