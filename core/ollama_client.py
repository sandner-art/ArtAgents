# ArtAgent/core/ollama_client.py
"""
Small, dependency-free client for the Ollama REST API.

- /api/chat with system prompts, images and streaming (content and "thinking" kept apart)
- /api/tags + /api/show for model discovery (vision / thinking / image capabilities)
- /api/ps, /api/version and model unloading

Everything raises OllamaError with a message that can be shown in the UI as-is.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Iterator
from urllib.parse import urlparse

import requests

DEFAULT_BASE_URL = "http://localhost:11434"

# Options accepted by current Ollama releases (api.Options + api.Runner).
# Older keys such as tfs_z, mirostat*, penalize_newline, numa, low_vram, f16_kv,
# vocab_only and use_mlock are ignored by the server, so they are not sent.
VALID_OPTIONS = {
    "num_keep", "seed", "num_predict", "top_k", "top_p", "min_p", "typical_p",
    "repeat_last_n", "temperature", "repeat_penalty", "presence_penalty",
    "frequency_penalty", "stop",
    "num_ctx", "num_batch", "num_gpu", "main_gpu", "use_mmap", "num_thread",
}

THINK_SETTINGS = ("off", "auto", "on")  # settings["ollama_think"]

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.S | re.I)

_caps_cache: dict[tuple[str, str], list[str]] = {}
_caps_lock = threading.Lock()
_warned_options: set[str] = set()


class OllamaError(Exception):
    """Raised for connection problems and errors reported by the Ollama server."""


def base_url_from_settings(settings: dict | None) -> str:
    """
    Returns scheme://host:port for the Ollama server.

    Accepts the new "ollama_base_url" setting or the legacy "ollama_url"
    (which used to hold the full .../api/generate URL).
    """
    settings = settings or {}
    raw = (settings.get("ollama_base_url") or settings.get("ollama_url") or DEFAULT_BASE_URL).strip()
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return DEFAULT_BASE_URL
    return f"{parsed.scheme}://{parsed.netloc}"


def strip_think_blocks(text: str) -> str:
    """Removes <think>...</think> sections some reasoning models put into the answer text."""
    if not isinstance(text, str) or "<think" not in text.lower():
        return text
    return _THINK_BLOCK_RE.sub("", text).strip()


def filter_options(options: dict | None) -> dict:
    """Drops options current Ollama no longer understands (warns once per key)."""
    clean = {}
    for key, value in (options or {}).items():
        if key in VALID_OPTIONS:
            clean[key] = value
        elif key not in _warned_options:
            _warned_options.add(key)
            print(f"Info: Ollama option '{key}' is no longer supported by Ollama and is not sent.")
    return clean


def clear_capability_cache() -> None:
    with _caps_lock:
        _caps_cache.clear()


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 10.0, generation_timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.generation_timeout = generation_timeout

    @classmethod
    def from_settings(cls, settings: dict | None, **kwargs) -> "OllamaClient":
        return cls(base_url_from_settings(settings), **kwargs)

    # ---------- low-level helpers ----------
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @staticmethod
    def _error_from_response(resp: requests.Response) -> str:
        try:
            data = resp.json()
            if isinstance(data, dict) and data.get("error"):
                return str(data["error"])
        except ValueError:
            pass
        return (resp.text or resp.reason or "").strip()[:500]

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        try:
            resp = requests.request(method, self._url(path), **kwargs)
        except requests.exceptions.ConnectionError as e:
            raise OllamaError(f"Could not connect to Ollama at {self.base_url}. Is it running?") from e
        except requests.exceptions.Timeout as e:
            raise OllamaError(f"Ollama at {self.base_url} did not respond in time.") from e
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Error communicating with Ollama: {e}") from e
        if resp.status_code >= 400:
            detail = self._error_from_response(resp)
            resp.close()
            raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {detail}")
        return resp

    # ---------- discovery ----------
    def version(self) -> str:
        return self._request("GET", "/api/version").json().get("version", "unknown")

    def list_models(self) -> list[dict]:
        """Installed models from /api/tags (name, size, details, capabilities if reported)."""
        data = self._request("GET", "/api/tags").json()
        return [m for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]

    def show(self, model: str) -> dict:
        return self._request("POST", "/api/show", json={"model": model}).json()

    def capabilities(self, model: str, use_cache: bool = True) -> list[str]:
        """e.g. ["completion", "vision", "thinking", "tools"]; [] if unknown."""
        key = (self.base_url, model)
        if use_cache:
            with _caps_lock:
                if key in _caps_cache:
                    return _caps_cache[key]
        try:
            caps = list(self.show(model).get("capabilities") or [])
        except OllamaError:
            return []
        with _caps_lock:
            _caps_cache[key] = caps
        return caps

    def running(self) -> list[dict]:
        return self._request("GET", "/api/ps").json().get("models", [])

    def unload(self, model: str) -> str:
        self._request("POST", "/api/generate", json={"model": model, "keep_alive": 0}, timeout=30).close()
        return f"Model '{model}' unloaded."

    # ---------- chat ----------
    def think_value(self, model: str, think_setting: str | bool | None) -> bool | None:
        """
        Maps the app's think setting to the request's "think" field.

        Only thinking-capable models get the field (sending it to other models
        can be rejected). "off" -> False, "on" -> True, "auto"/None -> model default.
        """
        if think_setting in (None, "auto"):
            return None
        if "thinking" not in self.capabilities(model):
            return None
        if isinstance(think_setting, bool):
            return think_setting
        return think_setting == "on"

    def chat_stream(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
        think: bool | str | None = None,
        keep_alive: str | int | None = None,
    ) -> Iterator[tuple[str, str]]:
        """
        Streams a chat completion.

        Yields ("thinking", text) and ("content", text) pieces as they arrive.
        Raises OllamaError for transport errors and errors reported in the stream.
        """
        payload: dict = {"model": model, "messages": messages, "stream": True}
        clean_options = filter_options(options)
        if clean_options:
            payload["options"] = clean_options
        if think is not None:
            payload["think"] = think
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        resp = self._request("POST", "/api/chat", json=payload, stream=True, timeout=(self.timeout, self.generation_timeout))
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as e:
                    raise OllamaError(f"Unreadable response from Ollama (check the Ollama server log): {e}") from e
                if chunk.get("error"):
                    raise OllamaError(f"Ollama returned an error: {chunk['error']}")
                message = chunk.get("message") or {}
                if message.get("thinking"):
                    yield "thinking", message["thinking"]
                if message.get("content"):
                    yield "content", message["content"]
                if chunk.get("done"):
                    break
        except requests.exceptions.RequestException as e:
            raise OllamaError(f"Connection to Ollama was interrupted: {e}") from e
        finally:
            resp.close()

    def chat(self, model: str, messages: list[dict], **kwargs) -> str:
        """Non-streaming convenience wrapper: returns the answer text only."""
        return strip_think_blocks("".join(text for kind, text in self.chat_stream(model, messages, **kwargs) if kind == "content"))


def build_messages(system: str | None, user: str, images_b64: list[str] | None = None, history: list[dict] | None = None) -> list[dict]:
    """Builds an /api/chat message list: optional system prompt, prior turns, then the user turn."""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    for turn in history or []:
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant") and isinstance(turn.get("content"), str):
            messages.append({"role": turn["role"], "content": turn["content"]})
    user_msg: dict = {"role": "user", "content": user}
    if images_b64:
        user_msg["images"] = list(images_b64)
    messages.append(user_msg)
    return messages
