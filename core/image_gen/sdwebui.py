# ArtAgent/core/image_gen/sdwebui.py
"""
AUTOMATIC1111 / Forge / SD.Next backend (they share the /sdapi/v1 API).

The WebUI must be started with --api (A1111/Forge). txt2img runs in a worker
thread while /sdapi/v1/progress is polled for live step progress.
"""
from __future__ import annotations

import base64
import binascii
import json
import threading
import time
from typing import Iterator

import requests

from core.image_utils import open_image_bytes
from .base import (
    GeneratedImage, GenProgress, GenResult, ImageBackend, ImageGenError, ImageGenRequest,
    response_error_detail,
)

API_NOT_ENABLED = (
    "API not enabled — start the WebUI with --api (Forge/A1111) or check the URL. "
    "Note ArtAgents itself also defaults to port 7860."
)


def decode_b64_image(data: str):
    """Decodes a base64 image string (optionally a data: URL) with the PNG/JPEG/WEBP whitelist."""
    if not isinstance(data, str):
        raise ImageGenError("Server returned a non-text image payload.")
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=False)
    except (binascii.Error, ValueError) as e:
        raise ImageGenError(f"Server returned invalid base64 image data: {e}") from None
    try:
        return open_image_bytes(raw)
    except Exception as e:
        raise ImageGenError(f"Server returned an unreadable image: {e}") from None


class SDWebUIBackend(ImageBackend):
    label = "Forge / A1111 / SD.Next"
    url_suffixes = ("/sdapi/v1", "/docs")

    def __init__(self, base_url: str, timeout: float = 30.0, generation_timeout: float = 600.0):
        super().__init__(base_url, timeout)
        self.generation_timeout = generation_timeout

    def _api_url(self, path: str) -> str:
        return f"{self.base_url}/sdapi/v1{path}"

    def _api(self, method: str, path: str, **kwargs) -> requests.Response:
        resp = self._call(getattr(requests, method), self._api_url(path), **kwargs)
        if resp.status_code == 404:
            raise ImageGenError(API_NOT_ENABLED)
        return resp

    def _json(self, resp: requests.Response, what: str):
        if resp.status_code != 200:
            raise ImageGenError(f"{self.label} {what} failed (HTTP {resp.status_code}): {response_error_detail(resp)}")
        try:
            return resp.json()
        except ValueError:
            raise ImageGenError(f"{self.label} {what} returned invalid JSON (is this really a WebUI API URL?).") from None

    def check(self) -> tuple[bool, str]:
        try:
            models = self._json(self._api("get", "/sd-models"), "model list")
        except ImageGenError as e:
            return False, str(e)
        count = len(models) if isinstance(models, list) else 0
        return True, f"Connected to {self.label} API ({count} checkpoint(s))"

    def list_models(self) -> list[str]:
        models = self._json(self._api("get", "/sd-models"), "model list")
        return [m.get("title") or m.get("model_name") for m in models if isinstance(m, dict) and (m.get("title") or m.get("model_name"))]

    def list_samplers(self) -> list[str]:
        samplers = self._json(self._api("get", "/samplers"), "sampler list")
        return [s["name"] for s in samplers if isinstance(s, dict) and s.get("name")]

    def interrupt(self) -> None:
        try:
            self._api("post", "/interrupt")
        except ImageGenError:
            pass

    @staticmethod
    def build_payload(req: ImageGenRequest) -> dict:
        payload = {
            "prompt": req.prompt or "",
            "negative_prompt": req.negative_prompt or "",
            "width": int(req.width),
            "height": int(req.height),
            "steps": int(req.steps) if req.steps is not None else None,
            "cfg_scale": float(req.cfg_scale) if req.cfg_scale is not None else None,
            "seed": int(req.seed) if req.seed is not None else -1,
            "sampler_name": req.sampler or None,
            "batch_size": max(1, int(req.batch_size or 1)),
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        if req.model:
            payload["override_settings"] = {"sd_model_checkpoint": req.model}
            payload["override_settings_restore_afterwards"] = True
        return payload

    def _progress(self) -> GenProgress:
        try:
            data = self._json(self._api("get", "/progress", params={"skip_current_image": "true"}, timeout=5), "progress")
        except ImageGenError:
            return GenProgress(None, "Generating…")
        state = data.get("state") or {}
        step, total = state.get("sampling_step", 0), state.get("sampling_steps", 0)
        fraction = data.get("progress")
        fraction = float(fraction) if isinstance(fraction, (int, float)) else None
        if not total:
            return GenProgress(fraction, "Waiting for the WebUI to start sampling…")
        eta = data.get("eta_relative")
        eta_text = f", eta {eta:.0f}s" if isinstance(eta, (int, float)) and eta > 0 else ""
        job_no, job_count = state.get("job_no", 0), state.get("job_count", 0)
        job_text = f" (image {job_no + 1}/{job_count})" if isinstance(job_count, int) and job_count > 1 else ""
        return GenProgress(fraction, f"step {step}/{total}{eta_text}{job_text}")

    def _parse_result(self, resp: requests.Response, req: ImageGenRequest) -> list[GeneratedImage]:
        data = self._json(resp, "txt2img")
        encoded = data.get("images") or []
        if not encoded:
            raise ImageGenError(f"{self.label} returned no images.")
        info: dict = {}
        raw_info = data.get("info")
        if isinstance(raw_info, str):
            try:
                info = json.loads(raw_info)
            except ValueError:
                info = {}
        elif isinstance(raw_info, dict):
            info = raw_info
        seeds = info.get("all_seeds") or ([info["seed"]] if "seed" in info else [])
        if seeds and len(encoded) == len(seeds) + 1:
            encoded = encoded[1:]  # Leading grid image for batches
        images = []
        for index, item in enumerate(encoded):
            seed = seeds[index] if index < len(seeds) else (seeds[-1] if seeds else None)
            images.append(GeneratedImage(image=decode_b64_image(item), seed=seed, info={
                "backend": self.label, "batch_index": index,
                "sd_model_name": info.get("sd_model_name"), "sampler_name": info.get("sampler_name"),
                "steps": info.get("steps"), "cfg_scale": info.get("cfg_scale"),
            }))
        return images

    def generate_iter(self, req: ImageGenRequest) -> Iterator[GenProgress | GenResult]:
        payload = self.build_payload(req)
        url = self._api_url("/txt2img")
        outcome: dict = {}

        def worker():
            try:
                outcome["resp"] = requests.post(url, json=payload, timeout=self.generation_timeout)
            except Exception as e:  # handed back to the generator below
                outcome["error"] = e

        thread = threading.Thread(target=worker, name="sdwebui-txt2img", daemon=True)
        thread.start()
        finished = False
        try:
            yield GenProgress(0.0, f"Sent to {self.label}…")
            while thread.is_alive():
                thread.join(self.poll_interval)
                if thread.is_alive():
                    yield self._progress()
            if "error" in outcome:
                error = outcome["error"]
                if isinstance(error, requests.exceptions.ConnectionError):
                    raise ImageGenError(f"Could not connect to {self.label} at {self.base_url}. Is it running with --api?")
                if isinstance(error, requests.exceptions.Timeout):
                    raise ImageGenError(f"{self.label} did not finish within {self.generation_timeout:.0f}s.")
                raise ImageGenError(f"Request to {self.label} failed: {error}")
            resp = outcome["resp"]
            if resp.status_code == 404:
                raise ImageGenError(API_NOT_ENABLED)
            images = self._parse_result(resp, req)
            finished = True
            yield GenResult(images=images)
        except GeneratorExit:
            if not finished:
                self.interrupt()
            raise
