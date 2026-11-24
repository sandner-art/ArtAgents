# ArtAgent/core/image_gen/gradient.py
"""
GRADIENT backend (https://github.com/sandner-art/gradient).

GRADIENT runs diffusers models behind a job queue: every image is one
`text_to_image` job (POST /api/v1/queue), polled with GET /api/v1/queue/{id}; the
finished PNG comes from GET /api/v1/history/outputs/{id}/image.

Why it suits LoRA tests: a real no-LoRA baseline (`loras: []` disables LoRA layers),
LoRAs stay loaded between jobs, LoRA files can be given by absolute path, and every
image has its own seeded generator.

Backend-specific request options (ImageGenRequest.extra):
  quantization, vae_id   GRADIENT's per-render precision and VAE override
  guidance               a CFG schedule string ('linear(9,3)', '9@0,4@20'); overrides cfg_scale
  label, group, tags, origin   how the job appears in GRADIENT's queue and gallery
                               (group, tags and origin need GRADIENT 0.9.1+; older
                               versions ignore them)

Steps and CFG left unset come from GRADIENT's recommended settings for the model
(GET /api/v1/models/local), else GRADIENT's own defaults (25 steps, CFG 7.5). GRADIENT
0.9.1+ reports LoRAs that matched no layer of the model (`loras_skipped`).

Optional API token for a GRADIENT on another machine: ARTAGENTS_GRADIENT_TOKEN or the
`token` argument (sent as a Bearer token).
"""
from __future__ import annotations

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import requests
from PIL import Image

from .base import (
    GeneratedImage, GenProgress, GenResult, ImageBackend, ImageGenError, ImageGenRequest, resolve_seed,
    response_error_detail,
)

TOKEN_ENV = "ARTAGENTS_GRADIENT_TOKEN"
API = "/api/v1"
LABEL_PREFIX = "ArtAgents"
DONE, FAILED, CANCELLED = "completed", "failed", "cancelled"
GRADIENT_STEPS, GRADIENT_CFG = 25, 7.5  # What GRADIENT uses when neither the request nor the model says
DEFAULTS_TTL = 300.0
_defaults_cache: dict[tuple[str, str], tuple[float, dict]] = {}


def round8(value: int) -> int:
    """GRADIENT needs width and height in multiples of 8."""
    return max(8, (int(value) // 8) * 8)


class GradientBackend(ImageBackend):
    label = "GRADIENT"
    url_suffixes = ("/api/v1", "/api", "/docs")

    def __init__(self, base_url: str, timeout: float = 30.0, generation_timeout: float = 3600.0, token: str | None = None):
        super().__init__(base_url, timeout)
        self.generation_timeout = generation_timeout
        self.token = (token or os.environ.get(TOKEN_ENV, "")).strip()
        self._active: list[str] = []  # Job ids submitted by this instance and not finished

    # --- HTTP ------------------------------------------------------------------
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _api(self, method: str, path: str, **kwargs) -> requests.Response:
        kwargs.setdefault("headers", {}).update(self._headers())
        resp = self._call(getattr(requests, method), f"{self.base_url}{API}{path}", **kwargs)
        if resp.status_code in (401, 403):
            raise ImageGenError("GRADIENT refused the request: its API token is missing or wrong "
                                f"(set {TOKEN_ENV} or the token field).")
        return resp

    def _json(self, resp: requests.Response, what: str):
        if resp.status_code >= 400:
            raise ImageGenError(f"GRADIENT {what} failed (HTTP {resp.status_code}): {response_error_detail(resp)}")
        try:
            return resp.json()
        except ValueError as e:
            raise ImageGenError(f"GRADIENT returned an unreadable {what}.") from e

    # --- Information -----------------------------------------------------------
    def check(self) -> tuple[bool, str]:
        try:
            status = self._json(self._api("get", "/status"), "status")
            models = self.list_models()
        except ImageGenError as e:
            return False, str(e)
        device = ""
        try:
            info = self._json(self._api("get", "/system/info"), "system info")
            dev = info.get("device") or {}
            device = dev.get("name") or dev.get("device") or ""
        except ImageGenError:
            pass
        bits = [f"GRADIENT {status.get('version', '?')}"]
        if device:
            bits.append(device)
        bits.append(f"{len(models)} model(s)")
        return True, " · ".join(bits)

    def list_models(self) -> list[str]:
        return list(self._json(self._api("get", "/system/models/diffusion"), "model list") or [])

    def list_loras(self) -> list[str]:
        return list(self._json(self._api("get", "/system/models/loras"), "LoRA list") or [])

    def list_samplers(self) -> list[str]:
        """GRADIENT calls them schedulers (diffusers scheduler classes)."""
        return list(self._json(self._api("get", "/system/schedulers"), "scheduler list") or [])

    def list_vaes(self) -> list[str]:
        return list(self._json(self._api("get", "/system/models/vae"), "VAE list") or [])

    def list_precisions(self, model: str | None = None) -> list[tuple[str, str]]:
        """[(label, value)] of precisions that work on this machine (and model)."""
        params = {"model_id": model} if model else None
        data = self._json(self._api("get", "/system/quantization", params=params), "precision list")
        return [(o.get("label") or o.get("value"), o.get("value")) for o in (data or {}).get("options", [])
                if o.get("available") and o.get("value")]

    def model_defaults(self, model: str | None) -> dict:
        """GRADIENT's recommended settings for a model (steps, guidance_scale, width, height,
        scheduler; from its model manifest), {} when it has none. Cached for a few minutes."""
        if not model:
            return {}
        key = (self.base_url, model)
        hit = _defaults_cache.get(key)
        if hit and time.monotonic() - hit[0] < DEFAULTS_TTL:
            return dict(hit[1])
        found: dict = {}
        for kind in ("diffusers", "checkpoints"):
            try:
                data = self._json(self._api("get", "/models/local", params={"type": kind}), "model details")
            except ImageGenError:
                continue
            entry = next((m for m in data or [] if isinstance(m, dict) and m.get("name") == model), None)
            if entry is not None:
                found = {k: v for k, v in (entry.get("defaults") or {}).items() if v not in (None, "")}
                break
        _defaults_cache[key] = (time.monotonic(), found)
        return dict(found)

    @staticmethod
    def _wants_defaults(req: ImageGenRequest) -> bool:
        guidance = (req.extra or {}).get("guidance")
        return not req.steps or (req.cfg_scale is None and guidance in (None, ""))

    # --- Jobs ------------------------------------------------------------------
    def build_payload(self, req: ImageGenRequest, defaults: dict | None = None) -> dict:
        """The text_to_image payload; `defaults` (model_defaults) fill steps and CFG left unset."""
        if not req.model:
            raise ImageGenError("Pick a GRADIENT model first.")
        extra = req.extra or {}
        defaults = defaults or {}
        payload = {
            "model_id": req.model, "prompt": req.prompt or "", "negative_prompt": req.negative_prompt or "",
            "width": round8(req.width), "height": round8(req.height), "seed": resolve_seed(req.seed),
            "num_outputs": max(1, min(4, int(req.batch_size or 1))),  # GRADIENT renders up to 4 per job
            "loras": [{"name": str(name), "weight": float(weight)} for name, weight in (req.loras or [])],
        }
        if req.steps:
            payload["num_inference_steps"] = int(req.steps)
        elif defaults.get("steps"):
            payload["num_inference_steps"] = int(defaults["steps"])
        guidance = (extra.get("guidance") or "").strip() if isinstance(extra.get("guidance"), str) else extra.get("guidance")
        if guidance not in (None, ""):
            try:
                payload["guidance_scale"] = float(guidance)
            except (TypeError, ValueError):
                payload["guidance_scale"] = guidance  # A schedule: 'linear(9,3)', '9@0,4@20'
        elif req.cfg_scale is not None:
            payload["guidance_scale"] = float(req.cfg_scale)
        elif defaults.get("guidance_scale") is not None:
            payload["guidance_scale"] = float(defaults["guidance_scale"])
        if req.sampler:
            payload["scheduler_id"] = req.sampler
        for key in ("quantization", "vae_id"):
            if extra.get(key):
                payload[key] = extra[key]
        return payload

    def submit(self, req: ImageGenRequest) -> str:
        """Queues one image in GRADIENT; returns the job id."""
        extra = req.extra or {}
        payload = self.build_payload(req, self.model_defaults(req.model) if self._wants_defaults(req) else None)
        body = {"kind": "text_to_image", "payload": payload, "count": 1, "vary_seed": False,
                "label": extra.get("label") or f"{LABEL_PREFIX} · {' '.join((req.prompt or '').split())[:48]}"}
        for key in ("group", "tags", "origin"):
            if extra.get(key):
                body[key] = extra[key]
        data = self._json(self._api("post", "/queue", json=body), "job submission")
        try:
            job_id = data["jobs"][0]["id"]
        except (KeyError, IndexError, TypeError) as e:
            raise ImageGenError("GRADIENT accepted the job but returned no job id.") from e
        self._active.append(job_id)
        return job_id

    def wait_iter(self, job_id: str) -> Iterator[GenProgress | GenResult]:
        """Polls one job until it finishes; yields progress, then one GenResult."""
        deadline = time.monotonic() + self.generation_timeout
        while True:
            job = self._json(self._api("get", f"/queue/{job_id}"), "job status")
            status = job.get("status")
            if status == DONE:
                break
            if status == FAILED:
                self._forget(job_id)
                raise ImageGenError(f"GRADIENT job failed: {job.get('error') or 'no reason given'}")
            if status == CANCELLED:
                self._forget(job_id)
                raise ImageGenError("The job was cancelled in GRADIENT.")
            progress = job.get("progress") or {}
            if status == "running":
                step, total = progress.get("step"), progress.get("total")
                fraction = (float(step) / float(total)) if step is not None and total else None
                phase = progress.get("phase") or "rendering"
                yield GenProgress(fraction, f"GRADIENT: {phase}" + (f" {step}/{total}" if total else ""))
            else:
                position = job.get("position")
                yield GenProgress(None, "Waiting in GRADIENT's queue" + (f" (position {position + 1})" if isinstance(position, int) else ""))
            if time.monotonic() > deadline:
                self.cancel(job_id)
                raise ImageGenError("GRADIENT did not finish the job in time.")
            time.sleep(self.poll_interval)
        self._forget(job_id)
        result = job.get("result") or {}
        outputs = result.get("outputs") or []
        if not outputs:
            raise ImageGenError("GRADIENT finished the job but returned no image.")
        images = []
        for out in outputs:
            resp = self._api("get", f"/history/outputs/{out['id']}/image")
            if resp.status_code >= 400:
                raise ImageGenError(f"Could not fetch GRADIENT image {out['id']} (HTTP {resp.status_code}).")
            try:
                img = Image.open(io.BytesIO(resp.content), formats=["PNG", "JPEG", "WEBP"])
                img.load()
            except Exception as e:
                raise ImageGenError(f"GRADIENT image {out['id']} could not be read: {e}") from e
            images.append(GeneratedImage(img, out.get("seed"), {
                "output_id": out.get("id"), "run_uuid": result.get("run_uuid") or job.get("run_uuid"),
                "gradient_job": job_id, "loras_skipped": list(result.get("loras_skipped") or []),
            }))
        yield GenResult(images)

    def generate_iter(self, req: ImageGenRequest) -> Iterator[GenProgress | GenResult]:
        job_id = self.submit(req)
        finished = False
        try:
            yield GenProgress(0.0, "Queued in GRADIENT")
            yield from self.wait_iter(job_id)
            finished = True
        finally:
            if not finished:  # Stopped, closed early or failed: don't leave the job running
                self.cancel(job_id)

    def cancel(self, job_id: str) -> None:
        try:
            self._api("post", f"/queue/{job_id}/cancel")
        except ImageGenError:
            pass
        self._forget(job_id)

    def cancel_many(self, job_ids: list[str]) -> None:
        """Cancels several jobs, eight at a time (GRADIENT cancels one job per call)."""
        ids = [j for j in job_ids if j]
        if ids:
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(self.cancel, ids))

    def _forget(self, job_id: str) -> None:
        if job_id in self._active:
            self._active.remove(job_id)

    def interrupt(self) -> None:
        """Cancels this instance's jobs, and any waiting or running ArtAgents job in GRADIENT's queue."""
        ids = list(self._active)
        try:
            data = self._json(self._api("get", "/queue"), "queue")
            ids += [job["id"] for job in (data or {}).get("jobs", [])
                    if job.get("status") in ("queued", "running") and str(job.get("label", "")).startswith(LABEL_PREFIX)
                    and job["id"] not in ids]
        except ImageGenError:
            pass
        self.cancel_many(ids)
