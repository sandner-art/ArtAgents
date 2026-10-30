# ArtAgent/core/image_gen/comfyui.py
"""
ComfyUI backend.

Workflows are ComfyUI *API-format* JSON files in image_workflows/comfyui/.
Values are injected either through %placeholders% or, if the workflow has no
%prompt% placeholder, by auto-detecting the sampler, text-encoder, latent and
checkpoint nodes. See image_workflows/README.md.
"""
from __future__ import annotations

import copy
import json
import os
import time
import uuid
from typing import Any, Iterator

import requests

from core.image_utils import open_image_bytes
from .base import (
    GeneratedImage, GenProgress, GenResult, ImageBackend, ImageGenError, ImageGenRequest,
    resolve_seed, response_error_detail,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKFLOW_DIR = os.path.join(PROJECT_ROOT, "image_workflows", "comfyui")
DEFAULT_WORKFLOW = "basic_txt2img.json"

# Fallbacks used when a workflow has a placeholder but the UI left the value unset.
PLACEHOLDER_DEFAULTS = {"steps": 20, "cfg": 7.0, "sampler": "euler", "batch_size": 1}

SAMPLER_CLASSES = {"KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"}
# Text-encoder inputs that hold the prompt text (CLIPTextEncode, SDXL, Flux, SD3, HiDream, ...).
TEXT_INPUTS = ("text", "text_g", "text_l", "clip_l", "clip_g", "t5xxl", "llama")
# Inputs followed upstream when looking for the text encoder behind a conditioning link.
CONDITIONING_INPUTS = ("conditioning", "conditioning_1", "conditioning_to", "positive", "negative")
CHECKPOINT_CLASSES = {"CheckpointLoaderSimple", "CheckpointLoader", "ImageOnlyCheckpointLoader",
                      "CheckpointLoaderNF4", "unCLIPCheckpointLoader"}


# --- Workflow files ------------------------------------------------------------

def list_workflows(directory: str = WORKFLOW_DIR) -> list[str]:
    """Workflow filenames available in image_workflows/comfyui/ (sorted)."""
    try:
        return sorted(f for f in os.listdir(directory) if f.lower().endswith(".json"))
    except FileNotFoundError:
        return []


def load_workflow(name: str, directory: str = WORKFLOW_DIR) -> dict:
    """Loads and validates an API-format workflow by filename (no paths allowed)."""
    if not name or os.path.basename(name) != name or not name.lower().endswith(".json") or name.startswith("."):
        raise ImageGenError(f"Invalid workflow name: {name!r}")
    path = os.path.join(directory, name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ImageGenError(f"Workflow '{name}' not found in image_workflows/comfyui/.") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ImageGenError(f"Workflow '{name}' is not valid JSON: {e}") from None
    validate_api_workflow(data, name)
    return data


def validate_api_workflow(data: Any, name: str = "workflow") -> None:
    if isinstance(data, dict) and "nodes" in data and "links" in data:
        raise ImageGenError(
            f"'{name}' is a ComfyUI UI-format workflow. In ComfyUI use Workflow > Export (API) "
            "(or 'Save (API Format)' with dev mode enabled) and put that file in image_workflows/comfyui/."
        )
    if not isinstance(data, dict) or not data or not all(
        isinstance(node, dict) and "class_type" in node and isinstance(node.get("inputs", {}), dict)
        for node in data.values()
    ):
        raise ImageGenError(f"'{name}' is not a ComfyUI API-format workflow (expected {{node_id: {{class_type, inputs}}}}).")


# --- Parameter injection -------------------------------------------------------

def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int))


def _has_placeholder(workflow: dict, placeholder: str) -> bool:
    return any(
        isinstance(v, str) and placeholder in v
        for node in workflow.values() for v in node.get("inputs", {}).values()
    )


def _apply_placeholders(workflow: dict, req: ImageGenRequest, seed: int) -> None:
    values: dict[str, Any] = {
        "%seed%": seed,
        "%width%": int(req.width),
        "%height%": int(req.height),
        "%steps%": int(req.steps) if req.steps is not None else PLACEHOLDER_DEFAULTS["steps"],
        "%cfg%": float(req.cfg_scale) if req.cfg_scale is not None else PLACEHOLDER_DEFAULTS["cfg"],
        "%batch_size%": max(1, int(req.batch_size or 1)),
        "%sampler%": req.sampler or PLACEHOLDER_DEFAULTS["sampler"],
    }
    text_values = {"%prompt%": req.prompt or "", "%negative%": req.negative_prompt or ""}
    for node_id, node in workflow.items():
        inputs = node.get("inputs", {})
        for key, value in list(inputs.items()):
            if not isinstance(value, str):
                continue
            if value == "%checkpoint%":
                if not req.model:
                    raise ImageGenError("This workflow needs a checkpoint: pick one in the Model dropdown.")
                inputs[key] = req.model
            elif value in values:
                inputs[key] = values[value]
            elif any(p in value for p in text_values):
                # Text placeholders may be embedded, e.g. "photo of %prompt%, film grain".
                for placeholder, text in text_values.items():
                    value = value.replace(placeholder, text)
                inputs[key] = value


def _set_text(node: dict, text: str) -> bool:
    inputs = node.get("inputs", {})
    changed = False
    for key in TEXT_INPUTS:
        if isinstance(inputs.get(key), str):
            inputs[key] = text
            changed = True
    return changed


def _find_text_encoders(workflow: dict, link: Any, depth: int = 0, seen: set | None = None) -> list[str]:
    """Follows a conditioning link upstream to the node(s) that hold prompt text."""
    seen = seen if seen is not None else set()
    if not _is_link(link) or depth > 8:
        return []
    node_id = str(link[0])
    if node_id in seen or node_id not in workflow:
        return []
    seen.add(node_id)
    node = workflow[node_id]
    if "ZeroOut" in node.get("class_type", ""):
        return []  # e.g. Flux negative = ConditioningZeroOut(positive): never write there
    inputs = node.get("inputs", {})
    if any(isinstance(inputs.get(k), str) for k in TEXT_INPUTS):
        return [node_id]
    found: list[str] = []
    for key in CONDITIONING_INPUTS:
        found += _find_text_encoders(workflow, inputs.get(key), depth + 1, seen)
    return found


def _linked_node(workflow: dict, link: Any) -> dict | None:
    return workflow.get(str(link[0])) if _is_link(link) else None


def _set_number(workflow: dict, node: dict, keys: tuple[str, ...], value: Any) -> bool:
    """Sets the first matching input; follows one link hop (e.g. a seed primitive node)."""
    inputs = node.get("inputs", {})
    for key in keys:
        if key not in inputs:
            continue
        if _is_link(inputs[key]):
            target = _linked_node(workflow, inputs[key])
            if target:
                for target_key in keys + ("value", "int", "number"):
                    if isinstance(target.get("inputs", {}).get(target_key), (int, float)):
                        target["inputs"][target_key] = value
                        return True
            return False
        inputs[key] = value
        return True
    return False


def _auto_inject(workflow: dict, req: ImageGenRequest, seed: int, name: str) -> None:
    samplers = [n for n in workflow.values() if n.get("class_type") in SAMPLER_CLASSES]
    positive_ids: set[str] = set()
    negative_ids: set[str] = set()
    for sampler in samplers:
        inputs = sampler.get("inputs", {})
        cond_sources = [inputs]
        guider = _linked_node(workflow, inputs.get("guider"))
        if guider:
            cond_sources.append(guider.get("inputs", {}))
            if req.cfg_scale is not None and "cfg" in guider.get("inputs", {}):
                guider["inputs"]["cfg"] = float(req.cfg_scale)
            # BasicGuider (Flux) has a single "conditioning" input = positive.
            if "conditioning" in guider.get("inputs", {}):
                positive_ids.update(_find_text_encoders(workflow, guider["inputs"]["conditioning"]))
        for source in cond_sources:
            positive_ids.update(_find_text_encoders(workflow, source.get("positive")))
            negative_ids.update(_find_text_encoders(workflow, source.get("negative")))

        # Seed / steps / cfg / sampler on the sampler or its helper nodes.
        if not _set_number(workflow, sampler, ("seed", "noise_seed"), seed):
            noise = _linked_node(workflow, inputs.get("noise"))
            if noise:
                _set_number(workflow, noise, ("noise_seed", "seed"), seed)
        if req.steps is not None and not _set_number(workflow, sampler, ("steps",), int(req.steps)):
            sched = _linked_node(workflow, inputs.get("sigmas"))
            if sched:
                _set_number(workflow, sched, ("steps",), int(req.steps))
        if req.cfg_scale is not None:
            _set_number(workflow, sampler, ("cfg",), float(req.cfg_scale))
        if req.sampler:
            if "sampler_name" in inputs and not _is_link(inputs["sampler_name"]):
                inputs["sampler_name"] = req.sampler
            else:
                select = _linked_node(workflow, inputs.get("sampler"))
                if select and "sampler_name" in select.get("inputs", {}):
                    select["inputs"]["sampler_name"] = req.sampler

    if not positive_ids:
        raise ImageGenError(
            f"Couldn't find where to put the prompt in '{name}'. Set the positive prompt text to "
            "%prompt% in ComfyUI before exporting (see image_workflows/README.md)."
        )
    for node_id in positive_ids:
        _set_text(workflow[node_id], req.prompt or "")
    if req.negative_prompt:  # Empty negative keeps the workflow's own negative prompt
        for node_id in negative_ids - positive_ids:
            _set_text(workflow[node_id], req.negative_prompt)

    for node in workflow.values():
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if class_type.startswith("Empty") and "Latent" in class_type:
            if "width" in inputs and not _is_link(inputs["width"]):
                inputs["width"] = int(req.width)
            if "height" in inputs and not _is_link(inputs["height"]):
                inputs["height"] = int(req.height)
            if "batch_size" in inputs and not _is_link(inputs["batch_size"]):
                inputs["batch_size"] = max(1, int(req.batch_size or 1))
        if req.model and class_type in CHECKPOINT_CLASSES and "ckpt_name" in inputs:
            inputs["ckpt_name"] = req.model


def prepare_workflow(workflow: dict, req: ImageGenRequest, seed: int, name: str = "workflow") -> dict:
    """Returns a copy of the workflow with the request's values filled in."""
    prepared = copy.deepcopy(workflow)
    if _has_placeholder(prepared, "%prompt%"):
        _apply_placeholders(prepared, req, seed)
    else:
        _auto_inject(prepared, req, seed, name)
    return prepared


def format_prompt_error(data: Any) -> str:
    """Turns ComfyUI's /prompt 400 body ({error, node_errors}) into a readable message."""
    if not isinstance(data, dict):
        return f"ComfyUI rejected the workflow: {str(data)[:300]}"
    error = data.get("error")
    message = error.get("message") if isinstance(error, dict) else str(error or "invalid workflow")
    parts = [f"ComfyUI rejected the workflow: {message}"]
    for node_id, node_err in (data.get("node_errors") or {}).items():
        for err in (node_err or {}).get("errors", [])[:3]:
            detail = err.get("details") or ""
            text = err.get("message", "error") + (f": {detail}" if detail else "")
            parts.append(f"Node {node_id} ({node_err.get('class_type', '?')}): {text[:200]}")
    return "\n".join(parts)


def _format_execution_error(status: dict) -> str:
    for message in status.get("messages") or []:
        if not (isinstance(message, (list, tuple)) and len(message) == 2):
            continue
        event, data = message
        if event == "execution_error" and isinstance(data, dict):
            return (f"ComfyUI error in node {data.get('node_id')} ({data.get('node_type')}): "
                    f"{data.get('exception_message', '').strip()[:300]}")
        if event == "execution_interrupted":
            return "ComfyUI job was interrupted."
    return "ComfyUI reported an error while running the workflow."


def _combo_options(spec: Any) -> list[str]:
    """Reads combo choices from /object_info (legacy [[...], {...}] or ["COMBO", {"options": [...]}])."""
    if isinstance(spec, list) and spec:
        if isinstance(spec[0], list):
            return [str(v) for v in spec[0]]
        if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            return [str(v) for v in spec[1].get("options", [])]
    return []


class ComfyUIBackend(ImageBackend):
    label = "ComfyUI"
    url_suffixes = ("/prompt",)

    def __init__(self, base_url: str, timeout: float = 30.0, generation_timeout: float = 600.0,
                 workflow_dir: str = WORKFLOW_DIR):
        super().__init__(base_url, timeout)
        self.generation_timeout = generation_timeout
        self.workflow_dir = workflow_dir

    @staticmethod
    def list_workflows(directory: str = WORKFLOW_DIR) -> list[str]:
        return list_workflows(directory)

    def _get(self, path: str, **kwargs) -> requests.Response:
        return self._call(requests.get, self.base_url + path, **kwargs)

    def _post(self, path: str, **kwargs) -> requests.Response:
        return self._call(requests.post, self.base_url + path, **kwargs)

    def check(self) -> tuple[bool, str]:
        try:
            resp = self._get("/system_stats")
        except ImageGenError as e:
            return False, str(e)
        if resp.status_code != 200:
            return False, f"ComfyUI at {self.base_url} answered HTTP {resp.status_code}."
        try:
            stats = resp.json()
            version = stats.get("system", {}).get("comfyui_version", "")
            devices = ", ".join(d.get("name", "?") for d in stats.get("devices", [])[:2])
        except (ValueError, AttributeError):
            version, devices = "", ""
        detail = " ".join(x for x in (f"v{version}" if version else "", f"({devices})" if devices else "") if x)
        return True, f"Connected to ComfyUI {detail}".strip()

    def _object_info_combo(self, node_class: str, input_name: str) -> list[str]:
        resp = self._get(f"/object_info/{node_class}")
        if resp.status_code != 200:
            raise ImageGenError(f"ComfyUI /object_info/{node_class} answered HTTP {resp.status_code}.")
        try:
            spec = resp.json()[node_class]["input"]["required"][input_name]
        except (ValueError, KeyError, TypeError):
            return []
        return _combo_options(spec)

    def list_models(self) -> list[str]:
        return self._object_info_combo("CheckpointLoaderSimple", "ckpt_name")

    def list_samplers(self) -> list[str]:
        return self._object_info_combo("KSampler", "sampler_name")

    def interrupt(self) -> None:
        try:
            self._post("/interrupt", json={})
        except ImageGenError:
            pass

    def _cancel(self, prompt_id: str) -> None:
        """Removes a pending job and interrupts it if it is running."""
        for path, body in (("/queue", {"delete": [prompt_id]}), ("/interrupt", {"prompt_id": prompt_id})):
            try:
                self._post(path, json=body)
            except ImageGenError:
                pass

    def _queue_status(self, prompt_id: str) -> str | None:
        try:
            resp = self._get("/queue")
            data = resp.json()
        except (ImageGenError, ValueError):
            return None
        if any(len(item) > 1 and item[1] == prompt_id for item in data.get("queue_running", [])):
            return "running"
        pending = sorted((item for item in data.get("queue_pending", []) if len(item) > 1),
                         key=lambda item: item[0] if isinstance(item[0], (int, float)) else 0)
        for position, item in enumerate(pending, 1):
            if item[1] == prompt_id:
                return f"queued (position {position})"
        return None

    def _download_outputs(self, entry: dict, seed: int) -> list[GeneratedImage]:
        files = []
        for node_id, output in (entry.get("outputs") or {}).items():
            for image in (output or {}).get("images", []) or []:
                if isinstance(image, dict) and image.get("filename"):
                    files.append((node_id, image))
        preferred = [f for f in files if f[1].get("type") == "output"] or [f for f in files if f[1].get("type") == "temp"]
        images = []
        for batch_index, (node_id, file_info) in enumerate(preferred):
            params = {"filename": file_info["filename"], "subfolder": file_info.get("subfolder", ""),
                      "type": file_info.get("type", "output")}
            resp = self._get("/view", params=params)
            if resp.status_code != 200:
                raise ImageGenError(f"ComfyUI could not return image {file_info['filename']} (HTTP {resp.status_code}).")
            try:
                img = open_image_bytes(resp.content)
            except Exception as e:
                raise ImageGenError(f"ComfyUI returned an unreadable image ({file_info['filename']}): {e}") from None
            images.append(GeneratedImage(image=img, seed=seed, info={
                "backend": self.label, "node_id": node_id, "batch_index": batch_index,
                "server_filename": file_info["filename"],
            }))
        return images

    def generate_iter(self, req: ImageGenRequest) -> Iterator[GenProgress | GenResult]:
        name = req.workflow or DEFAULT_WORKFLOW
        workflow = load_workflow(name, self.workflow_dir)
        seed = resolve_seed(req.seed)
        prepared = prepare_workflow(workflow, req, seed, name)

        resp = self._post("/prompt", json={"prompt": prepared, "client_id": uuid.uuid4().hex})
        if resp.status_code != 200:
            try:
                raise ImageGenError(format_prompt_error(resp.json()))
            except ValueError:
                raise ImageGenError(f"ComfyUI rejected the workflow (HTTP {resp.status_code}): "
                                    f"{response_error_detail(resp)}") from None
        try:
            prompt_id = resp.json()["prompt_id"]
        except (ValueError, KeyError, TypeError):
            raise ImageGenError("ComfyUI accepted the job but returned no prompt_id.") from None

        started = time.monotonic()
        finished = False
        try:
            yield GenProgress(0.0, f"Queued in ComfyUI (seed {seed})")
            while True:
                hist_resp = self._get(f"/history/{prompt_id}")
                entry = None
                if hist_resp.status_code == 200:
                    try:
                        entry = (hist_resp.json() or {}).get(prompt_id)
                    except ValueError:
                        entry = None
                if entry:  # ComfyUI adds a history entry only once the job has finished
                    status = entry.get("status") or {}
                    if status.get("status_str") == "error":
                        raise ImageGenError(_format_execution_error(status))
                    images = self._download_outputs(entry, seed)
                    if not images:
                        raise ImageGenError(f"Workflow '{name}' finished but produced no images "
                                            "(does it end in a SaveImage or PreviewImage node?).")
                    finished = True
                    yield GenResult(images=images)
                    return
                elapsed = time.monotonic() - started
                if elapsed > self.generation_timeout:
                    self._cancel(prompt_id)
                    raise ImageGenError(f"ComfyUI did not finish within {self.generation_timeout:.0f}s; job cancelled.")
                state = self._queue_status(prompt_id) or "running"
                yield GenProgress(None, f"ComfyUI: {state}, {elapsed:.0f}s elapsed (seed {seed})")
                time.sleep(self.poll_interval)
        except GeneratorExit:
            if not finished:
                self._cancel(prompt_id)
            raise
