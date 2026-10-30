# ArtAgent/tests/test_image_gen_comfyui.py
import io
import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image

from core.image_gen import comfyui
from core.image_gen.base import GenProgress, GenResult, ImageGenError, ImageGenRequest
from core.image_gen.comfyui import (
    ComfyUIBackend, format_prompt_error, list_workflows, load_workflow, prepare_workflow,
)


# --- helpers -------------------------------------------------------------------

class FakeResp:
    def __init__(self, status=200, data=None, content=b"", text=None):
        self.status_code = status
        self._data = data
        self.content = content
        self.text = text if text is not None else (json.dumps(data) if data is not None else "")

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


def png(color=(10, 20, 30), size=(8, 8), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def make_requests_mock(get_routes=None, post_routes=None):
    """Mock of the `requests` module; routes map URL path -> FakeResp | callable(url, **kw)."""
    mock = MagicMock()
    mock.exceptions = requests.exceptions
    calls = []

    def route(routes, method):
        def handler(url, **kwargs):
            path = url.split("127.0.0.1:8188", 1)[-1]
            calls.append((method, path, kwargs))
            for prefix, value in (routes or {}).items():
                if path == prefix or (prefix.endswith("*") and path.startswith(prefix[:-1])):
                    return value(url, **kwargs) if callable(value) else value
            return FakeResp(404, {"error": "not found"})
        return handler

    mock.get.side_effect = route(get_routes, "GET")
    mock.post.side_effect = route(post_routes, "POST")
    mock.calls = calls
    return mock


BASIC = {
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "%checkpoint%"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": "%width%", "height": "%height%", "batch_size": "%batch_size%"}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "%prompt%", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "%negative%", "clip": ["4", 1]}},
    "3": {"class_type": "KSampler", "inputs": {
        "seed": "%seed%", "steps": "%steps%", "cfg": "%cfg%", "sampler_name": "%sampler%", "scheduler": "normal",
        "denoise": 1, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ArtAgents", "images": ["8", 0]}},
}

SD15_NO_PLACEHOLDERS = {
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5.safetensors"}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "beautiful scenery", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "text, watermark", "clip": ["4", 1]}},
    "3": {"class_type": "KSampler", "inputs": {
        "seed": 1, "steps": 20, "cfg": 8.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1,
        "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
    "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}},
}


@pytest.fixture
def wf_dir(tmp_path):
    d = tmp_path / "comfyui"
    d.mkdir()
    (d / "basic.json").write_text(json.dumps(BASIC), encoding="utf-8")
    (d / "sd15.json").write_text(json.dumps(SD15_NO_PLACEHOLDERS), encoding="utf-8")
    (d / "notes.txt").write_text("not a workflow", encoding="utf-8")
    return str(d)


def backend(wf_dir, **kw):
    b = ComfyUIBackend("http://127.0.0.1:8188", workflow_dir=wf_dir, **kw)
    b.poll_interval = 0
    return b


# --- workflow files --------------------------------------------------------------

def test_list_workflows_sorted_json_only(wf_dir):
    assert list_workflows(wf_dir) == ["basic.json", "sd15.json"]
    assert ComfyUIBackend.list_workflows(wf_dir) == ["basic.json", "sd15.json"]


def test_list_workflows_missing_dir(tmp_path):
    assert list_workflows(str(tmp_path / "nope")) == []


def test_bundled_workflow_is_valid_api_format():
    assert "basic_txt2img.json" in list_workflows()
    wf = load_workflow("basic_txt2img.json")
    assert wf["9"]["class_type"] == "SaveImage"


@pytest.mark.parametrize("name", ["../secret.json", "sub/x.json", "x.txt", "", ".hidden.json", "..\\x.json"])
def test_load_workflow_rejects_paths_and_non_json(wf_dir, name):
    with pytest.raises(ImageGenError):
        load_workflow(name, wf_dir)


def test_load_workflow_rejects_ui_format(wf_dir, tmp_path):
    ui = {"last_node_id": 9, "nodes": [{"id": 3, "type": "KSampler"}], "links": [[1, 4, 0, 3, 0, "MODEL"]]}
    (tmp_path / "comfyui" / "ui.json").write_text(json.dumps(ui), encoding="utf-8")
    with pytest.raises(ImageGenError, match="Export \\(API\\)"):
        load_workflow("ui.json", wf_dir)


def test_load_workflow_rejects_garbage(wf_dir, tmp_path):
    (tmp_path / "comfyui" / "bad.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "comfyui" / "list.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ImageGenError, match="not valid JSON"):
        load_workflow("bad.json", wf_dir)
    with pytest.raises(ImageGenError, match="API-format"):
        load_workflow("list.json", wf_dir)
    with pytest.raises(ImageGenError, match="not found"):
        load_workflow("missing.json", wf_dir)


# --- placeholder injection ---------------------------------------------------------

def test_placeholders_replaced_with_typed_values():
    req = ImageGenRequest(prompt="a red chair", negative_prompt="blurry", width=768, height=512, steps=30,
                          cfg_scale=5.5, batch_size=3, model="sdxl.safetensors", sampler="dpmpp_2m")
    wf = prepare_workflow(BASIC, req, 1234)
    ks = wf["3"]["inputs"]
    assert ks["seed"] == 1234 and ks["steps"] == 30 and ks["cfg"] == 5.5 and ks["sampler_name"] == "dpmpp_2m"
    assert wf["5"]["inputs"] == {"width": 768, "height": 512, "batch_size": 3}
    assert wf["6"]["inputs"]["text"] == "a red chair" and wf["7"]["inputs"]["text"] == "blurry"
    assert wf["4"]["inputs"]["ckpt_name"] == "sdxl.safetensors"
    assert BASIC["6"]["inputs"]["text"] == "%prompt%"  # original untouched


def test_placeholder_defaults_when_unset():
    wf = prepare_workflow(BASIC, ImageGenRequest(prompt="x", model="m.safetensors"), 7)
    ks = wf["3"]["inputs"]
    assert (ks["steps"], ks["cfg"], ks["sampler_name"]) == (20, 7.0, "euler")
    assert wf["7"]["inputs"]["text"] == ""


def test_placeholder_checkpoint_requires_model():
    with pytest.raises(ImageGenError, match="checkpoint"):
        prepare_workflow(BASIC, ImageGenRequest(prompt="x"), 1)


def test_embedded_text_placeholder():
    wf = json.loads(json.dumps(BASIC))
    wf["6"]["inputs"]["text"] = "photo of %prompt%, film grain"
    wf["4"]["inputs"]["ckpt_name"] = "fixed.safetensors"
    out = prepare_workflow(wf, ImageGenRequest(prompt="a fox"), 1)
    assert out["6"]["inputs"]["text"] == "photo of a fox, film grain"
    assert out["4"]["inputs"]["ckpt_name"] == "fixed.safetensors"


# --- auto-detection ------------------------------------------------------------------

def test_auto_detect_ksampler_workflow():
    req = ImageGenRequest(prompt="a lighthouse", negative_prompt="people", width=640, height=448, steps=12,
                          cfg_scale=4.0, batch_size=2, model="other.safetensors", sampler="euler_ancestral")
    wf = prepare_workflow(SD15_NO_PLACEHOLDERS, req, 99)
    assert wf["6"]["inputs"]["text"] == "a lighthouse"
    assert wf["7"]["inputs"]["text"] == "people"
    ks = wf["3"]["inputs"]
    assert (ks["seed"], ks["steps"], ks["cfg"], ks["sampler_name"]) == (99, 12, 4.0, "euler_ancestral")
    assert wf["5"]["inputs"] == {"width": 640, "height": 448, "batch_size": 2}
    assert wf["4"]["inputs"]["ckpt_name"] == "other.safetensors"


def test_auto_detect_keeps_workflow_values_when_unset():
    wf = prepare_workflow(SD15_NO_PLACEHOLDERS, ImageGenRequest(prompt="x"), 5)
    ks = wf["3"]["inputs"]
    assert (ks["seed"], ks["steps"], ks["cfg"], ks["sampler_name"]) == (5, 20, 8.0, "euler")
    assert wf["7"]["inputs"]["text"] == "text, watermark"  # empty negative keeps the workflow's
    assert wf["4"]["inputs"]["ckpt_name"] == "v1-5.safetensors"


def test_auto_detect_sdxl_text_encoder():
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl.safetensors"}},
        "2": {"class_type": "CLIPTextEncodeSDXL", "inputs": {"text_g": "old", "text_l": "old", "width": 1024, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncodeSDXL", "inputs": {"text_g": "bad", "text_l": "bad", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "5": {"class_type": "KSamplerAdvanced", "inputs": {"noise_seed": 1, "steps": 25, "cfg": 7,
              "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0], "model": ["1", 0]}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}},
    }
    out = prepare_workflow(wf, ImageGenRequest(prompt="castle", negative_prompt="fog"), 77)
    assert out["2"]["inputs"]["text_g"] == out["2"]["inputs"]["text_l"] == "castle"
    assert out["3"]["inputs"]["text_g"] == "fog"
    assert out["5"]["inputs"]["noise_seed"] == 77
    assert out["2"]["inputs"]["width"] == 1024  # non-text inputs untouched


def test_auto_detect_flux_sampler_custom_advanced():
    wf = {
        "10": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux1-dev.safetensors"}},
        "11": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "t5", "clip_name2": "l"}},
        "6": {"class_type": "CLIPTextEncodeFlux", "inputs": {"clip_l": "old", "t5xxl": "old", "guidance": 3.5, "clip": ["11", 0]}},
        "26": {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5, "conditioning": ["6", 0]}},
        "22": {"class_type": "BasicGuider", "inputs": {"model": ["10", 0], "conditioning": ["26", 0]}},
        "25": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}},
        "16": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "17": {"class_type": "BasicScheduler", "inputs": {"scheduler": "simple", "steps": 20, "denoise": 1, "model": ["10", 0]}},
        "27": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["25", 0], "guider": ["22", 0],
               "sampler": ["16", 0], "sigmas": ["17", 0], "latent_image": ["27", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["13", 0]}},
    }
    req = ImageGenRequest(prompt="neon city", width=832, height=1216, steps=8, sampler="dpmpp_2m", model="ignored")
    out = prepare_workflow(wf, req, 4242)
    assert out["6"]["inputs"]["clip_l"] == out["6"]["inputs"]["t5xxl"] == "neon city"
    assert out["25"]["inputs"]["noise_seed"] == 4242
    assert out["17"]["inputs"]["steps"] == 8
    assert out["16"]["inputs"]["sampler_name"] == "dpmpp_2m"
    assert (out["27"]["inputs"]["width"], out["27"]["inputs"]["height"]) == (832, 1216)
    assert out["10"]["inputs"]["unet_name"] == "flux1-dev.safetensors"  # no checkpoint loader -> untouched


def test_auto_detect_zero_out_negative_does_not_overwrite_positive():
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux.safetensors"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "old", "clip": ["1", 1]}},
        "35": {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5, "conditioning": ["6", 0]}},
        "33": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["6", 0]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "3": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 1, "sampler_name": "euler",
              "positive": ["35", 0], "negative": ["33", 0], "latent_image": ["5", 0], "model": ["1", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
    }
    out = prepare_workflow(wf, ImageGenRequest(prompt="a cat", negative_prompt="dogs"), 3)
    assert out["6"]["inputs"]["text"] == "a cat"


def test_auto_detect_seed_via_linked_primitive():
    wf = json.loads(json.dumps(SD15_NO_PLACEHOLDERS))
    wf["50"] = {"class_type": "Seed (rgthree)", "inputs": {"seed": 0}}
    wf["3"]["inputs"]["seed"] = ["50", 0]
    out = prepare_workflow(wf, ImageGenRequest(prompt="x"), 31337)
    assert out["50"]["inputs"]["seed"] == 31337
    assert out["3"]["inputs"]["seed"] == ["50", 0]


def test_auto_detect_without_prompt_node_errors():
    wf = {"1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
          "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}}}
    with pytest.raises(ImageGenError, match="%prompt%"):
        prepare_workflow(wf, ImageGenRequest(prompt="x"), 1)


# --- error formatting ------------------------------------------------------------------

def test_format_prompt_error_lists_node_errors():
    body = {"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
            "node_errors": {"4": {"class_type": "CheckpointLoaderSimple", "errors": [
                {"message": "Value not in list", "details": "ckpt_name: 'x.safetensors' not in ['a.safetensors']"}]}}}
    msg = format_prompt_error(body)
    assert "Prompt outputs failed validation" in msg
    assert "Node 4 (CheckpointLoaderSimple): Value not in list: ckpt_name: 'x.safetensors'" in msg


# --- server calls -----------------------------------------------------------------------

def test_check_and_lists(wf_dir):
    mock = make_requests_mock(get_routes={
        "/system_stats": FakeResp(200, {"system": {"comfyui_version": "0.3.60"}, "devices": [{"name": "RTX 4090"}]}),
        "/object_info/CheckpointLoaderSimple": FakeResp(200, {"CheckpointLoaderSimple": {"input": {"required": {
            "ckpt_name": [["a.safetensors", "b.safetensors"], {"tooltip": "x"}]}}}}),
        "/object_info/KSampler": FakeResp(200, {"KSampler": {"input": {"required": {
            "sampler_name": ["COMBO", {"options": ["euler", "dpmpp_2m"]}]}}}}),
    })
    with patch.object(comfyui, "requests", mock):
        b = backend(wf_dir)
        ok, msg = b.check()
        assert ok and "0.3.60" in msg and "RTX 4090" in msg
        assert b.list_models() == ["a.safetensors", "b.safetensors"]
        assert b.list_samplers() == ["euler", "dpmpp_2m"]


def test_check_connection_error_is_friendly(wf_dir):
    mock = make_requests_mock()
    mock.get.side_effect = requests.exceptions.ConnectionError("refused")
    with patch.object(comfyui, "requests", mock):
        ok, msg = backend(wf_dir).check()
    assert not ok and "Could not connect to ComfyUI" in msg


def _job_mock(history_after=2, outputs=None, status=None, view=None, prompt_resp=None):
    state = {"polls": 0}
    outputs = outputs if outputs is not None else {"9": {"images": [
        {"filename": "ArtAgents_00001_.png", "subfolder": "", "type": "output"},
        {"filename": "ArtAgents_00002_.png", "subfolder": "", "type": "output"},
        {"filename": "preview.png", "subfolder": "", "type": "temp"}]}}

    def history(url, **kw):
        state["polls"] += 1
        if state["polls"] <= history_after:
            return FakeResp(200, {})
        return FakeResp(200, {"p1": {"outputs": outputs, "status": status or {"status_str": "success", "completed": True}}})

    return make_requests_mock(
        get_routes={
            "/history/p1": history,
            "/queue": FakeResp(200, {"queue_running": [], "queue_pending": [[5, "other", {}, {}, []], [6, "p1", {}, {}, []]]}),
            "/view": view or (lambda url, **kw: FakeResp(200, content=png((1, 2, 3)))),
        },
        post_routes={"/prompt": prompt_resp or FakeResp(200, {"prompt_id": "p1", "number": 6, "node_errors": {}}),
                     "/queue": FakeResp(200), "/interrupt": FakeResp(200)},
    )


def test_generate_polls_until_done_and_downloads_outputs(wf_dir):
    mock = _job_mock()
    req = ImageGenRequest(prompt="a red chair", model="a.safetensors", workflow="basic.json", seed=-1, batch_size=2)
    with patch.object(comfyui, "requests", mock):
        items = list(backend(wf_dir).generate_iter(req))
    progress = [i for i in items if isinstance(i, GenProgress)]
    assert isinstance(items[-1], GenResult) and sum(isinstance(i, GenResult) for i in items) == 1
    assert any("queued (position 2)" in p.message for p in progress)
    images = items[-1].images
    assert len(images) == 2  # temp preview ignored when output images exist
    seed = images[0].seed
    assert 0 <= seed <= 2**32 - 1 and images[1].seed == seed
    assert images[0].image.size == (8, 8) and images[0].info["server_filename"] == "ArtAgents_00001_.png"
    posted = next(kw["json"] for method, path, kw in mock.calls if method == "POST" and path == "/prompt")
    assert posted["prompt"]["3"]["inputs"]["seed"] == seed and posted["prompt"]["6"]["inputs"]["text"] == "a red chair"
    assert posted["client_id"]
    view_params = [kw["params"] for method, path, kw in mock.calls if path == "/view"]
    assert view_params[0] == {"filename": "ArtAgents_00001_.png", "subfolder": "", "type": "output"}


def test_generate_uses_default_workflow_name():
    mock = _job_mock(history_after=0)
    with patch.object(comfyui, "requests", mock):
        b = ComfyUIBackend("http://127.0.0.1:8188")
        b.poll_interval = 0
        images = b.generate(ImageGenRequest(prompt="x", model="a.safetensors", seed=5))
    assert len(images) == 2 and images[0].seed == 5


def test_generate_node_errors_raise_readable_message(wf_dir):
    body = {"error": {"message": "Prompt outputs failed validation"},
            "node_errors": {"4": {"class_type": "CheckpointLoaderSimple", "errors": [{"message": "Value not in list", "details": "ckpt_name"}]}}}
    mock = _job_mock(prompt_resp=FakeResp(400, body))
    with patch.object(comfyui, "requests", mock), pytest.raises(ImageGenError, match="Node 4 \\(CheckpointLoaderSimple\\)"):
        list(backend(wf_dir).generate_iter(ImageGenRequest(prompt="x", model="m", workflow="basic.json")))


def test_generate_execution_error_from_history(wf_dir):
    status = {"status_str": "error", "completed": False, "messages": [
        ["execution_start", {}],
        ["execution_error", {"node_id": "3", "node_type": "KSampler", "exception_message": "CUDA out of memory"}]]}
    mock = _job_mock(history_after=0, status=status, outputs={})
    with patch.object(comfyui, "requests", mock), pytest.raises(ImageGenError, match="node 3 \\(KSampler\\): CUDA out of memory"):
        list(backend(wf_dir).generate_iter(ImageGenRequest(prompt="x", model="m", workflow="basic.json")))


def test_generate_without_images_errors(wf_dir):
    mock = _job_mock(history_after=0, outputs={"9": {"text": ["hello"]}})
    with patch.object(comfyui, "requests", mock), pytest.raises(ImageGenError, match="produced no images"):
        list(backend(wf_dir).generate_iter(ImageGenRequest(prompt="x", model="m", workflow="basic.json")))


def test_generate_rejects_non_image_download(wf_dir):
    mock = _job_mock(history_after=0, view=lambda url, **kw: FakeResp(200, content=png(fmt="GIF")))
    with patch.object(comfyui, "requests", mock), pytest.raises(ImageGenError, match="unreadable image"):
        list(backend(wf_dir).generate_iter(ImageGenRequest(prompt="x", model="m", workflow="basic.json")))


def test_closing_generator_cancels_job(wf_dir):
    mock = _job_mock(history_after=100)
    with patch.object(comfyui, "requests", mock):
        gen = backend(wf_dir).generate_iter(ImageGenRequest(prompt="x", model="m", workflow="basic.json"))
        assert isinstance(next(gen), GenProgress)
        assert isinstance(next(gen), GenProgress)
        gen.close()
    posts = [(path, kw.get("json")) for method, path, kw in mock.calls if method == "POST"]
    assert ("/queue", {"delete": ["p1"]}) in posts
    assert ("/interrupt", {"prompt_id": "p1"}) in posts


def test_closing_after_result_does_not_cancel(wf_dir):
    mock = _job_mock(history_after=0)
    with patch.object(comfyui, "requests", mock):
        gen = backend(wf_dir).generate_iter(ImageGenRequest(prompt="x", model="m", workflow="basic.json"))
        items = [next(gen), next(gen)]
        assert isinstance(items[-1], GenResult)
        gen.close()
    assert not any(path in ("/queue", "/interrupt") for method, path, kw in mock.calls if method == "POST")


def test_timeout_cancels_and_raises(wf_dir):
    mock = _job_mock(history_after=100)
    with patch.object(comfyui, "requests", mock):
        b = backend(wf_dir, generation_timeout=0)
        with pytest.raises(ImageGenError, match="did not finish"):
            list(b.generate_iter(ImageGenRequest(prompt="x", model="m", workflow="basic.json")))
    assert any(path == "/interrupt" for method, path, kw in mock.calls if method == "POST")


def test_base_url_normalisation():
    assert ComfyUIBackend("127.0.0.1:8188/").base_url == "http://127.0.0.1:8188"
    assert ComfyUIBackend("http://host:8188/prompt").base_url == "http://host:8188"
    with pytest.raises(ImageGenError):
        ComfyUIBackend("  ")
