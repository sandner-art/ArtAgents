# ArtAgent/tests/test_lab_image_gen_lora.py
"""Tier 3: LoRA support in the image backends and caption-set output in caption generation."""
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import captioning_logic as cl
from core.dataset import caption_io, provenance, structured
from core.image_gen import comfyui, sdwebui
from core.image_gen.base import ImageGenError, ImageGenRequest
from core.image_gen.comfyui import ComfyUIBackend, prepare_workflow, validate_api_workflow
from core.image_gen.sdwebui import SDWebUIBackend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LORA_WF_PATH = os.path.join(ROOT, "image_workflows", "comfyui", "lora_txt2img.json")


def lora_wf():
    with open(LORA_WF_PATH, encoding="utf-8") as f:
        return json.load(f)


BASIC = {
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "%checkpoint%"}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "%prompt%", "clip": ["4", 1]}},
    "3": {"class_type": "KSampler", "inputs": {"seed": "%seed%", "steps": "%steps%", "cfg": "%cfg%",
                                               "sampler_name": "%sampler%", "model": ["4", 0], "positive": ["6", 0]}},
}


# ---------- ImageGenRequest ----------

def test_request_loras_default_is_independent_empty_list():
    a, b = ImageGenRequest(prompt="x"), ImageGenRequest(prompt="y")
    assert a.loras == [] and b.loras == []
    a.loras.append(("l", 1.0))
    assert b.loras == []


# ---------- ComfyUI ----------

def test_lora_workflow_is_valid_api_format_and_wired():
    wf = lora_wf()
    validate_api_workflow(wf, "lora_txt2img.json")
    loaders = {k: n for k, n in wf.items() if n["class_type"] == "LoraLoader"}
    assert len(loaders) == 1
    lid, loader = next(iter(loaders.items()))
    ckpt = next(k for k, n in wf.items() if n["class_type"] == "CheckpointLoaderSimple")
    assert loader["inputs"]["model"] == [ckpt, 0] and loader["inputs"]["clip"] == [ckpt, 1]
    assert loader["inputs"]["lora_name"] == "%lora_name%"
    assert loader["inputs"]["strength_model"] == "%lora_strength%"
    assert loader["inputs"]["strength_clip"] == "%lora_clip_strength%"
    encoders = [n for n in wf.values() if n["class_type"] == "CLIPTextEncode"]
    assert len(encoders) == 2 and all(e["inputs"]["clip"] == [lid, 1] for e in encoders)
    sampler = next(n for n in wf.values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["model"] == [lid, 0]


def test_lora_placeholders_filled_with_real_numbers():
    req = ImageGenRequest(prompt="a cat", model="base.safetensors", loras=[("style_e04.safetensors", 0.8)])
    wf = prepare_workflow(lora_wf(), req, 5)
    loader = next(n for n in wf.values() if n["class_type"] == "LoraLoader")["inputs"]
    assert loader["lora_name"] == "style_e04.safetensors"
    assert loader["strength_model"] == 0.8 and isinstance(loader["strength_model"], float)
    assert loader["strength_clip"] == 0.8
    texts = [n["inputs"]["text"] for n in wf.values() if n["class_type"] == "CLIPTextEncode"]
    assert "a cat" in texts
    # The template itself is not modified
    assert next(n for n in lora_wf().values() if n["class_type"] == "LoraLoader")["inputs"]["lora_name"] == "%lora_name%"


def test_lora_baseline_strength_zero():
    req = ImageGenRequest(prompt="a cat", model="base.safetensors", loras=[("style.safetensors", 0)])
    loader = next(n for n in prepare_workflow(lora_wf(), req, 1).values() if n["class_type"] == "LoraLoader")["inputs"]
    assert loader["strength_model"] == 0.0 and loader["strength_clip"] == 0.0


def test_only_first_lora_used_in_comfyui_slot():
    req = ImageGenRequest(prompt="x", model="m", loras=[("one.safetensors", 0.5), ("two.safetensors", 1.0)])
    loader = next(n for n in prepare_workflow(lora_wf(), req, 1).values() if n["class_type"] == "LoraLoader")["inputs"]
    assert loader["lora_name"] == "one.safetensors" and loader["strength_model"] == 0.5


def test_loras_without_placeholder_raise():
    req = ImageGenRequest(prompt="x", model="m", loras=[("l.safetensors", 1.0)])
    with pytest.raises(ImageGenError, match="no %lora_name% placeholder"):
        prepare_workflow(BASIC, req, 1, name="basic.json")


def test_lora_placeholder_without_lora_raises():
    with pytest.raises(ImageGenError, match="expects a LoRA"):
        prepare_workflow(lora_wf(), ImageGenRequest(prompt="x", model="m"), 1, name="lora_txt2img.json")


def test_workflow_without_lora_and_request_without_lora_is_untouched():
    wf = prepare_workflow(BASIC, ImageGenRequest(prompt="x", model="m"), 3)
    assert wf["3"]["inputs"]["seed"] == 3


class FakeResp:
    def __init__(self, status=200, data=None):
        self.status_code = status
        self._data = data
        self.text = json.dumps(data) if data is not None else ""

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


def _requests_mock(get):
    mock = MagicMock()
    mock.exceptions = requests.exceptions
    mock.get.side_effect = get
    return mock


def test_comfyui_list_loras():
    def get(url, **kw):
        assert url.endswith("/object_info/LoraLoader")
        return FakeResp(200, {"LoraLoader": {"input": {"required": {"lora_name": [["a.safetensors", "b.safetensors"], {}]}}}})
    with patch.object(comfyui, "requests", _requests_mock(get)):
        assert ComfyUIBackend("http://127.0.0.1:8188").list_loras() == ["a.safetensors", "b.safetensors"]


def test_comfyui_list_loras_combo_format_and_http_error():
    combo = FakeResp(200, {"LoraLoader": {"input": {"required": {"lora_name": ["COMBO", {"options": ["x.safetensors"]}]}}}})
    with patch.object(comfyui, "requests", _requests_mock(lambda url, **kw: combo)):
        assert ComfyUIBackend("http://127.0.0.1:8188").list_loras() == ["x.safetensors"]
    with patch.object(comfyui, "requests", _requests_mock(lambda url, **kw: FakeResp(500, {}))):
        with pytest.raises(ImageGenError, match="LoraLoader"):
            ComfyUIBackend("http://127.0.0.1:8188").list_loras()


# ---------- Forge / A1111 / SD.Next ----------

def test_sdwebui_list_loras_sorted_names():
    def get(url, **kw):
        assert url.endswith("/sdapi/v1/loras")
        return FakeResp(200, [{"name": "zeta"}, {"name": "alpha", "path": "/x"}, {"alias": "no-name"}, "junk"])
    with patch.object(sdwebui, "requests", _requests_mock(get)):
        assert SDWebUIBackend("http://127.0.0.1:7861").list_loras() == ["alpha", "zeta"]


def test_sdwebui_list_loras_api_disabled():
    with patch.object(sdwebui, "requests", _requests_mock(lambda url, **kw: FakeResp(404, {"detail": "Not Found"}))):
        with pytest.raises(ImageGenError):
            SDWebUIBackend("http://127.0.0.1:7861").list_loras()


@pytest.mark.parametrize("loras, expected", [
    ([("style_e04.safetensors", 0.8)], "a cat <lora:style_e04:0.8>"),
    ([("style_e04", 1.0)], "a cat <lora:style_e04:1>"),
    ([("Style.PT", 0.5)], "a cat <lora:Style:0.5>"),
    ([("model.v2.ckpt", 1.25)], "a cat <lora:model.v2:1.25>"),
    ([("style.safetensors", 0)], "a cat"),                       # Baseline: no tag
    ([("a.safetensors", 0.6), ("b", -0.5)], "a cat <lora:a:0.6> <lora:b:-0.5>"),
    ([], "a cat"),
])
def test_sdwebui_payload_lora_tags(loras, expected):
    assert SDWebUIBackend.build_payload(ImageGenRequest(prompt="a cat", loras=loras))["prompt"] == expected


def test_sdwebui_payload_lora_with_empty_prompt():
    assert SDWebUIBackend.build_payload(ImageGenRequest(prompt="", loras=[("l", 1)]))["prompt"] == "<lora:l:1>"


# ---------- caption generation into a caption set ----------

EXECUTE_PATH = "core.captioning_logic.execute_chat_or_team"
SET_EXT = ".short.txt"


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    Image.new("RGB", (40, 30), "white").save(d / "a.png")
    Image.new("RGB", (40, 30), "black").save(d / "b.png")
    return d


@pytest.fixture(autouse=True)
def _no_digest():
    with patch("core.captioning_logic._model_digest", return_value="d"):
        yield


def _gen(folder, mode="Overwrite", captions=None, config=None, reply="A cat on a sofa."):
    paths = {n: str(folder / n) for n in ("a.png", "b.png")}
    cfg = {"extension": SET_EXT}
    cfg.update(config or {})
    with patch(EXECUTE_PATH, return_value=(reply, "", None, [])) as ex:
        out = cl.generate_captions_for_selected(
            "a.png", "(Direct Agent Call)", "vision-model (VISION)", mode, paths, dict(captions or {}), {},
            [{"name": "vision-model", "vision": True}], {}, {}, {}, [], [], caption_instruction="Describe.",
            caption_config=cfg)
    return out, ex


def test_set_output_writes_set_file_only(folder):
    (folder / "a.txt").write_text("active caption", encoding="utf-8")
    (status, captions, shown, _), _ = _gen(folder, captions={"a.png": "active caption"})
    assert "Status: Success" in status
    assert (folder / "a.short.txt").read_text(encoding="utf-8") == "A cat on a sofa."
    assert (folder / "a.txt").read_text(encoding="utf-8") == "active caption"
    assert captions == {"a.png": "active caption"}
    assert shown == "A cat on a sofa."


def test_set_output_does_not_add_key_to_captions(folder):
    (status, captions, _, _), _ = _gen(folder, captions={})
    assert "a.png" not in captions and not (folder / "a.txt").exists()


def test_set_output_append_reads_existing_set_file(folder):
    (folder / "a.txt").write_text("active", encoding="utf-8")
    (folder / "a.short.txt").write_text("old set caption.", encoding="utf-8")
    (status, captions, shown, _), _ = _gen(folder, mode="Append", captions={"a.png": "active"})
    assert (folder / "a.short.txt").read_text(encoding="utf-8") == "old set caption. A cat on a sofa."
    assert (folder / "a.txt").read_text(encoding="utf-8") == "active"


def test_set_output_skip_uses_set_file_existence(folder):
    (folder / "a.short.txt").write_text("existing", encoding="utf-8")
    (status, _, _, _), ex = _gen(folder, mode="Skip")
    assert "Skipped" in status and not ex.called
    # Active caption present but no set file: not skipped
    (folder / "b.txt").write_text("x", encoding="utf-8")
    (folder / "a.short.txt").unlink()
    (folder / "a.txt").write_text("active", encoding="utf-8")
    (status, _, _, _), ex = _gen(folder, mode="Skip", captions={"a.png": "active"})
    assert "Success" in status and (folder / "a.short.txt").exists()


def test_set_output_writes_no_provenance(folder):
    _gen(folder)
    assert provenance.read_provenance(str(folder / "a.png")) == {}
    assert not os.path.exists(os.path.join(folder, ".artagents", "provenance"))


def test_set_output_structured_saves_no_fields(folder):
    reply = json.dumps({"subject": "a cat", "setting": "a sofa"})
    with patch("core.dataset.vision.ask", return_value=reply):
        paths = {n: str(folder / n) for n in ("a.png", "b.png")}
        status, captions, shown, _ = cl.generate_captions_for_selected(
            "a.png", "(Direct Agent Call)", "vision-model (VISION)", "Overwrite", paths, {}, {}, [], {}, {}, {}, [], [],
            caption_config={"extension": SET_EXT, "source": cl.SOURCE_STRUCTURED,
                            "structured": {"fields": ["subject", "setting"]}})
    assert "Success" in status, status
    assert (folder / "a.short.txt").read_text(encoding="utf-8").startswith("A cat")
    assert structured.load_fields(str(folder / "a.png")) == {}
    assert provenance.read_provenance(str(folder / "a.png")) == {}


def test_active_output_still_writes_provenance(folder):
    (status, captions, _, _), _ = _gen(folder, config={"extension": ".txt"})
    assert captions["a.png"] == "A cat on a sofa."
    assert (folder / "a.txt").exists()
    assert provenance.read_provenance(str(folder / "a.png"))["latest"]["caption_hash"] == provenance.text_hash("A cat on a sofa.")


def test_default_extension_is_active(folder):
    paths = {n: str(folder / n) for n in ("a.png", "b.png")}
    with patch(EXECUTE_PATH, return_value=("A dog.", "", None, [])):
        _, captions, _, _ = cl.generate_captions_for_selected(
            "a.png", "(Direct Agent Call)", "vision-model (VISION)", "Overwrite", paths, {}, {},
            [{"name": "vision-model", "vision": True}], {}, {}, {}, [], [])
    assert captions["a.png"] == "A dog." and caption_io.read_caption(paths["a.png"]) == "A dog."
