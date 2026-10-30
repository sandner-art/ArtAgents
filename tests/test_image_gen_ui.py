# ArtAgent/tests/test_image_gen_ui.py
"""Tests for core/image_gen_ui.py (Image Generation tab callbacks) with fake backends."""
import os
import sys

import pytest
from PIL import Image

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(test_dir))

from core import image_gen_ui
from core.image_gen.base import GeneratedImage, GenProgress, GenResult, ImageGenError

SETTINGS = {"image_generation": {"comfyui_url": "http://comfy:8188", "sdwebui_url": "http://forge:7861",
                                 "output_dir": "outputs/test-images", "save_prompt_txt": False}}


class FakeBackend:
    label = "ComfyUI"
    supports_negative_prompt = True
    supports_steps = True
    supports_cfg = True
    supports_sampler = True
    events = []
    error = None
    check_result = (True, "ComfyUI 0.4 is running.")
    models = ["sdxl.safetensors", "flux.safetensors"]
    samplers = ["euler", "dpmpp_2m"]
    created = []

    def __init__(self, base_url, timeout=30.0):
        self.base_url = base_url
        self.requests = []
        type(self).created.append(self)

    def check(self):
        return self.check_result

    def list_models(self):
        if isinstance(self.models, Exception):
            raise self.models
        return list(self.models)

    def list_samplers(self):
        return list(self.samplers)

    def generate_iter(self, req):
        self.requests.append(req)
        for event in self.events:
            yield event
        if self.error:
            raise self.error


class FakeOllamaBackend(FakeBackend):
    label = "Ollama (experimental)"
    supports_negative_prompt = supports_steps = supports_cfg = supports_sampler = False


@pytest.fixture
def backends(monkeypatch):
    FakeBackend.created = []
    FakeBackend.events = []
    FakeBackend.error = None
    FakeBackend.check_result = (True, "ComfyUI 0.4 is running.")
    FakeBackend.models = ["sdxl.safetensors", "flux.safetensors"]
    registry = {"ComfyUI": FakeBackend, "Forge / A1111 / SD.Next": FakeBackend, "Ollama (experimental)": FakeOllamaBackend}
    monkeypatch.setattr(image_gen_ui, "BACKENDS", registry)
    monkeypatch.setattr(image_gen_ui, "get_backend", lambda label, url: registry[label](url))
    monkeypatch.setattr(image_gen_ui, "list_workflows", lambda: ["basic_txt2img.json", "flux.json"])
    saved = []

    def fake_save(images, req, backend_label, out_root, save_prompt_txt=True):
        saved.append(dict(images=images, req=req, backend=backend_label, out_root=out_root, txt=save_prompt_txt))
        return [os.path.join(out_root, "2026-10-30", f"img_{i}.png") for i in range(len(images))]
    monkeypatch.setattr(image_gen_ui, "save_images", fake_save)
    return saved


def _generate(prompt="a red fox", backend="ComfyUI", gallery=None, **overrides):
    args = dict(prompt=prompt, negative_prompt=" blurry ", backend_label=backend, model="sdxl.safetensors",
                workflow="basic_txt2img.json", sampler="", width=832, height=1216, steps=0, cfg_scale=0,
                seed=1234, batch_size=2, settings=SETTINGS, gallery_items=gallery)
    args.update(overrides)
    return list(image_gen_ui.iter_generate_images(**args))


def _images(n, seed0=1234):
    return [GeneratedImage(Image.new("RGB", (8, 8)), seed0 + i, {}) for i in range(n)]


# --- small helpers ---

def test_clean_prompt_text_strips_progress_lines():
    text = "⏳ Team 'X' · Step 2/3: Artist\nA misty forest\n⏳ more\nwith fireflies "
    assert image_gen_ui.clean_prompt_text(text) == "A misty forest\nwith fireflies"
    assert image_gen_ui.clean_prompt_text(None) == ""


def test_send_response_to_image_tab():
    prompt, tabs = image_gen_ui.send_response_to_image_tab("⏳ working\nA castle at dawn")
    assert prompt == "A castle at dawn"
    assert getattr(tabs, "selected", None) == "image_gen"


def test_default_backend(backends):
    assert image_gen_ui.default_backend({}) == "ComfyUI"
    assert image_gen_ui.default_backend({"image_generation": {"backend": "Forge / A1111 / SD.Next"}}) == "Forge / A1111 / SD.Next"
    assert image_gen_ui.default_backend({"image_generation": {"backend": "Nope"}}) == "ComfyUI"


def test_clear_gallery():
    assert image_gen_ui.clear_gallery() == ([], [], "Gallery cleared (saved files are kept).")


# --- refresh_backend_options ---

def test_refresh_comfyui_lists_models_workflows_samplers(backends):
    updates = image_gen_ui.refresh_backend_options("ComfyUI", SETTINGS)
    assert len(updates) == 7
    model_u, workflow_u, sampler_u, neg_u, steps_u, cfg_u, status = updates
    assert model_u["choices"] == ["sdxl.safetensors", "flux.safetensors"] and model_u["value"] == "sdxl.safetensors"
    assert workflow_u["visible"] is True and workflow_u["choices"] == ["basic_txt2img.json", "flux.json"]
    assert sampler_u["choices"] == ["", "euler", "dpmpp_2m"] and sampler_u["interactive"] is True
    assert neg_u["interactive"] is True and steps_u["interactive"] is True and cfg_u["interactive"] is True
    assert status == "✅ ComfyUI at http://comfy:8188: ComfyUI 0.4 is running."
    assert FakeBackend.created[-1].base_url == "http://comfy:8188"


def test_refresh_non_comfy_hides_workflow_and_disables_unsupported_fields(backends):
    _, workflow_u, sampler_u, neg_u, steps_u, cfg_u, _ = image_gen_ui.refresh_backend_options("Ollama (experimental)", SETTINGS)
    assert workflow_u["visible"] is False and workflow_u["choices"] == []
    assert neg_u["interactive"] is False and steps_u["interactive"] is False
    assert cfg_u["interactive"] is False and sampler_u["interactive"] is False
    *_, status = image_gen_ui.refresh_backend_options("Forge / A1111 / SD.Next", SETTINGS)
    assert "http://forge:7861" in status


def test_refresh_unreachable_backend(backends):
    FakeBackend.check_result = (False, "Could not connect.")
    model_u, *_, status = image_gen_ui.refresh_backend_options("ComfyUI", SETTINGS)
    assert model_u["choices"] == [] and model_u["value"] is None
    assert status.startswith("⚠️ ComfyUI at http://comfy:8188: Could not connect.")


def test_refresh_model_list_error_and_empty(backends):
    FakeBackend.models = ImageGenError("object_info failed")
    *_, status = image_gen_ui.refresh_backend_options("ComfyUI", SETTINGS)
    assert "(could not list models: object_info failed)" in status and status.endswith("No models found.")


def test_refresh_unknown_backend(backends):
    updates = image_gen_ui.refresh_backend_options("Nope", SETTINGS)
    assert updates[1]["visible"] is False and updates[-1] == "Unknown backend 'Nope'."


# --- iter_generate_images ---

@pytest.mark.parametrize("prompt", ["", "   ", None])
def test_empty_prompt(backends, prompt):
    updates = _generate(prompt=prompt, gallery=[("x", "old")])
    assert updates == [("Enter a prompt first (or use 'Use last LLM response').", [("x", "old")], [("x", "old")])]
    assert FakeBackend.created == []


def test_unknown_backend_label(backends):
    assert _generate(backend="Nope")[-1][0] == "Unknown backend 'Nope'."


def test_success_streams_progress_saves_and_prepends(backends):
    images = _images(2)
    FakeBackend.events = [GenProgress(None, "queued (position 1)"), GenProgress(0.5, "step 10/20"), GenResult(images)]
    old = [(Image.new("RGB", (2, 2)), "seed 1")]
    updates = _generate(gallery=old)

    statuses = [u[0] for u in updates]
    assert statuses[0] == "⏳ ComfyUI · seed 1234 · 832x1216: sending job..."
    assert statuses[1].startswith("⏳ ComfyUI · seed 1234 · 832x1216: queued (position 1)")
    assert statuses[2].startswith("⏳ ComfyUI · seed 1234 · 832x1216: 50% step 10/20")
    assert statuses[-1].startswith("✅ 2 image(s) in ")
    assert "Saved to " + os.path.join(image_gen_ui.resolve_output_dir(SETTINGS), "2026-10-30") in statuses[-1]

    gallery = updates[-1][1]
    assert updates[-1][2] == gallery
    assert [label for _, label in gallery] == ["seed 1234", "seed 1235", "seed 1"] # Newest first
    assert gallery[0][0] is images[0].image

    req = FakeBackend.created[0].requests[0]
    assert (req.prompt, req.negative_prompt, req.width, req.height) == ("a red fox", "blurry", 832, 1216)
    assert req.steps is None and req.cfg_scale is None and req.sampler is None # 0 / "" mean "backend default"
    assert (req.seed, req.batch_size, req.model, req.workflow) == (1234, 2, "sdxl.safetensors", "basic_txt2img.json")
    assert backends[0]["backend"] == "ComfyUI" and backends[0]["txt"] is False
    assert backends[0]["out_root"] == image_gen_ui.resolve_output_dir(SETTINGS)


def test_random_seed_is_chosen_and_shown(backends):
    FakeBackend.events = [GenResult([GeneratedImage(Image.new("RGB", (4, 4)), None, {})])]
    updates = _generate(seed=-1, steps=30, cfg_scale=6.5, sampler="euler", batch_size=0)
    req = FakeBackend.created[0].requests[0]
    assert 0 <= req.seed < 2**31
    assert f"seed {req.seed}" in updates[0][0]
    assert req.steps == 30 and req.cfg_scale == 6.5 and req.sampler == "euler" and req.batch_size == 1
    assert updates[-1][1][0][1] == "ComfyUI" # No seed reported by the backend -> backend label


def test_gallery_capped(backends):
    FakeBackend.events = [GenResult(_images(3))]
    old = [(f"old{i}", f"o{i}") for i in range(image_gen_ui.MAX_GALLERY_ITEMS)]
    gallery = _generate(gallery=old)[-1][1]
    assert len(gallery) == image_gen_ui.MAX_GALLERY_ITEMS
    assert [g[1] for g in gallery[:3]] == ["seed 1234", "seed 1235", "seed 1236"]
    assert gallery[-1] == ("old44", "o44")


def test_backend_error_is_shown_and_gallery_kept(backends):
    FakeBackend.events = [GenProgress(None, "running")]
    FakeBackend.error = ImageGenError("Ollama removed experimental image generation in v0.32.6.")
    updates = _generate(gallery=[("keep", "me")])
    assert updates[-1] == ("⚠️ Ollama removed experimental image generation in v0.32.6.", [("keep", "me")], [("keep", "me")])
    assert backends == [] # Nothing saved


def test_unexpected_backend_exception(backends):
    FakeBackend.error = KeyError("outputs")
    assert _generate()[-1][0].startswith("⚠️ Unexpected error from ComfyUI:")


def test_no_images_returned(backends):
    FakeBackend.events = [GenResult([])]
    assert _generate()[-1][0] == "⚠️ ComfyUI finished but returned no images."


def test_save_failure_still_shows_images(backends, monkeypatch):
    def broken_save(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(image_gen_ui, "save_images", broken_save)
    FakeBackend.events = [GenResult(_images(1))]
    status, gallery, _ = _generate()[-1]
    assert status.startswith("✅ 1 image(s)") and "Could not save files: disk full" in status
    assert len(gallery) == 1


def test_stop_closes_backend_generator(backends, monkeypatch):
    """UI Stop closes iter_generate_images, which must close the backend generator (server-side cancel)."""
    closed = []

    def gen(self, req):
        try:
            yield GenProgress(None, "running")
            yield GenProgress(None, "still running")
        finally:
            closed.append(True)
    monkeypatch.setattr(FakeBackend, "generate_iter", gen) # Restored after the test
    it = image_gen_ui.iter_generate_images("p", "", "ComfyUI", None, None, None, 512, 512, 0, 0, 1, 1, SETTINGS, [])
    next(it); next(it)
    it.close()
    assert closed == [True]
