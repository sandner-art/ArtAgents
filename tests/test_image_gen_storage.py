# ArtAgent/tests/test_image_gen_storage.py
import io
import json
import os

import pytest
from PIL import Image, UnidentifiedImageError

import core.image_gen as image_gen
from core.image_gen import (
    BACKENDS, DEFAULT_URLS, ComfyUIBackend, GeneratedImage, ImageGenError, ImageGenRequest, OllamaImageBackend,
    SDWebUIBackend, get_backend, resolve_seed, save_images,
)
from core.image_gen.storage import build_parameters_text
from core.image_utils import open_image_bytes


def gen(seed=123, mode="RGB", size=(12, 10)):
    return GeneratedImage(image=Image.new(mode, size), seed=seed, info={"backend": "test"})


def test_save_images_writes_png_metadata_and_sidecar(tmp_path):
    req = ImageGenRequest(prompt="a red chair, studio light", negative_prompt="blurry", steps=25, cfg_scale=6.0,
                          sampler="euler", model="sdxl.safetensors", workflow="basic_txt2img.json")
    paths = save_images([gen(42), gen(42)], req, "ComfyUI", str(tmp_path / "out"))
    assert len(paths) == 2 and all(os.path.isabs(p) and p.endswith(".png") for p in paths)
    day_dir = os.path.dirname(paths[0])
    assert os.path.dirname(day_dir) == str(tmp_path / "out")
    assert len(os.path.basename(day_dir)) == 10  # YYYY-MM-DD
    name = os.path.basename(paths[0])
    assert "_seed42_1" in name and name != os.path.basename(paths[1])

    with Image.open(paths[0]) as img:
        params = img.text["parameters"]
        meta = json.loads(img.text["artagents"])
    assert params.splitlines()[0] == "a red chair, studio light"
    assert params.splitlines()[1] == "Negative prompt: blurry"
    assert "Steps: 25, Sampler: euler, CFG scale: 6.0, Seed: 42, Size: 12x10, Model: sdxl.safetensors" in params
    assert "Backend: ComfyUI" in params
    assert meta["seed"] == 42 and meta["backend"] == "ComfyUI" and meta["workflow"] == "basic_txt2img.json"
    with open(paths[0][:-4] + ".txt", encoding="utf-8") as f:
        assert f.read() == "a red chair, studio light"


def test_save_images_unique_names_without_sidecar(tmp_path):
    req = ImageGenRequest(prompt="x")
    first = save_images([gen(None)], req, "Ollama (experimental)", str(tmp_path), save_prompt_txt=False)
    second = save_images([gen(None)], req, "Ollama (experimental)", str(tmp_path), save_prompt_txt=False)
    assert first[0] != second[0] and "seed" not in os.path.basename(first[0])
    assert not os.path.exists(first[0][:-4] + ".txt")


def test_save_images_converts_unsaveable_modes(tmp_path):
    paths = save_images([gen(1, mode="CMYK")], ImageGenRequest(prompt="x"), "B", str(tmp_path))
    with Image.open(paths[0]) as img:
        assert img.mode == "RGB"


def test_parameters_text_uses_backend_info_when_request_unset():
    image = GeneratedImage(Image.new("RGB", (8, 4)), seed=7, info={"steps": 30, "sampler_name": "Euler a",
                                                                   "cfg_scale": 5, "sd_model_name": "juggernaut"})
    text = build_parameters_text(ImageGenRequest(prompt="p"), image, "Forge / A1111 / SD.Next")
    assert text == "p\nSteps: 30, Sampler: Euler a, CFG scale: 5, Seed: 7, Size: 8x4, Model: juggernaut, Backend: Forge / A1111 / SD.Next"


def test_registry_and_get_backend():
    assert set(BACKENDS) == set(DEFAULT_URLS) == {"ComfyUI", "Forge / A1111 / SD.Next", "Ollama (experimental)"}
    assert isinstance(get_backend("ComfyUI", ""), ComfyUIBackend)
    assert get_backend("ComfyUI", "").base_url == "http://127.0.0.1:8188"
    assert isinstance(get_backend("Forge / A1111 / SD.Next", "http://h:7861"), SDWebUIBackend)
    assert isinstance(get_backend("Ollama (experimental)"), OllamaImageBackend)
    for label, cls in BACKENDS.items():
        assert cls.label == label
    with pytest.raises(ImageGenError):
        get_backend("Midjourney", "x")
    assert image_gen.list_workflows() == sorted(image_gen.list_workflows())


def test_resolve_seed():
    assert resolve_seed(17) == 17
    for value in (-1, None):
        seed = resolve_seed(value)
        assert 0 <= seed <= 2**32 - 1


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_open_image_bytes_accepts_generated_formats(fmt):
    buf = io.BytesIO()
    Image.new("RGB", (5, 4), (1, 2, 3)).save(buf, format=fmt)
    img = open_image_bytes(buf.getvalue())
    assert img.size == (5, 4)
    img.load()  # detached from the buffer


@pytest.mark.parametrize("fmt", ["TIFF", "GIF", "BMP"])
def test_open_image_bytes_rejects_other_formats(fmt):
    buf = io.BytesIO()
    Image.new("RGB", (5, 4)).save(buf, format=fmt)
    with pytest.raises(UnidentifiedImageError):
        open_image_bytes(buf.getvalue())
