# ArtAgent/tests/test_image_gen_ollama.py
import base64
import io
import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image

from core.image_gen import ollama_images
from core.image_gen.base import GenProgress, GenResult, ImageGenError, ImageGenRequest
from core.image_gen.ollama_images import OLLAMA_IMAGE_GEN_REMOVED, OllamaImageBackend

BASE = "http://localhost:11434"


class FakeResp:
    def __init__(self, status=200, data=None, text=None):
        self.status_code = status
        self._data = data
        self.text = text if text is not None else (json.dumps(data) if data is not None else "")

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


def b64png(size=(16, 8)):
    buf = io.BytesIO()
    Image.new("RGB", size, (9, 9, 9)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_requests_mock(get_routes=None, post_routes=None):
    mock = MagicMock()
    mock.exceptions = requests.exceptions
    calls = []

    def route(routes, method):
        def handler(url, **kwargs):
            path = url.replace(BASE, "")
            calls.append((method, path, kwargs))
            value = (routes or {}).get(path)
            if value is None:
                return FakeResp(404, {"error": "not found"})
            return value(url, **kwargs) if callable(value) else value
        return handler

    mock.get.side_effect = route(get_routes, "GET")
    mock.post.side_effect = route(post_routes, "POST")
    mock.calls = calls
    return mock


def test_flags_and_url_normalisation():
    b = OllamaImageBackend(BASE + "/api/generate")
    assert b.base_url == BASE
    assert OllamaImageBackend("localhost:11434/v1/").base_url == BASE
    assert not (b.supports_negative_prompt or b.supports_steps or b.supports_cfg or b.supports_sampler or b.supports_seed)


def test_list_models_filters_image_capability():
    tags = {"models": [
        {"name": "gemma4:e4b", "capabilities": ["completion", "vision"]},
        {"name": "x/z-image-turbo:latest", "capabilities": ["image"]},
        {"name": "x/flux2-klein:4b", "capabilities": ["image"]},
    ]}
    with patch.object(ollama_images, "requests", make_requests_mock(get_routes={"/api/tags": FakeResp(200, tags)})):
        assert OllamaImageBackend(BASE).list_models() == ["x/z-image-turbo:latest", "x/flux2-klein:4b"]


def test_list_models_falls_back_to_name_prefixes():
    tags = {"models": [{"name": "llama3:latest"}, {"name": "x/z-image-turbo:fp8"}, {"name": "x/flux2-klein:latest"}]}
    with patch.object(ollama_images, "requests", make_requests_mock(get_routes={"/api/tags": FakeResp(200, tags)})):
        assert OllamaImageBackend(BASE).list_models() == ["x/z-image-turbo:fp8", "x/flux2-klein:latest"]


def test_list_models_openai_compatible_server():
    mock = make_requests_mock(get_routes={"/v1/models": FakeResp(200, {"data": [{"id": "sdxl-turbo"}, {"id": "flux"}]})})
    with patch.object(ollama_images, "requests", mock):
        assert OllamaImageBackend(BASE).list_models() == ["sdxl-turbo", "flux"]


def test_check_reports_version_or_failure():
    with patch.object(ollama_images, "requests", make_requests_mock(get_routes={"/api/version": FakeResp(200, {"version": "0.40.1"})})):
        ok, msg = OllamaImageBackend(BASE).check()
        assert ok and "0.40.1" in msg
    mock = make_requests_mock()
    mock.get.side_effect = requests.exceptions.ConnectionError()
    with patch.object(ollama_images, "requests", mock):
        ok, msg = OllamaImageBackend(BASE).check()
        assert not ok and "Could not connect" in msg


def test_generate_reports_removed_feature():
    removed = FakeResp(400, {"error": "image generation models are not currently supported"})
    mock = make_requests_mock(post_routes={"/v1/images/generations": removed})
    with patch.object(ollama_images, "requests", mock), pytest.raises(ImageGenError) as exc:
        list(OllamaImageBackend(BASE).generate_iter(ImageGenRequest(prompt="cat", model="x/z-image-turbo")))
    assert str(exc.value) == OLLAMA_IMAGE_GEN_REMOVED
    assert "v0.32.6" in str(exc.value) and "ComfyUI" in str(exc.value)


def test_generate_success_loops_batch_and_sends_size():
    mock = make_requests_mock(post_routes={"/v1/images/generations": FakeResp(200, {"data": [{"b64_json": b64png()}]})})
    req = ImageGenRequest(prompt="cat", negative_prompt="ignored", width=768, height=512, batch_size=2,
                          model="x/z-image-turbo", seed=5)
    with patch.object(ollama_images, "requests", mock):
        items = list(OllamaImageBackend(BASE).generate_iter(req))
    assert sum(isinstance(i, GenProgress) for i in items) == 2 and isinstance(items[-1], GenResult)
    images = items[-1].images
    assert len(images) == 2 and images[0].seed is None and images[0].image.size == (16, 8)
    bodies = [kw["json"] for method, path, kw in mock.calls if method == "POST"]
    assert len(bodies) == 2
    assert bodies[0] == {"model": "x/z-image-turbo", "prompt": "cat", "size": "768x512", "response_format": "b64_json"}


def test_generate_requires_model_and_b64_payload():
    with pytest.raises(ImageGenError, match="Pick an image model"):
        list(OllamaImageBackend(BASE).generate_iter(ImageGenRequest(prompt="x")))
    mock = make_requests_mock(post_routes={"/v1/images/generations": FakeResp(200, {"data": [{"url": "http://x/y.png"}]})})
    with patch.object(ollama_images, "requests", mock), pytest.raises(ImageGenError, match="b64_json"):
        list(OllamaImageBackend(BASE).generate_iter(ImageGenRequest(prompt="x", model="m")))


def test_generate_other_http_errors():
    mock = make_requests_mock(post_routes={"/v1/images/generations": FakeResp(500, {"error": "boom"})})
    with patch.object(ollama_images, "requests", mock), pytest.raises(ImageGenError, match=r"HTTP 500\): boom"):
        list(OllamaImageBackend(BASE).generate_iter(ImageGenRequest(prompt="x", model="m")))
    with patch.object(ollama_images, "requests", make_requests_mock()), pytest.raises(ImageGenError, match="images/generations"):
        list(OllamaImageBackend(BASE).generate_iter(ImageGenRequest(prompt="x", model="m")))
