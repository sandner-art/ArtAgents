# ArtAgent/tests/test_image_gen_sdwebui.py
import base64
import io
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image

from core.image_gen import sdwebui
from core.image_gen.base import GenProgress, GenResult, ImageGenError, ImageGenRequest
from core.image_gen.sdwebui import API_NOT_ENABLED, SDWebUIBackend, decode_b64_image

BASE = "http://127.0.0.1:7861"


class FakeResp:
    def __init__(self, status=200, data=None, text=None):
        self.status_code = status
        self._data = data
        self.text = text if text is not None else (json.dumps(data) if data is not None else "")

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


def b64png(color=(1, 2, 3), size=(8, 8), fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
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
                return FakeResp(404, {"detail": "Not Found"})
            return value(url, **kwargs) if callable(value) else value
        return handler

    mock.get.side_effect = route(get_routes, "GET")
    mock.post.side_effect = route(post_routes, "POST")
    mock.calls = calls
    return mock


def make_backend(**kw):
    b = SDWebUIBackend(BASE + "/sdapi/v1/", **kw)
    b.poll_interval = 0.02
    return b


def txt2img_result(seeds, with_grid=False):
    images = [b64png((s % 255, 0, 0)) for s in seeds]
    if with_grid:
        images.insert(0, b64png((255, 255, 255), size=(16, 8)))
    info = {"all_seeds": seeds, "seed": seeds[0], "sd_model_name": "juggernaut", "sampler_name": "Euler a",
            "steps": 24, "cfg_scale": 6}
    return FakeResp(200, {"images": images, "parameters": {}, "info": json.dumps(info)})


def test_url_normalisation():
    assert make_backend().base_url == BASE
    assert SDWebUIBackend("127.0.0.1:7860").base_url == "http://127.0.0.1:7860"


def test_build_payload_omits_unset_and_sets_checkpoint_override():
    payload = SDWebUIBackend.build_payload(ImageGenRequest(prompt="p", width=640, height=480, batch_size=2))
    assert payload == {"prompt": "p", "negative_prompt": "", "width": 640, "height": 480, "seed": -1, "batch_size": 2}
    full = SDWebUIBackend.build_payload(ImageGenRequest(prompt="p", negative_prompt="n", steps=30, cfg_scale=6.5,
                                                         seed=9, sampler="DPM++ 2M", model="sdxl [abc]"))
    assert full["steps"] == 30 and full["cfg_scale"] == 6.5 and full["sampler_name"] == "DPM++ 2M" and full["seed"] == 9
    assert full["override_settings"] == {"sd_model_checkpoint": "sdxl [abc]"}
    assert full["override_settings_restore_afterwards"] is True


def test_check_lists_and_404_message():
    ok_mock = make_requests_mock(get_routes={
        "/sdapi/v1/sd-models": FakeResp(200, [{"title": "a.safetensors [1]", "model_name": "a"}, {"model_name": "b"}]),
        "/sdapi/v1/samplers": FakeResp(200, [{"name": "Euler a"}, {"name": "DPM++ 2M"}]),
    })
    with patch.object(sdwebui, "requests", ok_mock):
        b = make_backend()
        ok, msg = b.check()
        assert ok and "2 checkpoint" in msg
        assert b.list_models() == ["a.safetensors [1]", "b"]
        assert b.list_samplers() == ["Euler a", "DPM++ 2M"]

    with patch.object(sdwebui, "requests", make_requests_mock()):
        b = make_backend()
        ok, msg = b.check()
        assert not ok and msg == API_NOT_ENABLED and "--api" in msg and "7860" in msg
        with pytest.raises(ImageGenError, match="--api"):
            b.list_models()


def test_non_json_response_is_explained():
    mock = make_requests_mock(get_routes={"/sdapi/v1/sd-models": FakeResp(200, None, text="<html>")})
    with patch.object(sdwebui, "requests", mock), pytest.raises(ImageGenError, match="invalid JSON"):
        make_backend().list_models()


def test_generate_yields_progress_and_parses_seeds():
    progress_steps = iter(range(1, 100))

    def slow_txt2img(url, **kw):
        time.sleep(0.25)
        return txt2img_result([100, 101], with_grid=True)

    def progress(url, **kw):
        step = next(progress_steps)
        return FakeResp(200, {"progress": min(1.0, step / 24), "eta_relative": 2.0,
                              "state": {"sampling_step": step, "sampling_steps": 24, "job_no": 0, "job_count": 2}})

    mock = make_requests_mock(get_routes={"/sdapi/v1/progress": progress},
                              post_routes={"/sdapi/v1/txt2img": slow_txt2img})
    req = ImageGenRequest(prompt="lamp", steps=24, seed=100, batch_size=2, model="m [1]")
    with patch.object(sdwebui, "requests", mock):
        items = list(make_backend().generate_iter(req))
    assert isinstance(items[-1], GenResult)
    progress_items = [i for i in items if isinstance(i, GenProgress)]
    assert any(p.message.startswith("step ") and "/24" in p.message and "eta 2s" in p.message for p in progress_items)
    assert any(isinstance(p.fraction, float) and p.fraction > 0 for p in progress_items)
    images = items[-1].images
    assert [g.seed for g in images] == [100, 101]  # grid dropped
    assert images[0].image.size == (8, 8) and images[0].info["sd_model_name"] == "juggernaut"
    sent = next(kw for method, path, kw in mock.calls if path == "/sdapi/v1/txt2img")
    assert sent["json"]["override_settings"] == {"sd_model_checkpoint": "m [1]"}
    assert any(kw.get("params") == {"skip_current_image": "true"} for method, path, kw in mock.calls if path == "/sdapi/v1/progress")


def test_progress_failure_is_not_fatal():
    def txt2img(url, **kw):
        time.sleep(0.1)
        return txt2img_result([5])

    mock = make_requests_mock(get_routes={"/sdapi/v1/progress": FakeResp(500, None, text="boom")},
                              post_routes={"/sdapi/v1/txt2img": txt2img})
    with patch.object(sdwebui, "requests", mock):
        images = make_backend().generate(ImageGenRequest(prompt="x", seed=5))
    assert [g.seed for g in images] == [5]


def test_generate_404_and_http_errors():
    with patch.object(sdwebui, "requests", make_requests_mock(get_routes={"/sdapi/v1/progress": FakeResp(200, {})})):
        with pytest.raises(ImageGenError, match="--api"):
            list(make_backend().generate_iter(ImageGenRequest(prompt="x")))
    mock = make_requests_mock(post_routes={"/sdapi/v1/txt2img": FakeResp(500, {"error": "OutOfMemoryError", "detail": "CUDA OOM"})})
    with patch.object(sdwebui, "requests", mock), pytest.raises(ImageGenError, match="HTTP 500.*OutOfMemoryError"):
        list(make_backend().generate_iter(ImageGenRequest(prompt="x")))


def test_generate_connection_error_is_friendly():
    mock = make_requests_mock()
    mock.post.side_effect = requests.exceptions.ConnectionError("refused")
    with patch.object(sdwebui, "requests", mock), pytest.raises(ImageGenError, match="Could not connect.*--api"):
        list(make_backend().generate_iter(ImageGenRequest(prompt="x")))


def test_closing_generator_interrupts_server():
    released = threading.Event()

    def blocking_txt2img(url, **kw):
        released.wait(5)
        return txt2img_result([1])

    def interrupt(url, **kw):
        released.set()
        return FakeResp(200, {})

    mock = make_requests_mock(
        get_routes={"/sdapi/v1/progress": FakeResp(200, {"progress": 0.1, "state": {"sampling_step": 2, "sampling_steps": 20}})},
        post_routes={"/sdapi/v1/txt2img": blocking_txt2img, "/sdapi/v1/interrupt": interrupt})
    with patch.object(sdwebui, "requests", mock):
        gen = make_backend().generate_iter(ImageGenRequest(prompt="x"))
        next(gen)
        next(gen)
        gen.close()
        assert released.wait(2)
    assert any(path == "/sdapi/v1/interrupt" for method, path, kw in mock.calls if method == "POST")


def test_decode_b64_image_variants():
    assert decode_b64_image("data:image/png;base64," + b64png()).size == (8, 8)
    assert decode_b64_image(b64png(fmt="JPEG")).mode == "RGB"
    with pytest.raises(ImageGenError, match="unreadable"):
        decode_b64_image(b64png(fmt="TIFF"))
    with pytest.raises(ImageGenError, match="unreadable"):
        decode_b64_image(base64.b64encode(b"not an image").decode())
    with pytest.raises(ImageGenError):
        decode_b64_image(None)
