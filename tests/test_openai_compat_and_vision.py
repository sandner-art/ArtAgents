# ArtAgent/tests/test_openai_compat_and_vision.py
"""Tests for core/openai_compat.py, Ollama's `format` field, and core/dataset/vision.py."""
import base64
import io
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import openai_compat as oc
from core.dataset import vision
from core.ollama_client import OllamaClient
from tests._ollama_fakes import REQUEST_PATH, FakeOllama, chat_lines

POST = "core.openai_compat.requests.post"
GET = "core.openai_compat.requests.get"


def _resp(status=200, data=None, text=None):
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.reason = "OK" if status < 400 else "Bad"
    if data is not None:
        r.json.return_value = data
        r.text = json.dumps(data)
    else:
        r.json.side_effect = ValueError("no json")
        r.text = text or ""
    return r


def _answer(content):
    return _resp(data={"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]})


# --- normalize_base_url ---

@pytest.mark.parametrize("url, expected", [
    ("http://localhost:1234", "http://localhost:1234/v1"),
    ("http://localhost:1234/", "http://localhost:1234/v1"),
    ("http://localhost:1234/v1", "http://localhost:1234/v1"),
    ("http://localhost:1234/v1/", "http://localhost:1234/v1"),
    ("localhost:8000", "http://localhost:8000/v1"),
    ("https://example.com/api/openai", "https://example.com/api/openai"),  # Paths are kept as given
    ("  http://h:8080  ", "http://h:8080/v1"),
])
def test_normalize_base_url(url, expected):
    assert oc.normalize_base_url(url) == expected


@pytest.mark.parametrize("url", ["", "   ", None])
def test_normalize_base_url_empty(url):
    with pytest.raises(oc.OpenAICompatError, match="Enter the server URL"):
        oc.normalize_base_url(url)


def test_normalize_base_url_rejects_other_schemes():
    with pytest.raises(oc.OpenAICompatError, match="Not a valid server URL"):
        oc.normalize_base_url("ftp://host/v1")


# --- image_part / build_messages ---

def _decode(part):
    url = part["image_url"]["url"]
    header, data = url.split(",", 1)
    return header, Image.open(io.BytesIO(base64.b64decode(data)))


def test_image_part_small_image_is_png():
    part = oc.image_part(Image.new("RGB", (100, 80), (10, 20, 30)))
    assert part["type"] == "image_url"
    header, img = _decode(part)
    assert header == "data:image/png;base64"
    assert img.format == "PNG" and img.size == (100, 80)


def test_image_part_large_image_is_jpeg_and_rgba_converted():
    part = oc.image_part(Image.new("RGBA", (1000, 600), (10, 20, 30, 128)))
    header, img = _decode(part)
    assert header == "data:image/jpeg;base64"
    assert img.format == "JPEG" and img.mode == "RGB" and img.size == (1000, 600)


def test_image_part_keeps_grayscale():
    _, img = _decode(oc.image_part(Image.new("L", (50, 50), 128)))
    assert img.mode == "L"


def test_build_messages_text_only_and_system():
    msgs = oc.build_messages("sys", "hello")
    assert msgs == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    assert oc.build_messages(None, "hi") == [{"role": "user", "content": "hi"}]


def test_build_messages_images_before_text():
    msgs = oc.build_messages(None, "describe", [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))])
    content = msgs[0]["content"]
    assert [p["type"] for p in content] == ["image_url", "image_url", "text"]
    assert content[-1]["text"] == "describe"


# --- client ---

def test_client_auth_header_only_with_key(monkeypatch):
    monkeypatch.delenv(oc.API_KEY_ENV, raising=False)
    assert "Authorization" not in oc.OpenAICompatClient("http://h:1")._headers()
    assert oc.OpenAICompatClient("http://h:1", "sk-1")._headers()["Authorization"] == "Bearer sk-1"


def test_client_api_key_from_env(monkeypatch):
    monkeypatch.setenv(oc.API_KEY_ENV, " env-key ")
    assert oc.OpenAICompatClient("http://h:1")._headers()["Authorization"] == "Bearer env-key"
    # An explicit key wins
    assert oc.OpenAICompatClient("http://h:1", "own")._headers()["Authorization"] == "Bearer own"


def test_client_never_uses_openai_api_key(monkeypatch):
    monkeypatch.delenv(oc.API_KEY_ENV, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai")
    assert "Authorization" not in oc.OpenAICompatClient("http://h:1")._headers()


def test_list_models_sorted(monkeypatch):
    monkeypatch.delenv(oc.API_KEY_ENV, raising=False)
    data = {"data": [{"id": "zeta"}, {"id": "alpha"}, {"no_id": 1}, "junk"]}
    with patch(GET, return_value=_resp(data=data)) as get:
        assert oc.OpenAICompatClient("http://h:1").list_models() == ["alpha", "zeta"]
    assert get.call_args[0][0] == "http://h:1/v1/models"


def test_list_models_http_error():
    with patch(GET, return_value=_resp(401, data={"error": {"message": "bad key"}})), \
            pytest.raises(oc.OpenAICompatError, match="HTTP 401: bad key"):
        oc.OpenAICompatClient("http://h:1").list_models()


def test_list_models_not_json():
    with patch(GET, return_value=_resp(200, text="<html>")), pytest.raises(oc.OpenAICompatError, match="did not return JSON"):
        oc.OpenAICompatClient("http://h:1").list_models()


def test_list_models_connection_error():
    with patch(GET, side_effect=requests.exceptions.ConnectionError("refused")), \
            pytest.raises(oc.OpenAICompatError, match="Cannot connect"):
        oc.OpenAICompatClient("http://h:1").list_models()


def test_chat_payload_and_answer():
    with patch(POST, return_value=_answer("A cat.")) as post:
        out = oc.OpenAICompatClient("http://h:1").chat(" m ", [{"role": "user", "content": "x"}], temperature=0,
                                                       seed=5, max_tokens=99)
    assert out == "A cat."
    url = post.call_args[0][0]
    payload = post.call_args[1]["json"]
    assert url == "http://h:1/v1/chat/completions"
    assert payload == {"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": False,
                       "temperature": 0.0, "seed": 5, "max_tokens": 99}


def test_chat_omits_unset_options():
    with patch(POST, return_value=_answer("x")) as post:
        oc.OpenAICompatClient("http://h:1").chat("m", [])
    payload = post.call_args[1]["json"]
    assert set(payload) == {"model", "messages", "stream"}


def test_chat_requires_model():
    with pytest.raises(oc.OpenAICompatError, match="model name"):
        oc.OpenAICompatClient("http://h:1").chat("  ", [])


def test_chat_strips_think_blocks():
    with patch(POST, return_value=_answer("<think>long reasoning</think>Final caption.")):
        assert oc.OpenAICompatClient("http://h:1").chat("m", []) == "Final caption."


def test_chat_content_as_parts():
    parts = [{"type": "text", "text": "A "}, {"type": "text", "text": "dog."}, "junk"]
    with patch(POST, return_value=_answer(parts)):
        assert oc.OpenAICompatClient("http://h:1").chat("m", []) == "A dog."


def test_chat_null_content_is_empty():
    with patch(POST, return_value=_answer(None)):
        assert oc.OpenAICompatClient("http://h:1").chat("m", []) == ""


def test_chat_json_schema_sent():
    schema = {"type": "object"}
    with patch(POST, return_value=_answer("{}")) as post:
        oc.OpenAICompatClient("http://h:1").chat("m", [], json_schema=schema)
    rf = post.call_args[1]["json"]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["schema"] == schema


def test_chat_400_with_schema_retries_without_response_format():
    sent, answers = [], [_resp(400, data={"error": "response_format unsupported"}), _answer('{"a": 1}')]

    def fake_post(url, json=None, **kwargs):
        sent.append(dict(json))  # Copy: the client reuses the payload dict for the retry
        return answers.pop(0)

    with patch(POST, side_effect=fake_post):
        out = oc.OpenAICompatClient("http://h:1").chat("m", [], json_schema={"type": "object"})
    assert out == '{"a": 1}'
    assert len(sent) == 2
    assert "response_format" in sent[0]
    assert "response_format" not in sent[1]


def test_chat_400_without_schema_is_error():
    with patch(POST, return_value=_resp(400, data={"error": {"message": "context too long"}})) as post, \
            pytest.raises(oc.OpenAICompatError, match="HTTP 400 .*context too long"):
        oc.OpenAICompatClient("http://h:1").chat("m", [])
    assert post.call_count == 1


@pytest.mark.parametrize("data, text, expected", [
    ({"error": {"message": "nope"}}, None, "nope"),
    ({"error": "plain"}, None, "plain"),
    ({"detail": "fastapi style"}, None, "fastapi style"),
    (None, "raw body", "raw body"),
])
def test_chat_http_error_messages(data, text, expected):
    with patch(POST, return_value=_resp(500, data=data, text=text)), pytest.raises(oc.OpenAICompatError, match=expected):
        oc.OpenAICompatClient("http://h:1").chat("m", [])


def test_chat_unexpected_answer_shape():
    with patch(POST, return_value=_resp(200, data={"choices": []})), \
            pytest.raises(oc.OpenAICompatError, match="Unexpected answer"):
        oc.OpenAICompatClient("http://h:1").chat("m", [])


@pytest.mark.parametrize("exc, match", [
    (requests.exceptions.ConnectionError("x"), "Cannot connect"),
    (requests.exceptions.Timeout("x"), "did not answer in time"),
    (requests.exceptions.RequestException("x"), "failed"),
])
def test_chat_transport_errors(exc, match):
    with patch(POST, side_effect=exc), pytest.raises(oc.OpenAICompatError, match=match):
        oc.OpenAICompatClient("http://h:1").chat("m", [])


# --- Ollama format field ---

def test_ollama_chat_stream_sends_format_only_when_given():
    fake = FakeOllama(chat=chat_lines("{}"))
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    with patch(REQUEST_PATH, side_effect=fake):
        list(OllamaClient("http://h:1").chat_stream("m", [], format=schema))
        list(OllamaClient("http://h:1").chat_stream("m", [], format="json"))
        list(OllamaClient("http://h:1").chat_stream("m", []))
        list(OllamaClient("http://h:1").chat_stream("m", [], format=None))
    payloads = fake.payloads("/api/chat")
    assert payloads[0]["format"] == schema
    assert payloads[1]["format"] == "json"
    assert "format" not in payloads[2] and "format" not in payloads[3]


# --- vision.ask ---

def test_vision_target_label_and_clean_name():
    assert vision.VisionTarget(vision.BACKEND_OLLAMA, "gemma4:e4b").label == "gemma4:e4b"
    assert vision.VisionTarget(vision.BACKEND_OPENAI, "jc", "http://h:1/v1").label == "jc @ http://h:1/v1"
    assert vision.clean_model_name("gemma4:e4b (VISION)") == "gemma4:e4b"
    assert vision.clean_model_name(None) == ""


def test_ask_requires_model():
    with pytest.raises(vision.VisionError, match="Select a vision model"):
        vision.ask(vision.VisionTarget(vision.BACKEND_OLLAMA, " (VISION)"), {}, "x")


def test_ask_ollama_payload():
    fake = FakeOllama(chat=chat_lines('{"subject": "cat"}'))
    img = Image.new("RGB", (16, 16), (200, 0, 0))
    schema = {"type": "object"}
    with patch(REQUEST_PATH, side_effect=fake):
        out = vision.ask(vision.VisionTarget(vision.BACKEND_OLLAMA, "gemma4:e4b (VISION)"), {"ollama_url": "http://h:9"},
                         "describe", [img], system="sys", temperature=0, seed=3, max_tokens=50, json_schema=schema)
    assert out == '{"subject": "cat"}'
    payload = fake.payloads("/api/chat")[0]
    assert payload["model"] == "gemma4:e4b"
    assert payload["options"] == {"temperature": 0.0, "seed": 3, "num_predict": 50}
    assert payload["format"] == schema
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    assert payload["messages"][-1]["content"] == "describe" and len(payload["messages"][-1]["images"]) == 1
    assert "think" not in payload  # Model without the thinking capability


def test_ask_ollama_no_options_no_format():
    fake = FakeOllama(chat=chat_lines("hi"))
    with patch(REQUEST_PATH, side_effect=fake):
        vision.ask(vision.VisionTarget(vision.BACKEND_OLLAMA, "m"), None, "x")
    payload = fake.payloads("/api/chat")[0]
    assert "options" not in payload and "format" not in payload
    assert "images" not in payload["messages"][-1]


def test_ask_ollama_error_becomes_vision_error():
    fake = FakeOllama(chat_status=404, chat_json={"error": "model 'm' not found"})
    with patch(REQUEST_PATH, side_effect=fake), pytest.raises(vision.VisionError, match="not found"):
        vision.ask(vision.VisionTarget(vision.BACKEND_OLLAMA, "m"), {}, "x")


def test_ask_openai_backend():
    with patch("core.dataset.vision.OpenAICompatClient") as cls:
        cls.return_value.chat.return_value = "A dog."
        out = vision.ask(vision.VisionTarget(vision.BACKEND_OPENAI, "jc (VISION)", "http://h:1", "k"), {}, "describe",
                         [Image.new("RGB", (8, 8))], system="s", temperature=0.2, seed=1, max_tokens=10,
                         json_schema={"type": "object"})
    assert out == "A dog."
    cls.assert_called_once_with("http://h:1", "k")
    args, kwargs = cls.return_value.chat.call_args
    assert args[0] == "jc"
    assert args[1][0] == {"role": "system", "content": "s"}
    assert [p["type"] for p in args[1][1]["content"]] == ["image_url", "text"]
    assert kwargs == {"temperature": 0.2, "seed": 1, "max_tokens": 10, "json_schema": {"type": "object"}}


def test_ask_openai_error_becomes_vision_error():
    with patch(POST, side_effect=requests.exceptions.ConnectionError("x")), \
            pytest.raises(vision.VisionError, match="Cannot connect"):
        vision.ask(vision.VisionTarget(vision.BACKEND_OPENAI, "m", "http://h:1"), {}, "x")


def test_ask_openai_bad_url_becomes_vision_error():
    with pytest.raises(vision.VisionError, match="server URL"):
        vision.ask(vision.VisionTarget(vision.BACKEND_OPENAI, "m", ""), {}, "x")


# --- parse_json_object ---

@pytest.mark.parametrize("text", [
    '{"a": "b"}',
    '```json\n{"a": "b"}\n```',
    '```\n{"a": "b"}\n```',
    'Sure! Here it is: {"a": "b"} Hope this helps.',
    '  {"a": "b"}  ',
])
def test_parse_json_object_variants(text):
    assert vision.parse_json_object(text) == {"a": "b"}


@pytest.mark.parametrize("text, match", [
    ("no json here", "did not return a JSON object"),
    ("", "did not return a JSON object"),
    (None, "did not return a JSON object"),
    ("[1, 2]", "not an object"),
    ('"just a string"', "not an object"),
])
def test_parse_json_object_errors(text, match):
    with pytest.raises(ValueError, match=match):
        vision.parse_json_object(text)


def test_parse_json_object_broken_json_raises_value_error():
    with pytest.raises(ValueError):
        vision.parse_json_object('prefix {"a": } suffix')
