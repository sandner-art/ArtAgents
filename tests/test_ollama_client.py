# ArtAgent/tests/test_ollama_client.py
"""Tests for core/ollama_client.py."""
import json
import os
import sys
from unittest.mock import patch

import pytest
import requests

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(test_dir))

from core import ollama_client
from core.ollama_client import (
    DEFAULT_BASE_URL, OllamaClient, OllamaError, base_url_from_settings, build_messages,
    filter_options, strip_think_blocks,
)
from tests._ollama_fakes import REQUEST_PATH, FakeOllama, chat_lines, make_response


@pytest.fixture(autouse=True)
def _clear_cache():
    ollama_client.clear_capability_cache()
    yield
    ollama_client.clear_capability_cache()


# --- base_url_from_settings ---

@pytest.mark.parametrize("settings, expected", [
    ({"ollama_base_url": "http://gpu-box:11434"}, "http://gpu-box:11434"),
    ({"ollama_base_url": "http://gpu-box:11434/"}, "http://gpu-box:11434"),
    ({"ollama_url": "http://localhost:11434/api/generate"}, "http://localhost:11434"), # Legacy setting
    ({"ollama_base_url": "https://a:1", "ollama_url": "http://b:2/api/generate"}, "https://a:1"), # New key wins
    ({"ollama_url": "localhost:11434"}, DEFAULT_BASE_URL), # No scheme
    ({"ollama_url": "ftp://host/x"}, DEFAULT_BASE_URL),
    ({}, DEFAULT_BASE_URL),
    (None, DEFAULT_BASE_URL),
])
def test_base_url_from_settings(settings, expected):
    assert base_url_from_settings(settings) == expected


def test_from_settings_uses_base_url():
    assert OllamaClient.from_settings({"ollama_url": "http://x:1/api/generate"}).base_url == "http://x:1"


# --- helpers ---

def test_filter_options_drops_obsolete_keys(capsys):
    ollama_client._warned_options.discard("tfs_z")
    clean = filter_options({"temperature": 0.7, "tfs_z": 0.9, "num_ctx": 8192, "mirostat": 0})
    assert clean == {"temperature": 0.7, "num_ctx": 8192}
    assert "'tfs_z' is no longer supported" in capsys.readouterr().out


def test_filter_options_warns_once_per_key(capsys):
    ollama_client._warned_options.discard("penalize_newline")
    filter_options({"penalize_newline": False})
    filter_options({"penalize_newline": False})
    assert capsys.readouterr().out.count("penalize_newline") == 1


def test_filter_options_none():
    assert filter_options(None) == {}


@pytest.mark.parametrize("text, expected", [
    ("<think>reasoning</think>Answer", "Answer"),
    ("<THINK>\nmulti\nline\n</THINK>\n  Answer ", "Answer"),
    ("No tags here", "No tags here"),
    ("", ""),
    (None, None),
])
def test_strip_think_blocks(text, expected):
    assert strip_think_blocks(text) == expected


def test_build_messages_full():
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "ignored"}, # Only user/assistant turns are replayed
        {"role": "user", "content": None},        # Invalid content dropped
        "garbage",
    ]
    msgs = build_messages("sys", "q2", ["b64img"], history)
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2", "images": ["b64img"]},
    ]


def test_build_messages_minimal():
    assert build_messages(None, "hi") == [{"role": "user", "content": "hi"}]


# --- chat_stream ---

def test_chat_stream_splits_content_and_thinking():
    fake = FakeOllama(chat=chat_lines("Hel", "lo", thinking=["hmm"]))
    client = OllamaClient("http://h:1")
    with patch(REQUEST_PATH, side_effect=fake):
        pieces = list(client.chat_stream("m", [{"role": "user", "content": "x"}], options={"temperature": 1, "tfs_z": 1},
                                         think=False, keep_alive="5m"))
    assert pieces == [("thinking", "hmm"), ("content", "Hel"), ("content", "lo")]
    payload = fake.payloads("/api/chat")[0]
    assert payload["options"] == {"temperature": 1}
    assert payload["think"] is False and payload["keep_alive"] == "5m"
    fake.chat_responses[0].close.assert_called_once()


def test_chat_stream_omits_empty_fields():
    fake = FakeOllama()
    with patch(REQUEST_PATH, side_effect=fake):
        list(OllamaClient("http://h:1").chat_stream("m", []))
    payload = fake.payloads("/api/chat")[0]
    assert "options" not in payload and "think" not in payload and "keep_alive" not in payload


def test_chat_stream_stops_at_done():
    lines = chat_lines("one") + [json.dumps({"message": {"content": "after done"}, "done": False})]
    fake = FakeOllama(chat=lines)
    with patch(REQUEST_PATH, side_effect=fake):
        pieces = list(OllamaClient("http://h:1").chat_stream("m", []))
    assert pieces == [("content", "one")]


def test_chat_stream_error_chunk_raises():
    fake = FakeOllama(chat=['{"error": "model requires more system memory"}'])
    with patch(REQUEST_PATH, side_effect=fake), pytest.raises(OllamaError, match="more system memory"):
        list(OllamaClient("http://h:1").chat_stream("m", []))
    fake.chat_responses[0].close.assert_called_once()


def test_chat_stream_bad_json_raises():
    fake = FakeOllama(chat=["{not json"])
    with patch(REQUEST_PATH, side_effect=fake), pytest.raises(OllamaError, match="Unreadable response"):
        list(OllamaClient("http://h:1").chat_stream("m", []))


def test_http_404_reports_server_error_text():
    fake = FakeOllama(chat_status=404, chat_json={"error": "model not found"})
    with patch(REQUEST_PATH, side_effect=fake), pytest.raises(OllamaError) as exc:
        list(OllamaClient("http://h:1").chat_stream("m", []))
    assert "HTTP 404" in str(exc.value) and "model not found" in str(exc.value)


def test_http_error_without_json_uses_text():
    with patch(REQUEST_PATH, return_value=make_response(status_code=500, text="Internal boom")), \
         pytest.raises(OllamaError, match="HTTP 500: Internal boom"):
        OllamaClient("http://h:1").list_models()


def test_connection_error_message():
    with patch(REQUEST_PATH, side_effect=requests.exceptions.ConnectionError("refused")), \
         pytest.raises(OllamaError, match=r"Could not connect to Ollama at http://h:1\. Is it running\?"):
        OllamaClient("http://h:1").version()


def test_generic_request_exception_message():
    with patch(REQUEST_PATH, side_effect=requests.exceptions.RequestException("weird")), \
         pytest.raises(OllamaError, match="Error communicating with Ollama: weird"):
        OllamaClient("http://h:1").version()


def test_interrupted_stream_raises_ollama_error():
    resp = make_response(lines=[])
    resp.iter_lines.side_effect = requests.exceptions.ChunkedEncodingError("cut")
    with patch(REQUEST_PATH, return_value=resp), pytest.raises(OllamaError, match="interrupted"):
        list(OllamaClient("http://h:1").chat_stream("m", []))
    resp.close.assert_called_once()


def test_chat_returns_content_only_without_think_tags():
    fake = FakeOllama(chat=chat_lines("<think>x</think>", "Answer", thinking=["t"]))
    with patch(REQUEST_PATH, side_effect=fake):
        assert OllamaClient("http://h:1").chat("m", []) == "Answer"


# --- discovery ---

def test_capabilities_are_cached_per_server_and_model():
    fake = FakeOllama(capabilities={"m": ["completion", "vision"]})
    client = OllamaClient("http://h:1")
    with patch(REQUEST_PATH, side_effect=fake):
        assert client.capabilities("m") == ["completion", "vision"]
        assert client.capabilities("m") == ["completion", "vision"]
        assert len(fake.payloads("/api/show")) == 1
        client.capabilities("m", use_cache=False)
        assert len(fake.payloads("/api/show")) == 2
        OllamaClient("http://other:2").capabilities("m") # Different server -> separate entry
        assert len(fake.payloads("/api/show")) == 3


def test_capabilities_unknown_on_error_and_not_cached():
    client = OllamaClient("http://h:1")
    with patch(REQUEST_PATH, side_effect=requests.exceptions.ConnectionError("x")):
        assert client.capabilities("m") == []
    fake = FakeOllama(capabilities={"m": ["completion"]})
    with patch(REQUEST_PATH, side_effect=fake):
        assert client.capabilities("m") == ["completion"]


def test_list_models_filters_invalid_entries():
    fake = FakeOllama(tags=[{"name": "a"}, {"model": "no-name"}, "junk", {"name": "b"}])
    with patch(REQUEST_PATH, side_effect=fake):
        assert [m["name"] for m in OllamaClient("http://h:1").list_models()] == ["a", "b"]


def test_running_version_and_unload():
    fake = FakeOllama(ps=[{"name": "gemma4:e4b"}])
    client = OllamaClient("http://h:1")
    with patch(REQUEST_PATH, side_effect=fake):
        assert client.running() == [{"name": "gemma4:e4b"}]
        assert client.version() == "0.40.1"
        assert client.unload("gemma4:e4b") == "Model 'gemma4:e4b' unloaded."
    assert fake.payloads("/api/generate") == [{"model": "gemma4:e4b", "keep_alive": 0}]


# --- think_value ---

@pytest.mark.parametrize("setting, caps, expected", [
    (None, ["thinking"], None),
    ("auto", ["thinking"], None),
    ("off", ["thinking"], False),
    ("on", ["thinking"], True),
    (True, ["thinking"], True),
    (False, ["thinking"], False),
    ("on", ["completion"], None),
    ("off", [], None),
])
def test_think_value(setting, caps, expected):
    client = OllamaClient("http://h:1")
    with patch.object(OllamaClient, "capabilities", return_value=caps):
        assert client.think_value("m", setting) is expected
