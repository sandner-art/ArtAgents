# ArtAgent/tests/test_agent.py
"""Tests for agents/ollama_agent.py (Ollama /api/chat via core.ollama_client)."""
import base64
import os
import sys
from unittest.mock import patch

import pytest
import requests

test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir)
sys.path.insert(0, project_root)

from PIL import Image as PILImageModule

from agents.ollama_agent import (
    build_effective_options, get_llm_response, is_error_response, stream_llm_response,
)
from core import ollama_client
from tests._ollama_fakes import REQUEST_PATH, FakeOllama, chat_lines, make_response

ENCODE_IMAGE_PATH = 'agents.ollama_agent.encode_image_for_ollama'

# --- Test Data ---
MOCK_SETTINGS = {
    "ollama_url": "http://fake-test-ollama:11434/api/generate", # Legacy full URL still works
    "ollama_api_options": {"seed": 1, "temperature": 0.5, "num_ctx": 2048, "num_predict": 512},
    "use_ollama_api_options": False,
    "ollama_api_prompt_to_console": False,
}
MOCK_ROLES_DATA = {
    "Tester": {"description": "A test role.", "ollama_api_options": {"temperature": 0.8, "top_k": 10}},
    "NoOptionsRole": {"description": "Another role."},
    "PredictRole": {"description": "Role defining num_predict.", "ollama_api_options": {"num_predict": 777}},
}
DEFAULT_ARGS = {
    "role": "Tester", "prompt": "Test prompt", "model": "test-model",
    "settings": MOCK_SETTINGS, "roles_data": MOCK_ROLES_DATA, "images": None,
    "max_tokens": 1500, "ollama_api_options": None,
}


@pytest.fixture(autouse=True)
def _clear_caps_cache():
    ollama_client.clear_capability_cache()
    yield
    ollama_client.clear_capability_cache()


def _run(fake, **overrides):
    with patch(REQUEST_PATH, side_effect=fake):
        return get_llm_response(**{**DEFAULT_ARGS, **overrides})


def _chat_payload(fake):
    payloads = fake.payloads("/api/chat")
    assert len(payloads) == 1
    return payloads[0]


# --- Success paths ---

def test_get_llm_response_success_text_only():
    fake = FakeOllama(chat=chat_lines("Hello", " world"))
    result = _run(fake)
    assert result == "Hello world"
    method, url, kwargs = [c for c in fake.calls if c[1].endswith("/api/chat")][0]
    assert method == "POST"
    assert url == "http://fake-test-ollama:11434/api/chat" # Base URL derived from the legacy setting
    assert kwargs["stream"] is True
    payload = kwargs["json"]
    assert payload["model"] == "test-model"
    assert payload["stream"] is True
    assert payload["messages"] == [{"role": "user", "content": "Test prompt"}]
    fake.chat_responses[0].close.assert_called() # Streaming connection released


def test_system_prompt_and_history_are_sent_as_messages():
    fake = FakeOllama()
    history = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"}]
    _run(fake, system="You are a tester.", history=history)
    assert _chat_payload(fake)["messages"] == [
        {"role": "system", "content": "You are a tester."},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "Test prompt"},
    ]


def test_stream_llm_response_yields_accumulated_text():
    fake = FakeOllama(chat=chat_lines("A", "B", "C"))
    with patch(REQUEST_PATH, side_effect=fake):
        updates = list(stream_llm_response(**DEFAULT_ARGS))
    assert updates[:3] == ["A", "AB", "ABC"]
    assert updates[-1] == "ABC"


def test_thinking_is_not_part_of_the_answer():
    fake = FakeOllama(chat=chat_lines("Final", thinking=["let me think", " more"]))
    assert _run(fake) == "Final"


def test_think_tags_in_content_are_stripped():
    fake = FakeOllama(chat=chat_lines("<think>hidden reasoning</think>", "Visible answer"))
    assert _run(fake) == "Visible answer"


def test_empty_stream_returns_placeholder_message():
    fake = FakeOllama(chat=chat_lines())
    assert _run(fake) == "No response text received from LLM stream."


# --- Options ---

def test_global_options_not_sent_when_disabled():
    """With 'Use Advanced Ollama API Options' off, only Max Tokens and role options are sent."""
    fake = FakeOllama()
    _run(fake, use_global_options=False)
    options = _chat_payload(fake)["options"]
    assert options == {"num_predict": 1500, "temperature": 0.8, "top_k": 10}


def test_global_options_follow_settings_flag_by_default():
    fake = FakeOllama()
    _run(fake, settings={**MOCK_SETTINGS, "use_ollama_api_options": True})
    options = _chat_payload(fake)["options"]
    assert options["seed"] == 1 and options["num_ctx"] == 2048
    assert options["temperature"] == 0.8 # Role beats global
    assert options["num_predict"] == 1500 # Max Tokens beats the global num_predict


def test_role_num_predict_beats_max_tokens():
    fake = FakeOllama()
    _run(fake, role="PredictRole", max_tokens=100)
    assert _chat_payload(fake)["options"]["num_predict"] == 777


def test_direct_overrides_have_highest_priority():
    fake = FakeOllama()
    _run(fake, ollama_api_options={"temperature": 0.1, "num_predict": 42}, use_global_options=True)
    options = _chat_payload(fake)["options"]
    assert options["temperature"] == 0.1 and options["num_predict"] == 42


def test_obsolete_options_are_not_sent():
    fake = FakeOllama()
    settings = {**MOCK_SETTINGS, "ollama_api_options": {"tfs_z": 0.8, "mirostat": 0, "f16_kv": True, "top_p": 0.9}}
    _run(fake, settings=settings, use_global_options=True)
    options = _chat_payload(fake)["options"]
    assert "tfs_z" not in options and "mirostat" not in options and "f16_kv" not in options
    assert options["top_p"] == 0.9


def test_build_effective_options_unknown_role_and_no_max_tokens():
    opts = build_effective_options({"ollama_api_options": {"seed": 3}}, MOCK_ROLES_DATA, "Missing", None, None, True)
    assert opts == {"seed": 3}


# --- Thinking models ---

@pytest.mark.parametrize("setting, caps, expected", [
    ("off", ["completion", "thinking"], False),
    ("on", ["completion", "thinking"], True),
    ("auto", ["completion", "thinking"], None),
    ("off", ["completion"], None), # Non-thinking model never gets the field
])
def test_think_field_only_for_thinking_models(setting, caps, expected):
    fake = FakeOllama(capabilities={"test-model": caps})
    _run(fake, settings={**MOCK_SETTINGS, "ollama_think": setting})
    payload = _chat_payload(fake)
    if expected is None:
        assert "think" not in payload
    else:
        assert payload["think"] is expected


def test_default_think_setting_is_off():
    fake = FakeOllama(capabilities={"test-model": ["completion", "thinking"]})
    _run(fake) # MOCK_SETTINGS has no ollama_think
    assert _chat_payload(fake)["think"] is False


# --- Errors ---

def test_connection_error():
    with patch(REQUEST_PATH, side_effect=requests.exceptions.ConnectionError("refused")):
        result = get_llm_response(**DEFAULT_ARGS)
    assert result.startswith("⚠️ Error:")
    assert "Could not connect to Ollama at http://fake-test-ollama:11434" in result


def test_timeout():
    with patch(REQUEST_PATH, side_effect=requests.exceptions.Timeout("slow")):
        result = get_llm_response(**DEFAULT_ARGS)
    assert result.startswith("⚠️ Error:") and "did not respond in time" in result


def test_http_error_includes_ollama_message():
    fake = FakeOllama(chat_status=404, chat_json={"error": "model 'test-model' not found, try pulling it first"})
    result = _run(fake)
    assert result.startswith("⚠️ Error:")
    assert "HTTP 404" in result and "not found, try pulling it first" in result


def test_error_chunk_in_stream():
    fake = FakeOllama(chat=['{"message": {"content": "Part"}, "done": false}', '{"error": "out of memory"}'])
    result = _run(fake)
    assert result == "⚠️ Error: Ollama returned an error: out of memory"
    fake.chat_responses[0].close.assert_called()


def test_json_decode_error_in_stream():
    fake = FakeOllama(chat=['{"message": {"content": "Part 1."}, "done": false}', '{"invalid json chunk'])
    result = _run(fake)
    assert result.startswith("⚠️ Error:") and "Unreadable response from Ollama" in result
    assert "Part 1." not in result # Errors replace the partial answer


def test_stream_yields_partial_then_error():
    fake = FakeOllama(chat=['{"message": {"content": "Part"}, "done": false}', '{"error": "boom"}'])
    with patch(REQUEST_PATH, side_effect=fake):
        updates = list(stream_llm_response(**DEFAULT_ARGS))
    assert updates[0] == "Part"
    assert is_error_response(updates[-1])


def test_unexpected_exception_is_reported_as_error():
    with patch("agents.ollama_agent.OllamaClient.chat_stream", side_effect=RuntimeError("bug")), \
         patch("agents.ollama_agent.OllamaClient.think_value", return_value=None):
        result = get_llm_response(**DEFAULT_ARGS)
    assert result.startswith("⚠️ Error: Unexpected error") and "bug" in result


# --- Images ---

def test_with_image_attaches_base64_to_user_message():
    fake = FakeOllama(chat=chat_lines("Image analyzed."))
    img = PILImageModule.new("RGB", (8, 8), "red")
    assert _run(fake, images=[img]) == "Image analyzed."
    user_msg = _chat_payload(fake)["messages"][-1]
    assert len(user_msg["images"]) == 1
    assert base64.b64decode(user_msg["images"][0])[:2] == b"\xff\xd8" # JPEG


def test_with_palette_image_is_converted():
    """GIF/palette images used to fail with 'cannot write mode P as JPEG'."""
    fake = FakeOllama()
    img = PILImageModule.new("P", (8, 8), 3)
    result = _run(fake, images=[img])
    assert not is_error_response(result)
    assert base64.b64decode(_chat_payload(fake)["messages"][-1]["images"][0])[:2] == b"\xff\xd8"


def test_with_invalid_image_object_is_skipped(capsys):
    fake = FakeOllama()
    _run(fake, images=["not an image"])
    assert "images" not in _chat_payload(fake)["messages"][-1]
    assert "is not a PIL Image object" in capsys.readouterr().out


@patch(ENCODE_IMAGE_PATH, side_effect=Exception("PIL Save Error"))
def test_image_processing_error(mock_encode):
    fake = FakeOllama()
    result = _run(fake, images=[PILImageModule.new("RGB", (4, 4))])
    assert result.startswith("⚠️ Error: Failed to process image data") and "PIL Save Error" in result
    assert fake.payloads("/api/chat") == [] # Nothing sent


# --- Logging / helpers ---

def test_console_logging(capsys):
    fake = FakeOllama()
    _run(fake, settings={**MOCK_SETTINGS, "ollama_api_prompt_to_console": True}, system="Persona")
    out = capsys.readouterr().out
    assert "--- Ollama Request ---" in out and '"model": "test-model"' in out and "Persona" in out


def test_no_console_logging_when_disabled(capsys):
    _run(FakeOllama())
    assert "--- Ollama Request ---" not in capsys.readouterr().out


@pytest.mark.parametrize("text, expected", [
    ("⚠️ Error: x", True), ("Error: y", True), ("fine", False), (None, False), (42, False),
])
def test_is_error_response(text, expected):
    assert is_error_response(text) is expected
