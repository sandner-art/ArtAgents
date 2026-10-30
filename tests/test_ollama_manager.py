# ArtAgent/tests/test_ollama_manager.py
"""Tests for core/ollama_manager.py (unloading models and model discovery)."""
import os
import sys
from unittest.mock import patch

import pytest
import requests

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(test_dir))

from core import ollama_client
from core import ollama_manager
from core.ollama_manager import (
    discover_models, model_choices, model_display_name, release_all_models_logic, release_model,
    vision_model_choices, worker_model_choices,
)
from tests._ollama_fakes import REQUEST_PATH, FakeOllama

LOAD_JSON_PATH = "core.ollama_manager.load_json"
SETTINGS = {"ollama_base_url": "http://fake-ollama:11434"}
CURATED = [
    {"name": "gemma4:e4b", "vision": True, "notes": "Curated note"},
    {"name": "llama3:latest", "vision": False},
]


@pytest.fixture(autouse=True)
def _clear_cache():
    ollama_client.clear_capability_cache()
    yield
    ollama_client.clear_capability_cache()


# --- release_model ---

def test_release_model_with_settings_dict():
    fake = FakeOllama()
    with patch(REQUEST_PATH, side_effect=fake):
        msg = release_model("gemma4:e4b", SETTINGS)
    assert msg == "Model 'gemma4:e4b' unloaded."
    method, url, kwargs = fake.calls[0]
    assert (method, url) == ("POST", "http://fake-ollama:11434/api/generate")
    assert kwargs["json"] == {"model": "gemma4:e4b", "keep_alive": 0}


@pytest.mark.parametrize("url", ["http://fake-ollama:11434/api/generate", "http://fake-ollama:11434"])
def test_release_model_with_legacy_or_base_url(url):
    fake = FakeOllama()
    with patch(REQUEST_PATH, side_effect=fake):
        release_model("m", url)
    assert fake.calls[0][1] == "http://fake-ollama:11434/api/generate"


@pytest.mark.parametrize("name, target", [("", SETTINGS), ("m", None), ("m", "")])
def test_release_model_missing_inputs(name, target):
    with patch(REQUEST_PATH) as mock_request:
        msg = release_model(name, target)
    assert msg.startswith("Skipping release")
    mock_request.assert_not_called()


def test_release_model_error_is_reported():
    with patch(REQUEST_PATH, side_effect=requests.exceptions.ConnectionError("down")):
        msg = release_model("m", SETTINGS)
    assert msg.startswith("Error releasing model 'm':") and "Could not connect" in msg


# --- release_all_models_logic ---

def test_release_all_unloads_running_models():
    fake = FakeOllama(ps=[{"name": "a:1"}, {"model": "b:2"}, {}])
    with patch(REQUEST_PATH, side_effect=fake):
        summary = release_all_models_logic(SETTINGS)
    assert [p["model"] for p in fake.payloads("/api/generate")] == ["a:1", "b:2"]
    assert "Model 'a:1' unloaded." in summary and "Model 'b:2' unloaded." in summary


def test_release_all_when_nothing_loaded():
    fake = FakeOllama(ps=[])
    with patch(REQUEST_PATH, side_effect=fake):
        assert release_all_models_logic(SETTINGS) == "No models are currently loaded in Ollama."
    assert fake.payloads("/api/generate") == []


def test_release_all_falls_back_to_models_json_when_ps_fails():
    fake = FakeOllama(ps=requests.exceptions.ConnectionError("no ps"))
    with patch(REQUEST_PATH, side_effect=fake), \
         patch(LOAD_JSON_PATH, return_value=[{"name": "x"}, {"vision": True}, {"name": "y"}]):
        summary = release_all_models_logic(SETTINGS)
    assert [p["model"] for p in fake.payloads("/api/generate")] == ["x", "y"]
    assert "Model 'x' unloaded." in summary and "Model 'y' unloaded." in summary


def test_release_all_reports_error_when_ps_fails_and_no_models_json():
    with patch(REQUEST_PATH, side_effect=requests.exceptions.ConnectionError("down")), \
         patch(LOAD_JSON_PATH, return_value={}):
        summary = release_all_models_logic(SETTINGS)
    assert summary.startswith("Cannot release models:") and "Could not connect" in summary


# --- discover_models ---

def test_discover_installed_models_with_capabilities():
    fake = FakeOllama(
        tags=[
            {"name": "llama3:latest", "details": {"parameter_size": "8B"}},
            {"name": "gemma4:e4b"},
            {"name": "embeddinggemma:300m"},
            {"name": "x/z-image-turbo:latest", "capabilities": ["image"]}, # Reported by /api/tags
            {"name": "qwen3:8b"},
        ],
        capabilities={
            "llama3:latest": ["completion"],
            "gemma4:e4b": ["completion", "vision", "thinking"],
            "embeddinggemma:300m": ["embedding"],
            "qwen3:8b": ["completion", "thinking", "tools"],
        },
    )
    with patch(REQUEST_PATH, side_effect=fake):
        models, status = discover_models(SETTINGS, CURATED)
    assert [m["name"] for m in models] == ["gemma4:e4b", "llama3:latest", "qwen3:8b"] # Vision first, then by name
    gemma = models[0]
    assert gemma == {"name": "gemma4:e4b", "vision": True, "thinking": True, "installed": True, "notes": "Curated note"}
    assert models[1]["notes"] == "8B" # From /api/tags details when not curated
    assert models[2]["thinking"] is True and models[2]["vision"] is False
    assert status == "Found 3 installed Ollama model(s), 1 with vision."
    # /api/tags capabilities are used directly (no /api/show for the image model)
    assert {"model": "x/z-image-turbo:latest"} not in fake.payloads("/api/show")


def test_discover_uses_curated_vision_flag_when_capabilities_unknown():
    fake = FakeOllama(tags=[{"name": "gemma4:e4b"}], capabilities={"gemma4:e4b": []})
    with patch(REQUEST_PATH, side_effect=fake):
        models, _ = discover_models(SETTINGS, CURATED)
    assert models[0]["vision"] is True and models[0]["thinking"] is False


def test_discover_falls_back_to_curated_when_unreachable():
    with patch(REQUEST_PATH, side_effect=requests.exceptions.ConnectionError("down")):
        models, status = discover_models(SETTINGS, CURATED)
    assert [m["name"] for m in models] == ["gemma4:e4b", "llama3:latest"]
    assert all(m["installed"] is False for m in models)
    assert status.startswith("Ollama not reachable") and "models.json (2 models)" in status


def test_discover_with_no_chat_models_installed():
    fake = FakeOllama(tags=[{"name": "nomic-embed-text"}], capabilities={"nomic-embed-text": ["embedding"]})
    with patch(REQUEST_PATH, side_effect=fake):
        models, status = discover_models(SETTINGS, CURATED)
    assert [m["name"] for m in models] == ["gemma4:e4b", "llama3:latest"]
    assert "no chat models installed" in status and "ollama pull gemma4" in status


def test_discover_loads_models_json_by_default():
    with patch(REQUEST_PATH, side_effect=requests.exceptions.ConnectionError("down")), \
         patch(LOAD_JSON_PATH, return_value=[{"name": "from-file", "vision": False}, "junk"]):
        models, _ = discover_models(SETTINGS)
    assert [m["name"] for m in models] == ["from-file"]


def test_discover_clears_capability_cache():
    fake = FakeOllama(tags=[{"name": "m"}], capabilities={"m": ["completion"]})
    with patch(REQUEST_PATH, side_effect=fake):
        discover_models(SETTINGS, [])
        fake.capabilities = {"m": ["completion", "vision"]} # Model updated in Ollama
        models, _ = discover_models(SETTINGS, [])
    assert models[0]["vision"] is True


# --- choices ---

MODELS = [
    {"name": "gemma4:e4b", "vision": True},
    {"name": "llama3:latest", "vision": False},
    {"name": "gemma4:e4b", "vision": True}, # Duplicate name collapses in worker list
]


def test_model_display_name():
    assert model_display_name({"name": "a", "vision": True}) == "a (VISION)"
    assert model_display_name({"name": "b"}) == "b"


def test_choice_helpers():
    assert model_choices(MODELS) == ["gemma4:e4b (VISION)", "llama3:latest", "gemma4:e4b (VISION)"]
    assert vision_model_choices(MODELS) == ["gemma4:e4b (VISION)", "gemma4:e4b (VISION)"]
    assert worker_model_choices(MODELS) == ["gemma4:e4b", "llama3:latest"]


def test_load_curated_models_handles_bad_file():
    with patch(LOAD_JSON_PATH, return_value={"not": "a list"}):
        assert ollama_manager.load_curated_models() == []
