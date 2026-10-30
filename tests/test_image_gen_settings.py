# ArtAgent/tests/test_image_gen_settings.py
"""Tests for core/image_gen_settings.py (connection + image server settings)."""
import os
import sys
from unittest.mock import patch

import pytest

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(test_dir))

from core import image_gen_settings as igs
from core.image_gen_settings import (
    IMAGE_GEN_DEFAULTS, backend_url, image_gen_settings, is_valid_http_url, resolve_output_dir,
    save_connection_settings_callback,
)
from core.utils import PROJECT_ROOT

LOAD_PATH = "core.image_gen_settings.load_json"
SAVE_PATH = "core.image_gen_settings.save_json"


def test_defaults_when_missing():
    assert image_gen_settings(None) == IMAGE_GEN_DEFAULTS
    assert image_gen_settings({"image_generation": "garbage"}) == IMAGE_GEN_DEFAULTS


def test_stored_values_override_defaults_but_blanks_do_not():
    merged = image_gen_settings({"image_generation": {"comfyui_url": "http://gpu:8188", "sdwebui_url": "", "output_dir": None,
                                                      "save_prompt_txt": False}})
    assert merged["comfyui_url"] == "http://gpu:8188"
    assert merged["sdwebui_url"] == IMAGE_GEN_DEFAULTS["sdwebui_url"]
    assert merged["output_dir"] == IMAGE_GEN_DEFAULTS["output_dir"]
    assert merged["save_prompt_txt"] is False # False is a real value, not "blank"
    assert IMAGE_GEN_DEFAULTS["comfyui_url"] == "http://127.0.0.1:8188" # Defaults not mutated


@pytest.mark.parametrize("label, key", [
    ("ComfyUI", "comfyui_url"),
    ("Forge / A1111 / SD.Next", "sdwebui_url"),
    ("Ollama (experimental)", "ollama_images_url"),
    ("Unknown backend", "comfyui_url"),
])
def test_backend_url(label, key):
    settings = {"image_generation": {key: f"http://custom-{key}:1"}}
    assert backend_url(settings, label) == f"http://custom-{key}:1"


def test_backend_labels_match_image_gen_package():
    from core.image_gen import BACKENDS
    assert set(igs.BACKEND_URL_KEYS) == set(BACKENDS)


def test_resolve_output_dir_relative_and_absolute(tmp_path):
    assert resolve_output_dir({}) == os.path.join(PROJECT_ROOT, "outputs/images")
    absolute = str(tmp_path / "renders")
    assert resolve_output_dir({"image_generation": {"output_dir": absolute}}) == absolute


@pytest.mark.parametrize("url, ok", [
    ("http://127.0.0.1:8188", True), ("https://host", True), (" http://host:1 ", True),
    ("127.0.0.1:8188", False), ("file:///etc/passwd", False), ("javascript:alert(1)", False), ("", False), (None, False),
])
def test_is_valid_http_url(url, ok):
    assert is_valid_http_url(url) is ok


def _save(**overrides):
    args = dict(previous_status="Settings saved successfully.", ollama_url="http://gpu-box:11434/api/generate", think_mode="auto",
                comfyui_url="http://127.0.0.1:8189", sdwebui_url="http://127.0.0.1:7861", ollama_images_url="http://localhost:11434",
                output_dir="  renders  ", save_prompt_txt=0)
    args.update(overrides)
    return save_connection_settings_callback(*args.values())


def test_save_connection_settings_writes_and_reloads():
    stored = {"theme": "Soft", "ollama_url": "http://old/api/generate"}
    written = {}

    def fake_save(path, data, is_relative=False):
        written.update(data)
        return True
    with patch(LOAD_PATH, side_effect=[stored, {"reloaded": True}]), patch(SAVE_PATH, side_effect=fake_save) as mock_save:
        status, reloaded = _save()
    assert status == "Settings saved successfully.\nConnection settings saved and applied."
    assert reloaded == {"reloaded": True} # Running app gets the file's new content
    assert mock_save.call_args.args[0] == "settings.json" and mock_save.call_args.kwargs == {"is_relative": True}
    assert written["theme"] == "Soft" # Other settings untouched
    assert written["ollama_base_url"] == "http://gpu-box:11434"
    assert written["ollama_url"] == "http://gpu-box:11434" # Legacy key kept in sync
    assert written["ollama_think"] == "auto"
    assert written["image_generation"] == {
        "backend": "ComfyUI", "comfyui_url": "http://127.0.0.1:8189", "sdwebui_url": "http://127.0.0.1:7861",
        "ollama_images_url": "http://localhost:11434", "output_dir": "renders", "save_prompt_txt": False,
    }


def test_save_rejects_non_http_urls():
    with patch(LOAD_PATH, return_value={"keep": 1}), patch(SAVE_PATH) as mock_save:
        status, settings = _save(comfyui_url="file:///etc", sdwebui_url="localhost:7860")
    assert status.startswith("Settings saved successfully.\nNot saved:")
    assert "ComfyUI URL must start with http:// or https:// ('file:///etc')" in status
    assert "Forge/A1111 URL" in status
    assert settings == {"keep": 1}
    mock_save.assert_not_called()


def test_save_invalid_think_mode_and_blank_values_use_defaults():
    written = {}
    with patch(LOAD_PATH, side_effect=[None, {}]), patch(SAVE_PATH, side_effect=lambda p, d, is_relative=False: written.update(d) or True):
        _save(ollama_url="", think_mode="maybe", comfyui_url="", output_dir="")
    assert "ollama_base_url" not in written # Blank Ollama URL leaves the setting alone
    assert written["ollama_think"] == "off"
    assert written["image_generation"]["comfyui_url"] == IMAGE_GEN_DEFAULTS["comfyui_url"]
    assert written["image_generation"]["output_dir"] == IMAGE_GEN_DEFAULTS["output_dir"]


def test_save_write_failure_is_reported():
    with patch(LOAD_PATH, return_value={}), patch(SAVE_PATH, return_value=False):
        status, settings = _save()
    assert status.endswith("Error: could not write settings.json.")
    assert settings["ollama_think"] == "auto" # Returns what it tried to save
