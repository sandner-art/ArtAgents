# ArtAgent/tests/test_utils.py

import pytest
import os
import json
import sys
import gradio as gr # Import gradio to test theme loading

# --- Adjust import path ---
# Add project root to the path to allow direct import of core modules
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from core.utils import (
        get_absolute_path,
        load_json,
        save_json,
        get_theme_object,
        AVAILABLE_THEMES, # Need this for testing
        PROJECT_ROOT # Also test the constant if needed
    )
except ImportError as e:
    pytest.skip(f"Skipping utils tests, core module not found: {e}", allow_module_level=True)

# --- Test Data ---
TEST_REL_PATH = "core/test_data_dir/test_file.json"
TEST_ABS_PATH_EXPECTED_FRAGMENT = os.path.join("ArtAgent", "core", "test_data_dir", "test_file.json")
TEST_JSON_DATA = {"key": "value", "list": [1, 2, "three"]}

# --- Tests for get_absolute_path ---

def test_get_absolute_path_basic():
    """Tests basic relative path conversion."""
    abs_path = get_absolute_path("some/relative/path.txt")
    assert os.path.isabs(abs_path)
    # Check if the end of the path matches the expected structure, more robust than exact match
    assert abs_path.endswith(os.path.join("ArtAgent", "some", "relative", "path.txt"))

def test_get_absolute_path_project_root():
    """Tests that the path is relative to the correct project root."""
    expected_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Parent of tests/
    assert PROJECT_ROOT == expected_root, f"PROJECT_ROOT constant mismatch: Expected {expected_root}, Got {PROJECT_ROOT}"
    abs_path = get_absolute_path("README.md")
    assert abs_path == os.path.join(expected_root, "README.md")

# --- Tests for load_json ---

def test_load_json_success(tmp_path):
    """Tests loading a valid JSON file."""
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test.json"
    p.write_text(json.dumps(TEST_JSON_DATA), encoding='utf-8')

    # Test loading with absolute path
    loaded_data = load_json(str(p), is_relative=False)
    assert loaded_data == TEST_JSON_DATA

def test_load_json_relative(tmp_path, monkeypatch):
    """Tests loading a valid JSON file using a relative path."""
    # Create a dummy file structure within the *actual* project structure
    # This is a bit tricky, maybe mocking os.path.join/PROJECT_ROOT is better,
    # but let's try direct creation first.
    core_dir = tmp_path / "core"
    core_dir.mkdir()
    test_data_dir = core_dir / "test_data_dir"
    test_data_dir.mkdir()
    p = test_data_dir / "test_file.json"
    p.write_text(json.dumps(TEST_JSON_DATA), encoding='utf-8')

    # Temporarily change the PROJECT_ROOT perceived by the function
    monkeypatch.setattr('core.utils.PROJECT_ROOT', str(tmp_path))

    # Now load using the relative path used in the main code
    loaded_data = load_json(TEST_REL_PATH, is_relative=True)
    assert loaded_data == TEST_JSON_DATA


def test_load_json_file_not_found():
    """Tests loading a non-existent file."""
    loaded_data = load_json("non_existent_file.json", is_relative=False)
    assert loaded_data == {} # Expect empty dict on file not found

def test_load_json_invalid_json(tmp_path):
    """Tests loading a file with invalid JSON content."""
    p = tmp_path / "invalid.json"
    p.write_text("{'key': 'value',", encoding='utf-8') # Invalid JSON syntax
    loaded_data = load_json(str(p), is_relative=False)
    assert loaded_data == {} # Expect empty dict on JSON decode error

def test_load_json_empty_file(tmp_path):
    """Tests loading an empty file."""
    p = tmp_path / "empty.json"
    p.write_text("", encoding='utf-8')
    loaded_data = load_json(str(p), is_relative=False)
    assert loaded_data == {} # Expect empty dict for empty file

# --- Tests for save_json ---

def test_save_json_success(tmp_path):
    """Tests saving data to a JSON file."""
    p = tmp_path / "output.json"
    result = save_json(str(p), TEST_JSON_DATA, is_relative=False)
    assert result is True
    assert p.exists()
    # Read back and verify content
    read_data = json.loads(p.read_text(encoding='utf-8'))
    assert read_data == TEST_JSON_DATA

def test_save_json_creates_directory(tmp_path):
    """Tests if save_json creates the parent directory."""
    p = tmp_path / "new_dir" / "output.json"
    assert not p.parent.exists() # Ensure dir doesn't exist initially
    result = save_json(str(p), TEST_JSON_DATA, is_relative=False)
    assert result is True
    assert p.parent.exists()
    assert p.exists()

def test_save_json_relative(tmp_path, monkeypatch):
    """Tests saving data using a relative path."""
    p_expected = tmp_path / "core" / "test_data_dir" / "test_file.json"
    # Temporarily change the PROJECT_ROOT
    monkeypatch.setattr('core.utils.PROJECT_ROOT', str(tmp_path))

    result = save_json(TEST_REL_PATH, TEST_JSON_DATA, is_relative=True)
    assert result is True
    assert p_expected.exists()
    assert p_expected.parent.exists()
    # Verify content
    read_data = json.loads(p_expected.read_text(encoding='utf-8'))
    assert read_data == TEST_JSON_DATA

# --- Tests for get_theme_object ---

def test_get_theme_object_valid():
    """Tests retrieving a valid theme object."""
    default_theme_obj = get_theme_object("Default")
    assert isinstance(default_theme_obj, gr.themes.Default)

    # Check another theme if available and imported correctly
    if "Soft" in AVAILABLE_THEMES:
         soft_theme_obj = get_theme_object("Soft")
         assert isinstance(soft_theme_obj, gr.themes.Soft)

def test_get_theme_object_invalid():
    """Tests retrieving an invalid theme name, expecting default."""
    invalid_theme_obj = get_theme_object("NonExistentThemeName")
    assert isinstance(invalid_theme_obj, gr.themes.Default) # Should return default

def test_available_themes_structure():
    """Checks if AVAILABLE_THEMES is a dict and values are theme objects."""
    assert isinstance(AVAILABLE_THEMES, dict)
    for name, theme_obj in AVAILABLE_THEMES.items():
         assert isinstance(name, str)
         # Check if it's a Gradio theme base class or specific instance
         assert isinstance(theme_obj, (gr.themes.base.ThemeClass, gr.themes.Default))