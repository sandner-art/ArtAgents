# ArtAgent/tests/test_ollama_manager.py

import pytest
import os
import sys
import requests
import json
from unittest.mock import patch, MagicMock, call # Import call for checking multiple calls

# --- Adjust import path ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from core.ollama_manager import (
        release_model,
        release_all_models_logic,
        MODELS_FILE # Import the constant
    )
    # Mocks for dependencies
    REQUESTS_POST_PATH = 'core.ollama_manager.requests.post'
    LOAD_JSON_PATH = 'core.ollama_manager.load_json'
    RELEASE_MODEL_FUNC_PATH = 'core.ollama_manager.release_model' # Path to the function within its own module
except ImportError as e:
    pytest.skip(f"Skipping ollama_manager tests, modules not found: {e}", allow_module_level=True)


# --- Test Data ---
MODEL_NAME = "test-model:latest"
OLLAMA_URL = "http://fake-manager-test:11434/api/generate"
SETTINGS_WITH_URL = {"ollama_url": OLLAMA_URL}
SETTINGS_NO_URL = {}
MODELS_DATA = [
    {"name": "model1:7b", "vision": False},
    {"name": "model2-vision:latest", "vision": True},
    {"name": "model3", "vision": False} # Model without tag
]
EMPTY_MODELS_DATA = []
INVALID_MODELS_DATA = {"not": "a list"} # Incorrect format

# --- Helper ---
def create_mock_response(status_code=200, text=""):
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = status_code
    mock_resp.text = text
    # Mock raise_for_status conditionally
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status_code} Error", response=mock_resp)
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp

# --- Tests for release_model ---

@patch(REQUESTS_POST_PATH)
def test_release_model_success(mock_post):
    """Test successful model release."""
    mock_post.return_value = create_mock_response(200)
    result = release_model(MODEL_NAME, OLLAMA_URL)

    expected_payload = {"model": MODEL_NAME, "keep_alive": 0}
    mock_post.assert_called_once_with(OLLAMA_URL, json=expected_payload, timeout=20)
    assert f"Model '{MODEL_NAME}' release request sent successfully" in result

def test_release_model_missing_args():
    """Test skipping release if model name or URL is missing."""
    result_no_name = release_model("", OLLAMA_URL)
    assert "Skipping release: Missing model name" in result_no_name
    result_no_url = release_model(MODEL_NAME, "")
    assert "Skipping release: Missing URL" in result_no_url
    result_both_none = release_model(None, None)
    assert "Skipping release: Missing model name" in result_both_none # Checks name first

@patch(REQUESTS_POST_PATH)
def test_release_model_timeout(mock_post):
    """Test handling of request timeout during release."""
    mock_post.side_effect = requests.exceptions.Timeout("Release timed out")
    result = release_model(MODEL_NAME, OLLAMA_URL)
    assert f"Error releasing model '{MODEL_NAME}': Request timed out" in result

@patch(REQUESTS_POST_PATH)
def test_release_model_http_error(mock_post):
    """Test handling of HTTP errors during release."""
    mock_resp = create_mock_response(404, text="Model not found")
    mock_post.return_value = mock_resp
    result = release_model(MODEL_NAME, OLLAMA_URL)
    assert f"Error releasing model '{MODEL_NAME}'" in result
    assert "Status: 404" in result
    assert "Body: Model not found" in result

@patch(REQUESTS_POST_PATH)
def test_release_model_request_exception(mock_post):
    """Test handling of other request exceptions during release."""
    mock_post.side_effect = requests.exceptions.RequestException("Network error")
    result = release_model(MODEL_NAME, OLLAMA_URL)
    assert f"Error releasing model '{MODEL_NAME}': Network error" in result

# --- Tests for release_all_models_logic ---

# Mock load_json to control models data and mock the release_model function itself
@patch(RELEASE_MODEL_FUNC_PATH)
@patch(LOAD_JSON_PATH)
def test_release_all_models_success(mock_load, mock_release):
    """Test successfully triggering release for all models in models.json."""
    mock_load.return_value = MODELS_DATA # Simulate successful load
    # Simulate successful release for each model
    mock_release.side_effect = [
        f"Model '{m['name']}' release request sent successfully." for m in MODELS_DATA
    ]

    summary = release_all_models_logic(SETTINGS_WITH_URL)

    # Check load_json call
    mock_load.assert_called_once_with(MODELS_FILE, is_relative=True)

    # Check release_model calls
    assert mock_release.call_count == len(MODELS_DATA)
    expected_calls = [
        call(MODELS_DATA[0]['name'], OLLAMA_URL),
        call(MODELS_DATA[1]['name'], OLLAMA_URL),
        call(MODELS_DATA[2]['name'], OLLAMA_URL),
    ]
    mock_release.assert_has_calls(expected_calls)

    # Check summary message
    assert f"Model '{MODELS_DATA[0]['name']}' release request sent successfully." in summary
    assert f"Model '{MODELS_DATA[1]['name']}' release request sent successfully." in summary
    assert f"Model '{MODELS_DATA[2]['name']}' release request sent successfully." in summary


@patch(RELEASE_MODEL_FUNC_PATH)
@patch(LOAD_JSON_PATH)
def test_release_all_models_some_errors(mock_load, mock_release):
    """Test handling when some release calls succeed and others fail."""
    mock_load.return_value = MODELS_DATA
    # Simulate success for model1, timeout for model2, success for model3
    mock_release.side_effect = [
        f"Model '{MODELS_DATA[0]['name']}' release request sent successfully.",
        f"Error releasing model '{MODELS_DATA[1]['name']}': Request timed out.",
        f"Model '{MODELS_DATA[2]['name']}' release request sent successfully."
    ]

    summary = release_all_models_logic(SETTINGS_WITH_URL)

    assert mock_release.call_count == len(MODELS_DATA)
    assert f"Model '{MODELS_DATA[0]['name']}' release request sent successfully." in summary
    assert f"Error releasing model '{MODELS_DATA[1]['name']}': Request timed out." in summary
    assert f"Model '{MODELS_DATA[2]['name']}' release request sent successfully." in summary

@patch(RELEASE_MODEL_FUNC_PATH)
@patch(LOAD_JSON_PATH)
def test_release_all_models_no_url_in_settings(mock_load, mock_release):
    """Test behavior when Ollama URL is missing in settings."""
    summary = release_all_models_logic(SETTINGS_NO_URL) # Pass settings without URL

    assert "Cannot release models: Ollama URL not found" in summary
    mock_load.assert_not_called() # Should exit before loading models
    mock_release.assert_not_called() # Should exit before attempting release


@patch(RELEASE_MODEL_FUNC_PATH)
@patch(LOAD_JSON_PATH)
def test_release_all_models_file_not_found(mock_load, mock_release):
    """Test behavior when models.json is not found (load_json returns {})."""
    mock_load.return_value = {} # Simulate file not found or empty
    summary = release_all_models_logic(SETTINGS_WITH_URL)

    assert "No models found or invalid format in models.json" in summary
    mock_load.assert_called_once_with(MODELS_FILE, is_relative=True)
    mock_release.assert_not_called()

@patch(RELEASE_MODEL_FUNC_PATH)
@patch(LOAD_JSON_PATH)
def test_release_all_models_invalid_format(mock_load, mock_release):
    """Test behavior when models.json has invalid format (not a list)."""
    mock_load.return_value = INVALID_MODELS_DATA # Simulate load returning a dict
    summary = release_all_models_logic(SETTINGS_WITH_URL)

    assert "No models found or invalid format in models.json" in summary
    mock_load.assert_called_once_with(MODELS_FILE, is_relative=True)
    mock_release.assert_not_called()

@patch(RELEASE_MODEL_FUNC_PATH)
@patch(LOAD_JSON_PATH)
def test_release_all_models_empty_list(mock_load, mock_release):
    """Test behavior when models.json contains an empty list."""
    mock_load.return_value = EMPTY_MODELS_DATA # Simulate load returning []
    summary = release_all_models_logic(SETTINGS_WITH_URL)

    # The code currently proceeds but finds no models to release
    assert "No valid model names found to release" in summary # Adjusted expectation
    mock_load.assert_called_once_with(MODELS_FILE, is_relative=True)
    mock_release.assert_not_called()

@patch(RELEASE_MODEL_FUNC_PATH)
@patch(LOAD_JSON_PATH)
def test_release_all_models_missing_name_key(mock_load, mock_release):
    """Test skipping models in the list that are missing the 'name' key."""
    models_missing_name = [
        {"vision": False}, # Missing name
        {"name": "model2:ok", "vision": True}
    ]
    mock_load.return_value = models_missing_name
    mock_release.return_value = "Model 'model2:ok' release request sent successfully." # Only one success expected

    summary = release_all_models_logic(SETTINGS_WITH_URL)

    assert mock_release.call_count == 1 # Only called for the valid model
    mock_release.assert_called_once_with("model2:ok", OLLAMA_URL)
    assert "Model 'model2:ok' release request sent successfully." in summary