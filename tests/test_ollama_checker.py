# ArtAgent/tests/test_ollama_checker.py

import pytest
import os
import sys
import requests
from unittest.mock import patch, MagicMock

# --- Adjust import path ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from core.ollama_checker import OllamaStatusChecker
    REQUESTS_GET_PATH = 'core.ollama_checker.requests.get'
except ImportError as e:
    pytest.skip(f"Skipping ollama_checker tests, core module not found: {e}", allow_module_level=True)


# --- Test Data ---
VALID_API_URL = "http://localhost:11434/api/generate"
VALID_BASE_URL = "http://localhost:11434"
CUSTOM_API_URL = "http://custom.host:8080/ollama/api"
CUSTOM_BASE_URL = "http://custom.host:8080"
INVALID_API_URL = "not_a_url"
DEFAULT_BASE_URL_FALLBACK = "http://localhost:11434"


# --- Helper ---
def create_mock_response(status_code=200):
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = status_code
    return mock_resp

# --- Tests ---

# Test URL derivation
def test_checker_derive_base_url_valid():
    checker = OllamaStatusChecker(VALID_API_URL)
    assert checker.base_url == VALID_BASE_URL

def test_checker_derive_base_url_custom():
    checker = OllamaStatusChecker(CUSTOM_API_URL)
    assert checker.base_url == CUSTOM_BASE_URL

def test_checker_derive_base_url_invalid():
    checker = OllamaStatusChecker(INVALID_API_URL)
    assert checker.base_url == DEFAULT_BASE_URL_FALLBACK # Check fallback

def test_checker_derive_base_url_no_scheme():
    checker = OllamaStatusChecker("localhost:11434/api/generate")
    assert checker.base_url == DEFAULT_BASE_URL_FALLBACK # Check fallback

# Test check() method outcomes
@patch(REQUESTS_GET_PATH)
def test_checker_check_success(mock_get):
    """Test successful connection to the base URL."""
    mock_get.return_value = create_mock_response(status_code=200)
    checker = OllamaStatusChecker(VALID_API_URL, timeout=1) # Use short timeout for test
    result = checker.check()

    assert result is True
    assert checker.available is True
    assert checker.is_available is True
    assert "Ollama responded" in checker.status_message
    assert f"status: 200" in checker.status_message
    assert VALID_BASE_URL in checker.status_message
    mock_get.assert_called_once_with(VALID_BASE_URL, timeout=1)
    assert checker.get_console_message() is None # No message on success

@patch(REQUESTS_GET_PATH)
def test_checker_check_server_error(mock_get):
    """Test connection success but server returns 5xx error."""
    mock_get.return_value = create_mock_response(status_code=503) # Service Unavailable
    checker = OllamaStatusChecker(VALID_API_URL, timeout=1)
    result = checker.check()

    assert result is False
    assert checker.available is False
    assert checker.is_available is False
    assert "Connected, but Ollama server returned status 503" in checker.status_message
    assert VALID_BASE_URL in checker.status_message
    mock_get.assert_called_once_with(VALID_BASE_URL, timeout=1)
    # Check console message generation
    console_msg = checker.get_console_message()
    assert console_msg is not None
    assert "Ollama Connection Check Failed" in console_msg
    assert "status 503" in console_msg

@patch(REQUESTS_GET_PATH)
def test_checker_check_timeout(mock_get):
    """Test connection timeout."""
    mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")
    checker = OllamaStatusChecker(VALID_API_URL, timeout=1)
    result = checker.check()

    assert result is False
    assert checker.available is False
    assert "Connection to Ollama timed out" in checker.status_message
    assert "(1s)" in checker.status_message
    assert VALID_BASE_URL in checker.status_message
    mock_get.assert_called_once_with(VALID_BASE_URL, timeout=1)
    console_msg = checker.get_console_message()
    assert console_msg is not None
    assert "timed out" in console_msg

@patch(REQUESTS_GET_PATH)
def test_checker_check_connection_error(mock_get):
    """Test connection refused or other connection errors."""
    mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
    checker = OllamaStatusChecker(VALID_API_URL, timeout=1)
    result = checker.check()

    assert result is False
    assert checker.available is False
    assert "Could not connect to Ollama base URL" in checker.status_message
    assert VALID_BASE_URL in checker.status_message
    mock_get.assert_called_once_with(VALID_BASE_URL, timeout=1)
    console_msg = checker.get_console_message()
    assert console_msg is not None
    assert "Could not connect" in console_msg

@patch(REQUESTS_GET_PATH)
def test_checker_check_request_exception(mock_get):
    """Test other generic request exceptions."""
    mock_get.side_effect = requests.exceptions.RequestException("Some weird network issue")
    checker = OllamaStatusChecker(VALID_API_URL, timeout=1)
    result = checker.check()

    assert result is False
    assert checker.available is False
    assert "An unexpected error occurred" in checker.status_message
    assert "Some weird network issue" in checker.status_message
    assert VALID_BASE_URL in checker.status_message
    mock_get.assert_called_once_with(VALID_BASE_URL, timeout=1)
    console_msg = checker.get_console_message()
    assert console_msg is not None
    assert "unexpected error" in console_msg


def test_checker_properties_before_check():
    """Test accessing properties before check() is called."""
    checker = OllamaStatusChecker(VALID_API_URL)
    assert checker.available is False # Default
    assert checker.is_available is False # Property reflects state
    assert "Check not performed yet" in checker.status_message
    # Check console message when check() hasn't run
    console_msg = checker.get_console_message()
    assert console_msg is not None # Should generate message if check hasn't run
    assert "Check not performed yet" in console_msg