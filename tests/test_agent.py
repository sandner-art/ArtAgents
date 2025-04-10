# ArtAgent/tests/test_agent.py

import pytest
import requests
import json
from unittest.mock import MagicMock, patch

# --- Adjust import paths based on the final structure ---
# Assuming agent logic is now in agents.ollama_agent
try:
    from agents.ollama_agent import get_llm_response
    # Define paths for patching based on the new structure
    REQUESTS_POST_PATH = 'agents.ollama_agent.requests.post'
    # If get_llm_response directly uses settings/roles passed in,
    # we don't need to patch loading functions within the agent module itself.
except ImportError as e:
    print(f"Import Error in test_agent.py: {e}. Ensure agents/ollama_agent.py exists.")
    # Define dummy function for test collection
    def get_llm_response(*args, **kwargs): raise NotImplementedError("Agent function not found/imported.")
    REQUESTS_POST_PATH = 'builtins.print' # Dummy path


# --- Test Data ---
# Mock settings and roles to be passed *into* get_llm_response
MOCK_SETTINGS = {
    "ollama_url": "http://fake-test-ollama:11434/api/generate",
    "ollama_api_options": {"seed": 1, "temperature": 0.5, "num_predict": 500}
}

MOCK_ROLES_DATA = {
    "Tester": {
        "description": "A test role.",
        "ollama_api_options": {"temperature": 0.8, "top_k": 10} # Role overrides temp
    },
    "NoOptionsRole": {
        "description": "Another role."
    }
}

DEFAULT_ARGS = {
    "role": "Tester",
    "prompt": "Test prompt",
    "model": "test-model",
    "settings": MOCK_SETTINGS,  # Pass mock settings
    "roles_data": MOCK_ROLES_DATA, # Pass mock roles
    "images": None,
    "max_tokens": 100, # Fallback for num_predict
    "ollama_api_options": None, # No direct overrides in default test case
}


# --- Helper to create mock streaming response ---
def mock_streaming_response(chunks, status_code=200, raise_for_status=None):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.iter_lines.return_value = (c.encode('utf-8') for c in chunks)
    mock_resp.raise_for_status.side_effect = raise_for_status if raise_for_status else None
    return mock_resp

# --- Tests ---

# Use pytest-mock's 'mocker' fixture implicitly where needed (via patch decorator)
@patch(REQUESTS_POST_PATH) # Patch requests.post within the agent module
def test_get_llm_response_success_text_only(mock_post):
    """ Test successful text-only response with streaming. """
    # Define stream chunks
    stream_chunks = [
        json.dumps({"response": "Hello ", "done": False}),
        json.dumps({"response": "Ollama!", "done": False}),
        json.dumps({"done": True}),
    ]
    mock_post.return_value = mock_streaming_response(stream_chunks)

    # Call the function with default args
    result = get_llm_response(**DEFAULT_ARGS)

    # Assertions
    assert result == "Hello Ollama!"
    mock_post.assert_called_once()
    call_args, call_kwargs = mock_post.call_args
    assert call_args[0] == MOCK_SETTINGS["ollama_url"]
    payload = call_kwargs.get('json', {})
    assert payload.get('model') == DEFAULT_ARGS['model']
    assert payload.get('prompt') == DEFAULT_ARGS['prompt']
    assert payload.get('stream') is True
    assert "images" not in payload
    # Check merged options (role temp 0.8 overrides global 0.5)
    assert payload.get('options', {}).get('temperature') == 0.8
    assert payload.get('options', {}).get('seed') == 1
    assert payload.get('options', {}).get('top_k') == 10
    # Check num_predict (uses max_tokens as fallback if not in global/role/direct)
    assert payload.get('options', {}).get('num_predict') == DEFAULT_ARGS['max_tokens']

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_options_override(mock_post):
    """ Test options merging: direct call > role > global settings. """
    stream_chunks = [json.dumps({"response": "Test", "done": False}), json.dumps({"done": True})]
    mock_post.return_value = mock_streaming_response(stream_chunks)

    # Pass direct ollama_api_options and different role
    direct_options = {"temperature": 0.99, "seed": 42, "num_predict": 999}
    # Override role and add direct options to default args
    args = {**DEFAULT_ARGS, "role": "Tester", "ollama_api_options": direct_options}

    get_llm_response(**args) # Call with modified args

    mock_post.assert_called_once()
    _, call_kwargs = mock_post.call_args
    payload = call_kwargs.get('json', {})
    # Direct options should override Role and Global
    assert payload.get('options', {}).get('temperature') == 0.99 # Direct override
    assert payload.get('options', {}).get('seed') == 42          # Direct override
    assert payload.get('options', {}).get('top_k') == 10          # Role option still present
    assert payload.get('options', {}).get('num_predict') == 999   # Direct override


@patch(REQUESTS_POST_PATH)
def test_get_llm_response_connection_error(mock_post):
    """ Test handling of requests.exceptions.ConnectionError. """
    mock_post.side_effect = requests.exceptions.ConnectionError("Test connection error")

    result = get_llm_response(**DEFAULT_ARGS)

    assert "⚠️ Error: Could not connect to Ollama" in result
    assert MOCK_SETTINGS["ollama_url"] in result # Check if URL is mentioned

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_timeout(mock_post):
    """ Test handling of requests.exceptions.Timeout. """
    mock_post.side_effect = requests.exceptions.Timeout("Test timeout")

    result = get_llm_response(**DEFAULT_ARGS)

    assert "⚠️ Error: Request to Ollama timed out" in result

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_http_error(mock_post):
    """ Test handling of HTTP errors via raise_for_status. """
    mock_resp = mock_streaming_response(
        [], status_code=500, raise_for_status=requests.exceptions.HTTPError("Server Error")
    )
    mock_post.return_value = mock_resp # Set return value before setting side effect on method
    # We mock raise_for_status specifically on the returned mock object
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")


    result = get_llm_response(**DEFAULT_ARGS)

    assert "⚠️ Error communicating with Ollama" in result
    assert "500 Server Error" in result # Check if original exception text is included

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_json_decode_error(mock_post):
    """ Test handling of invalid JSON during streaming. """
    # Chunks: valid, invalid, final (should not be processed)
    stream_chunks = [
        json.dumps({"response": "Part 1.", "done": False}),
        '{"response": "Part 2."...', # Invalid JSON
        json.dumps({"done": True}),
    ]
    mock_post.return_value = mock_streaming_response(stream_chunks)

    result = get_llm_response(**DEFAULT_ARGS)

    assert result.startswith("Part 1.")
    assert "[Error decoding stream chunk: Expecting" in result # Check for JSONDecodeError message

# --- TODO: Add image processing tests ---
# These might require mocking PIL, io.BytesIO, base64, etc. or using small test images.