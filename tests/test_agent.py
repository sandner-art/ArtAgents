# ArtAgent/tests/test_agent.py

import pytest
import requests
import json
import base64
import io
import sys
import os
from unittest.mock import patch, MagicMock, ANY # ANY helps match complex args like streams

# --- Adjust import path ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from agents.ollama_agent import get_llm_response
    # Mock PIL Image for image tests without requiring Pillow installed during testing
    # (or handle the import error gracefully if Pillow is expected)
    try:
        from PIL import Image
    except ImportError:
        # Create a dummy Image class if PIL is not available in test env
        class DummyImage:
            def __init__(self, mode="RGB", size=(10, 10)):
                self.mode = mode
                self.size = size
            def save(self, fp, format=None):
                 fp.write(b"dummy image data") # Write some bytes
            def close(self):
                 pass
        Image = DummyImage # Use the dummy class
        print("Warning: Pillow not found, using dummy Image class for tests.")

    REQUESTS_POST_PATH = 'agents.ollama_agent.requests.post'
    BASE64_B64ENCODE_PATH = 'agents.ollama_agent.base64.b64encode'
    IO_BYTESIO_PATH = 'agents.ollama_agent.io.BytesIO'
    PIL_IMAGE_PATH = 'agents.ollama_agent.Image' # Path to Image used within the module

except ImportError as e:
    pytest.skip(f"Skipping agent tests, modules not found: {e}", allow_module_level=True)
    # Define dummy function for test collection if main import fails
    def get_llm_response(*args, **kwargs): raise NotImplementedError("Agent function not found/imported.")
    REQUESTS_POST_PATH = 'builtins.print' # Dummy path
    BASE64_B64ENCODE_PATH = 'builtins.print'
    IO_BYTESIO_PATH = 'builtins.print'
    PIL_IMAGE_PATH = 'builtins.print'


# --- Test Data ---
# Mock settings and roles to be passed *into* get_llm_response
MOCK_SETTINGS = {
    "ollama_url": "http://fake-test-ollama:11434/api/generate",
    "ollama_api_options": {"seed": 1, "temperature": 0.5, "num_ctx": 2048}, # Added num_ctx
    "ollama_api_prompt_to_console": False # Disable logging during most tests
}

MOCK_ROLES_DATA = {
    "Tester": {
        "description": "A test role.",
        "ollama_api_options": {"temperature": 0.8, "top_k": 10} # Role overrides temp
    },
    "NoOptionsRole": {
        "description": "Another role."
    },
    "PredictRole": {
        "description": "Role defining num_predict.",
        "ollama_api_options": {"num_predict": 777}
    }
}

DEFAULT_ARGS = {
    "role": "Tester",
    "prompt": "Test prompt",
    "model": "test-model",
    "settings": MOCK_SETTINGS,
    "roles_data": MOCK_ROLES_DATA,
    "images": None,
    "max_tokens": 1500, # Fallback for num_predict only if not set elsewhere
    "ollama_api_options": None,
}


# --- Helper to create mock streaming response ---
def mock_streaming_response(chunks, status_code=200, raise_for_status=None):
    """Creates a mock requests.Response object for streaming."""
    mock_resp = MagicMock(spec=requests.Response) # Use spec for better mocking
    mock_resp.status_code = status_code
    # Simulate iter_lines yielding decoded strings
    mock_resp.iter_lines.return_value = (c for c in chunks)
    mock_resp.raise_for_status = MagicMock() # Create mock method
    if raise_for_status:
        mock_resp.raise_for_status.side_effect = raise_for_status
    return mock_resp

# --- Tests ---

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_success_text_only(mock_post):
    """ Test successful text-only response with streaming. """
    stream_chunks = [
        json.dumps({"response": "Hello ", "done": False}),
        json.dumps({"response": "Ollama!", "done": False}),
        json.dumps({"done": True, "context": [1,2,3]}), # Add final context
    ]
    mock_post.return_value = mock_streaming_response(stream_chunks)

    result = get_llm_response(**DEFAULT_ARGS)

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
    expected_options = {"seed": 1, "temperature": 0.8, "top_k": 10, "num_ctx": 2048, "num_predict": 1500}
    assert payload.get('options') == expected_options

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_num_predict_priority(mock_post):
    """ Test num_predict priority: Direct > Role > Global > max_tokens fallback """
    stream_chunks = [json.dumps({"response": "Test", "done": False}), json.dumps({"done": True})]
    mock_post.return_value = mock_streaming_response(stream_chunks)

    # 1. Test fallback to max_tokens (role has no num_predict)
    args1 = {**DEFAULT_ARGS, "role": "Tester", "max_tokens": 123}
    get_llm_response(**args1)
    payload1 = mock_post.call_args.kwargs.get('json', {})
    assert payload1.get('options', {}).get('num_predict') == 123

    # 2. Test role override
    mock_post.reset_mock()
    args2 = {**DEFAULT_ARGS, "role": "PredictRole", "max_tokens": 456} # Role defines 777
    get_llm_response(**args2)
    payload2 = mock_post.call_args.kwargs.get('json', {})
    assert payload2.get('options', {}).get('num_predict') == 777

    # 3. Test direct call override
    mock_post.reset_mock()
    direct_opts = {"num_predict": 999}
    args3 = {**DEFAULT_ARGS, "role": "PredictRole", "max_tokens": 456, "ollama_api_options": direct_opts}
    get_llm_response(**args3)
    payload3 = mock_post.call_args.kwargs.get('json', {})
    assert payload3.get('options', {}).get('num_predict') == 999


@patch(REQUESTS_POST_PATH)
def test_get_llm_response_options_override_priority(mock_post):
    """ Test general options merging priority: Direct > Role > Global settings. """
    stream_chunks = [json.dumps({"response": "Test", "done": False}), json.dumps({"done": True})]
    mock_post.return_value = mock_streaming_response(stream_chunks)

    direct_options = {"temperature": 0.99, "seed": 42, "top_p": 0.88}
    # Use "Tester" role (defines temp=0.8, top_k=10)
    # Global settings define seed=1, temp=0.5, num_ctx=2048
    args = {**DEFAULT_ARGS, "role": "Tester", "ollama_api_options": direct_options}

    get_llm_response(**args)

    mock_post.assert_called_once()
    _, call_kwargs = mock_post.call_args
    payload = call_kwargs.get('json', {})
    expected_options = {
        "seed": 42,          # Direct override
        "temperature": 0.99, # Direct override
        "top_k": 10,         # Role option (not overridden)
        "num_ctx": 2048,     # Global option (not overridden)
        "top_p": 0.88,       # Direct option added
        "num_predict": 1500  # Fallback from max_tokens
    }
    assert payload.get('options') == expected_options


@patch(REQUESTS_POST_PATH)
def test_get_llm_response_connection_error(mock_post):
    """ Test handling of requests.exceptions.ConnectionError. """
    mock_post.side_effect = requests.exceptions.ConnectionError("Test connection error")
    result = get_llm_response(**DEFAULT_ARGS)
    assert "⚠️ Error: Could not connect to Ollama" in result
    assert MOCK_SETTINGS["ollama_url"] in result

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_timeout(mock_post):
    """ Test handling of requests.exceptions.Timeout. """
    mock_post.side_effect = requests.exceptions.Timeout("Test timeout")
    result = get_llm_response(**DEFAULT_ARGS)
    assert "⚠️ Error: Request to Ollama timed out" in result

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_http_error(mock_post):
    """ Test handling of HTTP errors via raise_for_status. """
    mock_response = MagicMock(spec=requests.Response)
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error Detail"
    http_error = requests.exceptions.HTTPError("500 Server Error")
    http_error.response = mock_response # Attach response to error
    mock_response.raise_for_status.side_effect = http_error
    mock_post.return_value = mock_response

    result = get_llm_response(**DEFAULT_ARGS)
    assert "⚠️ Error communicating with Ollama" in result
    assert "Status: 500" in result
    assert "Response Text: Internal Server Error Detail" in result

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_generic_request_exception(mock_post):
    """ Test handling of other requests.exceptions.RequestException. """
    mock_post.side_effect = requests.exceptions.RequestException("Some other request error")
    result = get_llm_response(**DEFAULT_ARGS)
    assert "⚠️ Error communicating with Ollama: Some other request error" in result


@patch(REQUESTS_POST_PATH)
def test_get_llm_response_json_decode_error_in_stream(mock_post):
    """ Test handling of invalid JSON during streaming. """
    stream_chunks = [
        json.dumps({"response": "Part 1.", "done": False}),
        '{"invalid json chunk', # Invalid JSON
        json.dumps({"response": "Part 3 (ignored)", "done": False}),
        json.dumps({"done": True}),
    ]
    mock_post.return_value = mock_streaming_response(stream_chunks)

    result = get_llm_response(**DEFAULT_ARGS)
    assert result.startswith("Part 1.")
    # Check for the specific error message appended by the handler
    assert "[Error decoding stream chunk: Expecting value: line 1 column 2 (char 1)]" in result
    assert "Part 3" not in result # Ensure processing stopped


@patch(REQUESTS_POST_PATH)
@patch(BASE64_B64ENCODE_PATH)
@patch(IO_BYTESIO_PATH)
@patch(PIL_IMAGE_PATH, new_callable=MagicMock) # Mock the Image class itself
def test_get_llm_response_with_image(mock_pil_image_class, mock_bytesio, mock_b64encode, mock_post):
    """ Test successful call with a single image. """
    # Configure mocks
    stream_chunks = [json.dumps({"response": "Image analyzed.", "done": True})]
    mock_post.return_value = mock_streaming_response(stream_chunks)

    mock_image_instance = MagicMock(spec=Image) # Mock instance returned by Image.open/fromarray
    mock_image_instance.mode = "RGB"
    # We don't need to mock Image.open if we pass the mock object directly

    mock_buffer = MagicMock(spec=io.BytesIO)
    mock_bytesio.return_value = mock_buffer
    mock_buffer.getvalue.return_value = b"imagedata"

    mock_b64encode.return_value = b"encodedimagedata"

    # Prepare arguments with image
    args = {**DEFAULT_ARGS, "images": [mock_image_instance]} # Pass mock image object

    # Call function
    result = get_llm_response(**args)

    # Assertions
    assert result == "Image analyzed."
    mock_image_instance.save.assert_called_once_with(mock_buffer, format="JPEG") # Check format used
    mock_b64encode.assert_called_once_with(b"imagedata")
    mock_post.assert_called_once()
    payload = mock_post.call_args.kwargs.get('json', {})
    assert "images" in payload
    assert payload["images"] == ["encodedimagedata"] # base64 returns bytes, decode expected

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_with_invalid_image_object(mock_post, capsys):
    """ Test passing something other than a PIL Image in the images list. """
    stream_chunks = [json.dumps({"response": "Text response only.", "done": True})]
    mock_post.return_value = mock_streaming_response(stream_chunks)

    args = {**DEFAULT_ARGS, "images": ["not_an_image_object"]}

    result = get_llm_response(**args)

    assert result == "Text response only."
    # Check payload doesn't contain images key
    payload = mock_post.call_args.kwargs.get('json', {})
    assert "images" not in payload
    # Check for warning print
    captured = capsys.readouterr()
    assert "Warning: Item 1 in images list is not a PIL Image object" in captured.out

@patch(REQUESTS_POST_PATH)
@patch(PIL_IMAGE_PATH, new_callable=MagicMock)
def test_get_llm_response_image_processing_error(mock_pil_image_class, mock_post):
    """ Test handling of exceptions during image processing (e.g., save fails). """
    mock_image_instance = MagicMock(spec=Image)
    mock_image_instance.mode = "RGB"
    mock_image_instance.save.side_effect = Exception("PIL Save Error")

    args = {**DEFAULT_ARGS, "images": [mock_image_instance]}

    result = get_llm_response(**args)

    assert result.startswith("⚠️ Error: Failed to process image data.")
    assert "PIL Save Error" in result
    mock_post.assert_not_called() # API call should not happen if image processing fails


@patch(REQUESTS_POST_PATH)
def test_get_llm_response_empty_stream(mock_post):
    """Test response when the stream returns done immediately or is empty."""
    # Scenario 1: Only done=true
    stream1 = [json.dumps({"done": True})]
    mock_post.return_value = mock_streaming_response(stream1)
    result1 = get_llm_response(**DEFAULT_ARGS)
    assert "No response text received from LLM stream" in result1

    # Scenario 2: Empty stream iterable
    mock_post.reset_mock()
    stream2 = []
    mock_post.return_value = mock_streaming_response(stream2)
    result2 = get_llm_response(**DEFAULT_ARGS)
    assert "No response text received from LLM stream" in result2

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_console_logging(mock_post, capsys):
    """Test if request details are logged to console when enabled."""
    stream_chunks = [json.dumps({"response": "Logged", "done": True})]
    mock_post.return_value = mock_streaming_response(stream_chunks)

    # Enable logging in settings for this test
    settings_log_on = {**MOCK_SETTINGS, "ollama_api_prompt_to_console": True}
    args = {**DEFAULT_ARGS, "settings": settings_log_on, "prompt": "Short prompt"}

    get_llm_response(**args)

    captured = capsys.readouterr()
    log_output = captured.out
    assert "--- Ollama Request ---" in log_output
    assert '"model": "test-model"' in log_output
    assert '"prompt_start": "Short prompt"' in log_output # Check short prompt logging
    assert '"images_count": 0' in log_output
    assert '"effective_options": {' in log_output
    assert '"temperature": 0.8' in log_output # Check merged option
    assert "----------------------" in log_output