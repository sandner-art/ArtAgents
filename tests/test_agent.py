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
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir)
sys.path.insert(0, project_root)

try:
    from agents.ollama_agent import get_llm_response
    try:
        # Import the actual Image class for type checking if available
        from PIL import Image as PILImageModule
    except ImportError:
        # Create a dummy Image class if PIL is not available in test env
        class DummyImage:
            Image = None # Placeholder for Image.Image check below
            def __init__(self, mode="RGB", size=(10, 10)): self.mode = mode; self.size = size
            def save(self, fp, format=None): fp.write(b"dummy image data")
            def close(self): pass
        # Set the 'Image' attribute on the class itself
        DummyImage.Image = DummyImage
        PILImageModule = DummyImage # Use the dummy class
        print("Warning: Pillow not found, using dummy Image class for tests.")

    REQUESTS_POST_PATH = 'agents.ollama_agent.requests.post'
    BASE64_B64ENCODE_PATH = 'agents.ollama_agent.base64.b64encode'
    IO_BYTESIO_PATH = 'agents.ollama_agent.io.BytesIO' # Patch the class
    # Path to the *module* or *class* as it's imported/used in ollama_agent.py
    # Assuming ollama_agent.py does 'from PIL import Image'
    PIL_IMAGE_MODULE_PATH = 'agents.ollama_agent.Image'
    # Path to the specific Image class used in the `isinstance` check
    PIL_IMAGE_CLASS_PATH = 'agents.ollama_agent.Image.Image'


except ImportError as e:
    pytest.skip(f"Skipping agent tests, modules not found: {e}", allow_module_level=True)
    def get_llm_response(*args, **kwargs): raise NotImplementedError("Agent function not found/imported.")
    REQUESTS_POST_PATH = 'builtins.print'; BASE64_B64ENCODE_PATH = 'builtins.print'; IO_BYTESIO_PATH = 'builtins.print'; PIL_IMAGE_MODULE_PATH = 'builtins.print'; PIL_IMAGE_CLASS_PATH = 'builtins.print'


# --- Test Data ---
MOCK_SETTINGS = {
    "ollama_url": "http://fake-test-ollama:11434/api/generate",
    "ollama_api_options": {"seed": 1, "temperature": 0.5, "num_ctx": 2048},
    "ollama_api_prompt_to_console": False
}
MOCK_ROLES_DATA = {
    "Tester": {"description": "A test role.", "ollama_api_options": {"temperature": 0.8, "top_k": 10}},
    "NoOptionsRole": {"description": "Another role."},
    "PredictRole": {"description": "Role defining num_predict.", "ollama_api_options": {"num_predict": 777}}
}
DEFAULT_ARGS = {
    "role": "Tester", "prompt": "Test prompt", "model": "test-model",
    "settings": MOCK_SETTINGS, "roles_data": MOCK_ROLES_DATA, "images": None,
    "max_tokens": 1500, "ollama_api_options": None,
}

# --- Helper ---
def mock_streaming_response(chunks, status_code=200, raise_for_status=None):
    """Creates a mock requests.Response object for streaming."""
    mock_resp = MagicMock(spec=requests.Response)
    mock_resp.status_code = status_code
    # Simulate iter_lines yielding decoded strings
    mock_resp.iter_lines.return_value = (c for c in chunks)
    mock_resp.raise_for_status = MagicMock()
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
        json.dumps({"done": True, "context": [1,2,3]}),
    ]
    mock_post.return_value = mock_streaming_response(stream_chunks)
    result = get_llm_response(**DEFAULT_ARGS)
    assert result == "Hello Ollama!"
    mock_post.assert_called_once()
    payload = mock_post.call_args.kwargs.get('json', {})
    expected_options = {"seed": 1, "temperature": 0.8, "top_k": 10, "num_ctx": 2048, "num_predict": 1500}
    assert payload.get('options') == expected_options

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_num_predict_priority(mock_post):
    """ Test num_predict priority: Direct > Role > Global > max_tokens fallback """
    stream_chunks = [json.dumps({"response": "Test", "done": False}), json.dumps({"done": True})]
    mock_post.return_value = mock_streaming_response(stream_chunks)
    args1 = {**DEFAULT_ARGS, "role": "Tester", "max_tokens": 123}; get_llm_response(**args1); payload1 = mock_post.call_args.kwargs.get('json', {}); assert payload1.get('options', {}).get('num_predict') == 123
    mock_post.reset_mock(); args2 = {**DEFAULT_ARGS, "role": "PredictRole", "max_tokens": 456}; get_llm_response(**args2); payload2 = mock_post.call_args.kwargs.get('json', {}); assert payload2.get('options', {}).get('num_predict') == 777
    mock_post.reset_mock(); direct_opts = {"num_predict": 999}; args3 = {**DEFAULT_ARGS, "role": "PredictRole", "max_tokens": 456, "ollama_api_options": direct_opts}; get_llm_response(**args3); payload3 = mock_post.call_args.kwargs.get('json', {}); assert payload3.get('options', {}).get('num_predict') == 999

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_options_override_priority(mock_post):
    """ Test general options merging priority: Direct > Role > Global settings. """
    stream_chunks = [json.dumps({"response": "Test", "done": False}), json.dumps({"done": True})]
    mock_post.return_value = mock_streaming_response(stream_chunks)
    direct_options = {"temperature": 0.99, "seed": 42, "top_p": 0.88}
    args = {**DEFAULT_ARGS, "role": "Tester", "ollama_api_options": direct_options}
    get_llm_response(**args)
    payload = mock_post.call_args.kwargs.get('json', {})
    expected_options = {"seed": 42, "temperature": 0.99, "top_k": 10, "num_ctx": 2048, "top_p": 0.88, "num_predict": 1500}
    assert payload.get('options') == expected_options

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_connection_error(mock_post):
    """ Test handling of requests.exceptions.ConnectionError. """
    mock_post.side_effect = requests.exceptions.ConnectionError("Test connection error"); result = get_llm_response(**DEFAULT_ARGS)
    assert "⚠️ Error: Could not connect to Ollama" in result; assert MOCK_SETTINGS["ollama_url"] in result

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_timeout(mock_post):
    """ Test handling of requests.exceptions.Timeout. """
    mock_post.side_effect = requests.exceptions.Timeout("Test timeout"); result = get_llm_response(**DEFAULT_ARGS)
    assert "⚠️ Error: Request to Ollama timed out" in result

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_http_error(mock_post):
    """ Test handling of HTTP errors via raise_for_status. """
    mock_response = MagicMock(spec=requests.Response); mock_response.status_code = 500; mock_response.text = "Internal Server Error Detail"
    http_error = requests.exceptions.HTTPError("500 Server Error"); http_error.response = mock_response; mock_response.raise_for_status = MagicMock(side_effect=http_error)
    mock_post.return_value = mock_response
    result = get_llm_response(**DEFAULT_ARGS)
    assert "⚠️ Error communicating with Ollama" in result; assert "Status: 500" in result; assert "Response Text: Internal Server Error Detail" in result

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_generic_request_exception(mock_post):
    """ Test handling of other requests.exceptions.RequestException. """
    mock_post.side_effect = requests.exceptions.RequestException("Some other request error"); result = get_llm_response(**DEFAULT_ARGS)
    assert "⚠️ Error communicating with Ollama: Some other request error" in result


@patch(REQUESTS_POST_PATH)
def test_get_llm_response_json_decode_error_in_stream(mock_post):
    """ Test handling of invalid JSON during streaming. """
    stream_chunks = [json.dumps({"response": "Part 1.", "done": False}), '{"invalid json chunk', json.dumps({"done": True})]
    mock_post.return_value = mock_streaming_response(stream_chunks)
    result = get_llm_response(**DEFAULT_ARGS)
    assert result.startswith("Part 1.")
    assert "[Error decoding stream chunk:" in result
    # assert "Expecting value: line 1 column 1" in result # Old assertion
    assert "Unterminated string starting at" in result # Updated assertion matching the pytest output
    assert "Part 3" not in result

@patch('agents.ollama_agent.isinstance')
@patch(REQUESTS_POST_PATH)
@patch(BASE64_B64ENCODE_PATH)
@patch(IO_BYTESIO_PATH) # Patch the class io.BytesIO
@patch(PIL_IMAGE_CLASS_PATH) # Patch the class Image.Image used in isinstance
@patch(PIL_IMAGE_MODULE_PATH) # Patch the module Image used for saving etc.
def test_get_llm_response_with_image(
    mock_pil_image_module, mock_pil_image_class, mock_bytesio_class, mock_b64encode, mock_post,
    mock_isinstance # Add the new mock argument
):
    # Configure isinstance mock to return True specifically for the Image.Image check
    def isinstance_side_effect(obj, classinfo):
        # Check if the classinfo being compared against is the mocked PIL Image class
        if classinfo is mock_pil_image_class:
            return True # Pretend it's the right type
        # For other isinstance calls, delegate to the real isinstance
        return isinstance(obj, classinfo)
    mock_isinstance.side_effect = isinstance_side_effect


    """ Test successful call with a single image. """
    stream_chunks = [json.dumps({"response": "Image analyzed.", "done": True})]
    mock_post.return_value = mock_streaming_response(stream_chunks)

    # Create a standard MagicMock for the image instance
    mock_image_instance = MagicMock()
    # Make the isinstance check pass by setting the __class__ attribute
    # The class being patched is Image.Image within the ollama_agent module
    mock_image_instance.__class__ = mock_pil_image_class

    # Set attributes needed by the code
    mock_image_instance.mode = "RGB"
    mock_image_instance.save = MagicMock()

    # Setup BytesIO mock
    mock_buffer_instance = MagicMock(spec=io.BytesIO)
    mock_buffer_instance.getvalue = MagicMock(return_value=b"imagedata")
    mock_bytesio_class.return_value = mock_buffer_instance

    # Setup base64 mock
    mock_b64encode.return_value = b"ZW5jb2RlZGltYWdlZGF0YQ==" # base64 for 'encodedimagedata'

    args = {**DEFAULT_ARGS, "images": [mock_image_instance]}
    result = get_llm_response(**args)

    # Assertions
    assert result == "Image analyzed."
    mock_image_instance.save.assert_called_once_with(mock_buffer_instance, format="JPEG")
    mock_b64encode.assert_called_once_with(b"imagedata")
    mock_buffer_instance.getvalue.assert_called_once()
    payload = mock_post.call_args.kwargs.get('json', {})
    assert "images" in payload
    assert payload["images"] == ["ZW5jb2RlZGltYWdlZGF0YQ=="]


@patch(REQUESTS_POST_PATH)
def test_get_llm_response_with_invalid_image_object(mock_post, capsys):
    """ Test passing something other than a PIL Image in the images list. """
    stream_chunks = [json.dumps({"response": "Text response only.", "done": True})]
    mock_post.return_value = mock_streaming_response(stream_chunks)
    args = {**DEFAULT_ARGS, "images": ["not_an_image_object"]}
    result = get_llm_response(**args)
    assert result == "Text response only."
    payload = mock_post.call_args.kwargs.get('json', {})
    assert "images" not in payload
    captured = capsys.readouterr(); assert "Warning: Item 1 in images list is not a PIL Image object" in captured.out

@patch('agents.ollama_agent.isinstance') # Add this
@patch(REQUESTS_POST_PATH)
@patch(PIL_IMAGE_CLASS_PATH) # Patch Image.Image for isinstance check
@patch(PIL_IMAGE_MODULE_PATH) # Patch Image module for attributes like mode/save
def test_get_llm_response_image_processing_error(
    mock_pil_image_module, mock_pil_image_class, mock_post
):
    """ Test handling of exceptions during image processing (e.g., save fails). """
    # Create instance and link its class to the mocked class
    mock_image_instance = MagicMock()
    mock_image_instance.__class__ = mock_pil_image_class
    mock_image_instance.mode = "RGB"
    mock_image_instance.save = MagicMock(side_effect=Exception("PIL Save Error")) # Add save and make it fail
    args = {**DEFAULT_ARGS, "images": [mock_image_instance]}
    result = get_llm_response(**args)

    # Assert the specific error message returned by the function
    assert result.startswith("⚠️ Error: Failed to process image data.")
    assert "PIL Save Error" in result # Check original exception is included
    mock_post.assert_not_called()
    mock_image_instance.save.assert_called_once() # Check save was attempted


@patch(REQUESTS_POST_PATH)
def test_get_llm_response_empty_stream(mock_post):
    """Test response when the stream returns done immediately or is empty."""
    stream1 = [json.dumps({"done": True})]; mock_post.return_value = mock_streaming_response(stream1)
    result1 = get_llm_response(**DEFAULT_ARGS); assert "No response text received" in result1
    mock_post.reset_mock(); stream2 = []; mock_post.return_value = mock_streaming_response(stream2)
    result2 = get_llm_response(**DEFAULT_ARGS); assert "No response text received" in result2

@patch(REQUESTS_POST_PATH)
def test_get_llm_response_console_logging(mock_post, capsys):
    """Test if request details are logged to console when enabled."""
    stream_chunks = [json.dumps({"response": "Logged", "done": True})]
    mock_post.return_value = mock_streaming_response(stream_chunks)
    settings_log_on = {**MOCK_SETTINGS, "ollama_api_prompt_to_console": True}
    args = {**DEFAULT_ARGS, "settings": settings_log_on, "prompt": "Short prompt"}
    get_llm_response(**args)
    captured = capsys.readouterr(); log_output = captured.out
    assert "--- Ollama Request ---" in log_output; assert '"model": "test-model"' in log_output
    assert '"prompt_start": "Short prompt"' in log_output; assert '"images_count": 0' in log_output
    assert '"effective_options": {' in log_output; assert '"temperature": 0.8' in log_output
    assert "----------------------" in log_output