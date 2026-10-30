# ArtAgent/tests/test_agent.py

import pytest
import requests
import json
import base64
import sys
import os
from unittest.mock import patch, MagicMock

# --- Adjust import path ---
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir)
sys.path.insert(0, project_root)

try:
    from agents.ollama_agent import get_llm_response
    from PIL import Image as PILImageModule

    REQUESTS_POST_PATH = 'agents.ollama_agent.requests.post'
    ENCODE_IMAGE_PATH = 'agents.ollama_agent.encode_image_for_ollama'


except ImportError as e:
    pytest.skip(f"Skipping agent tests, modules not found: {e}", allow_module_level=True)
    def get_llm_response(*args, **kwargs): raise NotImplementedError("Agent function not found/imported.")
    REQUESTS_POST_PATH = 'builtins.print'; ENCODE_IMAGE_PATH = 'builtins.print'


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
    """ Invalid JSON mid-stream: the whole response becomes an error message (partial text is discarded). """
    stream_chunks = [json.dumps({"response": "Part 1.", "done": False}), '{"invalid json chunk', json.dumps({"response": "Part 3", "done": True})]
    mock_post.return_value = mock_streaming_response(stream_chunks)
    result = get_llm_response(**DEFAULT_ARGS)
    assert result.startswith("⚠️ Error: Error decoding JSON stream from Ollama after 2 chunks")
    assert "Unterminated string starting at" in result
    assert "Part 1." not in result
    assert "Part 3" not in result


@patch(REQUESTS_POST_PATH)
def test_get_llm_response_error_chunk_in_stream(mock_post):
    """ Ollama reports problems as {"error": "..."} chunks; these must surface as an error message. """
    stream_chunks = [json.dumps({"response": "Partial", "done": False}), json.dumps({"error": "model 'test-model' not found"}), json.dumps({"response": "never", "done": True})]
    mock_response = mock_streaming_response(stream_chunks)
    mock_post.return_value = mock_response
    result = get_llm_response(**DEFAULT_ARGS)
    assert result == "⚠️ Error: Ollama returned an error: model 'test-model' not found"
    mock_response.close.assert_called_once() # Streaming connection is released on early exit


@patch(REQUESTS_POST_PATH)
def test_get_llm_response_with_image(mock_post):
    """ Successful call with a single real PIL image: sent as base64 JPEG in the payload. """
    stream_chunks = [json.dumps({"response": "Image analyzed.", "done": True})]
    mock_post.return_value = mock_streaming_response(stream_chunks)
    image = PILImageModule.new("RGB", (16, 16), "red")

    result = get_llm_response(**{**DEFAULT_ARGS, "images": [image]})

    assert result == "Image analyzed."
    payload = mock_post.call_args.kwargs.get('json', {})
    assert len(payload.get("images", [])) == 1
    decoded = base64.b64decode(payload["images"][0])
    assert decoded[:3] == b"\xff\xd8\xff" # JPEG magic bytes


@patch(REQUESTS_POST_PATH)
def test_get_llm_response_with_palette_image(mock_post):
    """ Palette (GIF-style) images used to fail with 'cannot write mode P as JPEG'. """
    stream_chunks = [json.dumps({"response": "ok", "done": True})]
    mock_post.return_value = mock_streaming_response(stream_chunks)
    image = PILImageModule.new("P", (16, 16), 3)

    result = get_llm_response(**{**DEFAULT_ARGS, "images": [image]})

    assert result == "ok"
    payload = mock_post.call_args.kwargs.get('json', {})
    assert base64.b64decode(payload["images"][0])[:3] == b"\xff\xd8\xff"


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

@patch(ENCODE_IMAGE_PATH, side_effect=Exception("PIL Save Error"))
@patch(REQUESTS_POST_PATH)
def test_get_llm_response_image_processing_error(mock_post, mock_encode):
    """ Test handling of exceptions during image processing (e.g., save fails). """
    image = PILImageModule.new("RGB", (8, 8))
    result = get_llm_response(**{**DEFAULT_ARGS, "images": [image]})

    # Assert the specific error message returned by the function
    assert result.startswith("⚠️ Error: Failed to process image data.")
    assert "PIL Save Error" in result # Check original exception is included
    mock_post.assert_not_called()
    mock_encode.assert_called_once_with(image)


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