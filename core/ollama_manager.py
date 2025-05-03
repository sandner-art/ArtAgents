# ArtAgent/core/ollama_manager.py
import requests
from .utils import load_json # Use utils for loading

MODELS_FILE = 'models.json' # Relative path from root

def release_model(model_name: str, ollama_api_url: str):
    """Sends request to Ollama to release (unload) a model."""
    if not model_name or not ollama_api_url:
        return f"Skipping release: Missing model name ('{model_name}') or URL ('{ollama_api_url}')."

    payload = {"model": model_name, "keep_alive": 0}
    try:
        print(f"Sending release request for model: {model_name} to {ollama_api_url}")
        # Using POST to the generate endpoint is the documented way to unload
        response = requests.post(ollama_api_url, json=payload, timeout=20) # Short timeout for release
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
        msg = f"Model '{model_name}' release request sent successfully."
        print(msg)
        return msg
    except requests.exceptions.Timeout:
        msg = f"Error releasing model '{model_name}': Request timed out."
        print(msg); return msg
    except requests.exceptions.RequestException as e:
        error_detail = str(e)
        try: # Try get more info
            if e.response is not None: error_detail += f" | Status: {e.response.status_code} | Body: {e.response.text[:200]}"
        except Exception: pass
        msg = f"Error releasing model '{model_name}': {error_detail}"; print(msg); return msg

def release_all_models_logic(settings: dict):
    """
    Logic for releasing all models defined in models.json.
    Takes settings as input to get the URL.
    Returns a summary string.
    """
    print("Attempting to release all models specified in models.json...")
    ollama_url = settings.get("ollama_url")
    if not ollama_url:
        msg = "Cannot release models: Ollama URL not found in settings."
        print(msg); return msg

    # Use load_json from utils, specifying relative path
    models_to_release = load_json(MODELS_FILE, is_relative=True)
    if not models_to_release or not isinstance(models_to_release, list):
        msg = "No models found or invalid format in models.json."
        print(msg); return msg

    results = []
    for model_info in models_to_release:
        model_name = model_info.get("name")
        if model_name:
            # Call the release_model function also defined in this module
            results.append(release_model(model_name, ollama_url))

    summary = "\n".join(results) if results else "No valid model names found to release."
    print(f"Model release process finished.\nSummary:\n{summary}")
    return summary