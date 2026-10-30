# ArtAgent/core/ollama_manager.py
"""Model management helpers: unload models, discover installed models for the UI."""
from .utils import load_json
from .ollama_client import OllamaClient, OllamaError, base_url_from_settings, clear_capability_cache

MODELS_FILE = 'models.json' # Curated starter list (used when Ollama can't be reached)

# Capabilities that make a model unusable for chat/captioning dropdowns
_NON_CHAT_CAPABILITIES = {"embedding", "image"}


def _client_for(settings_or_url) -> OllamaClient:
    if isinstance(settings_or_url, dict):
        return OllamaClient.from_settings(settings_or_url, timeout=20)
    return OllamaClient(base_url_from_settings({"ollama_url": settings_or_url or ""}), timeout=20)


def release_model(model_name: str, settings_or_url) -> str:
    """
    Unloads a model from Ollama's memory (keep_alive=0).

    Args:
        model_name: Ollama model name.
        settings_or_url: the settings dict, or a server URL (base or legacy .../api/generate).
    """
    if not model_name or not settings_or_url:
        return f"Skipping release: Missing model name ('{model_name}') or Ollama URL."
    client = _client_for(settings_or_url)
    try:
        print(f"Sending release request for model: {model_name} to {client.base_url}")
        msg = client.unload(model_name)
    except OllamaError as e:
        msg = f"Error releasing model '{model_name}': {e}"
    print(msg)
    return msg


def release_all_models_logic(settings: dict) -> str:
    """Unloads every model Ollama currently has in memory (falls back to models.json)."""
    client = _client_for(settings)
    try:
        names = [m.get("name") or m.get("model") for m in client.running()]
        names = [n for n in names if n]
        if not names:
            return "No models are currently loaded in Ollama."
    except OllamaError as e:
        print(f"Could not list running models ({e}); trying models.json instead.")
        listed = load_json(MODELS_FILE, is_relative=True)
        names = [m.get("name") for m in listed if isinstance(m, dict) and m.get("name")] if isinstance(listed, list) else []
        if not names:
            return f"Cannot release models: {e}"
    results = [release_model(name, settings) for name in names]
    summary = "\n".join(results)
    print(f"Model release process finished.\nSummary:\n{summary}")
    return summary


def load_curated_models() -> list[dict]:
    data = load_json(MODELS_FILE, is_relative=True)
    return [m for m in data if isinstance(m, dict) and m.get("name")] if isinstance(data, list) else []


def discover_models(settings: dict, curated: list[dict] | None = None) -> tuple[list[dict], str]:
    """
    Builds the model list for the UI.

    When Ollama is reachable, lists the *installed* models with vision/thinking
    flags read from /api/show capabilities (curated notes from models.json are
    kept). Otherwise falls back to models.json.

    Returns:
        (models, status) where models are dicts: name, vision, thinking, installed, notes.
    """
    curated = load_curated_models() if curated is None else curated
    by_name = {m["name"]: m for m in curated}
    client = OllamaClient.from_settings(settings, timeout=5)
    clear_capability_cache()
    try:
        installed = client.list_models()
    except OllamaError as e:
        fallback = [dict(m, installed=False) for m in curated]
        return fallback, f"Ollama not reachable ({e}). Showing models.json ({len(fallback)} models)."

    models = []
    for entry in installed:
        name = entry["name"]
        caps = entry.get("capabilities") or client.capabilities(name)
        if caps and set(caps) & _NON_CHAT_CAPABILITIES and "completion" not in caps:
            continue # embedding or image-generation models can't chat
        known = by_name.get(name, {})
        vision = ("vision" in caps) if caps else bool(known.get("vision"))
        models.append({
            "name": name,
            "vision": vision,
            "thinking": "thinking" in caps,
            "installed": True,
            "notes": known.get("notes", entry.get("details", {}).get("parameter_size", "")),
        })
    models.sort(key=lambda m: (not m["vision"], m["name"].lower()))
    if not models:
        return [dict(m, installed=False) for m in curated], "Ollama is running but has no chat models installed. Showing models.json; pull one with e.g. 'ollama pull gemma4'."
    n_vision = sum(1 for m in models if m["vision"])
    return models, f"Found {len(models)} installed Ollama model(s), {n_vision} with vision."


def model_display_name(model: dict) -> str:
    return f"{model['name']} (VISION)" if model.get("vision") else model["name"]


def model_choices(models: list[dict]) -> list[str]:
    return [model_display_name(m) for m in models]


def vision_model_choices(models: list[dict]) -> list[str]:
    return [model_display_name(m) for m in models if m.get("vision")]


def worker_model_choices(models: list[dict]) -> list[str]:
    return sorted({m["name"] for m in models})
