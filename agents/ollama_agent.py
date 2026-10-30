# ArtAgent/agents/ollama_agent.py
"""
LLM calls for agents, built on core.ollama_client (/api/chat).

`stream_llm_response` yields the growing answer so the UI can show tokens as
they arrive; `get_llm_response` returns the final text. Both keep the app's
long-standing convention of returning errors as text starting with "⚠️ Error:".
"""
import json
from typing import Iterator

from PIL import Image

from core.image_utils import encode_image_for_ollama
from core.ollama_client import OllamaClient, OllamaError, build_messages, strip_think_blocks

ERROR_PREFIX = "⚠️ Error:"


def is_error_response(text) -> bool:
    return isinstance(text, str) and (text.startswith(ERROR_PREFIX) or text.startswith("Error:"))


def build_effective_options(
    settings: dict,
    roles_data: dict,
    role: str,
    max_tokens: int | None,
    ollama_api_options: dict | None = None,
    use_global_options: bool | None = None,
) -> dict:
    """
    Merges Ollama options. Priority (low -> high):
      global App Settings options (only if "Use Advanced Ollama API Options" is on)
      -> Max Tokens from the UI -> the agent role's own options -> direct overrides.
    """
    use_global = settings.get("use_ollama_api_options", False) if use_global_options is None else bool(use_global_options)
    options = dict(settings.get("ollama_api_options", {}) or {}) if use_global else {}
    if max_tokens:
        try:
            options["num_predict"] = int(max_tokens)
        except (TypeError, ValueError):
            pass
    role_options = (roles_data or {}).get(role, {}).get("ollama_api_options", {}) if isinstance((roles_data or {}).get(role), dict) else {}
    options.update(role_options or {})
    if ollama_api_options:
        options.update(ollama_api_options)
    return options


def _encode_images(images) -> list[str]:
    encoded = []
    for i, img in enumerate(images or []):
        if isinstance(img, Image.Image):
            encoded.append(encode_image_for_ollama(img))
        else:
            print(f"Warning: Item {i+1} in images list is not a PIL Image object, skipping.")
    return encoded


def stream_llm_response(
    role: str,
    prompt: str,
    model: str,
    settings: dict,
    roles_data: dict,
    images: list = None,
    max_tokens: int = 1500,
    ollama_api_options: dict = None,
    system: str | None = None,
    history: list[dict] | None = None,
    use_global_options: bool | None = None,
) -> Iterator[str]:
    """
    Streams an answer from Ollama's /api/chat.

    Yields the accumulated answer text after each received chunk. On failure the
    last item yielded is an error string starting with "⚠️ Error:".

    Args:
        role: Agent role name (used to look up role-specific Ollama options).
        prompt: The user message.
        system: Optional system prompt (agent persona / instructions).
        history: Optional earlier turns [{"role": "user"|"assistant", "content": str}].
        images: Optional list of PIL images attached to the user message.
    """
    settings = settings or {}
    client = OllamaClient.from_settings(settings)
    options = build_effective_options(settings, roles_data, role, max_tokens, ollama_api_options, use_global_options)

    try:
        images_b64 = _encode_images(images)
    except Exception as img_e:
        print(f"Error processing image for Ollama payload: {img_e}")
        yield f"{ERROR_PREFIX} Failed to process image data. Details: {img_e}"
        return

    messages = build_messages(system, prompt, images_b64, history)

    if settings.get("ollama_api_prompt_to_console", True):
        print("--- Ollama Request ---")
        print(json.dumps({
            "model": model, "role": role,
            "system_start": (system or "")[:120],
            "prompt_start": prompt[:200] + ("..." if len(prompt) > 200 else ""),
            "history_turns": len(history or []), "images_count": len(images_b64),
            "options": options,
        }, indent=2, ensure_ascii=False))
        print("----------------------")

    answer = ""
    thinking_chars = 0
    try:
        think = client.think_value(model, settings.get("ollama_think", "off"))
        print(f"Sending chat request to Ollama model '{model}' at {client.base_url}...")
        for kind, text in client.chat_stream(model, messages, options=options, think=think):
            if kind == "thinking":
                thinking_chars += len(text)
                continue
            answer += text
            yield answer
    except OllamaError as e:
        print(f"Error from Ollama: {e}")
        yield f"{ERROR_PREFIX} {e}"
        return
    except Exception as e:  # Unexpected bug; keep the UI responsive
        print(f"Unexpected error while calling Ollama: {e}")
        yield f"{ERROR_PREFIX} Unexpected error while calling Ollama: {e}"
        return

    if thinking_chars:
        print(f"  (model reasoning: {thinking_chars} chars, not included in the answer)")
    final = strip_think_blocks(answer)
    if not final:
        final = "No response text received from LLM stream."
    yield final


def get_llm_response(
    role: str,
    prompt: str,
    model: str,
    settings: dict,
    roles_data: dict,
    images: list = None,
    max_tokens: int = 1500,
    ollama_api_options: dict = None,
    system: str | None = None,
    history: list[dict] | None = None,
    use_global_options: bool | None = None,
) -> str:
    """Blocking variant of stream_llm_response: returns the final answer (or an error string)."""
    result = f"{ERROR_PREFIX} No response."
    for result in stream_llm_response(
        role, prompt, model, settings, roles_data, images=images, max_tokens=max_tokens,
        ollama_api_options=ollama_api_options, system=system, history=history,
        use_global_options=use_global_options,
    ):
        pass
    return result
