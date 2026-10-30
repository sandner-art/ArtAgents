# ArtAgent/core/refinement_logic.py
"""
Comment/Refine: continues the conversation that produced the current response.

The chat tab keeps a `conversation` state ({"model", "role", "system", "messages"})
from the last answer. A refinement is sent as the next user turn with the full
history, so the model sees its own previous answer and earlier refinements.
If there is no matching conversation (e.g. after a folder run or an app
restart), it falls back to a one-shot "revise this text" prompt.
"""
import time

from agents.roles_config import load_all_roles
from agents.ollama_agent import get_llm_response, stream_llm_response, is_error_response
from . import history_manager as history

FALLBACK_SYSTEM_PROMPT = (
    "You are an AI assistant specialized in refining and modifying existing text based on user instructions. "
    "Maintain the core essence of the original text unless the instruction explicitly asks for a fundamental change. "
    "Output only the revised text."
)


def _fallback_prompt(original_text: str, comment: str) -> str:
    return f"""**Original Text:**
---
{original_text}
---

**User's Refinement Instruction:**
---
{comment}
---

**Revised Text:**
"""


def iter_comment_logic(
    llm_response_text: str,
    comment: str,
    max_tokens_ui: int,
    use_ollama_api_options: bool,
    model_with_vision: str | None,
    current_settings: dict,
    file_agents_dict: dict,
    history_list_state: list,
    session_history_list_state: list,
    conversation_state: dict | None = None,
    ):
    """
    Streams a refinement of the current response.

    Yields (response_text, session_history_text, session_history_list, conversation).
    """
    history_list = list(history_list_state or [])
    current_session_history = list(session_history_list_state or [])

    if not comment or not model_with_vision:
        print(f"Comment ignored: No comment text provided or no model selected (Comment: '{comment}', Model: '{model_with_vision}').")
        yield llm_response_text, "\n---\n".join(current_session_history), current_session_history, conversation_state
        return

    actual_model_name = model_with_vision.replace(" (VISION)", "")
    roles_data_current = load_all_roles(current_settings, file_agents=file_agents_dict)

    conversation = conversation_state if isinstance(conversation_state, dict) else None
    messages = list(conversation.get("messages", [])) if conversation else []
    continues_chat = bool(messages) and messages[-1].get("role") == "assistant" and messages[-1].get("content") == llm_response_text

    if continues_chat:
        system = conversation.get("system")
        role = conversation.get("role") or "Refinement Assistant"
        prompt, chat_history = comment, messages
        mode = f"continuing conversation ({len(messages)} earlier messages)"
    else:
        system, role = FALLBACK_SYSTEM_PROMPT, "Refinement Assistant"
        prompt, chat_history = _fallback_prompt(llm_response_text, comment), None
        mode = "one-shot revision"
    print(f"Processing refinement using model {actual_model_name} ({mode})...")

    response = ""
    for response in stream_llm_response(
        role=role, prompt=prompt, model=actual_model_name, settings=current_settings,
        roles_data=roles_data_current, images=None, max_tokens=max_tokens_ui,
        system=system, history=chat_history, use_global_options=use_ollama_api_options,
    ):
        yield response, "\n---\n".join(current_session_history), current_session_history, None

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    entry = f"Timestamp: {timestamp}\nAction: Refine Text (Comment)\nModel: {actual_model_name}\nRefinement Instruction: {comment}\nContext: Previous response text\nResponse:\n{response}\n---\n"
    history_list = history.add_to_history(history_list, entry)
    current_session_history.append(entry)

    if is_error_response(response):
        new_conversation = conversation
    else:
        base_messages = messages if continues_chat else [{"role": "user", "content": prompt}]
        new_conversation = {
            "model": actual_model_name, "role": role, "system": system,
            "messages": base_messages + ([{"role": "user", "content": comment}] if continues_chat else []) + [{"role": "assistant", "content": response}],
        }
    yield response, "\n---\n".join(current_session_history), current_session_history, new_conversation


def comment_logic(*args, **kwargs) -> tuple[str, str, list]:
    """Blocking variant; returns (new_response, session_history_text, session_history_list)."""
    last = None
    for last in iter_comment_logic(*args, **kwargs):
        pass
    return tuple(last[:3])
