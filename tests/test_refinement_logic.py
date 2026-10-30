# ArtAgent/tests/test_refinement_logic.py
"""Tests for core/refinement_logic.py (Comment/Refine, multi-turn)."""
import os
import sys
from unittest.mock import patch

import pytest

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(test_dir))

from core import refinement_logic
from core.refinement_logic import FALLBACK_SYSTEM_PROMPT, comment_logic, iter_comment_logic

STREAM_PATH = 'core.refinement_logic.stream_llm_response'
ADD_HISTORY_PATH = 'core.refinement_logic.history.add_to_history'
LOAD_ROLES_PATH = 'core.refinement_logic.load_all_roles'
STRFTIME_PATH = 'core.refinement_logic.time.strftime'

SETTINGS = {"ollama_base_url": "http://x:1"}
ROLES = {"Agent1": {"description": "Does things."}}
SESSION = ["session_entry_1"]
MODEL_UI = "vision-model (VISION)"


def streaming(answer):
    return lambda **kw: iter([answer[: max(1, len(answer) // 2)], answer])


@pytest.fixture(autouse=True)
def _env():
    with patch(ADD_HISTORY_PATH, side_effect=lambda h, e: (h or []) + [e]) as mock_hist, \
         patch(LOAD_ROLES_PATH, return_value=ROLES) as mock_roles, \
         patch(STRFTIME_PATH, return_value="2024-01-01 10:00:00"):
        yield mock_hist, mock_roles


def test_one_shot_fallback_without_conversation(_env):
    mock_hist, mock_roles = _env
    with patch(STREAM_PATH, side_effect=streaming("Refined LLM Response")) as mock_stream:
        response, session_txt, session_list = comment_logic(
            "Original LLM Response", "Make it better.", 1024, False, MODEL_UI,
            SETTINGS, {}, ["persistent"], SESSION)

    assert response == "Refined LLM Response"
    assert len(session_list) == 2
    assert "Action: Refine Text (Comment)" in session_list[-1] and "Model: vision-model" in session_list[-1]
    assert session_txt == "\n---\n".join(session_list)
    assert SESSION == ["session_entry_1"] # Input state not mutated
    mock_roles.assert_called_once_with(SETTINGS, file_agents={})
    kwargs = mock_stream.call_args.kwargs
    assert kwargs["role"] == "Refinement Assistant"
    assert kwargs["model"] == "vision-model"
    assert kwargs["system"] == FALLBACK_SYSTEM_PROMPT
    assert kwargs["history"] is None
    assert kwargs["images"] is None and kwargs["max_tokens"] == 1024
    assert kwargs["use_global_options"] is False
    assert "**Original Text:**\n---\nOriginal LLM Response\n---" in kwargs["prompt"]
    assert "**User's Refinement Instruction:**\n---\nMake it better.\n---" in kwargs["prompt"]
    mock_hist.assert_called_once()


def test_fallback_builds_new_conversation():
    with patch(STREAM_PATH, side_effect=streaming("v2")):
        *_, conversation = list(iter_comment_logic("v1", "shorter", 100, True, "m", SETTINGS, {}, [], []))[-1]
    assert conversation["model"] == "m" and conversation["system"] == FALLBACK_SYSTEM_PROMPT
    assert conversation["messages"][0]["role"] == "user" and "v1" in conversation["messages"][0]["content"]
    assert conversation["messages"][-1] == {"role": "assistant", "content": "v2"}
    assert len(conversation["messages"]) == 2


def test_continues_existing_conversation():
    conversation = {
        "model": "text-model", "role": "Agent1", "system": "You are acting as the agent 'Agent1'.",
        "messages": [{"role": "user", "content": "a chair"}, {"role": "assistant", "content": "A red chair."}],
    }
    with patch(STREAM_PATH, side_effect=streaming("A blue chair.")) as mock_stream:
        items = list(iter_comment_logic("A red chair.", "make it blue", 500, False, "text-model", SETTINGS, {}, [], SESSION,
                                        conversation_state=conversation))
    response, _, session_list, new_conversation = items[-1]
    assert response == "A blue chair."
    kwargs = mock_stream.call_args.kwargs
    assert kwargs["prompt"] == "make it blue" # Comment becomes the next user turn
    assert kwargs["history"] == conversation["messages"]
    assert kwargs["system"] == "You are acting as the agent 'Agent1'."
    assert kwargs["role"] == "Agent1"
    assert new_conversation["messages"] == [
        {"role": "user", "content": "a chair"},
        {"role": "assistant", "content": "A red chair."},
        {"role": "user", "content": "make it blue"},
        {"role": "assistant", "content": "A blue chair."},
    ]
    assert len(conversation["messages"]) == 2 # Input state not mutated
    assert all(item[3] is None for item in items[:-1]) # Conversation only on the final update
    assert [item[0] for item in items[:-1]] == ["A blue", "A blue chair."]


def test_second_refine_extends_the_chain():
    conversation = {"model": "m", "role": None, "system": None,
                    "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a1"}]}
    with patch(STREAM_PATH, side_effect=streaming("a2")):
        *_, conv2 = list(iter_comment_logic("a1", "c1", 10, False, "m", SETTINGS, {}, [], [], conversation))[-1]
    with patch(STREAM_PATH, side_effect=streaming("a3")) as mock_stream:
        *_, conv3 = list(iter_comment_logic("a2", "c2", 10, False, "m", SETTINGS, {}, [], [], conv2))[-1]
    assert len(mock_stream.call_args.kwargs["history"]) == 4
    assert [m["content"] for m in conv3["messages"]] == ["q", "a1", "c1", "a2", "c2", "a3"]
    assert mock_stream.call_args.kwargs["role"] == "Refinement Assistant" # role None -> generic


def test_stale_conversation_falls_back_to_one_shot():
    """If the response box no longer shows the conversation's last answer, don't continue it."""
    conversation = {"model": "m", "system": "s",
                    "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "old answer"}]}
    with patch(STREAM_PATH, side_effect=streaming("revised")) as mock_stream:
        list(iter_comment_logic("folder run summary", "c", 10, False, "m", SETTINGS, {}, [], [], conversation))
    assert mock_stream.call_args.kwargs["history"] is None
    assert mock_stream.call_args.kwargs["system"] == FALLBACK_SYSTEM_PROMPT


def test_error_keeps_previous_conversation():
    conversation = {"model": "m", "system": None,
                    "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    with patch(STREAM_PATH, side_effect=lambda **kw: iter(["⚠️ Error: down"])):
        response, _, session_list, conv = list(iter_comment_logic("a", "c", 10, False, "m", SETTINGS, {}, [], [], conversation))[-1]
    assert response == "⚠️ Error: down"
    assert conv is conversation
    assert len(session_list) == 1 # The failed attempt is still logged


@pytest.mark.parametrize("comment, model", [("", "model"), ("A comment", None), (None, "m")])
def test_no_comment_or_model_is_a_no_op(comment, model):
    with patch(STREAM_PATH) as mock_stream:
        response, session_txt, session_list = comment_logic("Prev Resp", comment, 100, False, model, SETTINGS, {}, [], [])
        items = list(iter_comment_logic("Prev Resp", comment, 100, False, model, SETTINGS, {}, [], [], {"keep": 1}))
    assert response == "Prev Resp" and session_list == [] and session_txt == ""
    assert items == [("Prev Resp", "", [], {"keep": 1})] # Conversation passed through unchanged
    mock_stream.assert_not_called()
