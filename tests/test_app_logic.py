# ArtAgent/tests/test_app_logic.py

import pytest
import os
import sys
import json
import numpy as np
import time
import copy
from types import SimpleNamespace
import gradio as gr # Import gradio explicitly for gr.update checks
from unittest.mock import patch, MagicMock, ANY, mock_open, call

# --- Adjust import path ---
# Assuming the test file is in ArtAgent/tests/
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir) # Parent of tests/ -> ArtAgent/
sys.path.insert(0, project_root)


# --- Module Import ---
try:
    from core import app_logic
    from core import refinement_logic # comment_logic lives here now
    # Import history manager specifically for mocking save_history
    from core import history_manager
    from PIL import Image
except ImportError as e:
    pytest.skip(f"Skipping app_logic tests, core module or PIL not found: {e}", allow_module_level=True)


# --- Constants for Mock Paths ---
LOAD_ALL_ROLES_PATH = 'core.app_logic.load_all_roles'
GET_LLM_RESPONSE_PATH = 'core.app_logic.get_llm_response'
RUN_TEAM_WORKFLOW_PATH = 'core.app_logic.agent_manager.run_team_workflow'
RELEASE_MODEL_PATH = 'core.app_logic.ollama_manager.release_model'
ADD_TO_HISTORY_PATH = 'core.app_logic.history.add_to_history'
SAVE_TEAMS_PATH = 'core.app_logic.save_teams_to_file' # Path for mocking save_teams_to_file
SAVE_HISTORY_PATH = 'core.app_logic.history.save_history' # Path for mocking history save
LOAD_JSON_PATH = 'core.app_logic.load_json' # Used by settings callbacks
TIME_STRFTIME_PATH = 'core.app_logic.time.strftime'
GET_ACTUAL_ROLE_NAME_PATH = 'core.app_logic.get_actual_role_name'
OS_PATH_ISDIR_PATH = 'core.app_logic.os.path.isdir'
OS_LISTDIR_PATH = 'core.app_logic.os.listdir'
OS_PATH_EXISTS_PATH = 'core.app_logic.os.path.exists'
OS_PATH_SPLITEXT_PATH = 'core.app_logic.os.path.splitext'
OS_PATH_JOIN_PATH = 'core.app_logic.os.path.join' # Added for mocking
OS_PATH_BASENAME_PATH = 'core.app_logic.os.path.basename' # Added for mocking
OS_PATH_ISFILE_PATH = 'core.app_logic.os.path.isfile' # Added for folder processing fix
GET_ABSOLUTE_PATH = 'core.app_logic.get_absolute_path' # Used by save_settings
BUILTINS_OPEN_PATH = 'core.app_logic.open'
JSON_DUMP_PATH = 'core.app_logic.json.dump' # Used by save_settings
PIL_IMAGE_FROMARRAY_PATH = 'core.app_logic.Image.fromarray'
PIL_IMAGE_OPEN_PATH = 'core.app_logic.open_image' # Folder processing opens images via core.image_utils.open_image
GR_WARNING_PATH = 'core.app_logic.gr.Warning'
STREAM_LLM_PATH = 'core.app_logic.stream_llm_response'
ITER_TEAM_WORKFLOW_PATH = 'core.app_logic.agent_manager.iter_team_workflow'
ITER_CHAT_LOGIC_PATH = 'core.app_logic.iter_chat_logic'


def stream_returning(*answers):
    """side_effect for stream_llm_response: each call streams the next answer (a partial, then the full text)."""
    remaining = list(answers)

    def _fake(**kwargs):
        answer = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return iter([answer[: max(1, len(answer) // 2)], answer])
    return _fake


def final(gen):
    """Last item of a generator (and the list of all items)."""
    items = list(gen)
    return items[-1], items

# Patching 'core.app_logic.os.path.X' replaces os.path.X everywhere, so side effects must
# call the *real* functions captured here (calling os.path.splitext inside the mock recurses).
_REAL_SPLITEXT = os.path.splitext
_REAL_BASENAME = os.path.basename

# --- Test Data ---
MOCK_TIMESTAMP = "2024-01-01 10:00:00"
mock_settings = { # Simplified version used as *input* to callbacks
    "ollama_url": "http://mock-host:11434/api/generate", "max_tokens_slider": 4096,
    "using_default_agents": True, "using_custom_agents": False, "release_model_on_change": False,
    "ollama_api_prompt_to_console": False, "ollama_api_options": {"seed": 123, "temperature": 0.5, "num_ctx": 2048, "use_mmap": True}
}
# Represents settings loaded *from file* inside save_settings_callback
mock_loaded_settings_from_file = {
    "ollama_url": "http://old-url", "max_tokens_slider": 2048,
    "using_default_agents": False, "using_custom_agents": True, "release_model_on_change": True,
    "ollama_api_prompt_to_console": True, "gradio_theme": "Soft",
    "ollama_api_options": {"seed": 42, "temperature": 0.7, "num_ctx": 1024, "use_mmap": False, "top_k": 40} # Note different types/keys
}
mock_profiles_data = {
    "Fast": {"temperature": 0.6, "top_k": 30, "num_predict": 512, "num_ctx": 2048},
    "Creative": {"temperature": 0.9, "top_p": 0.9, "mirostat": 1}
}
mock_models_data = [{"name": "text-model", "vision": False}, {"name": "vision-model", "vision": True}]
mock_limiters_data = {"SDXL": {"limiter_token_slider": 200, "limiter_prompt_format": "SDXL style:"}}
mock_teams_data = {
    "TeamA": {"description": "Test Team A", "steps": [{"role":"Agent1"}], "assembly_strategy": "concatenate"},
    "TeamB": {"description": "Test Team B", "steps": [{"role":"Agent1"}, {"role":"Agent2"}], "assembly_strategy": "refine_last"}
}
mock_roles_data = {"Agent1": {"description": "Does agent things."}, "User": {"description": "Represents the user."}, "Agent2": {"description": "Does other things."}} # Added Agent2
mock_file_agents_dict = {}
mock_history_list = ["persistent_entry_1"]
mock_session_history = ["session_entry_1"]
ui_folder_path = "/fake/folder"
ui_user_input = "Test user query"
ui_model_vision = "vision-model (VISION)"
ui_model_text = "text-model"
ui_max_tokens = 1024
ui_file_handling = "Skip"
ui_limiter = "Off"
ui_single_image_np = np.zeros((10, 10, 3), dtype=np.uint8)
ui_use_ollama_options = False
ui_release_model = False
ui_role_agent1 = "Agent1"
ui_role_team_a = "[Team] TeamA"
mock_all_teams_state = mock_teams_data.copy() # For editor tests
mock_empty_editor_state = {"name": "", "description": "", "steps": [], "assembly_strategy": "concatenate"}
mock_editor_state_team_a = { # State as if TeamA was loaded
     "name": "TeamA", "description": "Test Team A",
     "steps": [{"role":"Agent1"}], "assembly_strategy": "concatenate"
}
# For save_settings callback - ordered keys based on mock_loaded_settings_from_file['ollama_api_options']
mock_api_option_keys = sorted(mock_loaded_settings_from_file['ollama_api_options'].keys()) # ['num_ctx', 'seed', 'temperature', 'top_k', 'use_mmap']


# --- Fixtures ---
@pytest.fixture(autouse=True)
def mock_time(monkeypatch):
    """Mock time.strftime."""
    monkeypatch.setattr(TIME_STRFTIME_PATH, lambda x: MOCK_TIMESTAMP)

# --- Tests for execute_chat_or_team / iter_execute_chat_or_team (Router) ---

ROUTER_STATES = (mock_settings, mock_models_data, mock_limiters_data, mock_teams_data,
                 None, mock_file_agents_dict, mock_history_list, mock_session_history)


def _router_args(selection, model=ui_model_vision, image=None, clean=False, user_input=ui_user_input):
    return (ui_folder_path, user_input, model, ui_max_tokens, ui_file_handling, ui_limiter, image,
            ui_use_ollama_options, ui_release_model, selection, clean) + ROUTER_STATES


def _team_events(final_text="Team Output"):
    from core.agent_manager import TeamProgress, TeamResult
    return [
        TeamProgress(1, 1, "Agent1", "", "Step 1/1: Agent1"),
        TeamProgress(1, 1, "Agent1", "Partial step text", "Step 1/1: Agent1"),
        TeamProgress(0, 1, "", "", "Assembling final output (concatenate)..."),
        TeamResult(final_text, ["persistent_entry_1", "team_log"], None),
    ]


@patch(ITER_TEAM_WORKFLOW_PATH)
@patch(LOAD_ALL_ROLES_PATH)
def test_execute_router_team_call(mock_load_roles, mock_iter_team, mock_time):
    """Team selections run agent_manager.iter_team_workflow and summarise the run in the session log."""
    mock_load_roles.return_value = mock_roles_data
    mock_iter_team.return_value = iter(_team_events())

    response, session_hist_txt, model_state, new_session_list = app_logic.execute_chat_or_team(*_router_args(ui_role_team_a))

    assert response == "Team Output"
    assert model_state is None # No single model state for teams
    assert "Workflow Run: 'TeamA'" in session_hist_txt and "session_entry_1" in session_hist_txt
    assert len(new_session_list) == 2
    mock_load_roles.assert_called_once_with(mock_settings, file_agents=mock_file_agents_dict)
    kwargs = mock_iter_team.call_args.kwargs
    assert kwargs['team_name'] == "TeamA"
    assert kwargs['team_definition'] == mock_teams_data['TeamA']
    assert kwargs['user_input'] == ui_user_input
    assert kwargs['initial_settings'] == mock_settings
    assert kwargs['all_roles_data'] == mock_roles_data
    assert kwargs['history_list'] == mock_history_list
    assert kwargs['worker_model_name'] == "vision-model" # ' (VISION)' suffix resolved
    assert kwargs['worker_model_has_vision'] is True
    assert kwargs['single_image_input'] is None


@patch(ITER_TEAM_WORKFLOW_PATH)
@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data)
def test_iter_router_team_streams_progress_and_conversation(mock_load_roles, mock_iter_team, mock_time):
    mock_iter_team.return_value = iter(_team_events("Final team text"))
    last, items = final(app_logic.iter_execute_chat_or_team(*_router_args(ui_role_team_a)))
    responses = [item[0] for item in items]
    assert responses[0] == "⏳ Team 'TeamA' · Step 1/1: Agent1"
    assert responses[1] == "⏳ Team 'TeamA' · Step 1/1: Agent1\n\nPartial step text"
    assert "Assembling final output" in responses[2]
    assert all(item[4] is None for item in items[:-1]) # No conversation while streaming
    response, _, model_state, session_list, conversation = last
    assert response == "Final team text"
    assert conversation["model"] == "vision-model"
    assert conversation["system"] == app_logic.REFINER_SYSTEM_PROMPT
    assert conversation["messages"][0] == {"role": "user", "content": f"Create the final output for this request: {ui_user_input}"}
    assert conversation["messages"][-1] == {"role": "assistant", "content": "Final team text"}


@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data)
def test_team_run_with_numpy_image_does_not_crash(mock_load_roles, mock_time):
    """Regression for 'The truth value of an array with more than one element is ambiguous'."""
    with patch('core.agent_manager.stream_llm_response', side_effect=lambda **kw: iter(["step out"])) as mock_stream, \
         patch('core.agent_manager.history.add_to_history', side_effect=lambda h, e: (h or []) + [e]):
        response, _, _, _ = app_logic.execute_chat_or_team(*_router_args(ui_role_team_a, image=ui_single_image_np))
    assert response == "step out"
    sent = mock_stream.call_args.kwargs["images"]
    assert len(sent) == 1 and isinstance(sent[0], Image.Image)


@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data)
def test_team_run_unknown_model(mock_load_roles, mock_time):
    response, *_ = app_logic.execute_chat_or_team(*_router_args(ui_role_team_a, model="ghost-model"))
    assert response == "Error: Could not determine worker model name for team from input 'ghost-model'."


def test_team_run_without_models_data(mock_time):
    args = list(_router_args(ui_role_team_a))
    args[12] = [] # models_data_state
    response, *_ = app_logic.execute_chat_or_team(*args)
    assert response == "Error: Models data state is missing."


@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data)
def test_team_run_bad_image_type(mock_load_roles, mock_time):
    response, *_ = app_logic.execute_chat_or_team(*_router_args(ui_role_team_a, image="not-an-image"))
    assert response.startswith("Error processing input image:")


@patch(ITER_CHAT_LOGIC_PATH)
def test_execute_router_agent_call(mock_iter_chat, mock_time):
    """Single-agent selections are routed to iter_chat_logic."""
    mock_iter_chat.return_value = iter([
        ("Agent Out", "s", "text-model", ["session_entry_1"], None),
        ("Agent Output", "session_entry_1\n---\nagent_log", "text-model", ["session_entry_1", "agent_log"], {"messages": []}),
    ])
    response, session_hist_txt, model_state, new_session_list = app_logic.execute_chat_or_team(*_router_args(ui_role_agent1, model=ui_model_text))
    assert response == "Agent Output"
    assert model_state == "text-model"
    assert "agent_log" in session_hist_txt and len(new_session_list) == 2
    kwargs = mock_iter_chat.call_args.kwargs
    assert kwargs['role_display_name'] == ui_role_agent1
    assert kwargs['model_with_vision'] == ui_model_text
    assert kwargs['session_history_list_state'] == mock_session_history


@patch(ITER_CHAT_LOGIC_PATH)
def test_execute_router_cleans_artifacts_when_requested(mock_iter_chat, mock_time):
    """clean_artifacts_flag=True strips '--- Output from Agent: ... ---' headers from the final text (and conversation)."""
    raw = "--- Output from Agent: Agent1 ---\nA red chair.\n--- Output from Agent: Agent2 ---\nOn a beach."
    conversation = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": raw}]}
    mock_iter_chat.side_effect = lambda **kw: iter([(raw, "", "text-model", [], copy.deepcopy(conversation))])

    last, items = final(app_logic.iter_execute_chat_or_team(*_router_args(ui_role_agent1, model=ui_model_text, clean=True)))
    untouched, *_ = app_logic.execute_chat_or_team(*_router_args(ui_role_agent1, model=ui_model_text, clean=False))

    assert len(items) == 2 # Raw final, then the cleaned version
    assert last[0] == "A red chair.\nOn a beach."
    assert last[4]["messages"][-1]["content"] == "A red chair.\nOn a beach."
    assert untouched == raw


@patch(ITER_CHAT_LOGIC_PATH)
def test_cleaning_skips_error_responses(mock_iter_chat, mock_time):
    error = "⚠️ Error: --- Output from Agent: x ---"
    mock_iter_chat.return_value = iter([(error, "", None, [], None)])
    last, items = final(app_logic.iter_execute_chat_or_team(*_router_args(ui_role_agent1, clean=True)))
    assert last[0] == error and len(items) == 1


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data)
def test_execute_router_direct_agent_call(mock_load_roles, mock_add_history, mock_time):
    """'(Direct Agent Call)' sends the prompt straight to the model, with no agent persona."""
    with patch(STREAM_LLM_PATH, side_effect=stream_returning("Plain model answer")) as mock_stream:
        response, session_hist_txt, model_state, new_session_list = app_logic.execute_chat_or_team(
            *_router_args("(Direct Agent Call)", model=ui_model_text))
    assert response == "Plain model answer"
    assert model_state == "text-model"
    kwargs = mock_stream.call_args.kwargs
    assert kwargs["system"] is None # No persona, no limiter
    assert kwargs["role"] == ""
    assert kwargs["prompt"] == ui_user_input
    assert "Role: (Direct Model Call)" in new_session_list[-1]
    assert len(new_session_list) == 2


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data)
def test_direct_call_with_limiter_uses_limiter_as_system(mock_load_roles, mock_add_history, mock_time):
    with patch(STREAM_LLM_PATH, side_effect=stream_returning("x")) as mock_stream:
        app_logic.chat_logic(None, "(Direct Agent Call)", ui_user_input, ui_model_text, 1000, ui_file_handling, "SDXL", None,
                             False, False, mock_settings, mock_models_data, mock_limiters_data, None,
                             mock_file_agents_dict, mock_history_list, mock_session_history)
    assert mock_stream.call_args.kwargs["system"] == "SDXL style:"
    assert mock_stream.call_args.kwargs["max_tokens"] == 200


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data)
def test_empty_selection_is_a_direct_call(mock_load_roles, mock_add_history, mock_time):
    with patch(STREAM_LLM_PATH, side_effect=stream_returning("answer")) as mock_stream:
        response, *_ = app_logic.execute_chat_or_team(*_router_args(None, model=ui_model_text))
    assert response == "answer" and mock_stream.call_args.kwargs["system"] is None


def test_execute_router_invalid_team_name(mock_time):
    """Test selecting a team name that doesn't exist in the state."""
    response, session_hist_txt, model_state, new_session_list = app_logic.execute_chat_or_team(
        *_router_args("[Team] NonExistentTeam", model=ui_model_text))
    assert "Error: Selected Agent Team 'NonExistentTeam' definition not found" in response
    assert model_state is None
    assert session_hist_txt == "session_entry_1"
    assert new_session_list == mock_session_history


def test_execute_router_without_any_output(mock_time):
    """The blocking wrapper never returns None, even if the stream produced nothing."""
    with patch(ITER_CHAT_LOGIC_PATH, return_value=iter([])):
        assert app_logic.execute_chat_or_team(*_router_args(ui_role_agent1)) == ("Error: Processing failed.", "", None, [])


# --- Tests for chat_logic (Single Agent Core Logic) ---

@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(RELEASE_MODEL_PATH)
@patch(LOAD_ALL_ROLES_PATH)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x) # Simple mock: returns input display name
def test_chat_logic_text_only(mock_get_actual_name, mock_load_roles, mock_release_model, mock_add_history, mock_time):
    """Text-only mode: the agent persona goes into the system prompt, the user input is the user turn."""
    mock_load_roles.return_value = mock_roles_data

    with patch(STREAM_LLM_PATH, side_effect=stream_returning("LLM Text Response")) as mock_stream:
        response, session_hist_txt, model_state, new_session_list = app_logic.chat_logic(
            None, ui_role_agent1, ui_user_input, ui_model_text, ui_max_tokens, # No folder path
            ui_file_handling, ui_limiter, None, ui_use_ollama_options, False, # No single image, no release model
            mock_settings, mock_models_data, mock_limiters_data,
            None, mock_file_agents_dict, mock_history_list, mock_session_history
        )

    assert response == "LLM Text Response"
    assert model_state == "text-model"
    assert session_hist_txt.endswith("---\nTimestamp: "+MOCK_TIMESTAMP+"\nRole: Agent1\nModel: text-model\nInput: Test user query\nImage: [None - Text Only]\nResponse:\nLLM Text Response\n---\n")
    assert len(new_session_list) == 2 # Original + new entry

    mock_get_actual_name.assert_called_once_with(ui_role_agent1)
    mock_load_roles.assert_called_once_with(mock_settings, file_agents=mock_file_agents_dict)
    mock_release_model.assert_not_called()
    mock_stream.assert_called_once()
    kwargs = mock_stream.call_args.kwargs
    assert kwargs['role'] == ui_role_agent1
    assert kwargs['model'] == ui_model_text
    assert kwargs['settings'] == mock_settings
    assert kwargs['roles_data'] == mock_roles_data
    assert kwargs['images'] is None
    assert kwargs['max_tokens'] == ui_max_tokens
    assert kwargs['use_global_options'] is ui_use_ollama_options # UI checkbox is honoured
    assert kwargs['prompt'] == ui_user_input
    assert kwargs['system'] == "You are acting as the agent 'Agent1'.\n\nDoes agent things."
    mock_add_history.assert_called_once()


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
def test_iter_chat_logic_streams_and_returns_conversation(mock_get_actual_name, mock_load_roles, mock_add_history, mock_time):
    with patch(STREAM_LLM_PATH, side_effect=lambda **kw: iter(["Hel", "Hello", "Hello there"])):
        last, items = final(app_logic.iter_chat_logic(
            None, ui_role_agent1, ui_user_input, ui_model_text, ui_max_tokens, ui_file_handling, ui_limiter, None,
            ui_use_ollama_options, False, mock_settings, mock_models_data, mock_limiters_data, None,
            mock_file_agents_dict, mock_history_list, mock_session_history))
    assert [i[0] for i in items] == ["Hel", "Hello", "Hello there", "Hello there"]
    assert all(i[4] is None for i in items[:-1])
    assert len(items[0][3]) == 1 # Session log only grows once the answer is complete
    conversation = last[4]
    assert conversation == {
        "model": "text-model", "role": "Agent1",
        "system": "You are acting as the agent 'Agent1'.\n\nDoes agent things.",
        "messages": [{"role": "user", "content": ui_user_input}, {"role": "assistant", "content": "Hello there"}],
    }


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
def test_iter_chat_logic_error_gives_no_conversation(mock_get_actual_name, mock_load_roles, mock_add_history, mock_time):
    with patch(STREAM_LLM_PATH, side_effect=lambda **kw: iter(["⚠️ Error: model not found"])):
        last, _ = final(app_logic.iter_chat_logic(
            None, ui_role_agent1, ui_user_input, ui_model_text, ui_max_tokens, ui_file_handling, ui_limiter, None,
            ui_use_ollama_options, False, mock_settings, mock_models_data, mock_limiters_data, None,
            mock_file_agents_dict, mock_history_list, mock_session_history))
    assert last[0] == "⚠️ Error: model not found"
    assert last[4] is None # Refine must not continue from an error


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(RELEASE_MODEL_PATH)
@patch(LOAD_ALL_ROLES_PATH)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
def test_chat_logic_single_image(mock_get_actual_name, mock_load_roles, mock_release_model, mock_add_history, mock_time):
    """A numpy image from the UI is converted to PIL and sent with the prompt to a vision model."""
    mock_load_roles.return_value = mock_roles_data

    with patch(STREAM_LLM_PATH, side_effect=stream_returning("LLM Image Response")) as mock_stream:
        response, session_hist_txt, model_state, new_session_list = app_logic.chat_logic(
            None, ui_role_agent1, ui_user_input, ui_model_vision, ui_max_tokens,
            ui_file_handling, ui_limiter, ui_single_image_np,
            ui_use_ollama_options, False,
            mock_settings, mock_models_data, mock_limiters_data,
            None, mock_file_agents_dict, mock_history_list, mock_session_history
        )

    assert response == "LLM Image Response"
    assert model_state == "vision-model"
    assert "Image: [Image Data Sent]" in session_hist_txt
    assert len(new_session_list) == 2
    kwargs = mock_stream.call_args.kwargs
    assert kwargs['model'] == "vision-model"
    assert kwargs['prompt'] == ui_user_input
    assert len(kwargs['images']) == 1
    assert isinstance(kwargs['images'][0], Image.Image) and kwargs['images'][0].size == (10, 10)
    mock_add_history.assert_called_once()


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
def test_chat_logic_image_with_non_vision_model_falls_back_to_text(mock_get_actual_name, mock_load_roles, mock_add_history, mock_time):
    """Used to return 'Processing...' forever; now the prompt is sent as text only."""
    with patch(STREAM_LLM_PATH, side_effect=stream_returning("Text answer")) as mock_stream:
        response, _, model_state, new_session_list = app_logic.chat_logic(
            None, ui_role_agent1, ui_user_input, ui_model_text, ui_max_tokens,
            ui_file_handling, ui_limiter, ui_single_image_np,
            ui_use_ollama_options, False,
            mock_settings, mock_models_data, mock_limiters_data,
            None, mock_file_agents_dict, mock_history_list, mock_session_history
        )
    assert response == "Text answer"
    assert model_state == "text-model"
    assert mock_stream.call_args.kwargs['images'] is None
    assert "Image: [Single Image Ignored - No Vision]" in new_session_list[-1]


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
def test_chat_logic_accepts_pil_image(mock_get_actual_name, mock_load_roles, mock_add_history, mock_time):
    pil = Image.new("RGB", (5, 5))
    with patch(STREAM_LLM_PATH, side_effect=stream_returning("ok")) as mock_stream:
        app_logic.chat_logic(None, ui_role_agent1, ui_user_input, ui_model_vision, ui_max_tokens, ui_file_handling,
                             ui_limiter, pil, False, False, mock_settings, mock_models_data, mock_limiters_data, None,
                             mock_file_agents_dict, mock_history_list, mock_session_history)
    assert mock_stream.call_args.kwargs['images'] == [pil]


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(RELEASE_MODEL_PATH)
@patch(GET_LLM_RESPONSE_PATH)
@patch(LOAD_ALL_ROLES_PATH)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
@patch(OS_PATH_ISDIR_PATH, return_value=True)
@patch(OS_LISTDIR_PATH, return_value=["img1.png", "img2.jpg", "other.txt"])
@patch(OS_PATH_EXISTS_PATH, return_value=False)
@patch(OS_PATH_SPLITEXT_PATH, side_effect=lambda p: _REAL_SPLITEXT(_REAL_BASENAME(p)))
@patch(OS_PATH_BASENAME_PATH, side_effect=os.path.basename) # ADDED BASENAME MOCK
@patch(OS_PATH_JOIN_PATH, side_effect=lambda *args: os.sep.join(args)) # Simple join mock # FIXED
@patch(OS_PATH_ISFILE_PATH, return_value=True) # Mock isfile
@patch(PIL_IMAGE_OPEN_PATH, return_value=MagicMock(spec=Image))
@patch(BUILTINS_OPEN_PATH, new_callable=mock_open)
def test_chat_logic_folder_processing_skip(
    mock_builtin_open, mock_pil_open, mock_isfile, mock_join, mock_basename, mock_splitext, mock_exists, mock_listdir, mock_isdir, # ADDED mocks
    mock_get_actual_name, mock_load_roles, mock_get_llm, mock_release_model, mock_add_history, mock_time):
    """'Skip' leaves existing .txt files alone; images without a caption file still get one written."""
    mock_load_roles.return_value = mock_roles_data
    mock_get_llm.side_effect = ["LLM Resp Img2"] # img1 is skipped before any model call
    mock_pil_image_instance = mock_pil_open.return_value
    mock_exists.side_effect = lambda p: p.endswith("img1.txt") # img1 already captioned, img2 not

    response, session_hist_txt, model_state, new_session_list = app_logic.chat_logic(
        ui_folder_path, ui_role_agent1, ui_user_input, ui_model_vision, ui_max_tokens,
        "Skip", ui_limiter, None,
        ui_use_ollama_options, False,
        mock_settings, mock_models_data, mock_limiters_data,
        None, mock_file_agents_dict, mock_history_list, mock_session_history
    )

    assert response.startswith("Folder processing complete (1 processed, 1 skipped):")
    assert "img1.png: Skipped (caption file exists)" in response
    assert "img2.jpg: Written -> img2.txt" in response
    assert model_state == "vision-model"
    assert len(new_session_list) == len(mock_session_history) + 1 # only img2 is logged

    mock_isdir.assert_called_once_with(ui_folder_path)
    mock_listdir.assert_called_once_with(ui_folder_path)
    assert mock_isfile.call_count == 3 # Checked for every listed entry, including other.txt
    assert mock_pil_open.call_count == 1 # img1 is never opened
    assert mock_get_llm.call_count == 1 # no wasted model call for img1
    call_args_2, call_kwargs_2 = mock_get_llm.call_args_list[0]
    assert call_kwargs_2['model'] == "vision-model"
    assert call_kwargs_2['images'][0] == mock_pil_image_instance
    assert "Image Context: Analyzing 'img2.jpg'" in call_kwargs_2['prompt']
    mock_builtin_open.assert_called_once_with(os.sep.join([ui_folder_path, "img2.txt"]), 'w', encoding='utf-8')
    mock_builtin_open().write.assert_called_once_with("LLM Resp Img2")
    assert mock_add_history.call_count == 1
    # Removed assert False


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(RELEASE_MODEL_PATH)
@patch(LOAD_ALL_ROLES_PATH)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
def test_chat_logic_model_release(
    mock_get_actual_name, mock_load_roles, mock_release_model, mock_add_history, mock_time):
    """'Unload Previous Model' releases the model used in the previous run (from model_state)."""
    mock_load_roles.return_value = mock_roles_data
    previous_model = "old-text-model"
    settings_release = {**mock_settings, "release_model_on_change": True}

    with patch(STREAM_LLM_PATH, side_effect=stream_returning("Response")):
        app_logic.chat_logic(
            None, ui_role_agent1, ui_user_input, ui_model_text, ui_max_tokens,
            ui_file_handling, ui_limiter, None, ui_use_ollama_options, True, # release_on_change UI input
            settings_release, mock_models_data, mock_limiters_data,
            previous_model, # Model used in the last run
            mock_file_agents_dict, mock_history_list, mock_session_history
        )

    mock_release_model.assert_called_once_with(previous_model, settings_release)


@pytest.mark.parametrize("previous", [None, "text-model", "text-model (VISION)"])
@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(RELEASE_MODEL_PATH)
@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
def test_chat_logic_no_release_for_same_or_missing_model(mock_get_actual_name, mock_load_roles, mock_release_model, mock_add_history, previous, mock_time):
    with patch(STREAM_LLM_PATH, side_effect=stream_returning("Response")):
        app_logic.chat_logic(None, ui_role_agent1, ui_user_input, ui_model_text, ui_max_tokens, ui_file_handling,
                             ui_limiter, None, False, True, mock_settings, mock_models_data, mock_limiters_data,
                             previous, mock_file_agents_dict, mock_history_list, mock_session_history)
    mock_release_model.assert_not_called()


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(LOAD_ALL_ROLES_PATH)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
def test_chat_logic_limiter(
    mock_get_actual_name, mock_load_roles, mock_add_history, mock_time):
    """Limiters cap max_tokens and add their prompt format to the system prompt."""
    mock_load_roles.return_value = mock_roles_data
    limiter_choice = "SDXL" # Defined in mock_limiters_data with tokens=200

    with patch(STREAM_LLM_PATH, side_effect=stream_returning("Response")) as mock_stream:
        app_logic.chat_logic(
            None, ui_role_agent1, ui_user_input, ui_model_text, 1000,
            ui_file_handling, limiter_choice, None, ui_use_ollama_options, False,
            mock_settings, mock_models_data, mock_limiters_data,
            None, mock_file_agents_dict, mock_history_list, mock_session_history
        )

    kwargs = mock_stream.call_args.kwargs
    assert kwargs['max_tokens'] == 200
    assert kwargs['system'] == "You are acting as the agent 'Agent1'.\n\nDoes agent things.\n\nSDXL style:"
    assert kwargs['prompt'] == ui_user_input


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(RELEASE_MODEL_PATH)
@patch(GET_LLM_RESPONSE_PATH)
@patch(LOAD_ALL_ROLES_PATH)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
@patch(OS_PATH_ISDIR_PATH, return_value=True)
@patch(OS_LISTDIR_PATH, return_value=["img1.png"])
@patch(OS_PATH_EXISTS_PATH, side_effect=[False, True])
@patch(OS_PATH_SPLITEXT_PATH, side_effect=lambda p: _REAL_SPLITEXT(_REAL_BASENAME(p)))
@patch(OS_PATH_BASENAME_PATH, side_effect=os.path.basename) # ADDED
@patch(OS_PATH_JOIN_PATH, side_effect=lambda *args: os.sep.join(args)) # FIXED
@patch(OS_PATH_ISFILE_PATH, return_value=True) # ADDED
@patch(PIL_IMAGE_OPEN_PATH, return_value=MagicMock(spec=Image))
@patch(BUILTINS_OPEN_PATH, new_callable=mock_open)
def test_chat_logic_folder_processing_overwrite(
    mock_builtin_open, mock_pil_open, mock_isfile, mock_join, mock_basename, mock_splitext, mock_exists, mock_listdir, mock_isdir, # ADDED mocks
    mock_get_actual_name, mock_load_roles, mock_get_llm, mock_release_model, mock_add_history, mock_time):
    """Test chat_logic with folder processing and 'Overwrite' file handling."""
    mock_load_roles.return_value = mock_roles_data
    mock_get_llm.return_value = "LLM Overwrite Resp"
    # Use the mocked join to determine expected path
    expected_txt_path = os.sep.join([ui_folder_path, "img1.txt"])

    # Scenario 1: File doesn't exist
    mock_exists.side_effect = [False]
    response1, _, _, _ = app_logic.chat_logic(ui_folder_path, ui_role_agent1, ui_user_input, ui_model_vision, ui_max_tokens, "Overwrite", ui_limiter, None, ui_use_ollama_options, False, mock_settings, mock_models_data, mock_limiters_data, None, mock_file_agents_dict, mock_history_list, mock_session_history)
    assert "img1.png: Written -> img1.txt" in response1 # FIXED
    mock_builtin_open.assert_called_with(expected_txt_path, 'w', encoding='utf-8')
    mock_builtin_open().write.assert_called_with("LLM Overwrite Resp")
    mock_builtin_open.reset_mock()
    mock_get_llm.reset_mock()
    mock_exists.reset_mock()

    # Scenario 2: File exists
    mock_exists.side_effect = [True]
    mock_get_llm.return_value = "LLM Overwrite Resp AGAIN"
    response2, _, _, _ = app_logic.chat_logic(ui_folder_path, ui_role_agent1, ui_user_input, ui_model_vision, ui_max_tokens, "Overwrite", ui_limiter, None, ui_use_ollama_options, False, mock_settings, mock_models_data, mock_limiters_data, None, mock_file_agents_dict, mock_history_list, mock_session_history)
    assert "img1.png: Overwritten -> img1.txt" in response2 # FIXED
    mock_builtin_open.assert_called_with(expected_txt_path, 'w', encoding='utf-8')
    mock_builtin_open().write.assert_called_with("LLM Overwrite Resp AGAIN")


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(RELEASE_MODEL_PATH)
@patch(GET_LLM_RESPONSE_PATH)
@patch(LOAD_ALL_ROLES_PATH)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
@patch(OS_PATH_ISDIR_PATH, return_value=True)
@patch(OS_LISTDIR_PATH, return_value=["img1.png"])
@patch(OS_PATH_EXISTS_PATH, return_value=True)
@patch(OS_PATH_SPLITEXT_PATH, side_effect=lambda p: _REAL_SPLITEXT(_REAL_BASENAME(p)))
@patch(OS_PATH_BASENAME_PATH, side_effect=os.path.basename) # ADDED
@patch(OS_PATH_JOIN_PATH, side_effect=lambda *args: os.sep.join(args)) # FIXED
@patch(OS_PATH_ISFILE_PATH, return_value=True) # ADDED
@patch(PIL_IMAGE_OPEN_PATH, return_value=MagicMock(spec=Image))
@patch(BUILTINS_OPEN_PATH, new_callable=mock_open, read_data="Original Content.")
def test_chat_logic_folder_processing_append(
    mock_builtin_open, mock_pil_open, mock_isfile, mock_join, mock_basename, mock_splitext, mock_exists, mock_listdir, mock_isdir, # ADDED mocks
    mock_get_actual_name, mock_load_roles, mock_get_llm, mock_release_model, mock_add_history, mock_time):
    """Test chat_logic with folder processing and 'Append' file handling."""
    mock_load_roles.return_value = mock_roles_data
    mock_get_llm.return_value = "LLM Append Resp"
    expected_txt_path = os.sep.join([ui_folder_path, "img1.txt"])

    response, _, _, _ = app_logic.chat_logic(ui_folder_path, ui_role_agent1, ui_user_input, ui_model_vision, ui_max_tokens, "Append", ui_limiter, None, ui_use_ollama_options, False, mock_settings, mock_models_data, mock_limiters_data, None, mock_file_agents_dict, mock_history_list, mock_session_history)

    assert "img1.png: Appended -> img1.txt" in response # FIXED
    assert mock_builtin_open.call_args_list == [
        call(expected_txt_path, 'r', encoding='utf-8'),
        call(expected_txt_path, 'w', encoding='utf-8'),
    ]
    mock_builtin_open().write.assert_called_once_with("Original Content.\n\n---\n\nLLM Append Resp")


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(RELEASE_MODEL_PATH)
@patch(GET_LLM_RESPONSE_PATH)
@patch(LOAD_ALL_ROLES_PATH)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
@patch(OS_PATH_ISDIR_PATH, return_value=True)
@patch(OS_LISTDIR_PATH, return_value=["img1.png"])
@patch(OS_PATH_EXISTS_PATH, return_value=True)
@patch(OS_PATH_SPLITEXT_PATH, side_effect=lambda p: _REAL_SPLITEXT(_REAL_BASENAME(p)))
@patch(OS_PATH_BASENAME_PATH, side_effect=os.path.basename) # ADDED
@patch(OS_PATH_JOIN_PATH, side_effect=lambda *args: os.sep.join(args)) # FIXED
@patch(OS_PATH_ISFILE_PATH, return_value=True) # ADDED
@patch(PIL_IMAGE_OPEN_PATH, return_value=MagicMock(spec=Image))
@patch(BUILTINS_OPEN_PATH, new_callable=mock_open, read_data="Original Content.") # Mock read for prepend
def test_chat_logic_folder_processing_prepend(
    mock_builtin_open, mock_pil_open, mock_isfile, mock_join, mock_basename, mock_splitext, mock_exists, mock_listdir, mock_isdir, # ADDED mocks
    mock_get_actual_name, mock_load_roles, mock_get_llm, mock_release_model, mock_add_history, mock_time):
    """Test chat_logic with folder processing and 'Prepend' file handling."""
    mock_load_roles.return_value = mock_roles_data
    mock_get_llm.return_value = "LLM Prepend Resp"
    expected_txt_path = os.sep.join([ui_folder_path, "img1.txt"])

    response, _, _, _ = app_logic.chat_logic(ui_folder_path, ui_role_agent1, ui_user_input, ui_model_vision, ui_max_tokens, "Prepend", ui_limiter, None, ui_use_ollama_options, False, mock_settings, mock_models_data, mock_limiters_data, None, mock_file_agents_dict, mock_history_list, mock_session_history)

    assert "img1.png: Prepended -> img1.txt" in response # FIXED
    # Prepend opens for read ('r'), then write ('w')
    expected_calls = [
        call(expected_txt_path, 'r', encoding='utf-8'),
        call(expected_txt_path, 'w', encoding='utf-8')
    ]
    mock_builtin_open.assert_has_calls(expected_calls, any_order=True) # Order depends on context manager use
    # Check the final write call content
    handle = mock_builtin_open()
    # Find the write call among potentially multiple calls to the handle
    write_call = next((c for c in handle.write.call_args_list if c.args[0] == "LLM Prepend Resp\n\n---\n\nOriginal Content."), None)
    assert write_call is not None


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(RELEASE_MODEL_PATH)
@patch(LOAD_ALL_ROLES_PATH)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
@patch(OS_PATH_ISDIR_PATH, return_value=True) # Folder exists
def test_chat_logic_folder_no_vision_model(
    mock_isdir, mock_get_actual_name, mock_load_roles, mock_release_model, mock_add_history, mock_time):
    """A folder with a non-vision model runs once as a text-only call."""
    mock_load_roles.return_value = mock_roles_data

    with patch(STREAM_LLM_PATH, side_effect=stream_returning("LLM Text Response (Folder Ignored)")) as mock_stream, \
         patch(GET_LLM_RESPONSE_PATH) as mock_get_llm:
        response, _, model_state, new_session_list = app_logic.chat_logic(
            ui_folder_path, ui_role_agent1, ui_user_input, ui_model_text,
            ui_max_tokens, "Skip", ui_limiter, None, ui_use_ollama_options, False,
            mock_settings, mock_models_data, mock_limiters_data,
            None, mock_file_agents_dict, mock_history_list, mock_session_history
        )

    assert response == "LLM Text Response (Folder Ignored)"
    assert model_state == "text-model"
    assert "Image: [None - Folder Ignored]" in new_session_list[-1]
    assert len(new_session_list) == 2
    mock_stream.assert_called_once()
    mock_get_llm.assert_not_called() # No per-image calls
    kwargs = mock_stream.call_args.kwargs
    assert kwargs['images'] is None
    assert "Image Context:" not in kwargs['prompt']


def test_chat_logic_error_model_not_found():
    """Test chat_logic when selected model isn't in models_data_state."""
    response, _, model_state, _ = app_logic.chat_logic(
        None, ui_role_agent1, ui_user_input, "non-existent-model", ui_max_tokens,
        ui_file_handling, ui_limiter, None, ui_use_ollama_options, False,
        mock_settings, mock_models_data, mock_limiters_data,
        None, mock_file_agents_dict, mock_history_list, mock_session_history
    )
    assert "Error: Selected model info not found for 'non-existent-model'." in response
    assert model_state is None # Model state should not be updated

@patch(PIL_IMAGE_FROMARRAY_PATH, side_effect=ValueError("Bad image data")) # Mock PIL error
def test_chat_logic_error_single_image_processing(mock_pil_fromarray):
    """Test chat_logic handling errors during single image processing."""
    response, _, model_state, _ = app_logic.chat_logic(
        None, ui_role_agent1, ui_user_input, ui_model_vision, ui_max_tokens,
        ui_file_handling, ui_limiter, ui_single_image_np, # Pass numpy image
        ui_use_ollama_options, False,
        mock_settings, mock_models_data, mock_limiters_data,
        None, mock_file_agents_dict, mock_history_list, mock_session_history
    )
    assert "Error processing single numpy image: Bad image data" in response
    assert model_state == "vision-model" # Model was found, error happened later

@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(OS_PATH_ISDIR_PATH, return_value=True)
@patch(OS_LISTDIR_PATH, return_value=["bad_image.png"])
@patch(OS_PATH_SPLITEXT_PATH, side_effect=lambda p: _REAL_SPLITEXT(_REAL_BASENAME(p)))
@patch(OS_PATH_BASENAME_PATH, side_effect=os.path.basename) # ADDED
@patch(OS_PATH_JOIN_PATH, side_effect=lambda *args: os.sep.join(args)) # FIXED
@patch(OS_PATH_ISFILE_PATH, return_value=True) # ADDED
@patch(PIL_IMAGE_OPEN_PATH, side_effect=IOError("Cannot open this image")) # Mock Image.open error
def test_chat_logic_error_folder_image_open(
    mock_pil_open, mock_isfile, mock_join, mock_basename, mock_splitext, mock_listdir, mock_isdir, mock_add_history): # ADDED mocks
    """Test chat_logic handling errors during Image.open in folder processing."""
    # Need to mock other dependencies even if not directly used in the error path
    with patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data), \
         patch(GET_LLM_RESPONSE_PATH, return_value="Should not be called"), \
         patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x):

        response, _, model_state, new_session_list = app_logic.chat_logic(
            ui_folder_path, ui_role_agent1, ui_user_input, ui_model_vision, ui_max_tokens,
            "Skip", ui_limiter, None, ui_use_ollama_options, False,
            mock_settings, mock_models_data, mock_limiters_data,
            None, mock_file_agents_dict, mock_history_list, mock_session_history
        )

    assert mock_add_history.call_count == 1 # Only one error entry added to history
    assert "ERROR: Cannot open this image" in new_session_list[-1] # Check session log
    assert response # Some message is returned to the UI


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(OS_PATH_ISDIR_PATH, return_value=True)
@patch(OS_LISTDIR_PATH, return_value=["bad_image.png"])
@patch(OS_PATH_ISFILE_PATH, return_value=True)
@patch(PIL_IMAGE_OPEN_PATH, side_effect=IOError("Cannot open this image"))
def test_chat_logic_folder_all_images_fail_reports_errors(mock_pil_open, mock_isfile, mock_listdir, mock_isdir, mock_add_history):
    """If every image fails to open, the UI response should still say which files failed and why."""
    with patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data), \
         patch(GET_LLM_RESPONSE_PATH, return_value="Should not be called"), \
         patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x):
        response, _, _, _ = app_logic.chat_logic(
            ui_folder_path, ui_role_agent1, ui_user_input, ui_model_vision, ui_max_tokens,
            "Skip", ui_limiter, None, ui_use_ollama_options, False,
            mock_settings, mock_models_data, mock_limiters_data,
            None, mock_file_agents_dict, mock_history_list, mock_session_history
        )
    assert "bad_image.png: Error - Cannot open this image" in response


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(OS_PATH_ISDIR_PATH, return_value=True)
@patch(OS_LISTDIR_PATH, return_value=["img1.png"])
@patch(OS_PATH_EXISTS_PATH, return_value=True)
@patch(OS_PATH_ISFILE_PATH, return_value=True)
@patch(PIL_IMAGE_OPEN_PATH, return_value=MagicMock(spec=Image))
def test_chat_logic_folder_skip_does_not_call_llm_for_existing_caption(mock_pil_open, mock_isfile, mock_exists, mock_listdir, mock_isdir, mock_add_history):
    """Skip mode should not spend a model call on an image whose caption file already exists."""
    with patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data), \
         patch(GET_LLM_RESPONSE_PATH, return_value="unused") as mock_get_llm, \
         patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x):
        app_logic.chat_logic(
            ui_folder_path, ui_role_agent1, ui_user_input, ui_model_vision, ui_max_tokens,
            "Skip", ui_limiter, None, ui_use_ollama_options, False,
            mock_settings, mock_models_data, mock_limiters_data,
            None, mock_file_agents_dict, mock_history_list, mock_session_history
        )
    mock_get_llm.assert_not_called()

@patch(OS_PATH_ISDIR_PATH, return_value=True)
@patch(OS_LISTDIR_PATH, side_effect=OSError("Permission denied listing folder")) # Mock listdir error
def test_chat_logic_error_folder_list(mock_listdir, mock_isdir):
    """Test chat_logic handling errors during os.listdir for folder processing."""
    response, _, model_state, _ = app_logic.chat_logic(
        ui_folder_path, ui_role_agent1, ui_user_input, ui_model_vision, ui_max_tokens,
        "Skip", ui_limiter, None, ui_use_ollama_options, False,
        mock_settings, mock_models_data, mock_limiters_data,
        None, mock_file_agents_dict, mock_history_list, mock_session_history
    )
    assert "Error listing folder: Permission denied listing folder" in response
    assert model_state == "vision-model" # Model was identified before listdir error


@patch(ADD_TO_HISTORY_PATH, side_effect=lambda h, e: (h or []) + [e])
@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data)
def test_iter_chat_logic_folder_yields_progress_per_image(mock_load_roles, mock_add_history, tmp_path, mock_time):
    """Folder mode reports progress before each image and writes real caption files."""
    for name in ("a.png", "b.png"):
        Image.new("RGB", (4, 4), "blue").save(tmp_path / name)
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")
    with patch(GET_LLM_RESPONSE_PATH, side_effect=["Caption A", "Caption B"]) as mock_get_llm:
        last, items = final(app_logic.iter_chat_logic(
            str(tmp_path), ui_role_agent1, "Describe", ui_model_vision, ui_max_tokens, "Overwrite", ui_limiter, None,
            False, False, mock_settings, mock_models_data, mock_limiters_data, None,
            mock_file_agents_dict, mock_history_list, mock_session_history))
    responses = [i[0] for i in items]
    assert responses[0].startswith("⏳ Processing image 1/2: a.png")
    assert responses[1].startswith("⏳ Processing image 2/2: b.png")
    assert "a.png: Written -> a.txt" in responses[1] # Earlier results stay visible
    assert last[0].startswith("Folder processing complete (2 processed, 0 skipped):")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "Caption A"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "Caption B"
    assert "Image Context: Analyzing 'b.png'" in mock_get_llm.call_args.kwargs["prompt"]
    assert mock_get_llm.call_args.kwargs["system"] == "You are acting as the agent 'Agent1'.\n\nDoes agent things."
    assert last[4] is None # Folder runs don't start a refine conversation


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data)
def test_iter_chat_logic_folder_with_error_response_skips_write(mock_load_roles, mock_add_history, tmp_path, mock_time):
    Image.new("RGB", (4, 4)).save(tmp_path / "a.png")
    with patch(GET_LLM_RESPONSE_PATH, return_value="⚠️ Error: model crashed"):
        response, *_ = app_logic.chat_logic(
            str(tmp_path), ui_role_agent1, "Describe", ui_model_vision, ui_max_tokens, "Overwrite", ui_limiter, None,
            False, False, mock_settings, mock_models_data, mock_limiters_data, None,
            mock_file_agents_dict, mock_history_list, mock_session_history)
    assert "a.png: Skipped (Empty/Error Response)" in response
    assert not (tmp_path / "a.txt").exists()


# --- Tests for Team Editor Callbacks ---

def test_load_team_for_editing_success():
    """Test loading an existing team into the editor."""
    name, desc, steps_json, strategy, editor_state = app_logic.load_team_for_editing(
        "TeamB", mock_all_teams_state
    )
    assert name == "TeamB"
    assert desc == "Test Team B"
    assert steps_json == mock_teams_data["TeamB"]["steps"]
    assert strategy == "refine_last"
    assert editor_state["name"] == "TeamB"
    assert editor_state["steps"] == mock_teams_data["TeamB"]["steps"]

def test_load_team_for_editing_not_found():
    """Test loading a non-existent team."""
    name, desc, steps_json, strategy, editor_state = app_logic.load_team_for_editing(
        "NonExistentTeam", mock_all_teams_state
    )
    assert name == "NonExistentTeam"
    assert "Error: Team data not found" in desc
    assert steps_json == []
    assert strategy == "concatenate" # Default
    assert "Error: Team data not found" in editor_state["description"]

def test_load_team_for_editing_no_selection():
    """Test loading with no team selected."""
    name, desc, steps_json, strategy, editor_state = app_logic.load_team_for_editing(
        None, mock_all_teams_state
    )
    assert name == ""
    assert desc == ""
    assert steps_json == []
    assert strategy == "concatenate"
    assert editor_state == mock_empty_editor_state

def test_clear_team_editor():
    """Test clearing the team editor fields."""
    name, desc, steps_json, strategy, editor_state = app_logic.clear_team_editor()
    assert name == ""
    assert desc == ""
    assert steps_json == []
    assert strategy == "concatenate"
    assert editor_state == mock_empty_editor_state

@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda name: name.replace("[File] ", "")) # Mock name cleaning
def test_add_step_to_editor_success(mock_get_name):
    """Test adding a new step to the editor state."""
    editor_state_in = copy.deepcopy(mock_editor_state_team_a) # Start with Team A state
    new_agent_display = "Agent2"

    new_state, steps_json = app_logic.add_step_to_editor(new_agent_display, editor_state_in)

    assert len(new_state["steps"]) == 2
    assert new_state["steps"][0] == {"role": "Agent1"}
    assert new_state["steps"][1] == {"role": "Agent2"} # Uses actual name
    assert steps_json == new_state["steps"]
    mock_get_name.assert_called_once_with(new_agent_display)


def test_add_step_to_editor_does_not_mutate_input_state():
    """Adding a step in the editor must not change the team definition until 'Save' is pressed."""
    teams_state = copy.deepcopy(mock_all_teams_state)
    *_, editor_state = app_logic.load_team_for_editing("TeamA", teams_state)
    app_logic.add_step_to_editor("Agent2", editor_state)
    assert teams_state["TeamA"]["steps"] == [{"role": "Agent1"}] # Unsaved edit leaked into teams state


@patch(GR_WARNING_PATH)
def test_add_step_to_editor_no_selection(mock_warning):
    """Test adding a step when no agent is selected."""
    editor_state_in = copy.deepcopy(mock_editor_state_team_a)
    new_state, steps_update = app_logic.add_step_to_editor(None, editor_state_in)
    assert new_state == editor_state_in # State should not change
    assert steps_update == gr.update() # No-op Gradio update
    mock_warning.assert_called_once()

def test_remove_step_from_editor_success():
    """Test removing an existing step by index."""
    # Start with Team B state which has 2 steps (roles Agent1, Agent2)
    editor_state_in = {
        "name": "TeamB", "description": "Test Team B",
        "steps": [{"role":"Agent1"}, {"role":"Agent2"}], "assembly_strategy": "refine_last"
    }
    index_to_remove = 1 # Remove the first step (Agent1)

    new_state, steps_json = app_logic.remove_step_from_editor(index_to_remove, editor_state_in)

    assert len(new_state["steps"]) == 1
    assert new_state["steps"][0] == {"role": "Agent2"} # Only Agent2 should remain
    assert steps_json == new_state["steps"]

@patch(GR_WARNING_PATH)
def test_remove_step_from_editor_invalid_index(mock_warning):
    """Test removing a step with an invalid index."""
    editor_state_in = copy.deepcopy(mock_editor_state_team_a) # Has 1 step
    # Try removing step 0 or step 2
    new_state_0, steps_update_0 = app_logic.remove_step_from_editor(0, editor_state_in)
    assert new_state_0 == editor_state_in # Unchanged
    assert isinstance(steps_update_0, dict) # Check it's a dict (Gradio update) # FIXED

    new_state_2, steps_update_2 = app_logic.remove_step_from_editor(2, editor_state_in)
    assert new_state_2 == editor_state_in # Unchanged
    assert isinstance(steps_update_2, dict) # Check it's a dict (Gradio update) # FIXED
    assert mock_warning.call_count == 2

@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data) # Mock role loading needed for dropdown update logic
@patch(SAVE_TEAMS_PATH, return_value=True) # Mock successful save
def test_save_team_from_editor_success_new(mock_save, mock_load_roles):
    """Test saving a new team definition."""
    editor_state_in = copy.deepcopy(mock_empty_editor_state)
    editor_state_in["steps"] = [{"role": "Agent1"}] # Add a step
    new_name = "NewTeam"
    new_desc = "A brand new team."
    new_strat = "concatenate"

    # Simulate necessary state inputs needed by the function signature
    current_teams_state = copy.deepcopy(mock_all_teams_state)
    current_settings_state = mock_settings.copy()
    current_file_agents_state = {}

    teams_state_out, editor_dd_update, chat_dd_update, caption_dd_update, status_msg = app_logic.save_team_from_editor(
        new_name, new_desc, new_strat, editor_state_in,
        current_teams_state, current_settings_state, current_file_agents_state
    )

    assert "saved successfully" in status_msg
    assert new_name in teams_state_out # New team added to state
    assert teams_state_out[new_name]["description"] == new_desc
    assert teams_state_out[new_name]["steps"] == [{"role": "Agent1"}]
    assert teams_state_out[new_name]["assembly_strategy"] == new_strat
    mock_save.assert_called_once_with(teams_state_out) # Check save call
    # Check Gradio updates (check specific relevant keys)
    assert isinstance(editor_dd_update, dict) # Check it's a dict (Gradio update)
    assert editor_dd_update.get('choices') is not None
    assert new_name in editor_dd_update['choices']
    assert editor_dd_update.get('value') == new_name
    assert isinstance(chat_dd_update, dict) # Check it's a dict (Gradio update)
    assert chat_dd_update.get('choices') is not None
    assert f"[Team] {new_name}" in chat_dd_update['choices']
    assert chat_dd_update.get('value') == f"[Team] {new_name}"
    assert caption_dd_update == chat_dd_update # Captions tab gets the same agent/team choices

@patch(GR_WARNING_PATH)
@patch(SAVE_TEAMS_PATH, return_value=False) # Mock failed save
def test_save_team_from_editor_save_fail(mock_save, mock_warning):
    """Test handling failure during team saving."""
    editor_state_in = copy.deepcopy(mock_editor_state_team_a)
    current_teams_state = copy.deepcopy(mock_all_teams_state)
    current_settings_state = mock_settings.copy()
    current_file_agents_state = {}

    teams_state_out, editor_dd_update, chat_dd_update, caption_dd_update, status_msg = app_logic.save_team_from_editor(
        "TeamA", "Desc", "concat", editor_state_in,
        current_teams_state, current_settings_state, current_file_agents_state
    )

    assert "Error: Failed to save team data" in status_msg
    mock_warning.assert_called_once_with(status_msg) # Shown as a toast (gr.Error() without raise was a no-op)
    assert teams_state_out == current_teams_state # State should not change on save fail
    assert isinstance(editor_dd_update, dict) # Check it's a dict (Gradio update) # FIXED
    assert editor_dd_update.get("choices") is None # No dropdown update
    assert isinstance(chat_dd_update, dict) # Check it's a dict (Gradio update) # FIXED
    assert chat_dd_update.get("choices") is None # No dropdown update
    assert caption_dd_update.get("choices") is None

@patch(SAVE_TEAMS_PATH)
@patch(GR_WARNING_PATH)
def test_save_team_from_editor_empty_name(mock_warning, mock_save):
    """Test saving with an empty team name."""
    editor_state_in = copy.deepcopy(mock_editor_state_team_a)
    current_teams_state = copy.deepcopy(mock_all_teams_state)
    current_settings_state = mock_settings.copy()
    current_file_agents_state = {}

    teams_state_out, editor_dd_update, chat_dd_update, caption_dd_update, status_msg = app_logic.save_team_from_editor(
        "  ", "Desc", "concat", editor_state_in, # Empty name
        current_teams_state, current_settings_state, current_file_agents_state
    )
    assert "Team Name cannot be empty" in status_msg
    assert teams_state_out == current_teams_state # State unchanged
    mock_save.assert_not_called()
    mock_warning.assert_called_once()

@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data) # Mock role loading needed for dropdown update logic
@patch(SAVE_TEAMS_PATH, return_value=True) # Mock successful save
def test_delete_team_logic_success(mock_save, mock_load_roles):
    """Test deleting an existing team."""
    team_to_delete = "TeamA"
    current_teams_state = copy.deepcopy(mock_all_teams_state)
    current_settings_state = mock_settings.copy()
    current_file_agents_state = {}

    # Returns 10 values: teams_state, editor_dd, chat_dd, caption_dd, status, + 5 editor-clear outputs
    outputs = app_logic.delete_team_logic(
        team_to_delete, current_teams_state,
        current_settings_state, current_file_agents_state
    )
    assert len(outputs) == 10
    teams_state_out, editor_dd_update, chat_dd_update, caption_dd_update, status_msg = outputs[0:5]
    clear_outputs = outputs[5:] # This is a list of 5 elements

    assert f"Team '{team_to_delete}' deleted successfully" in status_msg
    assert team_to_delete not in teams_state_out # Team removed from state
    assert "TeamB" in teams_state_out # Other teams remain
    mock_save.assert_called_once_with(teams_state_out) # Check save call with updated data
    # Check Gradio updates (check specific relevant keys)
    assert isinstance(editor_dd_update, dict) # Check it's a dict (Gradio update)
    assert editor_dd_update.get("choices") is not None
    assert team_to_delete not in editor_dd_update["choices"]
    assert editor_dd_update.get("value") is None # Value cleared
    assert isinstance(chat_dd_update, dict) # Check it's a dict (Gradio update)
    assert chat_dd_update.get("choices") is not None
    assert f"[Team] {team_to_delete}" not in chat_dd_update["choices"]
    assert chat_dd_update.get("value") == "(Direct Agent Call)" # Value reset
    assert caption_dd_update == chat_dd_update
    # Check clear outputs (match return of clear_team_editor)
    name_clr, desc_clr, steps_clr, strat_clr, state_clr = app_logic.clear_team_editor()
    # Compare elements individually # FIXED
    assert len(clear_outputs) == 5
    assert clear_outputs[0] == name_clr     # ""
    assert clear_outputs[1] == desc_clr     # ""
    assert clear_outputs[2] == steps_clr    # []
    assert clear_outputs[3] == strat_clr    # "concatenate"
    assert clear_outputs[4] == state_clr    # mock_empty_editor_state


@patch(GR_WARNING_PATH)
def test_delete_team_logic_not_found(mock_warning):
    """Test deleting a team that doesn't exist."""
    team_to_delete = "NonExistent"
    current_teams_state = copy.deepcopy(mock_all_teams_state)
    current_settings_state = mock_settings.copy()
    current_file_agents_state = {}

    outputs = app_logic.delete_team_logic(team_to_delete, current_teams_state, current_settings_state, current_file_agents_state)
    assert len(outputs) == 10
    status_msg = outputs[4]
    assert f"Team '{team_to_delete}' not found" in status_msg
    assert outputs[0] == current_teams_state # Nothing deleted

@patch(GR_WARNING_PATH)
def test_delete_team_logic_no_selection(mock_warning):
    """Test deleting with no team selected."""
    current_teams_state = copy.deepcopy(mock_all_teams_state)
    current_settings_state = mock_settings.copy()
    current_file_agents_state = {}

    outputs = app_logic.delete_team_logic(None, current_teams_state, current_settings_state, current_file_agents_state)
    assert len(outputs) == 10
    status_msg = outputs[4]
    assert "No team selected" in status_msg


# --- Tests for Settings Callbacks ---

def test_load_profile_options_callback_success():
    """Test loading a valid profile onto UI elements."""
    profile_name = "Fast"
    # Keys sorted alphabetically based on mock_profiles_data["Fast"]
    ordered_keys = sorted(mock_profiles_data[profile_name].keys())
    # Expected values in the same sorted order
    expected_values = [mock_profiles_data[profile_name][k] for k in ordered_keys]

    updates = app_logic.load_profile_options_callback(
        profile_name, mock_profiles_data, ordered_keys
    )

    assert isinstance(updates, list)
    assert len(updates) == len(ordered_keys)
    for i, update in enumerate(updates):
        assert isinstance(update, dict) # Check it's a dict (Gradio update) # FIXED
        assert update.get("value") == expected_values[i]

def test_load_profile_options_callback_partial_match():
    """Test loading profile where keys don't fully match expected UI keys."""
    profile_name = "Creative" # Defines temp, top_p, mirostat
    # Simulate UI having keys for temp, top_p, top_k (but profile doesn't have top_k)
    ordered_keys_ui = ["mirostat", "temperature", "top_k", "top_p"]

    updates = app_logic.load_profile_options_callback(
        profile_name, mock_profiles_data, ordered_keys_ui
    )

    assert len(updates) == 4
    # Check values loaded from profile
    assert updates[0].get("value") == mock_profiles_data[profile_name]["mirostat"]
    assert updates[1].get("value") == mock_profiles_data[profile_name]["temperature"]
    assert updates[3].get("value") == mock_profiles_data[profile_name]["top_p"]
    # Check value NOT loaded from profile (key mismatch) -> should be no-update
    assert updates[2].get("value") is None # top_k wasn't in Creative profile


def test_load_profile_options_callback_invalid_profile():
    """Test loading a profile name that doesn't exist."""
    ordered_keys_ui = ["temperature", "top_k"]
    updates = app_logic.load_profile_options_callback(
        "NonExistentProfile", mock_profiles_data, ordered_keys_ui
    )
    assert len(updates) == 2
    assert updates[0].get("value") is None # No update
    assert updates[1].get("value") is None # No update

@patch(JSON_DUMP_PATH) # Mock json.dump
@patch(BUILTINS_OPEN_PATH, new_callable=mock_open) # Mock file open
@patch(GET_ABSOLUTE_PATH, return_value="/fake/path/settings.json") # Mock path resolution
@patch(LOAD_JSON_PATH, return_value=mock_loaded_settings_from_file) # Mock loading existing settings
def test_save_settings_callback_success(mock_load, mock_abs_path, mock_file_open, mock_json_dump):
    """Test successfully saving settings with type conversions."""
    # Simulate UI inputs matching the order of mock_api_option_keys
    # ['num_ctx', 'seed', 'temperature', 'top_k', 'use_mmap']
    # Note: UI might provide strings or different types, test conversion based on original file type
    ui_api_option_values = (
        "4096",     # num_ctx (originally int) -> int
        "999",      # seed (originally int) -> int
        "0.65",     # temperature (originally float) -> float
        "50",       # top_k (originally int) -> int
        True        # use_mmap (originally bool) -> bool
    )
    ui_general_settings = (
        "http://new-url", # ollama_url_in
        8192,             # max_tokens_slider_range_in
        False,            # api_to_console_in
        True,             # use_default_in
        True,             # use_custom_in
        True,             # use_ollama_opts_default_in
        False,            # release_model_default_in
        "Glass"           # theme_select_in
    )

    status = app_logic.save_settings_callback(
        *ui_general_settings, mock_api_option_keys, *ui_api_option_values
    )

    assert "Settings saved successfully" in status
    assert "Restart application to apply theme change" in status # Theme changed

    # Check load was called twice: once for current, once for initial types inside save logic
    assert mock_load.call_count == 2
    mock_load.assert_called_with("/fake/path/settings.json", is_relative=False)

    mock_abs_path.assert_called_once_with(app_logic.SETTINGS_FILE)
    mock_file_open.assert_called_once_with("/fake/path/settings.json", 'w', encoding='utf-8')
    mock_json_dump.assert_called_once()

    # Check the data passed to json.dump
    saved_data = mock_json_dump.call_args[0][0]
    assert saved_data["ollama_url"] == "http://new-url"
    assert saved_data["max_tokens_slider"] == 8192
    assert saved_data["ollama_api_prompt_to_console"] is False
    assert saved_data["using_default_agents"] is True
    assert saved_data["using_custom_agents"] is True
    assert saved_data["use_ollama_api_options"] is True
    assert saved_data["release_model_on_change"] is False
    assert saved_data["gradio_theme"] == "Glass"

    # Check API options with type conversion
    saved_api_opts = saved_data["ollama_api_options"]
    assert saved_api_opts["num_ctx"] == 4096 # converted to int
    assert saved_api_opts["seed"] == 999 # converted to int
    assert saved_api_opts["temperature"] == 0.65 # converted to float
    assert saved_api_opts["top_k"] == 50 # converted to int
    assert saved_api_opts["use_mmap"] is True # converted to bool

# --- Tests for Other UI Callbacks ---

@patch(LOAD_JSON_PATH)
def test_update_max_tokens_on_limiter_change(mock_load_json):
    """Test max token slider update based on limiter."""
    mock_limiters = {"TestLimit": {"limiter_token_slider": 500}}
    mock_load_json.return_value = mock_limiters

    # Test update
    update = app_logic.update_max_tokens_on_limiter_change("TestLimit", 1000)
    assert isinstance(update, dict) # Check it's a dict (Gradio update) # FIXED
    assert update.get("value") == 500

    # Test no update if value matches
    update = app_logic.update_max_tokens_on_limiter_change("TestLimit", 500)
    assert isinstance(update, dict) # Check it's a dict (Gradio update) # FIXED
    assert update.get("value") is None # No change update

    # Test 'Off'
    update = app_logic.update_max_tokens_on_limiter_change("Off", 1000)
    assert isinstance(update, dict) # Check it's a dict (Gradio update) # FIXED
    assert update.get("value") is None # No change update

    # Test limiter not found
    update = app_logic.update_max_tokens_on_limiter_change("NotFound", 1000)
    assert isinstance(update, dict) # Check it's a dict (Gradio update) # FIXED
    assert update.get("value") is None # No change update

    # Test limiter has no token value
    mock_limiters = {"NoToken": {"limiter_prompt_format": "fmt"}}
    mock_load_json.return_value = mock_limiters
    update = app_logic.update_max_tokens_on_limiter_change("NoToken", 1000)
    assert isinstance(update, dict) # Check it's a dict (Gradio update) # FIXED
    assert update.get("value") is None # No change update


def test_clear_session_history_callback():
    """Test clearing session history."""
    display, new_state = app_logic.clear_session_history_callback(["entry1", "entry2"])
    assert display == ""
    assert new_state == []


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_handle_agent_file_upload_success(tmp_path):
    """Gradio 4+ passes the upload as a filepath string."""
    path = _write(tmp_path, "uploaded_agents.json", '{"AgentFromFile": {"description": "File Agent Desc"}}')

    result = app_logic.handle_agent_file_upload(path)

    assert len(result) == 2 # Matches the two outputs wired in app.py
    loaded_dict, filename = result
    assert filename == "uploaded_agents.json" # Should return basename
    assert loaded_dict == {"AgentFromFile": {"description": "File Agent Desc"}}


def test_handle_agent_file_upload_legacy_file_object(tmp_path):
    """Older Gradio passed a tempfile wrapper with a .name attribute; still supported."""
    path = _write(tmp_path, "legacy.json", '{"AgentFromFile": {"description": "File Agent Desc"}}')
    legacy_upload = SimpleNamespace(name=path) # (A MagicMock would look os.PathLike via __fspath__)

    loaded_dict, filename = app_logic.handle_agent_file_upload(legacy_upload)

    assert filename == "legacy.json"
    assert "AgentFromFile" in loaded_dict


def test_handle_agent_file_upload_none():
    """Test clearing the agent file upload."""
    assert app_logic.handle_agent_file_upload(None) == ({}, None)


@patch(GR_WARNING_PATH)
def test_handle_agent_file_upload_invalid_json(mock_warning, tmp_path):
    """Test uploading a file with invalid JSON content."""
    path = _write(tmp_path, "invalid.json", "invalid json")
    assert app_logic.handle_agent_file_upload(path) == ({}, None)
    assert "Invalid Agent File" in mock_warning.call_args.args[0]


@patch(GR_WARNING_PATH)
def test_handle_agent_file_upload_not_a_dict(mock_warning, tmp_path):
    """A JSON list (or other non-object) is rejected."""
    path = _write(tmp_path, "list.json", '[{"description": "x"}]')
    assert app_logic.handle_agent_file_upload(path) == ({}, None)
    assert "not a dict" in mock_warning.call_args.args[0]


@patch(GR_WARNING_PATH)
def test_handle_agent_file_upload_invalid_agent_format(mock_warning, tmp_path):
    """Valid JSON but no agent has a string 'description' -> nothing loaded."""
    path = _write(tmp_path, "bad_agents.json", '{"BadAgent": "just a string", "NoDesc": {"name": "x"}, "NumDesc": {"description": 5}}')
    assert app_logic.handle_agent_file_upload(path) == ({}, None)
    assert "No valid agent definitions" in mock_warning.call_args.args[0]


def test_handle_agent_file_upload_skips_invalid_entries(tmp_path):
    """Valid agents are kept while malformed entries are dropped."""
    path = _write(tmp_path, "mixed.json", '{"Good": {"description": "ok"}, "Bad": "nope"}')
    loaded_dict, filename = app_logic.handle_agent_file_upload(path)
    assert loaded_dict == {"Good": {"description": "ok"}}
    assert filename == "mixed.json"


@patch(GR_WARNING_PATH)
def test_handle_agent_file_upload_oversize(mock_warning, tmp_path):
    """Files over MAX_AGENT_FILE_BYTES are refused before being read/parsed."""
    big_description = "x" * (app_logic.MAX_AGENT_FILE_BYTES + 10)
    path = _write(tmp_path, "huge.json", json.dumps({"Agent": {"description": big_description}}))
    with patch(BUILTINS_OPEN_PATH, new_callable=mock_open) as mock_file:
        assert app_logic.handle_agent_file_upload(path) == ({}, None)
        mock_file.assert_not_called()
    assert "larger than" in mock_warning.call_args.args[0]


# --- Tests for History Callbacks ---

def test_show_clear_confirmation():
    update = app_logic.show_clear_confirmation()
    assert isinstance(update, dict) # Check it's a dict (Gradio update) # FIXED
    assert update.get("visible") is True

def test_hide_clear_confirmation():
    update = app_logic.hide_clear_confirmation()
    assert isinstance(update, dict) # Check it's a dict (Gradio update) # FIXED
    assert update.get("visible") is False

@patch(SAVE_HISTORY_PATH) # Mock the save function in history_manager
def test_clear_full_history_callback(mock_save_hist):
    """Test clearing the full persistent history."""
    initial_history = ["old_entry1", "old_entry2"]
    # Function returns 3 values: display_text, hide_group_update, new_history_state
    display, update_group, new_state = app_logic.clear_full_history_callback(initial_history)

    assert display == "" # Clears display box
    assert isinstance(update_group, dict) # Check it's a dict (Gradio update) # FIXED
    assert update_group.get("visible") is False # Check specific key
    assert new_state == [] # Returns empty list for state
    mock_save_hist.assert_called_once_with([]) # Ensures empty list was saved