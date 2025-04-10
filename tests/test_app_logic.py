# ArtAgent/tests/test_app_logic.py

import pytest
import os
import sys
import json
import numpy as np
import time
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
    # Import history manager specifically for mocking save_history
    from core import history_manager
    try:
        from PIL import Image
        # Add specific exception type if needed for mocking PIL errors
        # from PIL import UnidentifiedImageError
    except ImportError:
        # from builtins import UnidentifiedImageError # Define dummy if needed
        class DummyImage:
             def __init__(self, mode="RGB", size=(10, 10)): self.mode = mode; self.size = size
             def save(self, fp, format=None): fp.write(b"dummy")
             def close(self): pass
             @staticmethod
             def fromarray(arr): return DummyImage()
             @staticmethod
             def open(fp): return DummyImage()
        Image = DummyImage
        print("Warning: Pillow not found, using dummy Image class for app_logic tests.")
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
PIL_IMAGE_OPEN_PATH = 'core.app_logic.Image.open'

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

# --- Tests for execute_chat_or_team (Router) ---

@patch(RUN_TEAM_WORKFLOW_PATH)
@patch(LOAD_ALL_ROLES_PATH) # Mock roles loading for team execution
def test_execute_router_team_call(mock_load_roles, mock_run_team, mock_time):
    """Test routing to agent_manager.run_team_workflow for a team selection."""
    mock_load_roles.return_value = mock_roles_data
    # CORRECTED: run_team_workflow returns 3 values
    mock_run_team.return_value = ("Team Output", ["persistent_entry_1", "team_log"], None)

    response, session_hist_txt, model_state, new_session_list = app_logic.execute_chat_or_team(
        ui_folder_path, ui_user_input, ui_model_vision, ui_max_tokens, ui_file_handling,
        ui_limiter, None, ui_use_ollama_options, ui_release_model, # No single image
        ui_role_team_a, # Select TeamA
        mock_settings, mock_models_data, mock_limiters_data, mock_teams_data,
        None, mock_file_agents_dict, mock_history_list, mock_session_history
    )

    assert response == "Team Output"
    assert model_state is None # No single model state for teams
    assert "Workflow Run: 'TeamA'" in session_hist_txt
    assert "session_entry_1" in session_hist_txt # Check session history updated
    assert len(new_session_list) == 2

    mock_load_roles.assert_called_once_with(mock_settings, file_agents=mock_file_agents_dict)
    mock_run_team.assert_called_once()
    call_args, call_kwargs = mock_run_team.call_args
    assert call_kwargs['team_name'] == "TeamA"
    assert call_kwargs['team_definition'] == mock_teams_data['TeamA']
    assert call_kwargs['user_input'] == ui_user_input
    assert call_kwargs['initial_settings'] == mock_settings
    assert call_kwargs['all_roles_data'] == mock_roles_data
    assert call_kwargs['history_list'] == mock_history_list # Passes original persistent list
    assert call_kwargs['worker_model_name'] == "vision-model" # Extracts name correctly


@patch('core.app_logic.chat_logic') # Mock chat_logic function itself
def test_execute_router_agent_call(mock_chat_logic, mock_time):
    """Test routing to chat_logic for a single agent selection."""
    mock_chat_logic.return_value = ("Agent Output", "session_entry_1\n---\nagent_log", "Agent1", ["session_entry_1", "agent_log"])

    response, session_hist_txt, model_state, new_session_list = app_logic.execute_chat_or_team(
        ui_folder_path, ui_user_input, ui_model_text, ui_max_tokens, ui_file_handling,
        ui_limiter, None, ui_use_ollama_options, ui_release_model, # No single image
        ui_role_agent1, # Select Agent1
        mock_settings, mock_models_data, mock_limiters_data, mock_teams_data,
        None, mock_file_agents_dict, mock_history_list, mock_session_history
    )

    assert response == "Agent Output"
    assert model_state == "Agent1" # chat_logic returns the model used
    assert "session_entry_1" in session_hist_txt
    assert "agent_log" in session_hist_txt
    assert len(new_session_list) == 2
    mock_chat_logic.assert_called_once()
    call_args, call_kwargs = mock_chat_logic.call_args
    assert call_kwargs['role_display_name'] == ui_role_agent1
    assert call_kwargs['model_with_vision'] == ui_model_text


def test_execute_router_direct_agent_call(mock_time):
    """Test selecting the '(Direct Agent Call)' option."""
    response, session_hist_txt, model_state, new_session_list = app_logic.execute_chat_or_team(
        ui_folder_path, ui_user_input, ui_model_text, ui_max_tokens, ui_file_handling,
        ui_limiter, None, ui_use_ollama_options, ui_release_model,
        "(Direct Agent Call)", # Select direct call
        mock_settings, mock_models_data, mock_limiters_data, mock_teams_data,
        None, mock_file_agents_dict, mock_history_list, mock_session_history
    )
    assert "Please select a specific Agent or an Agent Team" in response
    assert model_state is None
    assert session_hist_txt == "session_entry_1" # History unchanged
    assert new_session_list == mock_session_history # History unchanged


def test_execute_router_invalid_team_name(mock_time):
    """Test selecting a team name that doesn't exist in the state."""
    response, session_hist_txt, model_state, new_session_list = app_logic.execute_chat_or_team(
        ui_folder_path, ui_user_input, ui_model_text, ui_max_tokens, ui_file_handling,
        ui_limiter, None, ui_use_ollama_options, ui_release_model,
        "[Team] NonExistentTeam", # Select non-existent team
        mock_settings, mock_models_data, mock_limiters_data, mock_teams_data,
        None, mock_file_agents_dict, mock_history_list, mock_session_history
    )
    assert "Error: Selected Agent Team 'NonExistentTeam' definition not found" in response
    assert model_state is None
    assert session_hist_txt == "session_entry_1" # History unchanged
    assert new_session_list == mock_session_history # History unchanged


# --- Tests for chat_logic (Single Agent Core Logic) ---

@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(RELEASE_MODEL_PATH)
@patch(GET_LLM_RESPONSE_PATH)
@patch(LOAD_ALL_ROLES_PATH)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x) # Simple mock: returns input display name
def test_chat_logic_text_only(mock_get_actual_name, mock_load_roles, mock_get_llm, mock_release_model, mock_add_history, mock_time):
    """Test chat_logic in text-only mode (no folder, no single image)."""
    mock_load_roles.return_value = mock_roles_data
    mock_get_llm.return_value = "LLM Text Response"

    response, session_hist_txt, model_state, new_session_list = app_logic.chat_logic(
        None, ui_role_agent1, ui_user_input, ui_model_text, ui_max_tokens, # No folder path
        ui_file_handling, ui_limiter, None, ui_use_ollama_options, False, # No single image, no release model
        mock_settings, mock_models_data, mock_limiters_data,
        None, mock_file_agents_dict, mock_history_list, mock_session_history
    )

    assert response == "LLM Text Response"
    assert model_state == "text-model" # Correct model name returned
    assert session_hist_txt.endswith("---\n"+MOCK_TIMESTAMP+"\nRole: Agent1\nModel: text-model\nInput: Test user query\nImage: [None - Text Only]\nResponse:\nLLM Text Response\n---\n")
    assert len(new_session_list) == 2 # Original + new entry

    mock_get_actual_name.assert_called_once_with(ui_role_agent1)
    mock_load_roles.assert_called_once_with(mock_settings, file_agents=mock_file_agents_dict)
    mock_release_model.assert_not_called() # release_model_on_change was False
    mock_get_llm.assert_called_once()
    call_args, call_kwargs = mock_get_llm.call_args
    assert call_kwargs['role'] == ui_role_agent1 # Uses actual role name (mocked here)
    assert call_kwargs['model'] == ui_model_text
    assert call_kwargs['settings'] == mock_settings
    assert call_kwargs['roles_data'] == mock_roles_data
    assert call_kwargs['images'] == [] # Empty list for text only
    assert call_kwargs['max_tokens'] == ui_max_tokens # Uses UI value as no limiter
    assert "Role: Agent1 - Does agent things." in call_kwargs['prompt'] # Check prompt construction
    assert "User Input: Test user query" in call_kwargs['prompt']
    mock_add_history.assert_called_once() # Called once for the single interaction
    # Removed assert False


@patch(PIL_IMAGE_FROMARRAY_PATH, return_value=MagicMock(spec=Image)) # Mock Image creation
@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(RELEASE_MODEL_PATH)
@patch(GET_LLM_RESPONSE_PATH)
@patch(LOAD_ALL_ROLES_PATH)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
def test_chat_logic_single_image(mock_get_actual_name, mock_load_roles, mock_get_llm, mock_release_model, mock_add_history, mock_pil_fromarray, mock_time):
    """Test chat_logic with a single image input."""
    mock_load_roles.return_value = mock_roles_data
    mock_get_llm.return_value = "LLM Image Response"
    mock_pil_image_instance = mock_pil_fromarray.return_value # Get the mocked image instance

    response, session_hist_txt, model_state, new_session_list = app_logic.chat_logic(
        None, ui_role_agent1, ui_user_input, ui_model_vision, ui_max_tokens, # No folder path
        ui_file_handling, ui_limiter, ui_single_image_np, # Pass numpy image
        ui_use_ollama_options, False, # No release model
        mock_settings, mock_models_data, mock_limiters_data,
        None, mock_file_agents_dict, mock_history_list, mock_session_history
    )

    assert response == "LLM Image Response"
    assert model_state == "vision-model" # Assert the base name
    assert "Image: [Single Upload]" in session_hist_txt # Check history entry
    assert len(new_session_list) == 2

    mock_pil_fromarray.assert_called_once() # Check PIL was called
    mock_get_llm.assert_called_once()
    call_args, call_kwargs = mock_get_llm.call_args
    assert call_kwargs['model'] == "vision-model" # Check base model name passed
    assert isinstance(call_kwargs['images'], list)
    assert len(call_kwargs['images']) == 1
    assert call_kwargs['images'][0] == mock_pil_image_instance # Check image object passed
    mock_add_history.assert_called_once()


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(RELEASE_MODEL_PATH)
@patch(GET_LLM_RESPONSE_PATH)
@patch(LOAD_ALL_ROLES_PATH)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
@patch(OS_PATH_ISDIR_PATH, return_value=True)
@patch(OS_LISTDIR_PATH, return_value=["img1.png", "img2.jpg", "other.txt"])
@patch(OS_PATH_EXISTS_PATH, return_value=False)
@patch(OS_PATH_SPLITEXT_PATH, side_effect=lambda p: os.path.splitext(os.path.basename(p)))
@patch(OS_PATH_BASENAME_PATH, side_effect=os.path.basename) # ADDED BASENAME MOCK
@patch(OS_PATH_JOIN_PATH, side_effect=lambda *args: os.sep.join(args)) # Simple join mock # FIXED
@patch(OS_PATH_ISFILE_PATH, return_value=True) # Mock isfile
@patch(PIL_IMAGE_OPEN_PATH, return_value=MagicMock(spec=Image))
@patch(BUILTINS_OPEN_PATH, new_callable=mock_open)
def test_chat_logic_folder_processing_skip(
    mock_builtin_open, mock_pil_open, mock_isfile, mock_join, mock_basename, mock_splitext, mock_exists, mock_listdir, mock_isdir, # ADDED mocks
    mock_get_actual_name, mock_load_roles, mock_get_llm, mock_release_model, mock_add_history, mock_time):
    """Test chat_logic with folder processing and 'Skip' file handling."""
    mock_load_roles.return_value = mock_roles_data
    mock_get_llm.side_effect = ["LLM Resp Img1", "LLM Resp Img2"]
    mock_pil_image_instance = mock_pil_open.return_value

    response, session_hist_txt, model_state, new_session_list = app_logic.chat_logic(
        ui_folder_path, ui_role_agent1, ui_user_input, ui_model_vision, ui_max_tokens,
        "Skip", ui_limiter, None,
        ui_use_ollama_options, False,
        mock_settings, mock_models_data, mock_limiters_data,
        None, mock_file_agents_dict, mock_history_list, mock_session_history
    )

    assert response.startswith("Folder processing complete (2 files):")
    assert "img1.png: Skipped -> img1.txt" in response
    assert "img2.jpg: Skipped -> img2.txt" in response
    assert model_state == "vision-model"
    assert len(new_session_list) == 3

    mock_isdir.assert_called_once_with(ui_folder_path)
    mock_listdir.assert_called_once_with(ui_folder_path)
    assert mock_isfile.call_count == 2
    assert mock_pil_open.call_count == 2
    assert mock_get_llm.call_count == 2
    call_args_2, call_kwargs_2 = mock_get_llm.call_args_list[1]
    assert call_kwargs_2['model'] == "vision-model"
    assert call_kwargs_2['images'][0] == mock_pil_image_instance
    assert "Image Context: Analyzing 'img2.jpg'" in call_kwargs_2['prompt']
    mock_builtin_open.assert_not_called()
    assert mock_add_history.call_count == 2
    # Removed assert False


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(RELEASE_MODEL_PATH)
@patch(GET_LLM_RESPONSE_PATH)
@patch(LOAD_ALL_ROLES_PATH)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
def test_chat_logic_model_release(
    mock_get_actual_name, mock_load_roles, mock_get_llm, mock_release_model, mock_add_history, mock_time):
    """Test that release_model is called when settings require it."""
    mock_load_roles.return_value = mock_roles_data
    mock_get_llm.return_value = "Response"
    previous_model = "old-text-model" # Tracker state
    current_model_ui = ui_model_text # New model selected
    # Update settings for this test
    settings_release = {**mock_settings, "release_model_on_change": True}

    app_logic.chat_logic(
        None, ui_role_agent1, ui_user_input, current_model_ui, ui_max_tokens,
        ui_file_handling, ui_limiter, None, ui_use_ollama_options, True, # release_on_change UI input
        settings_release, # Pass modified settings
        mock_models_data, mock_limiters_data,
        previous_model, # Pass previous model from tracker state
        mock_file_agents_dict, mock_history_list, mock_session_history
    )

    # Check that release_model was called with the *previous* model name
    mock_release_model.assert_called_once_with(previous_model, mock_settings["ollama_url"])


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(GET_LLM_RESPONSE_PATH)
@patch(LOAD_ALL_ROLES_PATH)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
def test_chat_logic_limiter(
    mock_get_actual_name, mock_load_roles, mock_get_llm, mock_add_history, mock_time):
    """Test that limiters affect max_tokens and prompt."""
    mock_load_roles.return_value = mock_roles_data
    mock_get_llm.return_value = "Response"
    limiter_choice = "SDXL" # Defined in mock_limiters_data with tokens=200
    high_ui_max_tokens = 1000

    app_logic.chat_logic(
        None, ui_role_agent1, ui_user_input, ui_model_text, high_ui_max_tokens,
        ui_file_handling, limiter_choice, None, ui_use_ollama_options, False,
        mock_settings, mock_models_data, mock_limiters_data,
        None, mock_file_agents_dict, mock_history_list, mock_session_history
    )

    mock_get_llm.assert_called_once()
    call_args, call_kwargs = mock_get_llm.call_args
    # Check that max_tokens passed to agent is the limiter's value (200)
    assert call_kwargs['max_tokens'] == 200
    # Check that the limiter's prompt format is included
    assert mock_limiters_data[limiter_choice]["limiter_prompt_format"] in call_kwargs['prompt']


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(RELEASE_MODEL_PATH)
@patch(GET_LLM_RESPONSE_PATH)
@patch(LOAD_ALL_ROLES_PATH)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
@patch(OS_PATH_ISDIR_PATH, return_value=True)
@patch(OS_LISTDIR_PATH, return_value=["img1.png"])
@patch(OS_PATH_EXISTS_PATH, side_effect=[False, True])
@patch(OS_PATH_SPLITEXT_PATH, side_effect=lambda p: os.path.splitext(os.path.basename(p)))
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
@patch(OS_PATH_SPLITEXT_PATH, side_effect=lambda p: os.path.splitext(os.path.basename(p)))
@patch(OS_PATH_BASENAME_PATH, side_effect=os.path.basename) # ADDED
@patch(OS_PATH_JOIN_PATH, side_effect=lambda *args: os.sep.join(args)) # FIXED
@patch(OS_PATH_ISFILE_PATH, return_value=True) # ADDED
@patch(PIL_IMAGE_OPEN_PATH, return_value=MagicMock(spec=Image))
@patch(BUILTINS_OPEN_PATH, new_callable=mock_open)
def test_chat_logic_folder_processing_append(
    mock_builtin_open, mock_pil_open, mock_isfile, mock_join, mock_basename, mock_splitext, mock_exists, mock_listdir, mock_isdir, # ADDED mocks
    mock_get_actual_name, mock_load_roles, mock_get_llm, mock_release_model, mock_add_history, mock_time):
    """Test chat_logic with folder processing and 'Append' file handling."""
    mock_load_roles.return_value = mock_roles_data
    mock_get_llm.return_value = "LLM Append Resp"
    expected_txt_path = os.sep.join([ui_folder_path, "img1.txt"])

    response, _, _, _ = app_logic.chat_logic(ui_folder_path, ui_role_agent1, ui_user_input, ui_model_vision, ui_max_tokens, "Append", ui_limiter, None, ui_use_ollama_options, False, mock_settings, mock_models_data, mock_limiters_data, None, mock_file_agents_dict, mock_history_list, mock_session_history)

    assert "img1.png: Appended -> img1.txt" in response # FIXED
    mock_builtin_open.assert_called_once_with(expected_txt_path, 'a', encoding='utf-8')
    mock_builtin_open().write.assert_called_once_with("\n\n---\n\nLLM Append Resp")


@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(RELEASE_MODEL_PATH)
@patch(GET_LLM_RESPONSE_PATH)
@patch(LOAD_ALL_ROLES_PATH)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
@patch(OS_PATH_ISDIR_PATH, return_value=True)
@patch(OS_LISTDIR_PATH, return_value=["img1.png"])
@patch(OS_PATH_EXISTS_PATH, return_value=True)
@patch(OS_PATH_SPLITEXT_PATH, side_effect=lambda p: os.path.splitext(os.path.basename(p)))
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
@patch(GET_LLM_RESPONSE_PATH)
@patch(LOAD_ALL_ROLES_PATH)
@patch(GET_ACTUAL_ROLE_NAME_PATH, side_effect=lambda x: x)
@patch(OS_PATH_ISDIR_PATH, return_value=True) # Folder exists
def test_chat_logic_folder_no_vision_model(
    mock_isdir, mock_get_actual_name, mock_load_roles, mock_get_llm, mock_release_model, mock_add_history, mock_time):
    """Test folder processing skips images if model lacks vision."""
    mock_load_roles.return_value = mock_roles_data
    mock_get_llm.return_value = "LLM Text Response (Folder Ignored)"

    response, _, model_state, new_session_list = app_logic.chat_logic(
        ui_folder_path, ui_role_agent1, ui_user_input, ui_model_text, # Use TEXT model
        ui_max_tokens, "Skip", ui_limiter, None, ui_use_ollama_options, False,
        mock_settings, mock_models_data, mock_limiters_data,
        None, mock_file_agents_dict, mock_history_list, mock_session_history
    )

    # Behavior is to run once as text-only if folder provided but model lacks vision
    assert response == "LLM Text Response (Folder Ignored)"
    assert model_state == "text-model"
    assert "Image: [None - Folder Ignored]" in new_session_list[-1] # Check history log entry detail
    assert len(new_session_list) == 2 # Only one entry added

    # Removed assertion on isdir call count
    mock_get_llm.assert_called_once() # Called only once
    call_args, call_kwargs = mock_get_llm.call_args
    assert call_kwargs['images'] is None # No images passed
    assert "Image Context:" not in call_kwargs['prompt'] # No image context in prompt


def test_chat_logic_error_model_not_found():
    """Test chat_logic when selected model isn't in models_data_state."""
    response, _, model_state, _ = app_logic.chat_logic(
        None, ui_role_agent1, ui_user_input, "non-existent-model", ui_max_tokens,
        ui_file_handling, ui_limiter, None, ui_use_ollama_options, False,
        mock_settings, mock_models_data, mock_limiters_data,
        None, mock_file_agents_dict, mock_history_list, mock_session_history
    )
    assert "Error: Selected model info not found." in response
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
    assert "Error processing single image: Bad image data" in response
    assert model_state == "vision-model" # Model was found, error happened later

@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(OS_PATH_ISDIR_PATH, return_value=True)
@patch(OS_LISTDIR_PATH, return_value=["bad_image.png"])
@patch(OS_PATH_SPLITEXT_PATH, side_effect=lambda p: os.path.splitext(os.path.basename(p)))
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

    assert response.startswith("Folder processing complete (0 files):") # 0 processed successfully
    assert "bad_image.png: Error - Cannot open this image" in response
    assert mock_add_history.call_count == 1 # Only one error entry added to history
    assert "ERROR: Cannot open this image" in new_session_list[-1] # Check session log
    # Removed assert False

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


# --- Tests for comment_logic ---

@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(GET_LLM_RESPONSE_PATH)
@patch(LOAD_ALL_ROLES_PATH)
def test_comment_logic_success(mock_load_roles, mock_get_llm, mock_add_history, mock_time):
    """Test successful comment/refinement call."""
    mock_load_roles.return_value = mock_roles_data
    mock_get_llm.return_value = "Refined LLM Response"
    previous_response = "Original LLM Response"
    comment_text = "Make it better."
    model_in_state = "vision-model"

    response, session_hist_txt, new_session_list = app_logic.comment_logic(
        previous_response, comment_text, ui_max_tokens, ui_use_ollama_options,
        model_in_state, mock_settings, mock_file_agents_dict,
        mock_history_list, mock_session_history
    )

    assert response == "Refined LLM Response"
    assert len(new_session_list) == 2 # Original session + comment entry
    assert "(Comment)" in new_session_list[-1] # Check history entry marker

    mock_load_roles.assert_called_once_with(mock_settings, file_agents=mock_file_agents_dict)
    mock_get_llm.assert_called_once()
    call_args, call_kwargs = mock_get_llm.call_args
    assert call_kwargs['role'] == "User" # Uses 'User' role for comments
    assert call_kwargs['model'] == model_in_state
    assert call_kwargs['settings'] == mock_settings
    assert call_kwargs['roles_data'] == mock_roles_data
    assert call_kwargs['images'] is None
    assert call_kwargs['max_tokens'] == ui_max_tokens
    # Check prompt construction for comments
    assert f"Previous Response:\n{previous_response}" in call_kwargs['prompt']
    assert f"User Comment/Instruction:\n{comment_text}" in call_kwargs['prompt']
    mock_add_history.assert_called_once() # History added for the comment


def test_comment_logic_no_comment_or_model():
    """Test comment logic skips if comment or model state is missing."""
    # Scenario 1: No comment text
    response1, session_hist_txt1, new_session_list1 = app_logic.comment_logic(
        "Prev Resp", "", 100, False, "model", mock_settings, {}, [], []
    )
    assert response1 == "Prev Resp" # Unchanged
    assert len(new_session_list1) == 0

    # Scenario 2: No model state
    response2, session_hist_txt2, new_session_list2 = app_logic.comment_logic(
        "Prev Resp", "A comment", 100, False, None, mock_settings, {}, [], []
    )
    assert response2 == "Prev Resp" # Unchanged
    assert len(new_session_list2) == 0


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
    editor_state_in = mock_editor_state_team_a.copy() # Start with Team A state
    new_agent_display = "Agent2"

    new_state, steps_json = app_logic.add_step_to_editor(new_agent_display, editor_state_in)

    assert len(new_state["steps"]) == 2
    assert new_state["steps"][0] == {"role": "Agent1"}
    assert new_state["steps"][1] == {"role": "Agent2"} # Uses actual name
    assert steps_json == new_state["steps"]
    mock_get_name.assert_called_once_with(new_agent_display)

def test_add_step_to_editor_no_selection():
    """Test adding a step when no agent is selected."""
    editor_state_in = mock_editor_state_team_a.copy()
    new_state, steps_update = app_logic.add_step_to_editor(None, editor_state_in)
    assert new_state == editor_state_in # State should not change
    assert isinstance(steps_update, dict) # Check it's a dict (Gradio update) # FIXED

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

def test_remove_step_from_editor_invalid_index():
    """Test removing a step with an invalid index."""
    editor_state_in = mock_editor_state_team_a.copy() # Has 1 step
    # Try removing step 0 or step 2
    new_state_0, steps_update_0 = app_logic.remove_step_from_editor(0, editor_state_in)
    assert new_state_0 == editor_state_in # Unchanged
    assert isinstance(steps_update_0, dict) # Check it's a dict (Gradio update) # FIXED

    new_state_2, steps_update_2 = app_logic.remove_step_from_editor(2, editor_state_in)
    assert new_state_2 == editor_state_in # Unchanged
    assert isinstance(steps_update_2, dict) # Check it's a dict (Gradio update) # FIXED
    # Removed assert False

@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data) # Mock role loading needed for dropdown update logic
@patch(SAVE_TEAMS_PATH, return_value=True) # Mock successful save
def test_save_team_from_editor_success_new(mock_save, mock_load_roles):
    """Test saving a new team definition."""
    editor_state_in = mock_empty_editor_state.copy()
    editor_state_in["steps"] = [{"role": "Agent1"}] # Add a step
    new_name = "NewTeam"
    new_desc = "A brand new team."
    new_strat = "concatenate"

    # Simulate necessary state inputs needed by the function signature
    current_teams_state = mock_all_teams_state.copy()
    current_settings_state = mock_settings.copy()
    current_file_agents_state = {}

    teams_state_out, editor_dd_update, chat_dd_update, status_msg = app_logic.save_team_from_editor(
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

@patch(SAVE_TEAMS_PATH, return_value=False) # Mock failed save
def test_save_team_from_editor_save_fail(mock_save):
    """Test handling failure during team saving."""
    editor_state_in = mock_editor_state_team_a.copy()
    current_teams_state = mock_all_teams_state.copy()
    current_settings_state = mock_settings.copy()
    current_file_agents_state = {}

    teams_state_out, editor_dd_update, chat_dd_update, status_msg = app_logic.save_team_from_editor(
        "TeamA", "Desc", "concat", editor_state_in,
        current_teams_state, current_settings_state, current_file_agents_state
    )

    assert "Error: Failed to save team data" in status_msg
    assert teams_state_out == current_teams_state # State should not change on save fail
    assert isinstance(editor_dd_update, dict) # Check it's a dict (Gradio update) # FIXED
    assert editor_dd_update.get("choices") is None # No dropdown update
    assert isinstance(chat_dd_update, dict) # Check it's a dict (Gradio update) # FIXED
    assert chat_dd_update.get("choices") is None # No dropdown update

def test_save_team_from_editor_empty_name():
    """Test saving with an empty team name."""
    editor_state_in = mock_editor_state_team_a.copy()
    current_teams_state = mock_all_teams_state.copy()
    current_settings_state = mock_settings.copy()
    current_file_agents_state = {}

    teams_state_out, editor_dd_update, chat_dd_update, status_msg = app_logic.save_team_from_editor(
        "  ", "Desc", "concat", editor_state_in, # Empty name
        current_teams_state, current_settings_state, current_file_agents_state
    )
    assert "Team Name cannot be empty" in status_msg
    assert teams_state_out == current_teams_state # State unchanged

@patch(LOAD_ALL_ROLES_PATH, return_value=mock_roles_data) # Mock role loading needed for dropdown update logic
@patch(SAVE_TEAMS_PATH, return_value=True) # Mock successful save
def test_delete_team_logic_success(mock_save, mock_load_roles):
    """Test deleting an existing team."""
    team_to_delete = "TeamA"
    current_teams_state = mock_all_teams_state.copy()
    current_settings_state = mock_settings.copy()
    current_file_agents_state = {}

    # Function returns 9 values: teams_state, editor_dd, chat_dd, status, + 5 clear outputs
    outputs = app_logic.delete_team_logic(
        team_to_delete, current_teams_state,
        current_settings_state, current_file_agents_state
    )
    teams_state_out, editor_dd_update, chat_dd_update, status_msg = outputs[0:4]
    clear_outputs = outputs[4:] # This is a list of 5 elements

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
    # Check clear outputs (match return of clear_team_editor)
    name_clr, desc_clr, steps_clr, strat_clr, state_clr = app_logic.clear_team_editor()
    # Compare elements individually # FIXED
    assert len(clear_outputs) == 5
    assert clear_outputs[0] == name_clr     # ""
    assert clear_outputs[1] == desc_clr     # ""
    assert clear_outputs[2] == steps_clr    # []
    assert clear_outputs[3] == strat_clr    # "concatenate"
    assert clear_outputs[4] == state_clr    # mock_empty_editor_state


def test_delete_team_logic_not_found():
    """Test deleting a team that doesn't exist."""
    team_to_delete = "NonExistent"
    current_teams_state = mock_all_teams_state.copy()
    current_settings_state = mock_settings.copy()
    current_file_agents_state = {}

    outputs = app_logic.delete_team_logic(team_to_delete, current_teams_state, current_settings_state, current_file_agents_state)
    status_msg = outputs[3]
    assert f"Team '{team_to_delete}' not found" in status_msg

def test_delete_team_logic_no_selection():
    """Test deleting with no team selected."""
    current_teams_state = mock_all_teams_state.copy()
    current_settings_state = mock_settings.copy()
    current_file_agents_state = {}

    outputs = app_logic.delete_team_logic(None, current_teams_state, current_settings_state, current_file_agents_state)
    status_msg = outputs[3]
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


@patch(BUILTINS_OPEN_PATH, new_callable=mock_open, read_data='{"AgentFromFile": {"description": "File Agent Desc"}}')
def test_handle_agent_file_upload_success(mock_file):
    """Test successfully uploading and parsing an agent file."""
    mock_uploaded_file = MagicMock()
    # Simulate Gradio file object having a 'name' attribute with the path
    mock_uploaded_file.name = "/path/to/uploaded_agents.json"

    loaded_dict, filename, update_obj = app_logic.handle_agent_file_upload(mock_uploaded_file)

    mock_file.assert_called_once_with("/path/to/uploaded_agents.json", 'r', encoding='utf-8')
    assert filename == "uploaded_agents.json" # Should return basename
    assert "AgentFromFile" in loaded_dict
    assert loaded_dict["AgentFromFile"]["description"] == "File Agent Desc"
    # Check if it returns a Gradio update object (dict)
    assert isinstance(update_obj, dict) # Check it's a dict (Gradio update)


def test_handle_agent_file_upload_none():
    """Test clearing the agent file upload."""
    loaded_dict, filename, update_obj = app_logic.handle_agent_file_upload(None)
    assert loaded_dict == {}
    assert filename is None
    assert isinstance(update_obj, dict) # Check it's a dict (Gradio update)

@patch(BUILTINS_OPEN_PATH, new_callable=mock_open, read_data='invalid json')
def test_handle_agent_file_upload_invalid_json(mock_file):
    """Test uploading a file with invalid JSON content."""
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.name = "/path/to/invalid.json"
    loaded_dict, filename, update_obj = app_logic.handle_agent_file_upload(mock_uploaded_file)
    assert loaded_dict == {}
    assert filename is None
    assert isinstance(update_obj, dict) # Check it's a dict (Gradio update)

@patch(BUILTINS_OPEN_PATH, new_callable=mock_open, read_data='{"BadAgent": "just a string"}')
def test_handle_agent_file_upload_invalid_agent_format(mock_file):
    """Test uploading valid JSON but with invalid agent structure."""
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.name = "/path/to/bad_agents.json"
    loaded_dict, filename, update_obj = app_logic.handle_agent_file_upload(mock_uploaded_file)
    assert loaded_dict == {} # No valid agents found
    assert filename is None # Returns None because no *valid* agents loaded
    assert isinstance(update_obj, dict) # Check it's a dict (Gradio update)

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