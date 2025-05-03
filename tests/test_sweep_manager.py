# ArtAgent/tests/test_sweep_manager.py

import pytest
import os
import sys
import json
import time
from unittest.mock import patch, MagicMock, call, ANY

# --- Adjust import path ---
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir)
sys.path.insert(0, project_root)

# --- Module Import ---
try:
    from core import sweep_manager
    # Import constants or other modules if sweep_manager uses them directly
    # e.g., from core.utils import save_json (if used explicitly)
except ImportError as e:
    pytest.skip(f"Skipping sweep_manager tests, core module not found: {e}", allow_module_level=True)

# --- Mock Paths ---
# Mock dependencies used *within* sweep_manager.py
GET_ABS_PATH = 'core.sweep_manager.get_absolute_path'
OS_MAKEDIRS_PATH = 'core.sweep_manager.os.makedirs'
SAVE_JSON_PATH = 'core.sweep_manager.save_json' # Assumes using utils.save_json
RUN_TEAM_WORKFLOW_PATH = 'core.sweep_manager.agent_manager.run_team_workflow'
LOAD_ALL_ROLES_PATH = 'core.sweep_manager.load_all_roles'
TIME_STRFTIME_PATH = 'core.sweep_manager.time.strftime' # Mock for folder name
TIME_GMTIME_PATH = 'core.sweep_manager.time.gmtime' # Mock for protocol timestamp
# Optional mocks if more detailed checking needed:
# HASH_MD5_PATH = 'core.sweep_manager.hashlib.md5'
# TRACEBACK_PRINT_PATH = 'core.sweep_manager.traceback.print_exc'


# --- Test Data ---
MOCK_FOLDER_TIMESTAMP = "20240101_112233"
MOCK_PROTOCOL_TIMESTAMP_STRUCT = time.struct_time((2024, 1, 1, 11, 22, 33, 1, 1, 0)) # UTC for protocol
MOCK_PROTOCOL_TIMESTAMP_STR = "2024-01-01T11:22:33Z" # Formatted string for protocol

MOCK_SETTINGS = {"setting1": "value1", "sweep_step_max_tokens": 600}
MOCK_TEAMS_DATA = {
    "TeamSweepA": {"description": "Sweep Team A", "steps": [{"role":"AgentSweep1"}], "assembly_strategy": "concatenate"},
    "TeamSweepB": {"description": "Sweep Team B", "steps": [{"role":"AgentSweep2"}], "assembly_strategy": "refine_last"}
}
MOCK_ROLES_DATA = {"AgentSweep1": {}, "AgentSweep2": {}} # Simplified for sweep test
MOCK_PROMPTS_TEXT = "Prompt One\nPrompt Two"
MOCK_PROMPTS_LIST = ["Prompt One", "Prompt Two"]
SELECTED_TEAMS = ["TeamSweepA", "TeamSweepB"]
SELECTED_MODELS = ["model-sweep-1", "model-sweep-2"]
OUTPUT_FOLDER_NAME = "my_test_sweep"
EXPECTED_BASE_OUTPUT_DIR = os.path.join(sweep_manager.SWEEP_OUTPUT_BASE_DIR, f"{MOCK_FOLDER_TIMESTAMP}_{OUTPUT_FOLDER_NAME}")
EXPECTED_ABS_OUTPUT_DIR = f"/fake/project/root/{EXPECTED_BASE_OUTPUT_DIR}" # Mocked absolute path

# --- Fixtures ---
@pytest.fixture(autouse=True)
def mock_time(monkeypatch):
    """Mock time functions used in sweep manager."""
    def mock_strftime(fmt, tm_tuple=None): # Accept optional tuple
        if fmt == "%Y%m%d_%H%M%S":
            return MOCK_FOLDER_TIMESTAMP
        elif fmt == "%Y-%m-%dT%H:%M:%SZ": # Used for protocol
            return MOCK_PROTOCOL_TIMESTAMP_STR
        else:
            # Fallback for unexpected formats
            return "UNEXPECTED_TIME_FORMAT"
    monkeypatch.setattr(TIME_STRFTIME_PATH, mock_strftime)
    # gmtime returns a time.struct_time object
    monkeypatch.setattr(TIME_GMTIME_PATH, lambda: MOCK_PROTOCOL_TIMESTAMP_STRUCT)
    # Mock time.time() just in case duration calculation is tested later
    monkeypatch.setattr('core.sweep_manager.time.time', MagicMock(side_effect=[1000.0, 1010.0])) # Example start/end times


# --- Tests ---

@patch(LOAD_ALL_ROLES_PATH, return_value=MOCK_ROLES_DATA)
@patch(RUN_TEAM_WORKFLOW_PATH)
@patch(SAVE_JSON_PATH, return_value=True) # Assume save succeeds
@patch(OS_MAKEDIRS_PATH)
@patch(GET_ABS_PATH, return_value=EXPECTED_ABS_OUTPUT_DIR)
def test_run_sweep_success_no_intermediate(
    mock_abs_path, mock_makedirs, mock_save_json, mock_run_workflow, mock_load_roles, mock_time): # mock_time is auto-applied
    """Test a successful sweep run without logging intermediate steps."""

    # Define side effects for run_team_workflow (8 runs total)
    mock_run_workflow.side_effect = [
        ("Output P1_TA_M1", [], None), ("Output P1_TA_M2", [], None),
        ("Output P1_TB_M1", [], None), ("Output P1_TB_M2", [], None),
        ("Output P2_TA_M1", [], None), ("Output P2_TA_M2", [], None),
        ("Output P2_TB_M1", [], None), ("Output P2_TB_M2", [], None),
    ]

    summary = sweep_manager.run_sweep(
        base_prompts_text=MOCK_PROMPTS_TEXT,
        selected_teams=SELECTED_TEAMS,
        selected_models=SELECTED_MODELS,
        output_folder_name=OUTPUT_FOLDER_NAME,
        log_intermediate=False, # Test without intermediate logs
        settings=MOCK_SETTINGS,
        all_teams_data=MOCK_TEAMS_DATA,
    )

    # --- Assertions ---
    # 1. Setup calls
    mock_abs_path.assert_called_once_with(EXPECTED_BASE_OUTPUT_DIR)
    mock_makedirs.assert_called_once_with(EXPECTED_ABS_OUTPUT_DIR, exist_ok=True)
    mock_load_roles.assert_called_once_with(MOCK_SETTINGS, file_agents={})

    # 2. Workflow calls (2 prompts * 2 teams * 2 models = 8 runs)
    assert mock_run_workflow.call_count == 8
    first_call_args, first_call_kwargs = mock_run_workflow.call_args_list[0]
    assert first_call_kwargs['team_name'] == "TeamSweepA"
    assert first_call_kwargs['user_input'] == MOCK_PROMPTS_LIST[0]
    assert first_call_kwargs['worker_model_name'] == SELECTED_MODELS[0]
    assert first_call_kwargs['return_intermediate_steps'] is False

    last_call_args, last_call_kwargs = mock_run_workflow.call_args_list[-1]
    assert last_call_kwargs['team_name'] == "TeamSweepB"
    assert last_call_kwargs['user_input'] == MOCK_PROMPTS_LIST[1]
    assert last_call_kwargs['worker_model_name'] == SELECTED_MODELS[1]
    assert last_call_kwargs['return_intermediate_steps'] is False

    # 3. Save JSON calls (one per run)
    assert mock_save_json.call_count == 8
    first_save_call_args, _ = mock_save_json.call_args_list[0]
    protocol_path1 = first_save_call_args[0]
    protocol_data1 = first_save_call_args[1]
    assert protocol_path1.startswith(EXPECTED_ABS_OUTPUT_DIR)
    assert protocol_path1.endswith(".json")
    # Check filename contains expected parts (hashes might differ slightly if prompt changes)
    assert "TeamSweepA" in protocol_path1 and "model-sweep-1" in protocol_path1

    # Check protocol structure
    assert protocol_data1['sweep_metadata']['base_user_prompt'] == MOCK_PROMPTS_LIST[0]
    assert protocol_data1['sweep_metadata']['timestamp_utc'] == MOCK_PROTOCOL_TIMESTAMP_STR # Check mocked time
    assert 'run_id' in protocol_data1['sweep_metadata'] # Check run_id exists
    assert protocol_data1['configuration']['agent_team_name'] == "TeamSweepA"
    assert protocol_data1['configuration']['worker_model'] == SELECTED_MODELS[0]
    assert protocol_data1['configuration']['log_intermediate_steps'] is False
    assert protocol_data1['execution_log'] is None # Not logged
    assert protocol_data1['final_output'] == "Output P1_TA_M1"

    # 4. Summary
    assert "Sweep Complete" in summary
    assert "Total Runs Attempted: 8/8" in summary
    assert f"Protocols saved to: {EXPECTED_ABS_OUTPUT_DIR}" in summary


@patch(LOAD_ALL_ROLES_PATH, return_value=MOCK_ROLES_DATA)
@patch(RUN_TEAM_WORKFLOW_PATH)
@patch(SAVE_JSON_PATH, return_value=True)
@patch(OS_MAKEDIRS_PATH)
@patch(GET_ABS_PATH, return_value=EXPECTED_ABS_OUTPUT_DIR)
def test_run_sweep_success_with_intermediate(
    mock_abs_path, mock_makedirs, mock_save_json, mock_run_workflow, mock_load_roles, mock_time): # mock_time auto-applied
    """Test a successful sweep run WITH logging intermediate steps."""

    mock_intermediate_data = {
        1: {"role": "AgentSweep1", "goal": "G1", "output": "Step1 Out", "error": None},
        2: {"role": "AgentSweep2", "goal": "G2", "output": "Step2 Out", "error": None},
    }
    mock_run_workflow.return_value = ("Final Output", [], mock_intermediate_data)

    summary = sweep_manager.run_sweep(
        base_prompts_text=MOCK_PROMPTS_TEXT,
        selected_teams=SELECTED_TEAMS,
        selected_models=SELECTED_MODELS,
        output_folder_name=OUTPUT_FOLDER_NAME,
        log_intermediate=True, # <<<< Test WITH intermediate logs
        settings=MOCK_SETTINGS,
        all_teams_data=MOCK_TEAMS_DATA,
    )

    # --- Assertions ---
    assert mock_run_workflow.call_count == 8
    assert mock_save_json.call_count == 8
    assert mock_run_workflow.call_args.kwargs['return_intermediate_steps'] is True

    first_save_call_args, _ = mock_save_json.call_args_list[0]
    protocol_data1 = first_save_call_args[1]
    assert protocol_data1['configuration']['log_intermediate_steps'] is True
    assert isinstance(protocol_data1['execution_log'], list)
    assert len(protocol_data1['execution_log']) == 2
    assert protocol_data1['execution_log'][0]['step'] == 1
    assert protocol_data1['execution_log'][0]['agent_role'] == "AgentSweep1"
    assert protocol_data1['execution_log'][0]['output'] == "Step1 Out"
    assert protocol_data1['execution_log'][1]['step'] == 2
    assert protocol_data1['execution_log'][1]['agent_role'] == "AgentSweep2"
    assert protocol_data1['execution_log'][1]['output'] == "Step2 Out"
    assert protocol_data1['final_output'] == "Final Output"


@patch(LOAD_ALL_ROLES_PATH, return_value=MOCK_ROLES_DATA)
@patch(RUN_TEAM_WORKFLOW_PATH)
@patch(SAVE_JSON_PATH, return_value=True)
@patch(OS_MAKEDIRS_PATH)
@patch(GET_ABS_PATH, return_value=EXPECTED_ABS_OUTPUT_DIR)
def test_run_sweep_workflow_error_handling(
    mock_abs_path, mock_makedirs, mock_save_json, mock_run_workflow, mock_load_roles, mock_time): # mock_time auto-applied
    """Test sweep continues and logs errors if one workflow run fails."""

    error_message = "Workflow Error During Run 3"
    mock_run_workflow.side_effect = [
        ("Output Run 1", [], None), ("Output Run 2", [], None),
        Exception(error_message), ("Output Run 4", [], None),
    ] * 2

    summary = sweep_manager.run_sweep(
        base_prompts_text=MOCK_PROMPTS_TEXT,
        selected_teams=SELECTED_TEAMS,
        selected_models=SELECTED_MODELS,
        output_folder_name=OUTPUT_FOLDER_NAME,
        log_intermediate=False,
        settings=MOCK_SETTINGS,
        all_teams_data=MOCK_TEAMS_DATA,
    )

    # --- Assertions ---
    assert mock_run_workflow.call_count == 8
    assert mock_save_json.call_count == 8

    failed_save_call_args, _ = mock_save_json.call_args_list[2]
    failed_protocol_data = failed_save_call_args[1]
    assert failed_protocol_data['final_output'].startswith("ERROR during execution:")
    assert error_message in failed_protocol_data['final_output']
    assert failed_protocol_data['execution_log'] is None

    success_save_call_args, _ = mock_save_json.call_args_list[3]
    success_protocol_data = success_save_call_args[1]
    assert success_protocol_data['final_output'] == "Output Run 4"

    assert "Sweep Complete" in summary
    assert "Total Runs Attempted: 8/8" in summary
    # Check specific status line for error (details might vary based on prompt hash etc)
    assert f"Error: {error_message}" in summary
    assert "Success" in summary # Check that success status is also present


# --- Test Input Validation and Setup Errors ---

def test_run_sweep_no_prompts():
    summary = sweep_manager.run_sweep("", SELECTED_TEAMS, SELECTED_MODELS, "f", False, {}, {})
    assert "Error: No base prompts provided." in summary

def test_run_sweep_no_teams():
    summary = sweep_manager.run_sweep(MOCK_PROMPTS_TEXT, [], SELECTED_MODELS, "f", False, {}, {})
    assert "Error: No Agent Teams selected." in summary

def test_run_sweep_no_models():
    summary = sweep_manager.run_sweep(MOCK_PROMPTS_TEXT, SELECTED_TEAMS, [], "f", False, {}, {})
    assert "Error: No Worker Models selected." in summary

@patch(GET_ABS_PATH, return_value=EXPECTED_ABS_OUTPUT_DIR)
@patch(OS_MAKEDIRS_PATH, side_effect=OSError("Cannot create dir"))
def test_run_sweep_makedirs_error(mock_makedirs, mock_abs_path, mock_time): # mock_time auto-applied
    summary = sweep_manager.run_sweep(MOCK_PROMPTS_TEXT, SELECTED_TEAMS, SELECTED_MODELS, OUTPUT_FOLDER_NAME, False, MOCK_SETTINGS, MOCK_TEAMS_DATA)
    assert "Error creating output directory" in summary
    assert "Cannot create dir" in summary

@patch(GET_ABS_PATH, return_value=EXPECTED_ABS_OUTPUT_DIR)
@patch(OS_MAKEDIRS_PATH)
@patch(LOAD_ALL_ROLES_PATH, side_effect=Exception("Roles loading failed"))
def test_run_sweep_load_roles_error(mock_load_roles, mock_makedirs, mock_abs_path, mock_time): # mock_time auto-applied
    summary = sweep_manager.run_sweep(MOCK_PROMPTS_TEXT, SELECTED_TEAMS, SELECTED_MODELS, OUTPUT_FOLDER_NAME, False, MOCK_SETTINGS, MOCK_TEAMS_DATA)
    assert "Error loading agent roles during sweep setup: Roles loading failed" in summary

@patch(LOAD_ALL_ROLES_PATH, return_value=MOCK_ROLES_DATA)
@patch(RUN_TEAM_WORKFLOW_PATH)
@patch(SAVE_JSON_PATH, return_value=False) # <<< Mock save failure
@patch(OS_MAKEDIRS_PATH)
@patch(GET_ABS_PATH, return_value=EXPECTED_ABS_OUTPUT_DIR)
def test_run_sweep_protocol_save_error(
    mock_abs_path, mock_makedirs, mock_save_json, mock_run_workflow, mock_load_roles, mock_time): # mock_time auto-applied
    """Test that save errors are reported in the summary."""
    mock_run_workflow.return_value = ("Output Run 1", [], None)

    summary = sweep_manager.run_sweep(
        base_prompts_text="One Prompt", # Just one run
        selected_teams=["TeamSweepA"],
        selected_models=["model-sweep-1"],
        output_folder_name=OUTPUT_FOLDER_NAME,
        log_intermediate=False,
        settings=MOCK_SETTINGS,
        all_teams_data=MOCK_TEAMS_DATA,
    )

    assert "Sweep Complete" in summary
    assert "Run 1:" in summary
    assert "Protocol Save Error:" in summary
    assert "save_json utility returned False" in summary