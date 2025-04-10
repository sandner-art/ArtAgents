# ArtAgent/tests/test_agent_manager.py

import pytest
import os
import sys
import time
from unittest.mock import patch, MagicMock, call # Import call for checking multiple calls

# --- Adjust import path ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from core.agent_manager import run_team_workflow
    # We will mock the dependencies called *by* run_team_workflow
    GET_LLM_RESPONSE_PATH = 'core.agent_manager.get_llm_response'
    ADD_TO_HISTORY_PATH = 'core.agent_manager.history.add_to_history'
    TIME_STRFTIME_PATH = 'core.agent_manager.time.strftime' # To make timestamps predictable
except ImportError as e:
    pytest.skip(f"Skipping agent_manager tests, modules not found: {e}", allow_module_level=True)


# --- Test Data ---

MOCK_SETTINGS = {
    "ollama_url": "http://fake-manager-test:11434/api/generate",
    "ollama_api_options": {"seed": 100},
    "sweep_step_max_tokens": 500, # For intermediate steps
    # Add other settings if run_team_workflow uses them directly
}

MOCK_ROLES_DATA = {
    "RoleA": {"description": "Performs task A."},
    "RoleB": {"description": "Performs task B."},
    "RoleC_Refiner": {"description": "Refines previous steps."}
}

# Team Definitions
TEAM_CONCAT = {
    "description": "Concatenate A and B.",
    "assembly_strategy": "concatenate",
    "steps": [
        {"role": "RoleA", "goal": "Generate part A"},
        {"role": "RoleB", "goal": "Generate part B based on A"}
    ]
}

TEAM_REFINE = {
    "description": "Refine A and B with C.",
    "assembly_strategy": "refine_last",
    "steps": [
        {"role": "RoleA", "goal": "Generate part A"},
        {"role": "RoleB", "goal": "Generate part B based on A"},
        {"role": "RoleC_Refiner", "goal": "Refine A and B"}
    ]
}

TEAM_MISSING_ROLE = {
    "description": "Team with a missing role.",
    "assembly_strategy": "concatenate",
    "steps": [
        {"role": "RoleA"},
        {"goal": "Step without a role"} # Missing "role" key
    ]
}

TEAM_INVALID_DEF = None # Test invalid input

USER_INPUT = "Initial user request."
WORKER_MODEL = "test-worker-model"
MOCK_TIMESTAMP = "2024-01-01 12:00:00" # For predictable logs


# --- Tests ---

@patch(TIME_STRFTIME_PATH, return_value=MOCK_TIMESTAMP)
@patch(ADD_TO_HISTORY_PATH, return_value=None) # Mock history logging
@patch(GET_LLM_RESPONSE_PATH) # Mock the agent call
def test_run_team_workflow_concat_success(mock_get_llm, mock_add_history, mock_strftime):
    """Test successful execution of a 'concatenate' workflow."""
    # Define agent outputs
    mock_get_llm.side_effect = [
        "Output from RoleA.",
        "Output from RoleB."
    ]

    final_output, history_list, intermediate = run_team_workflow(
        team_name="ConcatTeam",
        team_definition=TEAM_CONCAT,
        user_input=USER_INPUT,
        initial_settings=MOCK_SETTINGS,
        all_roles_data=MOCK_ROLES_DATA,
        history_list=[], # Start with empty persistent history list for this run
        worker_model_name=WORKER_MODEL,
        return_intermediate_steps=False
    )

    # Assertions on final output
    expected_output = "--- Contribution: RoleA ---\nOutput from RoleA.\n\n--- Contribution: RoleB ---\nOutput from RoleB."
    assert final_output == expected_output
    assert intermediate is None # Not requested

    # Assertions on mock calls
    assert mock_get_llm.call_count == 2
    calls = mock_get_llm.call_args_list

    # Call 1 (RoleA)
    args_a, kwargs_a = calls[0]
    assert kwargs_a['role'] == "RoleA"
    assert kwargs_a['model'] == WORKER_MODEL
    assert kwargs_a['settings'] == MOCK_SETTINGS
    assert kwargs_a['roles_data'] == MOCK_ROLES_DATA
    assert kwargs_a['max_tokens'] == MOCK_SETTINGS['sweep_step_max_tokens']
    prompt_a = kwargs_a['prompt']
    assert f"User Request: {USER_INPUT}" in prompt_a
    assert "Your Role: RoleA - Performs task A." in prompt_a
    assert "Your Goal for this step: Generate part A" in prompt_a
    assert "Step 1 (RoleA) Output:" not in prompt_a # First step has no prior output

    # Call 2 (RoleB)
    args_b, kwargs_b = calls[1]
    assert kwargs_b['role'] == "RoleB"
    prompt_b = kwargs_b['prompt']
    assert f"User Request: {USER_INPUT}" in prompt_b # Context carries over
    assert "Step 1 (RoleA) Output:\nOutput from RoleA." in prompt_b # Check context passing
    assert "Your Role: RoleB - Performs task B." in prompt_b
    assert "Your Goal for this step: Generate part B based on A" in prompt_b

    # Assertions on history logging (check number of calls and key contents)
    assert mock_add_history.call_count == 4 # Start + Step1 + Step2 + Final
    history_calls = mock_add_history.call_args_list
    assert "Workflow Start: 'ConcatTeam'" in history_calls[0].args[1]
    assert f"Workflow Step 1: 'RoleA'" in history_calls[1].args[1]
    assert "Output:\nOutput from RoleA." in history_calls[1].args[1]
    assert f"Workflow Step 2: 'RoleB'" in history_calls[2].args[1]
    assert "Output:\nOutput from RoleB." in history_calls[2].args[1]
    assert "Workflow End: 'ConcatTeam'" in history_calls[3].args[1]
    assert f"Final Output:\n{expected_output}" in history_calls[3].args[1]


@patch(TIME_STRFTIME_PATH, return_value=MOCK_TIMESTAMP)
@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(GET_LLM_RESPONSE_PATH)
def test_run_team_workflow_refine_success(mock_get_llm, mock_add_history, mock_strftime):
    """Test successful execution of a 'refine_last' workflow."""
    mock_get_llm.side_effect = [
        "Output from RoleA.",
        "Output from RoleB.",
        "Final refined output from RoleC." # Only this should be returned
    ]

    final_output, history_list, intermediate = run_team_workflow(
        team_name="RefineTeam",
        team_definition=TEAM_REFINE,
        user_input=USER_INPUT,
        initial_settings=MOCK_SETTINGS,
        all_roles_data=MOCK_ROLES_DATA,
        history_list=[],
        worker_model_name=WORKER_MODEL,
        return_intermediate_steps=False
    )

    # Assertions on final output
    assert final_output == "Final refined output from RoleC."
    assert intermediate is None

    # Assertions on mock calls
    assert mock_get_llm.call_count == 3
    calls = mock_get_llm.call_args_list

    # Check context passing to the last step
    args_c, kwargs_c = calls[2]
    assert kwargs_c['role'] == "RoleC_Refiner"
    prompt_c = kwargs_c['prompt']
    assert "Step 1 (RoleA) Output:\nOutput from RoleA." in prompt_c
    assert "Step 2 (RoleB) Output:\nOutput from RoleB." in prompt_c
    assert "Your Role: RoleC_Refiner - Refines previous steps." in prompt_c
    assert "Your Goal for this step: Refine A and B" in prompt_c

    # Assertions on history logging
    assert mock_add_history.call_count == 5 # Start + 3 Steps + Final
    history_calls = mock_add_history.call_args_list
    assert "Workflow Start: 'RefineTeam'" in history_calls[0].args[1]
    assert "Workflow Step 1: 'RoleA'" in history_calls[1].args[1]
    assert "Workflow Step 2: 'RoleB'" in history_calls[2].args[1]
    assert "Workflow Step 3: 'RoleC_Refiner'" in history_calls[3].args[1]
    assert "Output:\nFinal refined output from RoleC." in history_calls[3].args[1]
    assert "Workflow End: 'RefineTeam'" in history_calls[4].args[1]
    assert "Assembly Strategy: refine_last" in history_calls[4].args[1]
    assert "Final Output:\nFinal refined output from RoleC." in history_calls[4].args[1]


@patch(TIME_STRFTIME_PATH, return_value=MOCK_TIMESTAMP)
@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(GET_LLM_RESPONSE_PATH)
def test_run_team_workflow_return_intermediate(mock_get_llm, mock_add_history, mock_strftime):
    """Test returning the intermediate steps dictionary."""
    mock_get_llm.side_effect = ["Output A", "Output B"]

    _, _, intermediate = run_team_workflow( # Use _ for unused outputs
        team_name="IntermediateTest",
        team_definition=TEAM_CONCAT, # Use simple concat team
        user_input=USER_INPUT,
        initial_settings=MOCK_SETTINGS,
        all_roles_data=MOCK_ROLES_DATA,
        history_list=[],
        worker_model_name=WORKER_MODEL,
        return_intermediate_steps=True # Request intermediate steps
    )

    assert intermediate is not None
    assert isinstance(intermediate, dict)
    assert len(intermediate) == 2

    # Check step 1 data
    step1_data = intermediate.get(1)
    assert step1_data is not None
    assert step1_data['role'] == "RoleA"
    assert step1_data['goal'] == "Generate part A"
    assert step1_data['output'] == "Output A"
    assert step1_data['error'] is None

    # Check step 2 data
    step2_data = intermediate.get(2)
    assert step2_data is not None
    assert step2_data['role'] == "RoleB"
    assert step2_data['goal'] == "Generate part B based on A"
    assert step2_data['output'] == "Output B"
    assert step2_data['error'] is None


@patch(TIME_STRFTIME_PATH, return_value=MOCK_TIMESTAMP)
@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(GET_LLM_RESPONSE_PATH) # Still need to patch, though it won't be called
def test_run_team_workflow_invalid_definition(mock_get_llm, mock_add_history, mock_strftime):
    """Test passing an invalid team definition."""
    final_output, _, intermediate = run_team_workflow(
        team_name="InvalidTeam",
        team_definition=TEAM_INVALID_DEF, # Pass None
        user_input=USER_INPUT,
        initial_settings=MOCK_SETTINGS,
        all_roles_data=MOCK_ROLES_DATA,
        history_list=[],
        worker_model_name=WORKER_MODEL,
        return_intermediate_steps=True
    )

    assert "Error: Invalid team definition provided" in final_output
    assert intermediate is None # Should be None on early error exit
    mock_get_llm.assert_not_called() # Agent should not run

    # Check history log for the error
    assert mock_add_history.call_count == 1
    assert "Workflow Start Error: 'InvalidTeam'" in mock_add_history.call_args.args[1]
    assert "Error: Invalid team definition provided" in mock_add_history.call_args.args[1]


@patch(TIME_STRFTIME_PATH, return_value=MOCK_TIMESTAMP)
@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(GET_LLM_RESPONSE_PATH)
def test_run_team_workflow_missing_role_in_step(mock_get_llm, mock_add_history, mock_strftime):
    """Test workflow skips a step if 'role' is missing."""
    mock_get_llm.return_value = "Output from RoleA." # Only RoleA should run

    final_output, _, intermediate = run_team_workflow(
        team_name="MissingRoleTeam",
        team_definition=TEAM_MISSING_ROLE,
        user_input=USER_INPUT,
        initial_settings=MOCK_SETTINGS,
        all_roles_data=MOCK_ROLES_DATA,
        history_list=[],
        worker_model_name=WORKER_MODEL,
        return_intermediate_steps=True
    )

    # Assembly is concat, only RoleA ran. Step 2 was skipped.
    expected_output = "--- Contribution: RoleA ---\nOutput from RoleA."
    assert final_output == expected_output

    # Check agent calls
    mock_get_llm.assert_called_once() # Only RoleA called
    assert mock_get_llm.call_args.kwargs['role'] == "RoleA"

    # Check history
    assert mock_add_history.call_count == 3 # Start + Step1 (run) + Step2 (skip) + Final
    history_calls = mock_add_history.call_args_list
    assert "Workflow Start: 'MissingRoleTeam'" in history_calls[0].args[1]
    assert "Workflow Step 1: 'RoleA'" in history_calls[1].args[1]
    assert "Workflow Step 2 Skipped ('MissingRoleTeam')" in history_calls[2].args[1] # Check skip log
    assert "Reason: Missing 'role' definition" in history_calls[2].args[1]
    assert "Workflow End: 'MissingRoleTeam'" in history_calls[3].args[1] # Should still reach end

    # Check intermediate data
    assert intermediate is not None
    assert len(intermediate) == 2
    assert intermediate[1]['output'] == "Output from RoleA."
    assert intermediate[1]['error'] is None
    assert intermediate[2]['role'] is None # Role was missing
    assert intermediate[2]['output'] is None
    assert "missing 'role'" in intermediate[2]['error']


@patch(TIME_STRFTIME_PATH, return_value=MOCK_TIMESTAMP)
@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(GET_LLM_RESPONSE_PATH)
def test_run_team_workflow_agent_error(mock_get_llm, mock_add_history, mock_strftime):
    """Test workflow stops if an agent returns an error."""
    agent_error_message = "⚠️ Error: Agent B failed spectacularly."
    mock_get_llm.side_effect = [
        "Output from RoleA.",
        agent_error_message # RoleB returns an error
    ]

    final_output, _, intermediate = run_team_workflow(
        team_name="AgentErrorTeam",
        team_definition=TEAM_CONCAT, # Use 2-step concat team
        user_input=USER_INPUT,
        initial_settings=MOCK_SETTINGS,
        all_roles_data=MOCK_ROLES_DATA,
        history_list=[],
        worker_model_name=WORKER_MODEL,
        return_intermediate_steps=True
    )

    # Assertions on final output
    assert final_output.startswith("Workflow stopped due to error in step 2 (RoleB)")
    assert agent_error_message in final_output

    # Assertions on mock calls
    assert mock_get_llm.call_count == 2 # Called A and B (which failed)

    # Check history
    assert mock_add_history.call_count == 3 # Start + Step1 (success) + Step2 (error)
    history_calls = mock_add_history.call_args_list
    assert "Workflow Start: 'AgentErrorTeam'" in history_calls[0].args[1]
    assert "Workflow Step 1: 'RoleA'" in history_calls[1].args[1]
    assert "Workflow Step 2 Error ('RoleB')" in history_calls[2].args[1] # Check error log
    assert f"Error Message: {agent_error_message}" in history_calls[2].args[1]

    # Check intermediate data (should include up to the error)
    assert intermediate is not None
    assert len(intermediate) == 2
    assert intermediate[1]['output'] == "Output from RoleA."
    assert intermediate[1]['error'] is None
    assert intermediate[2]['role'] == "RoleB"
    assert intermediate[2]['output'] is None
    assert intermediate[2]['error'] == agent_error_message


@patch(TIME_STRFTIME_PATH, return_value=MOCK_TIMESTAMP)
@patch(ADD_TO_HISTORY_PATH, return_value=None)
@patch(GET_LLM_RESPONSE_PATH)
def test_run_team_workflow_no_successful_steps(mock_get_llm, mock_add_history, mock_strftime):
    """Test assembly when no steps succeed."""
    agent_error_message = "⚠️ Error: Agent failed."
    mock_get_llm.side_effect = [
        agent_error_message, # RoleA fails
        agent_error_message  # RoleB fails
    ]

    final_output, _, intermediate = run_team_workflow(
        team_name="AllFailTeam",
        team_definition=TEAM_CONCAT,
        user_input=USER_INPUT,
        initial_settings=MOCK_SETTINGS,
        all_roles_data=MOCK_ROLES_DATA,
        history_list=[],
        worker_model_name=WORKER_MODEL,
        return_intermediate_steps=True
    )

    # Final output comes from the *first* error encountered
    assert final_output.startswith("Workflow stopped due to error in step 1 (RoleA)")

    # Check intermediate steps (both should show errors)
    assert intermediate is not None
    assert len(intermediate) >= 1 # Only step 1 is guaranteed to be recorded before exit
    assert intermediate[1]['role'] == "RoleA"
    assert intermediate[1]['output'] is None
    assert intermediate[1]['error'] == agent_error_message
    # Step 2 won't be in intermediate dict as the workflow exited after step 1's error

    # Check history
    assert mock_add_history.call_count == 2 # Start + Step1 (error)
    assert "Workflow Step 1 Error ('RoleA')" in mock_add_history.call_args_list[1].args[1]