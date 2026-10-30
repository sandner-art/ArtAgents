# ArtAgent/tests/test_agent_manager.py
"""Tests for core/agent_manager.py (team workflows, streaming generator + blocking wrapper)."""
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(test_dir))

from core.agent_manager import TeamProgress, TeamResult, iter_team_workflow, load_agent_teams, run_team_workflow

STREAM_PATH = 'core.agent_manager.stream_llm_response'
GET_LLM_RESPONSE_PATH = 'core.agent_manager.get_llm_response'
ADD_TO_HISTORY_PATH = 'core.agent_manager.history.add_to_history'
TIME_STRFTIME_PATH = 'core.agent_manager.time.strftime'

MOCK_SETTINGS = {
    "ollama_base_url": "http://fake-manager-test:11434",
    "sweep_step_max_tokens": 500,
    "summary_step_max_tokens": 900,
}
MOCK_ROLES_DATA = {
    "RoleA": {"description": "Performs task A."},
    "RoleB": {"description": "Performs task B."},
    "RoleC_Refiner": {"description": "Refines previous steps."},
}
TEAM_CONCAT = {
    "description": "Concatenate A and B.", "assembly_strategy": "concatenate",
    "steps": [{"role": "RoleA", "goal": "Generate part A"}, {"role": "RoleB", "goal": "Generate part B based on A"}],
}
TEAM_REFINE = {
    "description": "Refine A and B with C.", "assembly_strategy": "refine_last",
    "steps": [{"role": "RoleA", "goal": "A"}, {"role": "RoleB", "goal": "B"}, {"role": "RoleC_Refiner", "goal": "Refine"}],
}
TEAM_MISSING_ROLE = {
    "description": "Team with a missing role.", "assembly_strategy": "concatenate",
    "steps": [{"role": "RoleA"}, {"goal": "Step without a role"}],
}
USER_INPUT = "Initial user request."
WORKER_MODEL = "test-worker-model"
MOCK_TIMESTAMP = "2024-01-01 12:00:00"


def streaming(*answers):
    """side_effect for stream_llm_response: each call streams one answer in two growing pieces."""
    remaining = list(answers)

    def _fake(**kwargs):
        answer = remaining.pop(0)
        if answer.startswith("⚠️ Error:"):
            return iter([answer])
        return iter([answer[: max(1, len(answer) // 2)], answer])
    return _fake


def run(team, **kwargs):
    args = dict(team_name="T", team_definition=team, user_input=USER_INPUT, initial_settings=MOCK_SETTINGS,
                all_roles_data=MOCK_ROLES_DATA, history_list=[], worker_model_name=WORKER_MODEL)
    args.update(kwargs)
    return run_team_workflow(**args)


def events(team, **kwargs):
    args = dict(team_name="T", team_definition=team, user_input=USER_INPUT, initial_settings=MOCK_SETTINGS,
                all_roles_data=MOCK_ROLES_DATA, history_list=[], worker_model_name=WORKER_MODEL)
    args.update(kwargs)
    return list(iter_team_workflow(**args))


@pytest.fixture(autouse=True)
def _quiet_history():
    with patch(ADD_TO_HISTORY_PATH, side_effect=lambda hist, entry: (hist or []) + [entry]) as mock_hist, \
         patch(TIME_STRFTIME_PATH, return_value=MOCK_TIMESTAMP):
        yield mock_hist


# --- Concatenate / step prompts ---

def test_concat_success_and_prompts():
    with patch(STREAM_PATH, side_effect=streaming("Output A", "Output B")) as mock_stream:
        final, hist, steps = run(TEAM_CONCAT)
    assert final == "Output A\n\nOutput B"
    assert steps is None # Not requested
    assert mock_stream.call_count == 2
    first, second = (c.kwargs for c in mock_stream.call_args_list)
    # Persona goes into the system prompt, not the user prompt
    assert first["system"] == "You are 'RoleA', one agent in a creative team workflow.\nPerforms task A."
    assert "Your Role" not in first["prompt"] and "RoleA" not in first["prompt"]
    assert "User Request: Initial user request." in first["prompt"]
    assert "Your Goal for this step: Generate part A" in first["prompt"]
    assert first["model"] == WORKER_MODEL and first["max_tokens"] == 500 and first["images"] is None
    # Step 2 sees step 1's output as context
    assert "Step 1 (RoleA) Output:\nOutput A" in second["prompt"]
    assert any("Workflow End: 'T'" in entry for entry in hist)


def test_iter_yields_progress_then_single_result():
    with patch(STREAM_PATH, side_effect=streaming("Output A", "Output B")):
        evs = events(TEAM_CONCAT)
    assert isinstance(evs[-1], TeamResult)
    assert sum(isinstance(e, TeamResult) for e in evs) == 1
    progress = [e for e in evs if isinstance(e, TeamProgress)]
    assert progress[0] == TeamProgress(1, 2, "RoleA", "", "Step 1/2: RoleA")
    assert TeamProgress(1, 2, "RoleA", "Outp", "Step 1/2: RoleA") in progress # Streamed partial
    assert any(p.step == 2 and p.partial_output == "Output B" for p in progress)
    assembling = progress[-1]
    assert assembling.step == 0 and "Assembling final output (concatenate)" in assembling.message


def test_unknown_role_gets_default_description():
    team = {"steps": [{"role": "Ghost"}]}
    with patch(STREAM_PATH, side_effect=streaming("boo")) as mock_stream:
        run(team)
    assert mock_stream.call_args.kwargs["system"].endswith("Perform your function.")
    assert "Your Goal for this step: Execute step 1" in mock_stream.call_args.kwargs["prompt"]


# --- Other strategies ---

def test_refine_last_returns_last_output():
    with patch(STREAM_PATH, side_effect=streaming("A", "B", "Refined final")):
        final, _, _ = run(TEAM_REFINE)
    assert final == "Refined final"


def test_structured_concatenate_adds_headers():
    team = dict(TEAM_CONCAT, assembly_strategy="structured_concatenate")
    with patch(STREAM_PATH, side_effect=streaming("A out", "B out")):
        final, _, _ = run(team)
    assert final == "--- Output from Agent: RoleA (Step 1) ---\nA out\n\n--- Output from Agent: RoleB (Step 2) ---\nB out"


def test_summarize_all_calls_summarizer():
    team = dict(TEAM_CONCAT, assembly_strategy="summarize_all")
    with patch(STREAM_PATH, side_effect=streaming("A out", "B out")), \
         patch(GET_LLM_RESPONSE_PATH, return_value="Summary") as mock_get:
        final, _, _ = run(team)
    assert final == "Summary"
    kwargs = mock_get.call_args.kwargs
    assert kwargs["max_tokens"] == 900 and "Output from Step 1 (RoleA):\nA out" in kwargs["prompt"]


def test_summarize_all_error_is_reported():
    team = dict(TEAM_CONCAT, assembly_strategy="summarize_all")
    with patch(STREAM_PATH, side_effect=streaming("A", "B")), \
         patch(GET_LLM_RESPONSE_PATH, return_value="⚠️ Error: down"):
        final, _, _ = run(team)
    assert final == "Error during final summarization step: ⚠️ Error: down"


@pytest.mark.parametrize("strategy, n_calls", [("metaphorical_synthesis", 2), ("stylistic_mashup", 2), ("conceptual_blend", 1)])
def test_creative_strategies_use_assembly_calls(strategy, n_calls):
    team = dict(TEAM_CONCAT, assembly_strategy=strategy)
    with patch(STREAM_PATH, side_effect=streaming("A", "B")), \
         patch(GET_LLM_RESPONSE_PATH, side_effect=['"a lighthouse"', "Creative final"][-n_calls:]) as mock_get:
        final, _, _ = run(team)
    assert final == "Creative final"
    assert mock_get.call_count == n_calls
    if n_calls == 2:
        assert "a lighthouse" in mock_get.call_args_list[1].kwargs["prompt"] # Quotes stripped, reused


def test_unknown_strategy_falls_back_to_concatenate():
    team = dict(TEAM_CONCAT, assembly_strategy="does-not-exist")
    with patch(STREAM_PATH, side_effect=streaming("A", "B")):
        final, _, _ = run(team)
    assert final == "A\n\nB"


# --- Intermediate steps, errors ---

def test_return_intermediate_steps():
    with patch(STREAM_PATH, side_effect=streaming("Out A", "Out B")):
        _, _, steps = run(TEAM_CONCAT, return_intermediate_steps=True)
    assert steps == {
        1: {"role": "RoleA", "goal": "Generate part A", "output": "Out A", "error": None},
        2: {"role": "RoleB", "goal": "Generate part B based on A", "output": "Out B", "error": None},
    }


@pytest.mark.parametrize("bad", [None, {}, {"steps": "nope"}])
def test_invalid_definition(bad):
    with patch(STREAM_PATH) as mock_stream:
        final, hist, steps = run(bad)
    assert final.startswith("Error: Invalid team definition")
    assert steps is None
    mock_stream.assert_not_called()
    assert any("Workflow Start Error" in e for e in hist)


def test_missing_role_step_is_skipped():
    with patch(STREAM_PATH, side_effect=streaming("Only A")) as mock_stream:
        final, hist, steps = run(TEAM_MISSING_ROLE, return_intermediate_steps=True)
    assert final == "Only A"
    assert mock_stream.call_count == 1
    assert steps[2]["output"] is None and "missing 'role'" in steps[2]["error"]
    assert any("Workflow Step 2 Skipped" in e for e in hist)


def test_agent_error_stops_workflow():
    with patch(STREAM_PATH, side_effect=streaming("A ok", "⚠️ Error: model crashed", "never")) as mock_stream:
        final, hist, steps = run(TEAM_REFINE, return_intermediate_steps=True)
    assert final == "Workflow stopped due to error in step 2 (RoleB): ⚠️ Error: model crashed"
    assert mock_stream.call_count == 2 # Step 3 never runs
    assert steps[2]["error"] == "⚠️ Error: model crashed"
    assert any("Workflow Step 2 Error" in e for e in hist)


def test_no_successful_steps():
    team = {"assembly_strategy": "concatenate", "steps": [{"goal": "no role"}]}
    with patch(STREAM_PATH) as mock_stream:
        final, _, _ = run(team)
    mock_stream.assert_not_called()
    assert final.startswith("Error: No successful outputs")


def test_run_team_workflow_without_result_event():
    with patch('core.agent_manager.iter_team_workflow', return_value=iter([TeamProgress(1, 1, "R", "", "")])):
        final, hist, steps = run_team_workflow(history_list=["h"])
    assert final.startswith("Error: Workflow ended without a result") and hist == ["h"] and steps is None


# --- Images ---

def test_numpy_image_does_not_crash_and_reaches_vision_model():
    """Regression: `if image:` on a numpy array raised 'The truth value of an array ... is ambiguous'."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with patch(STREAM_PATH, side_effect=streaming("A", "B")) as mock_stream:
        final, hist, _ = run(TEAM_CONCAT, single_image_input=image, worker_model_has_vision=True)
    assert final == "A\n\nB"
    sent = mock_stream.call_args_list[0].kwargs["images"]
    assert len(sent) == 1 and isinstance(sent[0], Image.Image) and sent[0].size == (8, 8)
    assert "Input includes a single image." in mock_stream.call_args_list[0].kwargs["prompt"]
    assert any("Image Provided: Yes" in e for e in hist)


def test_image_not_sent_to_non_vision_model():
    image = Image.new("RGB", (4, 4))
    with patch(STREAM_PATH, side_effect=streaming("A", "B")) as mock_stream:
        run(TEAM_CONCAT, single_image_input=image, worker_model_has_vision=False)
    assert all(c.kwargs["images"] is None for c in mock_stream.call_args_list)


def test_vision_falls_back_to_settings_models_data_when_flag_missing():
    settings = dict(MOCK_SETTINGS, loaded_models_data=[{"name": WORKER_MODEL, "vision": True}])
    with patch(STREAM_PATH, side_effect=streaming("A", "B")) as mock_stream:
        run(TEAM_CONCAT, initial_settings=settings, single_image_input=Image.new("RGB", (2, 2)))
    assert mock_stream.call_args_list[0].kwargs["images"] is not None


def test_empty_numpy_image_is_treated_as_no_image():
    with patch(STREAM_PATH, side_effect=streaming("A", "B")) as mock_stream:
        _, hist, _ = run(TEAM_CONCAT, single_image_input=np.zeros((0,), dtype=np.uint8), worker_model_has_vision=True)
    assert mock_stream.call_args_list[0].kwargs["images"] is None
    assert any("Image Provided: No" in e for e in hist)


# --- load_agent_teams ---

def test_load_agent_teams():
    with patch('core.agent_manager.load_json', return_value={"T": TEAM_CONCAT}):
        assert load_agent_teams() == {"T": TEAM_CONCAT}
    with patch('core.agent_manager.load_json', return_value=[]):
        assert load_agent_teams() == {}
