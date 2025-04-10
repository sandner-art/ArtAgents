# ArtAgent/core/agent_manager.py
import time
from .utils import load_json # Utility for loading team definitions if needed elsewhere
from agents.ollama_agent import get_llm_response # To call individual agents
from . import history_manager as history # To log steps to persistent history

AGENT_TEAMS_FILE = 'agent_teams.json' # Relative path from root

def load_agent_teams(filepath=AGENT_TEAMS_FILE):
    """Loads agent team definitions from a JSON file."""
    teams_data = load_json(filepath, is_relative=True)
    if not isinstance(teams_data, dict):
        print(f"Warning: Agent teams file '{filepath}' did not contain a valid dictionary. Returning empty.")
        return {}
    return teams_data

def run_team_workflow(
    team_name: str,
    team_definition: dict,
    user_input: str,
    initial_settings: dict,
    all_roles_data: dict, # Pass the combined dict of all available roles
    history_list: list, # Pass the current persistent history list for logging
    worker_model_name: str,
    return_intermediate_steps: bool = False # Argument to control return value
    ) -> tuple[str, list, dict | None]: # Updated return signature
    """
    Executes a defined agent team workflow.

    Args:
        team_name (str): The name of the team being executed.
        team_definition (dict): The dictionary defining the team's steps and strategy.
        user_input (str): The initial user request.
        initial_settings (dict): Current application settings (for URL, global options).
        all_roles_data (dict): Dictionary containing definitions for all available agent roles.
        history_list (list): The current persistent history list (passed by value, returns updated).
        worker_model_name (str): The model selected in the UI for worker agents.
        return_intermediate_steps (bool): If True, return dict of step outputs. Defaults to False.

    Returns:
        tuple[str, list, dict | None]: A tuple containing:
            - str: The final assembled output string (prompt).
            - list: The updated history_list with execution steps logged.
            - dict | None: Dictionary of intermediate step outputs if requested, else None.
                           Format: {step_index: {"role": ..., "goal": ..., "output": ..., "error": ...}}
    """
    print(f"\n--- Running Agent Team Workflow: {team_name} ---")
    if not team_definition or not isinstance(team_definition.get("steps"), list):
        msg = "Error: Invalid team definition provided (must be dict with a 'steps' list)."
        print(msg)
        # Log error to history as well
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        error_log = f"Timestamp: {timestamp}\nWorkflow Start Error: '{team_name}'\nError: {msg}\nUser Input: {user_input}\n---\n"
        history_list = history.add_to_history(history_list, error_log)
        return msg, history_list, None

    steps = team_definition.get("steps", [])
    assembly_strategy = team_definition.get("assembly_strategy", "concatenate") # Default to simple concat
    step_outputs_dict = {} # Store intermediate results {step_index: {details}}

    current_context = f"User Request: {user_input}\nWorkflow Goal: {team_definition.get('description', 'Generate detailed output.')}\n"
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    initial_log = f"Timestamp: {timestamp}\nWorkflow Start: '{team_name}'\nUser Input: {user_input}\n---\n"
    # Log start to persistent history
    history_list = history.add_to_history(history_list, initial_log)

    # --- Execute Sequence ---
    for i, step in enumerate(steps):
        step_idx = i + 1 # Use 1-based indexing for logging/keys
        step_role = step.get("role")
        step_goal = step.get("goal", f"Execute step {step_idx}") # Default goal if not specified
        # Initialize result dict for this step (for intermediate logging)
        step_result = {"role": step_role, "goal": step_goal, "output": None, "error": None}

        if not step_role:
            msg = f"Warning: Step {step_idx} in team '{team_name}' missing 'role'. Skipping."
            print(msg)
            step_result["error"] = msg
            step_outputs_dict[step_idx] = step_result # Store skipped step info
            # Log skip to persistent history
            skip_log = f"Timestamp: {timestamp}\nWorkflow Step {step_idx} Skipped ('{team_name}')\nReason: Missing 'role' definition.\n---\n"
            history_list = history.add_to_history(history_list, skip_log)
            continue # Continue to the next step

        print(f"\nStep {step_idx}/{len(steps)}: Running Agent '{step_role}'...")
        print(f"  Goal: {step_goal}")
        print(f"  Current Context Length: {len(current_context)}")

        # Construct prompt for this step's agent
        role_info = all_roles_data.get(step_role, {}) # Look up role details
        role_desc = role_info.get("description", "Perform your function.")
        # Provide context, define role/goal for the agent
        step_prompt = f"Context:\n{current_context}\n---\nYour Role: {step_role} - {role_desc}\nYour Goal for this step: {step_goal}\n\nBased *only* on the provided context and your goal, provide your specific output:"

        # --- Call the LLM using get_llm_response ---
        # Consider adding configurable token limit per step in team definition later
        step_max_tokens = initial_settings.get("sweep_step_max_tokens", 750) # Example: Default max for intermediate steps

        step_output_text = get_llm_response(
            role=step_role, # Use the actual role name from step definition
            prompt=step_prompt,
            model=worker_model_name,
            settings=initial_settings,
            roles_data=all_roles_data, # Pass full roles data for option merging
            max_tokens=step_max_tokens,
            # Pass other args like images if workflow handles them
            # ollama_api_options can be defined per-step in team definition too (future enhancement)
        )

        # Check if agent returned an error message (starts with warning emoji)
        if step_output_text.strip().startswith("⚠️ Error:"):
            error_msg = step_output_text.strip()
            print(f"  Agent '{step_role}' returned an error: {error_msg}")
            step_result["error"] = error_msg
            step_outputs_dict[step_idx] = step_result # Store error result

            # Log error to persistent history and stop the workflow
            error_log = f"Timestamp: {timestamp}\nWorkflow Step {step_idx} Error ('{step_role}')\nError Message: {error_msg}\nContext Provided (start):\n{current_context[:500]}...\n---\n"
            history_list = history.add_to_history(history_list, error_log)
            # Return error message, updated history, and collected step outputs so far
            return f"Workflow stopped due to error in step {step_idx} ({step_role}): {error_msg}", history_list, step_outputs_dict
        else:
            clean_output = step_output_text.strip()
            print(f"  Agent '{step_role}' Output Length: {len(clean_output)}")
            step_result["output"] = clean_output
            step_outputs_dict[step_idx] = step_result # Store successful result

            # Build context for the NEXT step
            current_context += f"\n---\nStep {step_idx} ({step_role}) Output:\n{clean_output}\n"

            # Log successful step to persistent history
            step_log = f"Timestamp: {timestamp}\nWorkflow Step {step_idx}: '{step_role}'\nGoal: {step_goal}\nOutput:\n{clean_output}\n---\n"
            history_list = history.add_to_history(history_list, step_log)

    # --- Assemble Final Output ---
    print("\n--- Assembling Final Workflow Output ---")
    final_output = ""
    # Use the step_outputs_dict which contains results (output or error) for assembly
    # Filter for steps that actually produced non-error output for assembly strategies
    successful_outputs = {idx: data["output"] for idx, data in step_outputs_dict.items() if data.get("output") is not None and data.get("error") is None}

    if assembly_strategy == "concatenate":
        print("Strategy: Concatenate")
        if not successful_outputs:
             final_output = "Error: No successful outputs generated by workflow steps to concatenate."
        else:
            # Concatenate in order of step index using successful outputs only
            assembly_list = []
            for idx in sorted(successful_outputs.keys()):
                 # Find role associated with this index
                 step_info = next((s for i, s in enumerate(steps) if i + 1 == idx), None)
                 role_name = step_info.get("role", f"Step {idx}") if step_info else f"Step {idx}"
                 assembly_list.append(f"--- Contribution: {role_name} ---\n{successful_outputs[idx]}")
            final_output = "\n\n".join(assembly_list)

    elif assembly_strategy == "refine_last":
        print("Strategy: Refine Last Step")
        # Find the output of the *last* step definition that ran successfully
        last_successful_idx = None
        for i in range(len(steps) - 1, -1, -1): # Iterate backwards through defined steps
             step_idx_check = i + 1
             if step_idx_check in successful_outputs: # Check if this step succeeded
                  last_successful_idx = step_idx_check
                  break

        if last_successful_idx:
             step_info = next((s for i, s in enumerate(steps) if i + 1 == last_successful_idx), None)
             role_name = step_info.get("role", f"Step {last_successful_idx}") if step_info else f"Step {last_successful_idx}"
             print(f"Using output from final successful Agent: '{role_name}' (Step {last_successful_idx})")
             final_output = successful_outputs[last_successful_idx]
             # Optional: Implement a separate "refiner" call here if the last step wasn't implicitly the refiner.
             # refiner_prompt = f"Context:\n{current_context}\n\nCombine into concise prompt..."
             # final_output = get_llm_response(role="PromptRefiner", ..., context=current_context)
        else:
             print("Warning: Refine strategy selected, but couldn't find output from any successful step. No output generated.")
             final_output = "Error: No valid output generated by workflow steps for refinement."

    else: # Default fallback to concatenate
        print(f"Warning: Unknown assembly strategy '{assembly_strategy}'. Concatenating.")
        if not successful_outputs:
             final_output = "Error: No successful outputs generated by workflow steps to concatenate."
        else:
            assembly_list = []
            for idx in sorted(successful_outputs.keys()):
                 step_info = next((s for i, s in enumerate(steps) if i + 1 == idx), None)
                 role_name = step_info.get("role", f"Step {idx}") if step_info else f"Step {idx}"
                 assembly_list.append(successful_outputs[idx]) # Just append output for simple concat
            final_output = "\n\n".join(assembly_list) # Join with double newline


    # Log final assembly to persistent history
    final_log = f"Timestamp: {timestamp}\nWorkflow End: '{team_name}'\nAssembly Strategy: {assembly_strategy}\nFinal Output:\n{final_output}\n---\n"
    history_list = history.add_to_history(history_list, final_log)

    print(f"--- Workflow {team_name} Finished. Output Length: {len(final_output)} ---")

    # Return final output, updated history list, and intermediate steps if requested
    # The intermediate steps dict includes errors if they occurred for logging purposes.
    return final_output, history_list, (step_outputs_dict if return_intermediate_steps else None)