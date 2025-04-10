# ArtAgent/core/agent_manager.py
import time
from .utils import load_json
from agents.ollama_agent import get_llm_response # To call individual agents
from . import history_manager as history # To log steps

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
    history_list: list, # Pass the current persistent history list
    # Add other necessary args like model if manager needs specific one?
    # For now, assume workers use the model selected in UI
    worker_model_name: str
    ) -> tuple[str, list]:
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

    Returns:
        tuple[str, list]: A tuple containing:
            - str: The final assembled output string (prompt).
            - list: The updated history_list with execution steps logged.
    """
    print(f"\n--- Running Agent Team Workflow: {team_name} ---")
    if not team_definition or "steps" not in team_definition:
        print("Error: Invalid team definition provided.")
        return "Error: Invalid team definition.", history_list

    steps = team_definition.get("steps", [])
    assembly_strategy = team_definition.get("assembly_strategy", "concatenate") # Default to simple concat
    # manager_role = team_definition.get("manager_role") # For future use

    step_outputs = {}
    current_context = f"User Request: {user_input}\nWorkflow Goal: {team_definition.get('description', 'Generate a detailed output.')}\n"
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    # Log initial user request for this workflow context
    initial_log = f"Timestamp: {timestamp}\nWorkflow Start: '{team_name}'\nUser Input: {user_input}\n---\n"
    history_list = history.add_to_history(history_list, initial_log)

    # --- Execute Sequence ---
    for i, step in enumerate(steps):
        step_role = step.get("role")
        step_goal = step.get("goal", f"Execute step {i+1}") # Default goal if not specified
        if not step_role:
            print(f"Warning: Step {i+1} in team '{team_name}' missing 'role'. Skipping.")
            continue

        print(f"\nStep {i+1}/{len(steps)}: Running Agent '{step_role}'...")
        print(f"  Goal: {step_goal}")
        print(f"  Current Context Length: {len(current_context)}")

        # Construct prompt for this step's agent
        # Include role description from all_roles_data
        role_info = all_roles_data.get(step_role, {})
        role_desc = role_info.get("description", "Perform your function.")
        step_prompt = f"Context:\n{current_context}\nYour Role: {step_role} - {role_desc}\nYour Goal for this step: {step_goal}\n\nBased *only* on the provided context and your goal, provide your specific output:"

        # --- Call the LLM ---
        # Use settings passed in, role data passed in, and specific worker model
        # Max tokens for intermediate steps might need adjustment? Use a default or from step definition?
        step_max_tokens = 500 # Default max tokens for intermediate steps? Or pass via settings?
        step_output = get_llm_response(
            role=step_role, # Use the actual role name
            prompt=step_prompt,
            model=worker_model_name,
            settings=initial_settings,
            roles_data=all_roles_data, # Pass full roles data
            max_tokens=step_max_tokens
            # Pass other necessary args like images if workflow supports them later
        )

        # Basic check if agent returned an error message
        if step_output.strip().startswith("⚠️ Error:"):
             print(f"  Agent '{step_role}' returned an error: {step_output}")
             # Log the error and potentially stop the workflow?
             error_log = f"Timestamp: {timestamp}\nWorkflow Step {i+1} Error ('{step_role}')\nError Message: {step_output}\nContext Provided:\n{current_context}\n---\n"
             history_list = history.add_to_history(history_list, error_log)
             # Option 1: Stop workflow on first error
             return f"Workflow stopped due to error in step {i+1} ({step_role}): {step_output}", history_list
             # Option 2: Store error and continue (might be messy)
             # step_outputs[step_role] = f"ERROR: {step_output}"
             # current_context += f"\nStep {i+1} ({step_role}) Output: ERROR\n"
             # continue # Continue to next step

        print(f"  Agent '{step_role}' Output Length: {len(step_output)}")
        step_outputs[step_role] = step_output.strip() # Store output, keyed by role for now
        current_context += f"\nStep {i+1} ({step_role}) Output:\n{step_output.strip()}\n"

        # Log intermediate step
        step_log = f"Timestamp: {timestamp}\nWorkflow Step {i+1}: '{step_role}'\nGoal: {step_goal}\nOutput:\n{step_output.strip()}\n---\n"
        history_list = history.add_to_history(history_list, step_log)

    # --- Assemble Final Output ---
    print("\n--- Assembling Final Workflow Output ---")
    final_output = ""
    if assembly_strategy == "concatenate":
        print("Strategy: Concatenate")
        # Simple concatenation in order of execution
        for role_name in [s.get("role") for s in steps if s.get("role")]: # Iterate in step order
            if role_name in step_outputs:
                final_output += f"--- Contribution: {role_name} ---\n"
                final_output += step_outputs[role_name] + "\n\n"
        final_output = final_output.strip()

    elif assembly_strategy == "refine_last":
        print("Strategy: Refine Last Step")
        # Find the role of the *last* step that successfully produced output
        refiner_role = None
        refiner_step_output = None
        for step in reversed(steps):
            role = step.get("role")
            if role and role in step_outputs: # Check if output exists for this role
                 refiner_role = role
                 refiner_step_output = step_outputs[role]
                 break # Use the last successful one

        if refiner_role and refiner_step_output:
            print(f"Using output from final Refiner Agent: '{refiner_role}'")
            # Perform an additional call to the refiner agent if needed?
            # The current prompt assumes the last step *is* the refiner.
            # For now, just return the output of that last step directly.
            final_output = refiner_step_output
            # Optionally, prepend context?
            # final_output = f"Final Prompt based on multi-agent process:\n{refiner_step_output}"

            # --- TODO: Implement actual refinement call if needed ---
            # Example: Call a 'PromptRefiner' agent again with all collected context
            # refiner_prompt = f"Context:\n{current_context}\n\nCombine the previous steps into a single, concise, effective text-to-image prompt (approx 1400 chars max)."
            # final_output = get_llm_response(role="PromptRefiner", prompt=refiner_prompt, ...)
        else:
             print("Warning: Refine strategy selected, but couldn't find output from the last step. Concatenating instead.")
             # Fallback to concatenation if refiner failed or wasn't last
             for role_name in [s.get("role") for s in steps if s.get("role")]:
                 if role_name in step_outputs: final_output += step_outputs[role_name] + "\n\n"
             final_output = final_output.strip()
             if not final_output: final_output = "Error: No valid output generated by workflow steps."

    else:
        print(f"Warning: Unknown assembly strategy '{assembly_strategy}'. Concatenating.")
        for role_name in [s.get("role") for s in steps if s.get("role")]:
             if role_name in step_outputs: final_output += step_outputs[role_name] + "\n\n"
        final_output = final_output.strip()

    # Log final assembly
    final_log = f"Timestamp: {timestamp}\nWorkflow End: '{team_name}'\nAssembly Strategy: {assembly_strategy}\nFinal Output:\n{final_output}\n---\n"
    history_list = history.add_to_history(history_list, final_log)

    print(f"--- Workflow {team_name} Finished. Final Output Length: {len(final_output)} ---")
    return final_output, history_list # Return final string and updated history list