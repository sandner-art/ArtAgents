# ArtAgent/core/sweep_manager.py
import gradio as gr # For progress updates potentially
import os
import time
import json
import hashlib # For hashing prompts in filenames if needed
import traceback # For logging detailed errors during workflow

# Import necessary functions/classes from sibling modules or agents
from .utils import get_absolute_path # Utility for path handling
# Assuming save_json is defined in utils or implement simple save here
try:
    from .utils import save_json
except ImportError:
    print("Warning: save_json not found in core.utils. Implementing basic save in sweep_manager.")
    def save_json(filepath, data, is_relative=False): # is_relative not used here
        """Basic JSON save function."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving JSON to {filepath}: {e}")
            return False
from . import agent_manager # Import the agent manager module
# Import the main workflow execution function
from .agent_manager import run_team_workflow
# Import function to load all roles (needed if run_team_workflow doesn't handle it internally based on settings)
from agents.roles_config import load_all_roles

SWEEP_OUTPUT_BASE_DIR = "sweep_runs" # Folder relative to project root

def run_sweep(
    base_prompts_text: str,
    selected_teams: list[str],
    selected_models: list[str],
    output_folder_name: str,
    log_intermediate: bool,
    # Need data passed from state via app.py
    settings: dict,
    all_teams_data: dict,
    # Pass file agents state if role loading depends on it dynamically here
    # file_agents_dict: dict, # Example if needed
    # Gradio progress object needs to be the *last* argument if used with type hints
    # progress=gr.Progress(track_tqdm=True)
    ) -> str: # Returns final status message
    """
    Runs the experiment sweep based on selected prompts, teams, and models.

    Args:
        base_prompts_text (str): Multiline string of base prompts.
        selected_teams (list[str]): List of team names to run.
        selected_models (list[str]): List of worker model names to run.
        output_folder_name (str): Name for the subfolder within SWEEP_OUTPUT_BASE_DIR.
        log_intermediate (bool): Whether to include intermediate step outputs in protocols.
        settings (dict): The current application settings dictionary.
        all_teams_data (dict): Dictionary containing definitions for all loaded teams.
        progress (gradio.Progress): Gradio progress tracker object.

    Returns:
        str: A summary message indicating completion status and output location.
    """
    start_time = time.time()
    print("\n--- Starting Experiment Sweep ---")

    # 1. Validate Inputs
    if not base_prompts_text or not base_prompts_text.strip():
        return "Error: No base prompts provided."
    if not selected_teams:
        return "Error: No Agent Teams selected."
    if not selected_models:
        return "Error: No Worker Models selected."

    # Sanitize folder name slightly
    safe_folder_name = "".join(c for c in output_folder_name.strip() if c.isalnum() or c in ('-', '_'))
    if not safe_folder_name: safe_folder_name = "sweep_results"

    prompts = [p.strip() for p in base_prompts_text.strip().splitlines() if p.strip()]
    if not prompts: return "Error: No valid prompts found after stripping."

    # 2. Prepare Output Directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"{timestamp}_{safe_folder_name}"
    output_dir = get_absolute_path(os.path.join(SWEEP_OUTPUT_BASE_DIR, run_folder_name))
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory created: {output_dir}")
    except Exception as e:
        return f"Error creating output directory '{output_dir}': {e}"

    # 3. Load necessary data (roles)
    # This assumes roles don't change during the sweep based on file uploads.
    # Pass empty dict for file_agents if not tracking that state during sweep.
    try:
        all_roles_data = load_all_roles(settings, file_agents={})
        if not all_roles_data:
             print("Warning: No agent roles loaded based on current settings. Workflows might fail.")
    except Exception as e:
        return f"Error loading agent roles during sweep setup: {e}"


    # 4. Execute Sweep Loops
    total_runs = len(prompts) * len(selected_teams) * len(selected_models)
    completed_runs = 0
    status_updates = [] # Store short status lines for final summary

    print(f"Total configurations to run: {total_runs}")
    # Initialize progress bar
    # progress(0, desc=f"Starting Sweep ({total_runs} runs)...")

    for p_idx, base_prompt in enumerate(prompts):
        prompt_label = f"Prompt {p_idx+1}/{len(prompts)}"
        # Use hash for filename part to avoid issues with long/special chars in prompts
        prompt_hash = hashlib.md5(base_prompt.encode()).hexdigest()[:8]

        for t_idx, team_name in enumerate(selected_teams):
            team_label = f"Team '{team_name}'"
            team_definition = all_teams_data.get(team_name)

            if not team_definition:
                msg = f"Skipping: Team definition for '{team_name}' not found."
                print(msg); status_updates.append(msg)
                # Increment completed runs by the number of models we are skipping for this team
                completed_runs += len(selected_models)
                continue # Skip to the next team

            for m_idx, model_name in enumerate(selected_models):
                model_label = f"Model '{model_name}'"
                run_label = f"{prompt_label}, {team_label}, {model_label}"
                run_id = f"{prompt_hash}_{team_name.replace(' ','-')}_{model_name.replace(':','-').replace('/','_')}"
                print(f"\nRunning Configuration {completed_runs + 1}/{total_runs}: {run_label}")

                # Update progress description for the current run
                # progress((completed_runs) / total_runs, desc=f"Running ({completed_runs+1}/{total_runs}): {model_label} on {team_label}...")

                # Prepare data for this specific run's protocol
                run_config = {
                    "agent_team_name": team_name,
                    "worker_model": model_name,
                    "log_intermediate_steps": log_intermediate,
                    # Optionally add effective ollama options used if needed
                }
                protocol = {
                    "sweep_metadata": {
                        "base_user_prompt": base_prompt,
                        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "run_id": run_id
                    },
                    "configuration": run_config,
                    "execution_log": [] if log_intermediate else None, # Initialize only if needed
                    "final_output": "Execution did not complete." # Default
                }

                # Execute the workflow
                run_status = "Unknown Error" # Default status
                try:
                    # Call run_team_workflow - assuming it handles history internally for logging
                    # but doesn't require the full list passed in for sweep mode.
                    # Expecting return: (final_output, updated_persistent_history, step_outputs_dict | None)
                    final_output, _, intermediate_steps = agent_manager.run_team_workflow(
                        team_name=team_name,
                        team_definition=team_definition,
                        user_input=base_prompt,
                        initial_settings=settings,
                        all_roles_data=all_roles_data, # Pass loaded roles
                        history_list=[], # Pass empty list, manager shouldn't rely on it
                        worker_model_name=model_name,
                        return_intermediate_steps=log_intermediate # Request intermediate steps if needed
                    )
                    protocol["final_output"] = final_output
                    run_status = "Success" # Mark success if no exception

                    # Log intermediate steps if requested and returned
                    if log_intermediate and intermediate_steps and isinstance(intermediate_steps, dict):
                        for step_idx, step_data in intermediate_steps.items():
                             protocol["execution_log"].append({
                                 "step": step_idx,
                                 "agent_role": step_data.get("role", "N/A"),
                                 "goal": step_data.get("goal", "N/A"),
                                 "output": step_data.get("output"), # Can be None if error occurred
                                 "error": step_data.get("error") # Will be None if no error
                             })
                    elif log_intermediate:
                         protocol["execution_log"] = "Intermediate steps requested but not returned/invalid."


                except Exception as e:
                    print(f"ERROR during workflow execution for {run_label}: {e}")
                    traceback.print_exc() # Print full traceback for debugging
                    error_str = f"ERROR during execution: {e}"
                    protocol["final_output"] = error_str
                    if protocol["execution_log"] is not None: # Add error to log if intermediate logging is on
                        protocol["execution_log"].append({"step": "FATAL_ERROR", "error": str(e)})
                    run_status = f"Error: {e}"

                # Save the protocol file
                protocol_filename = f"{run_id}.json"
                protocol_filepath = os.path.join(output_dir, protocol_filename)
                try:
                    # Use save_json utility if available and handles absolute paths
                    if not save_json(protocol_filepath, protocol, is_relative=False):
                         raise IOError("save_json utility returned False")
                    print(f"  Protocol saved: {protocol_filename}")
                except Exception as e_save:
                    save_error = f"Protocol Save Error: {e_save}"
                    print(f"  ERROR saving protocol file {protocol_filename}: {e_save}")
                    # Update run status without overwriting original execution status if it was success
                    if run_status == "Success": run_status = save_error
                    else: run_status += f" | {save_error}"

                # Append concise status update for final summary
                status_updates.append(f"Run {completed_runs + 1}: {run_label} -> {run_status[:100]}{'...' if len(run_status)>100 else ''}")
                completed_runs += 1
                # Update progress slightly more often inside the innermost loop
                # progress(completed_runs / total_runs)


    # 5. Final Summary
    end_time = time.time()
    duration = end_time - start_time
    final_summary = (
        f"--- Sweep Complete ---\n"
        f"Total Runs Attempted: {completed_runs}/{total_runs}\n"
        f"Total Duration: {duration:.2f} seconds\n"
        f"Protocols saved to: {output_dir}\n\n"
        f"Last {len(status_updates)} Run Statuses:\n" + "\n".join(status_updates)
    )
    print(final_summary)
    # Final progress update
    # progress(1.0, desc="Sweep Complete!")
    return final_summary