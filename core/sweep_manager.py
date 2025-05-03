# ArtAgent/core/sweep_manager.py
import gradio as gr # For progress updates potentially
import os
import time
import json
import hashlib # For hashing prompts in filenames if needed
import traceback # For logging detailed errors during workflow
import re # For sanitizing filenames

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

def sanitize_filename(name):
    """Removes or replaces characters unsafe for filenames."""
    # Remove leading/trailing whitespace
    name = name.strip()
    # Replace common problematic characters (:, /, \) with hyphens
    name = re.sub(r'[:/\\]', '-', name)
    # Remove other potentially problematic characters (allow letters, numbers, -, _)
    name = re.sub(r'[^\w\-.]', '', name)
    # Avoid empty names or names starting/ending with dots/hyphens
    if not name or name.startswith('.') or name.endswith('.') or name.startswith('-') or name.endswith('-'):
        return f"sanitized_{hashlib.md5(name.encode()).hexdigest()[:6]}" # Fallback
    return name


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
    Saves JSON protocols per run and separate TXT files of generated prompts per model.

    Args:
        base_prompts_text (str): Multiline string of base prompts.
        selected_teams (list[str]): List of team names to run.
        selected_models (list[str]): List of worker model names to run.
        output_folder_name (str): Name for the subfolder within SWEEP_OUTPUT_BASE_DIR.
        log_intermediate (bool): Whether to include intermediate step outputs in protocols.
        settings (dict): The current application settings dictionary.
        all_teams_data (dict): Dictionary containing definitions for all loaded teams.
        # progress (gradio.Progress): Gradio progress tracker object. # Uncomment if using progress

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
    try:
        # Pass empty dict for file_agents if sweep doesn't load specific files.
        # Ensure settings dict contains necessary flags for load_all_roles.
        all_roles_data = load_all_roles(settings, file_agents={})
        if not all_roles_data:
             print("Warning: No agent roles loaded based on current settings. Workflows might fail.")
    except Exception as e:
        return f"Error loading agent roles during sweep setup: {e}"

    # 4. Execute Sweep Loops
    total_runs = len(prompts) * len(selected_teams) * len(selected_models)
    completed_runs = 0
    status_updates = [] # Store short status lines for final summary
    prompt_file_handles = {} # Dictionary to keep prompt file handles open per model {model_name: file_handle}

    print(f"Total configurations to run: {total_runs}")
    # Initialize progress bar if used
    # progress(0, desc=f"Starting Sweep ({total_runs} runs)...")

    try: # Use try...finally to ensure prompt files are closed
        for p_idx, base_prompt in enumerate(prompts):
            prompt_label = f"Prompt {p_idx+1}/{len(prompts)}"
            prompt_hash = hashlib.md5(base_prompt.encode()).hexdigest()[:8]

            for t_idx, team_name in enumerate(selected_teams):
                team_label = f"Team '{team_name}'"
                team_definition = all_teams_data.get(team_name)

                if not team_definition:
                    msg = f"Skipping: Team definition for '{team_name}' not found."
                    print(msg); status_updates.append(msg)
                    completed_runs += len(selected_models)
                    continue

                for m_idx, model_name in enumerate(selected_models):
                    model_label = f"Model '{model_name}'"
                    run_label = f"{prompt_label}, {team_label}, {model_label}"
                    # Sanitize model name for filenames
                    sanitized_model_name = sanitize_filename(model_name)
                    run_id = f"{prompt_hash}_{sanitize_filename(team_name)}_{sanitized_model_name}"
                    print(f"\nRunning Configuration {completed_runs + 1}/{total_runs}: {run_label}")

                    # Update progress if used
                    # progress((completed_runs) / total_runs, desc=f"Running ({completed_runs+1}/{total_runs}): {model_label} on {team_label}...")

                    # Prepare data for this specific run's protocol
                    run_config = {
                        "agent_team_name": team_name,
                        "worker_model": model_name,
                        "log_intermediate_steps": log_intermediate,
                    }
                    protocol = {
                        "sweep_metadata": {
                            "base_user_prompt": base_prompt,
                            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "run_id": run_id
                        },
                        "configuration": run_config,
                        "execution_log": [] if log_intermediate else None,
                        "final_output": "Execution did not complete."
                    }

                    # Execute the workflow
                    run_status = "Unknown Error"
                    final_output = None # Initialize final_output
                    intermediate_steps = None # Initialize intermediate_steps

                    try:
                        # Pass empty list for history_list - sweep manager shouldn't modify persistent history directly during runs
                        final_output, _, intermediate_steps = agent_manager.run_team_workflow(
                            team_name=team_name,
                            team_definition=team_definition,
                            user_input=base_prompt,
                            initial_settings=settings,
                            all_roles_data=all_roles_data,
                            history_list=[], # Pass empty list for sweep mode
                            worker_model_name=model_name,
                            # Pass single_image_input=None as sweep currently doesn't handle image inputs
                            single_image_input=None,
                            return_intermediate_steps=log_intermediate
                        )
                        protocol["final_output"] = final_output
                        run_status = "Success"

                        # Log intermediate steps if requested and returned
                        if log_intermediate and intermediate_steps and isinstance(intermediate_steps, dict):
                            for step_idx, step_data in intermediate_steps.items():
                                 protocol["execution_log"].append({
                                     "step": step_idx,
                                     "agent_role": step_data.get("role", "N/A"),
                                     "goal": step_data.get("goal", "N/A"),
                                     "output": step_data.get("output"),
                                     "error": step_data.get("error")
                                 })
                        elif log_intermediate:
                             protocol["execution_log"] = "Intermediate steps requested but not returned/invalid."

                        # --- >>> NEW: Write successful prompt to model-specific file <<< ---
                        if final_output and not final_output.strip().startswith("Error:") and not final_output.strip().startswith("⚠️ Error:"):
                             prompt_filename = f"prompts_{sanitized_model_name}.txt"
                             prompt_filepath = os.path.join(output_dir, prompt_filename)
                             try:
                                 # Open file if not already open for this model
                                 if sanitized_model_name not in prompt_file_handles:
                                     prompt_file_handles[sanitized_model_name] = open(prompt_filepath, 'a+', encoding='utf-8')
                                     print(f"  Opened prompt file: {prompt_filename}")

                                 # Write the prompt
                                 prompt_file_handles[sanitized_model_name].write(final_output.strip() + '\n') # Ensure only one newline at the end

                             except Exception as e_prompt_write:
                                 write_error = f"Prompt Write Error: {e_prompt_write}"
                                 print(f"  ERROR writing to prompt file {prompt_filename}: {e_prompt_write}")
                                 if run_status == "Success": run_status = write_error # Update status if writing failed
                                 else: run_status += f" | {write_error}"
                        # --- <<< End NEW Prompt Writing Code >>> ---

                    except Exception as e:
                        print(f"ERROR during workflow execution for {run_label}: {e}")
                        traceback.print_exc()
                        error_str = f"ERROR during execution: {e}"
                        protocol["final_output"] = error_str
                        if protocol["execution_log"] is not None:
                            protocol["execution_log"].append({"step": "FATAL_ERROR", "error": str(e)})
                        run_status = f"Error: {e}"

                    # Save the protocol file
                    protocol_filename = f"{run_id}.json"
                    protocol_filepath = os.path.join(output_dir, protocol_filename)
                    try:
                        if not save_json(protocol_filepath, protocol, is_relative=False):
                             raise IOError("save_json utility returned False")
                        # print(f"  Protocol saved: {protocol_filename}") # Reduce console noise slightly
                    except Exception as e_save:
                        save_error = f"Protocol Save Error: {e_save}"
                        print(f"  ERROR saving protocol file {protocol_filename}: {e_save}")
                        if run_status == "Success": run_status = save_error
                        else: run_status += f" | {save_error}"

                    # Append concise status update
                    status_updates.append(f"Run {completed_runs + 1}: {run_label} -> {run_status[:100]}{'...' if len(run_status)>100 else ''}")
                    completed_runs += 1
                    # Update progress if used
                    # progress(completed_runs / total_runs)

    finally:
        # --- Ensure all prompt files are closed ---
        closed_count = 0
        for model_key, handle in prompt_file_handles.items():
            try:
                if handle and not handle.closed:
                    handle.close()
                    closed_count += 1
            except Exception as e_close:
                print(f"Warning: Error closing prompt file for model '{model_key}': {e_close}")
        if closed_count > 0: print(f"Closed {closed_count} prompt file(s).")


    # 5. Final Summary
    end_time = time.time()
    duration = end_time - start_time
    final_summary = (
        f"--- Sweep Complete ---\n"
        f"Total Runs Attempted: {completed_runs}/{total_runs}\n"
        f"Total Duration: {duration:.2f} seconds\n"
        f"Protocols and Prompt Files saved to: {output_dir}\n\n"
        f"Last {len(status_updates)} Run Statuses:\n" + "\n".join(status_updates)
    )
    print(final_summary)
    # Final progress update if used
    # progress(1.0, desc="Sweep Complete!")
    return final_summary