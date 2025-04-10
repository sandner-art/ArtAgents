# ArtAgent/core/app_logic.py
import gradio as gr
import json
import os
from PIL import Image # Ensure Image is imported
import numpy as np
import time

# Import necessary functions/classes from sibling modules or agents
from .utils import load_json, get_absolute_path
from . import history_manager as history
from agents.roles_config import load_all_roles, get_role_display_name, get_actual_role_name
from agents.ollama_agent import get_llm_response
# Import ollama_manager to call release_model and agent_manager for workflows
from . import ollama_manager
from . import agent_manager # Import the new manager

# --- Constants used by logic functions ---
SETTINGS_FILE = 'settings.json'
MODELS_FILE = 'models.json'
LIMITERS_FILE = 'limiters.json'
PROFILES_FILE = 'ollama_profiles.json' # Added for consistency if needed later
AGENT_TEAMS_FILE = 'agent_teams.json' # Define team file constant
DEFAULT_ROLES_FILE = 'agents/agent_roles.json' # Define here if needed by logic
CUSTOM_ROLES_FILE = 'agents/custom_agent_roles.json' # Define here if needed by logic

# --- Helper to save teams ---
def save_teams_to_file(teams_data):
    """Saves the teams dictionary to the JSON file."""
    full_path = get_absolute_path(AGENT_TEAMS_FILE)
    try:
        with open(full_path, 'w', encoding='utf-8') as file:
            json.dump(teams_data, file, indent=4, sort_keys=True) # Sort keys for consistency
        print(f"Agent teams saved successfully to {full_path}")
        return True
    except Exception as e:
        print(f"Error saving agent teams to {full_path}: {e}")
        return False

# --- Workflow Execution Router ---
def execute_chat_or_team(
    # UI Inputs (Common)
    folder_path, user_input, model_with_vision, max_tokens_ui,
    file_handling_option, limiter_handling_option, single_image_input, # Renamed from single_image_np
    use_ollama_api_options, release_model_on_change,
    # UI Inputs (Specific)
    selected_role_or_team, # This dropdown now selects EITHER a role OR a team
    # State Inputs
    current_settings,
    models_data_state,
    limiters_data_state,
    teams_data_state, # Loaded agent teams
    selected_model_tracker_value,
    file_agents_dict,
    history_list_state, # Persistent history list
    session_history_list_state # Current session history list
    ) -> tuple[str, str, str | None, list]: # Added type hints for return
    """
    Determines whether to run a single agent or an agent team workflow.

    Returns:
        tuple: (response_text, session_history_text, model_name_used_state, new_session_history_list)
    """
    print(f"\nExecuting chat/team with selection: '{selected_role_or_team}'")
    # Define return structure defaults
    response_text = "Error: Processing failed."
    session_history_text = "\n---\n".join(session_history_list_state)
    model_name_state_update = None # Will be None for team workflows
    # Create a working copy for the session history list
    new_session_history_list = list(session_history_list_state)

    # --- Check if it's a Team Workflow ---
    is_team_workflow = False
    team_name = None
    team_prefix = "[Team] "
    if selected_role_or_team and selected_role_or_team.startswith(team_prefix):
        team_name = selected_role_or_team[len(team_prefix):]
        if teams_data_state and team_name in teams_data_state:
            is_team_workflow = True
        else:
            print(f"Warning: Selected team '{team_name}' not found in loaded teams data.")
            response_text = f"Error: Selected Agent Team '{team_name}' definition not found."
            # Return immediately matching the 4-tuple signature
            return response_text, session_history_text, model_name_state_update, new_session_history_list

    if is_team_workflow:
        team_definition = teams_data_state[team_name]
        print(f"Running Agent Team Workflow: '{team_name}'")

        # Reload all roles data to pass to the manager (includes file agents)
        # Uses current_settings passed in from state
        all_roles_data = load_all_roles(current_settings, file_agents=file_agents_dict)

        # Determine the worker model (currently the one selected in UI)
        # Find actual model name from UI value (which might have '(VISION)' suffix)
        worker_model_name = None
        if not models_data_state: # Check if model data is available
             return "Error: Models data state is missing.", session_history_text, None, new_session_history_list
        for m in models_data_state:
            m_name = m.get("name")
            if not m_name: continue
            # Use model_with_vision passed to this function
            if m_name == model_with_vision or f"{m_name} (VISION)" == model_with_vision:
                worker_model_name = m_name
                break
        if not worker_model_name:
             # If called from captioning, model_with_vision might be None, try agent default? No, captioning passes model now.
             return f"Error: Could not determine worker model name for team from input '{model_with_vision}'.", session_history_text, None, new_session_history_list

        # Call the Agent Manager's workflow execution function
        # It returns the final output string, the *updated persistent history list*, and intermediate steps (or None)
        final_output, updated_persistent_history_list, _ = agent_manager.run_team_workflow(
            team_name=team_name,
            team_definition=team_definition,
            user_input=user_input,
            initial_settings=current_settings,
            all_roles_data=all_roles_data,
            history_list=list(history_list_state), # Pass copy of persistent history
            worker_model_name=worker_model_name,
            # TODO: Handle images if team workflow needs them (requires passing single_image_input)
        )
        # Persistent history state update is implicit via history.add_to_history inside manager

        # Update session history list with a summary of the workflow run
        summary_entry = f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\nWorkflow Run: '{team_name}' (Model: {worker_model_name})\nFinal Output Length: {len(final_output)}\n---\n(Full steps in Persistent History)"
        new_session_history_list.append(summary_entry)

        # Update return values
        response_text = final_output
        session_history_text = "\n---\n".join(new_session_history_list)
        model_name_state_update = None # No single model state for teams

    elif selected_role_or_team == "(Direct Agent Call)" or not selected_role_or_team:
        print("Info: Direct Agent Call selected, or no agent/team chosen.")
        response_text = "Please select a specific Agent or an Agent Team from the dropdown."
        # Use current session list for text display
        session_history_text = "\n---\n".join(new_session_history_list)

    else:
        # --- Assume it's a Single Agent Call ---
        print(f"Running Single Agent: '{selected_role_or_team}'")
        # Call the chat_logic function defined below
        # It returns: response_text, session_history_text, model_name, new_session_list
        response_text, session_history_text, model_name_state_update, new_session_history_list = chat_logic(
            # Pass all necessary arguments (including selected_role_or_team as role_display_name)
            folder_path=folder_path,
            role_display_name=selected_role_or_team, # The selected item is the role display name
            user_input=user_input,
            model_with_vision=model_with_vision, # Model name from UI/Caller
            max_tokens_ui=max_tokens_ui,
            file_handling_option=file_handling_option,
            limiter_handling_option=limiter_handling_option,
            single_image_input=single_image_input, # Pass the input image (PIL or Numpy)
            current_settings=current_settings, # Pass settings from state
            use_ollama_api_options=use_ollama_api_options,
            release_model_on_change=release_model_on_change,
            models_data_state=models_data_state,
            limiters_data_state=limiters_data_state,
            selected_model_tracker_value=selected_model_tracker_value,
            file_agents_dict=file_agents_dict,
            history_list_state=history_list_state,
            session_history_list_state=session_history_list_state # Pass current session list
        )

    # Return values matching the Gradio outputs binding for the submit button
    return response_text, session_history_text, model_name_state_update, new_session_history_list


# --- Single Agent Chat Logic ---
# CORRECTED to handle PIL Image input
def chat_logic(
    # UI Inputs
    folder_path, role_display_name, user_input, model_with_vision, max_tokens_ui,
    file_handling_option, limiter_handling_option, single_image_input, # Renamed for clarity
    use_ollama_api_options, release_model_on_change,
    # State Inputs
    current_settings, models_data_state, limiters_data_state,
    selected_model_tracker_value, file_agents_dict,
    history_list_state, session_history_list_state
    ) -> tuple[str, str, str | None, list]: # Added return type hint
    """Handles the core chat logic for a SINGLE agent, calling the Ollama agent."""
    # Use copies of mutable state lists
    history_list = list(history_list_state)
    current_session_history = list(session_history_list_state) # This is the list we modify

    # Get Actual Role Name from display name
    actual_role_name = get_actual_role_name(role_display_name)
    print(f"Single Agent Logic: Role='{actual_role_name}'")

    # Use settings PASSED IN to load roles relevant to this call
    roles_data_current = load_all_roles(current_settings, file_agents=file_agents_dict)

    # 1. Find Model Name and Info
    model_name = None; model_info = None
    if not models_data_state: return "Error: Models data not loaded.", "\n---\n".join(current_session_history), None, current_session_history
    if not model_with_vision: return "Error: No model specified for agent execution.", "\n---\n".join(current_session_history), None, current_session_history
    for m in models_data_state:
         m_name = m.get("name");
         if not m_name: continue
         # Match base name or display name (e.g., "llava:latest" or "llava:latest (VISION)")
         if m_name == model_with_vision or f"{m_name} (VISION)" == model_with_vision:
             model_name = m_name
             model_info = m
             break
    if not model_name or not model_info: return f"Error: Selected model info not found for '{model_with_vision}'.", "\n---\n".join(current_session_history), None, current_session_history
    print(f"  Using model: {model_name}")


    # 2. Handle Model Release
    if release_model_on_change and selected_model_tracker_value and selected_model_tracker_value != model_name:
        print(f"Requesting release of previous model: {selected_model_tracker_value}")
        ollama_manager.release_model(selected_model_tracker_value, current_settings.get("ollama_url"))

    # 3. Prepare Prompt and Options
    limiter_settings = limiters_data_state.get(limiter_handling_option, {})
    limiter_prompt_format = limiter_settings.get("limiter_prompt_format", "")
    limiter_token_slider = limiter_settings.get("limiter_token_slider")
    effective_max_tokens = min(max_tokens_ui, limiter_token_slider) if limiter_handling_option != "Off" and limiter_token_slider is not None else max_tokens_ui
    role_description = roles_data_current.get(actual_role_name, {}).get("description", "Unknown Role")
    # Construct prompt - ensure limiter format comes before user input if applicable
    prompt_parts = [f"Role: {actual_role_name} - {role_description}"]
    if limiter_prompt_format: prompt_parts.append(limiter_prompt_format)
    prompt_parts.append(f"User Input: {user_input}")
    prompt = "\n".join(prompt_parts) + "\n" # Add final newline

    agent_ollama_options = {} # Agent function will handle merging based on settings/role/etc

    # --- 4. Handle Image Input (REVISED LOGIC) ---
    pil_images_list = []
    is_single_image_mode = False
    image_source_info = "[None - Text Only]"

    # Check if input is not None and if the selected model supports vision
    if single_image_input is not None and model_info.get("vision"):
        # Check if input is already a PIL Image (coming from captioning logic)
        # Use PIL.Image directly for the check
        if isinstance(single_image_input, Image.Image):
            print("  Processing PIL Image input.")
            pil_images_list = [single_image_input]
            is_single_image_mode = True
            image_source_info = "[Single PIL Image]"
        # Check if input is a numpy array (coming from Gradio Image component)
        elif isinstance(single_image_input, np.ndarray):
            print("  Processing Numpy Image input.")
            try:
                pil_image = Image.fromarray(single_image_input.astype('uint8'))
                pil_images_list = [pil_image]
                is_single_image_mode = True
                image_source_info = "[Single Upload/Numpy]"
            except Exception as e:
                return f"Error processing single numpy image: {e}", "\n---\n".join(current_session_history), model_name, current_session_history
        else:
            # Handle unexpected input type if necessary
            print(f"Warning: Received single image input of unexpected type: {type(single_image_input)}. Ignoring.")

    elif single_image_input is not None and not model_info.get("vision"):
        print("Warning: Single image provided, but selected model lacks vision.")
        image_source_info = "[Single Image Ignored - No Vision]"
    # --- End Revised Image Handling ---

    # --- Add Debug Print for final arguments to ollama_agent ---
    print(f"DEBUG CHAT_LOGIC: Calling get_llm_response with:")
    print(f"  Role: {actual_role_name}")
    print(f"  Model: {model_name}")
    print(f"  Prompt Start: {prompt[:150]}...")
    print(f"  Images Type: {type(pil_images_list)}, Count: {len(pil_images_list)}")
    if pil_images_list: print(f"  First Image Type in list: {type(pil_images_list[0])}")
    print(f"  Max Tokens: {effective_max_tokens}")
    # --- End Debug Print ---

    # 5. Determine Mode and Call Agent
    final_response = "Processing..."
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    entry_prefix = f"Timestamp: {timestamp}\nRole: {actual_role_name}\nModel: {model_name}\nInput: {user_input}"

    if is_single_image_mode:
         # Single Image Call (PIL object is now in pil_images_list)
         print(f"Calling agent (Single Image): model={model_name}, role={actual_role_name}, image_source={image_source_info}")
         final_response = get_llm_response(
             role=actual_role_name, prompt=prompt, model=model_name, settings=current_settings,
             roles_data=roles_data_current, images=pil_images_list, max_tokens=effective_max_tokens,
             ollama_api_options=agent_ollama_options
         )
         # Replace potential base64 data with placeholder for history log display
         log_image_source_info = "[Image Data Sent]" if pil_images_list else image_source_info
         entry = f"{entry_prefix}\nImage: {log_image_source_info}\nResponse:\n{final_response}\n---\n"
         history_list = history.add_to_history(history_list, entry) # Updates persistent
         current_session_history.append(entry) # Updates session list copy

    elif folder_path and os.path.isdir(folder_path):
        # Folder Processing Call (Existing logic)
        if not model_info.get("vision"):
             print("Warning: Folder path provided, but model lacks vision. Processing text-only.")
             final_response = get_llm_response(
                 role=actual_role_name, prompt=prompt, model=model_name, settings=current_settings,
                 roles_data=roles_data_current, images=None, max_tokens=effective_max_tokens,
                 ollama_api_options=agent_ollama_options
            )
             entry = f"{entry_prefix}\nImage: [None - Folder Ignored]\nResponse:\n{final_response}\n---\n"
             history_list = history.add_to_history(history_list, entry)
             current_session_history.append(entry)
        else:
             # Process folder with vision model (image by image)
             print(f"Processing image folder: {folder_path}")
             confirmation_messages = []; processed_files = 0; base_prompt = prompt
             try:
                 files_in_folder = sorted([
                     f for f in os.listdir(folder_path)
                     if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))
                 ])
             except Exception as list_e: return f"Error listing folder: {list_e}", "\n---\n".join(current_session_history), model_name, current_session_history

             for file_name in files_in_folder:
                 file_path = os.path.join(folder_path, file_name)
                 loop_img = None # Initialize loop_img
                 try:
                      print(f"  Processing file: {file_name}")
                      loop_img = Image.open(file_path)
                      image_prompt = f"{base_prompt}\nImage Context: Analyzing '{file_name}'\n"
                      img_response = get_llm_response(
                           role=actual_role_name, prompt=image_prompt, model=model_name, settings=current_settings,
                           roles_data=roles_data_current, images=[loop_img], max_tokens=effective_max_tokens,
                           ollama_api_options=agent_ollama_options
                      )

                      # --- File Handling Logic ---
                      base_name = os.path.splitext(file_name)[0]
                      output_file = os.path.join(folder_path, f"{base_name}.txt")
                      action_taken = "Skipped"
                      file_exists = os.path.exists(output_file)
                      original_content = ""

                      if file_exists and file_handling_option != "Overwrite" and file_handling_option != "Skip":
                          try:
                              with open(output_file, 'r', encoding='utf-8') as f:
                                  original_content = f.read().strip()
                          except Exception as read_e:
                              print(f"  Warning: Could not read existing file {output_file}: {read_e}")

                      # Decide action based on mode and existence
                      if file_handling_option == "Overwrite" or not file_exists:
                          if img_response: # Only write if response is not empty
                              with open(output_file, 'w', encoding='utf-8') as f:
                                  f.write(img_response)
                              action_taken = "Written" if not file_exists else "Overwritten"
                          else: action_taken = "Skipped (Empty Response)"
                      elif file_handling_option == "Append":
                          if img_response:
                              separator = "\n\n---\n\n" if original_content else ""
                              with open(output_file, 'w', encoding='utf-8') as f:
                                  f.write(original_content + separator + img_response)
                              action_taken = "Appended"
                          else: action_taken = "Skipped (Empty Response)"
                      elif file_handling_option == "Prepend":
                          if img_response:
                              separator = "\n\n---\n\n" if original_content else ""
                              with open(output_file, 'w', encoding='utf-8') as f:
                                  f.write(img_response + separator + original_content)
                              action_taken = "Prepended"
                          else: action_taken = "Skipped (Empty Response)"
                      # Skip case is handled by default action_taken="Skipped"

                      # --- End File Handling ---

                      confirmation_messages.append(f"  - {file_name}: {action_taken} -> {base_name}.txt")
                      # History Update per Image
                      entry = f"{entry_prefix}\nImage: {file_name} [Data Sent]\nResponse:\n{img_response}\n---\n" # Placeholder for image data
                      history_list = history.add_to_history(history_list, entry); current_session_history.append(entry); processed_files += 1

                 except Exception as e:
                      error_msg = f"Error processing file '{file_name}': {e}"
                      print(f"  {error_msg}")
                      confirmation_messages.append(f"  - {file_name}: Error - {e}")
                      error_entry = f"Timestamp: {timestamp}\nRole: {actual_role_name}\nModel: {model_name}\nInput: {user_input}\nImage: {file_name}\nERROR: {e}\n---\n"
                      history_list = history.add_to_history(history_list, error_entry); current_session_history.append(error_entry)
                 finally:
                      if loop_img:
                          try: loop_img.close()
                          except Exception as e_close: print(f"  Warning: Error closing loop image {file_name}: {e_close}")

             if processed_files == 0: final_response = "No valid image files found or processed in the directory."
             else: final_response = f"Folder processing complete ({processed_files} files):\n" + "\n".join(confirmation_messages)

    elif single_image_input is None and (not folder_path or not os.path.isdir(folder_path)):
        # Text-Only Call (No image passed)
         print(f"Calling agent (Text Only): model={model_name}, role={actual_role_name}")
         final_response = get_llm_response(
             role=actual_role_name, prompt=prompt, model=model_name, settings=current_settings,
             roles_data=roles_data_current, images=None, max_tokens=effective_max_tokens, # Pass images=None
             ollama_api_options=agent_ollama_options
         )
         entry = f"{entry_prefix}\nImage: {image_source_info}\nResponse:\n{final_response}\n---\n"
         history_list = history.add_to_history(history_list, entry)
         current_session_history.append(entry)


    # Return response text, session history text, model name, and updated session list
    return final_response, "\n---\n".join(current_session_history), model_name, current_session_history


# --- Comment Logic ---
def comment_logic(
    llm_response_text, comment, max_tokens_ui,
    use_ollama_api_options,
    model_state_value, current_settings, file_agents_dict,
    history_list_state, session_history_list_state
    ) -> tuple[str, str, list]:
    """Handles the commenting logic."""
    history_list = list(history_list_state)
    current_session_history = list(session_history_list_state)

    if not comment or not model_state_value:
        print("Comment ignored: No comment text or model state.")
        # Return unchanged values, matching output signature (3 values)
        return llm_response_text, "\n---\n".join(current_session_history), current_session_history

    # Use settings PASSED IN
    roles_data_current = load_all_roles(current_settings, file_agents=file_agents_dict)
    print(f"Processing comment on model {model_state_value}...")
    role = "User"; role_description = roles_data_current.get(role, {}).get("description", "Follow up based on comment.")
    prompt = f"Previous Response:\n{llm_response_text}\n\nUser Comment/Instruction:\n{comment}\n\nPlease revise or continue based on the comment."
    agent_ollama_options = {} # Agent handles merging

    response = get_llm_response(
        role=role, prompt=prompt, model=model_state_value, settings=current_settings,
        roles_data=roles_data_current, images=None, max_tokens=max_tokens_ui,
        ollama_api_options=agent_ollama_options
    )

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    entry = f"Timestamp: {timestamp}\nRole: {role} (Comment)\nModel: {model_state_value}\nInput: {comment}\nContext: Previous response\nResponse:\n{response}\n---\n"
    history_list = history.add_to_history(history_list, entry)
    current_session_history.append(entry)

    # Return new response text, new session history text, and new session history list
    return response, "\n---\n".join(current_session_history), current_session_history


# --- UI Callbacks ---

def update_max_tokens_on_limiter_change(limiter_choice, current_max_tokens_value):
    """Updates the max token slider value based on limiter selection."""
    current_limiters_data = load_json(LIMITERS_FILE, is_relative=True)
    if limiter_choice == "Off": return gr.update()

    limiter_settings = current_limiters_data.get(limiter_choice, {})
    limiter_token_val = limiter_settings.get("limiter_token_slider")
    if limiter_token_val is not None:
         try:
             limiter_token_int = int(limiter_token_val)
             # Slider value might be float, handle potential conversion errors
             try:
                 current_max_tokens_int = int(float(current_max_tokens_value))
             except (ValueError, TypeError):
                 print(f"Warning: Could not convert current max_tokens slider value '{current_max_tokens_value}' to int.")
                 return gr.update() # Don't update if current value is invalid

             if limiter_token_int != current_max_tokens_int:
                  print(f"Limiter '{limiter_choice}' updating max_tokens slider to: {limiter_token_int}")
                  return gr.update(value=limiter_token_int)
         except (ValueError, TypeError): print(f"Warning: Invalid numeric value for limiter token value '{limiter_token_val}'.")
    return gr.update()


def clear_session_history_callback(session_history_list_state):
    """Clears the session history list state."""
    print("Session history cleared.")
    # Returns empty string for display, and empty list for the state
    return "", [] # Match outputs = [display, state]


# This callback's logic is now handled by the WRAPPER in app.py
# def update_role_dropdown_callback(use_default, use_custom, file_agents_dict, teams_data_state, current_settings_dict):
#     pass


def handle_agent_file_upload(uploaded_file):
    """Processes uploaded agent JSON file. Returns dict, filename, update obj for dropdown."""
    if uploaded_file is None:
        print("Agent file upload cleared.")
        # Needs to return 3 values matching outputs list in app.py upload handler
        # The third value signals Gradio to run the .then() chain
        return {}, None, gr.update()

    try:
        file_path = uploaded_file.name; file_basename = os.path.basename(file_path)
        print(f"Processing uploaded agent file: {file_basename}")
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read().strip()
        if not content: raise ValueError("Uploaded agent file is empty.")
        loaded_agents = json.loads(content)
        if not isinstance(loaded_agents, dict): raise ValueError("Uploaded file is not a dict.")
        # Basic validation
        valid_agents = {}
        for key, value in loaded_agents.items():
            if isinstance(value, dict) and 'description' in value: valid_agents[key] = value
            else: print(f"Warning: Agent '{key}' in file '{file_basename}' skipped (invalid format).")
        if not valid_agents: raise ValueError("No valid agent definitions found.")
        print(f"Successfully loaded {len(valid_agents)} agent(s) from {file_basename}.")
        # Return loaded dict, base filename, and trigger dropdown update via .then()
        # IMPORTANT: Return the update object directly
        return valid_agents, file_basename, gr.Dropdown.update()
    except (json.JSONDecodeError, ValueError) as e: error_msg = f"Invalid Agent File: {e}"; print(error_msg); gr.Warning(error_msg); return {}, None, gr.Dropdown.update()
    except Exception as e: error_msg = f"Error processing agent file: {e}"; print(error_msg); gr.Warning(error_msg); return {}, None, gr.Dropdown.update()


def load_profile_options_callback(profile_name_to_load, all_profiles_data_state, ordered_option_keys_state):
    """Loads profile options onto UI components. Returns list of updates."""
    print(f"Loading profile: {profile_name_to_load}")
    ordered_keys = ordered_option_keys_state
    num_outputs = len(ordered_keys) if ordered_keys else 0
    no_change_updates = [gr.update() for _ in range(num_outputs)] # Create list of no-op updates

    if not profile_name_to_load or not all_profiles_data_state or not ordered_keys:
        print("Cannot load profile: Missing profile name, data, or ordered keys.")
        return no_change_updates

    profile_options = all_profiles_data_state.get(profile_name_to_load)
    if not profile_options or not isinstance(profile_options, dict):
        print(f"Profile '{profile_name_to_load}' not found or invalid.")
        return no_change_updates

    updates_list = []
    for key in ordered_keys:
        if key in profile_options:
            new_value = profile_options[key]
            print(f"  Updating '{key}' UI to: {new_value}")
            updates_list.append(gr.update(value=new_value))
        else:
            # Send no-change update if key not in profile
            updates_list.append(gr.update())

    # Safety check
    if len(updates_list) != num_outputs:
         print(f"Error: Mismatch generating profile updates. Expected {num_outputs}, got {len(updates_list)}.")
         return no_change_updates

    return updates_list


def save_settings_callback(
    # General Settings Inputs
    ollama_url_in, max_tokens_slider_range_in, api_to_console_in,
    use_default_in, use_custom_in, use_ollama_opts_default_in,
    release_model_default_in, theme_select_in,
    # State Inputs
    ordered_option_keys_state, # Keys in order
    # API Options Inputs (passed via *args)
    *api_option_values):
    """Saves all settings back to settings.json."""
    print("Saving application settings...")
    try:
        settings_path = get_absolute_path(SETTINGS_FILE)
        # Use load_json which handles errors and returns dict
        current_settings = load_json(settings_path, is_relative=False)
        if not isinstance(current_settings, dict): current_settings = {}

        previous_theme = current_settings.get("gradio_theme", "Default")
        # Update general values
        current_settings.update({
            "ollama_url": ollama_url_in, "max_tokens_slider": max_tokens_slider_range_in,
            "ollama_api_prompt_to_console": api_to_console_in, "using_default_agents": use_default_in,
            "using_custom_agents": use_custom_in, "use_ollama_api_options": use_ollama_opts_default_in,
            "release_model_on_change": release_model_default_in, "gradio_theme": theme_select_in
        })
        # Update ollama_api_options
        current_settings["ollama_api_options"] = current_settings.get("ollama_api_options", {}) # Ensure exists
        ordered_keys = ordered_option_keys_state
        num_expected = len(ordered_keys); num_received = len(api_option_values)
        # Reload initial options just before saving to get original types accurately
        initial_options_for_types = load_json(settings_path, is_relative=False).get("ollama_api_options", {})

        if ordered_keys and num_expected == num_received:
            api_options_updates = current_settings["ollama_api_options"].copy() # Work on a copy
            for key, value in zip(ordered_keys, api_option_values):
                original_value = initial_options_for_types.get(key)
                try: # Attempt type conversion
                    # Check explicit types first for safer conversion
                    if isinstance(original_value, bool): api_options_updates[key] = bool(value)
                    elif isinstance(original_value, int): api_options_updates[key] = int(value)
                    elif isinstance(original_value, float): api_options_updates[key] = float(value)
                    else: api_options_updates[key] = value # Save as received type otherwise
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert saved value '{value}' for option '{key}' based on original type {type(original_value)}. Saving as received. Error: {e}")
                    api_options_updates[key] = value
            current_settings["ollama_api_options"] = api_options_updates # Assign updated dict
            print(f" Ollama API options updated ({num_expected} values).")
        elif not ordered_keys and num_received == 0:
             print("No API options defined or received, skipping update.")
        else: print(f"Warning: API Options count mismatch ({num_expected} vs {num_received}). Options NOT saved.")

        with open(settings_path, 'w', encoding='utf-8') as file: json.dump(current_settings, file, indent=4)
        print(f"Settings saved successfully to {settings_path}")

        save_msg = "Settings saved successfully."
        if theme_select_in != previous_theme: save_msg += " Restart application to apply theme change."
        return save_msg
    except Exception as e:
        error_msg = f"Error saving settings: {e}"; print(error_msg); return error_msg


# --- Team Editor Callbacks ---

def load_team_for_editing(team_name_to_load, all_teams_data_state):
    """Loads a selected team's data into the editor fields."""
    print(f"Loading team '{team_name_to_load}' into editor.")
    empty_state = {"name": "", "description": "", "steps": [], "assembly_strategy": "concatenate"}
    if not team_name_to_load or not all_teams_data_state:
        # Clear editor if no team selected
        return "", "", [], "concatenate", empty_state

    team_data = all_teams_data_state.get(team_name_to_load)
    if not team_data or not isinstance(team_data, dict):
        print(f"Error: Team data not found or invalid for '{team_name_to_load}'.")
        error_state = {"name": team_name_to_load, "description": "Error: Team data not found.", "steps": [], "assembly_strategy": "concatenate"}
        # Match the 5 return values expected by the UI binding
        return team_name_to_load, "Error: Team data not found.", [], "concatenate", error_state

    # Ensure steps is always a list, even if missing/null in JSON
    steps = team_data.get("steps", [])
    if not isinstance(steps, list): steps = []

    editor_state = {
        "name": team_name_to_load,
        "description": team_data.get("description", ""),
        "steps": steps,
        "assembly_strategy": team_data.get("assembly_strategy", "concatenate")
    }
    print(f" Loaded: {editor_state}")
    # Outputs: name_tb, desc_tb, steps_json, strategy_radio, editor_state
    return (
        editor_state["name"], editor_state["description"],
        editor_state["steps"], editor_state["assembly_strategy"], editor_state
    )


def clear_team_editor():
    """Clears the editor fields for creating a new team."""
    print("Clearing team editor.")
    empty_state = {"name": "", "description": "", "steps": [], "assembly_strategy": "concatenate"}
    # Outputs: name_tb, desc_tb, steps_json, strategy_radio, editor_state
    return "", "", [], "concatenate", empty_state


def add_step_to_editor(agent_role_display_name_to_add, current_editor_state):
    """Adds the selected agent as a new step to the current editor state."""
    if not agent_role_display_name_to_add:
        print("Add Step: No agent selected."); gr.Warning("Please select an agent role to add.")
        # Return state and no-update for JSON display
        return current_editor_state, gr.update()

    # Ensure state is a dictionary, default if not
    editor_state = current_editor_state.copy() if isinstance(current_editor_state, dict) else {"name": "", "description": "", "steps": [], "assembly_strategy": "concatenate"}

    if "steps" not in editor_state or not isinstance(editor_state["steps"], list):
        editor_state["steps"] = []

    actual_role_name = get_actual_role_name(agent_role_display_name_to_add) # Use actual name
    new_step = {"role": actual_role_name} # Add goal later if UI exists
    editor_state["steps"].append(new_step)
    print(f"Add Step: Added '{actual_role_name}'. Steps: {len(editor_state['steps'])}")
    # Outputs: editor_state, steps_json_display
    return editor_state, editor_state["steps"]


def remove_step_from_editor(step_index_to_remove, current_editor_state):
    """Removes a step at the specified index (1-based) from the editor state."""
    # Ensure state is a dictionary
    editor_state = current_editor_state.copy() if isinstance(current_editor_state, dict) else {"name": "", "description": "", "steps": [], "assembly_strategy": "concatenate"}
    steps = editor_state.get("steps", [])
    if not isinstance(steps, list): steps = [] # Ensure list

    try: # Validate index
        index_0_based = int(step_index_to_remove) - 1
        if not (0 <= index_0_based < len(steps)): # Check bounds concisely
            raise IndexError
    except (ValueError, TypeError, IndexError):
        msg = f"Invalid step number '{step_index_to_remove}' (must be 1 to {len(steps)})."
        print(f"Remove Step: {msg}"); gr.Warning(msg)
        # Return state and no-update for JSON display
        return editor_state, gr.update()

    removed_step = steps.pop(index_0_based);
    editor_state["steps"] = steps # Update steps in the state copy
    print(f"Remove Step: Removed step {step_index_to_remove} ('{removed_step.get('role')}'). Steps: {len(steps)}")
    # Outputs: editor_state, steps_json_display
    return editor_state, editor_state["steps"]


# --- save_team_from_editor: Adjustments needed to return 5 outputs ---
def save_team_from_editor(
    # Current editor UI values
    team_name_in, description_in, assembly_strategy_in,
    # State values
    current_editor_state, # Contains the steps list
    all_teams_data_state,
    current_settings_state, # Needed for reloading roles for dropdown update
    file_agents_dict_state  # Needed for reloading roles for dropdown update
    ) -> tuple[dict, dict, dict, dict, str]: # Added return type hint matching 5 outputs
    """Saves the team currently defined in the editor fields. Returns 5 values."""
    print("Attempting to save team...")
    team_name = team_name_in.strip()
    # Define default outputs for 5 return values
    default_outputs = [all_teams_data_state, gr.update(), gr.update(), gr.update(), "Save failed."]

    if not team_name: msg = "Team Name cannot be empty."; print(msg); gr.Error(msg); return default_outputs[0:4] + [msg]

    steps = current_editor_state.get("steps", [])
    if not steps: msg = f"Save Warning: Team '{team_name}' has no steps."; print(msg); gr.Warning(msg) # Allow save

    all_teams_data = all_teams_data_state.copy() if isinstance(all_teams_data_state, dict) else {}
    team_data = {"description": description_in.strip(), "steps": steps, "assembly_strategy": assembly_strategy_in}
    all_teams_data[team_name] = team_data # Add or overwrite

    if save_teams_to_file(all_teams_data):
        msg = f"Team '{team_name}' saved successfully."
        print(msg)
        # Reload roles using current settings and file agents state
        combined_roles = load_all_roles(current_settings_state, file_agents=file_agents_dict_state)
        file_agent_keys = list(file_agents_dict_state.keys()) if file_agents_dict_state else []
        role_display_choices = sorted([get_role_display_name(name, file_agent_keys) for name in combined_roles.keys()])
        new_team_choices = sorted(list(all_teams_data.keys()))
        new_chat_choices = ["(Direct Agent Call)"] + sorted([f"[Team] {name}" for name in new_team_choices]) + role_display_choices
        # Outputs: teams_data_state, editor_team_dd, chat_role_dd, caption_agent_dd, status_textbox # FIXED
        return (
            all_teams_data,
            gr.Dropdown.update(choices=new_team_choices, value=team_name),
            gr.Dropdown.update(choices=new_chat_choices, value=f"[Team] {team_name}"),
            gr.Dropdown.update(choices=new_chat_choices, value=f"[Team] {team_name}"), # Update caption DD too
            msg
        )
    else:
        msg = f"Error: Failed to save team data."; print(msg); gr.Error(msg)
        return default_outputs[0:4] + [msg] # Return original state, update status

# --- delete_team_logic: Adjustments needed to return 10 outputs ---
def delete_team_logic(
    team_name_to_delete,
    all_teams_data_state,
    current_settings_state, # Needed for reloading roles for dropdown update
    file_agents_dict_state  # Needed for reloading roles for dropdown update
    ) -> tuple[dict, dict, dict, dict, str, str, str, list, str, dict]: # Added return type hint matching 10 outputs
    """Deletes the selected team. Returns 10 values."""
    print(f"Attempting to delete team: '{team_name_to_delete}'")
    # Define default clear values and no-change outputs for 10 returns
    clear_name, clear_desc, clear_steps, clear_strat, clear_state = clear_team_editor()
    no_change_outputs = [
        all_teams_data_state, gr.update(), gr.update(), gr.update(), # States/DDs
        "Delete failed.", # Status
        clear_name, clear_desc, clear_steps, clear_strat, clear_state # Clear outputs
    ]

    if not team_name_to_delete:
        msg = "No team selected."; print(msg); gr.Warning(msg)
        return no_change_outputs[0:4] + [msg] + no_change_outputs[5:]
    all_teams_data = all_teams_data_state.copy() if isinstance(all_teams_data_state, dict) else {}
    if team_name_to_delete not in all_teams_data:
        msg = f"Team '{team_name_to_delete}' not found."; print(msg); gr.Warning(msg)
        return no_change_outputs[0:4] + [msg] + no_change_outputs[5:]

    del all_teams_data[team_name_to_delete]

    if save_teams_to_file(all_teams_data):
        msg = f"Team '{team_name_to_delete}' deleted successfully."
        print(msg)
        # Reload roles/teams for dropdown updates
        combined_roles = load_all_roles(current_settings_state, file_agents=file_agents_dict_state)
        file_agent_keys = list(file_agents_dict_state.keys()) if file_agents_dict_state else []
        role_display_choices = sorted([get_role_display_name(name, file_agent_keys) for name in combined_roles.keys()])
        new_team_choices = sorted(list(all_teams_data.keys()))
        new_chat_choices = ["(Direct Agent Call)"] + sorted([f"[Team] {name}" for name in new_team_choices]) + role_display_choices
        # Return updates: teams_state, editor_dd, chat_dd, caption_dd, status_msg, + clear outputs # FIXED
        return (
            all_teams_data,
            gr.Dropdown.update(choices=new_team_choices, value=None),
            gr.Dropdown.update(choices=new_chat_choices, value="(Direct Agent Call)"),
            gr.Dropdown.update(choices=new_chat_choices, value="(Direct Agent Call)"), # Update caption DD too
            msg,
            clear_name, clear_desc, clear_steps, clear_strat, clear_state
        )
    else:
        msg = f"Error: Failed to save teams after deletion."; print(msg); gr.Error(msg)
        # Don't update state/dropdowns if save failed, just status and editor clear
        return no_change_outputs[0:4] + [msg] + no_change_outputs[5:]


# --- History Callbacks ---
def show_clear_confirmation(): return gr.update(visible=True)
def hide_clear_confirmation(): return gr.update(visible=False)
def clear_full_history_callback(history_list_state):
    """Clears the persistent history file and returns updates."""
    print("Full history file cleared.")
    history.save_history([]) # Save empty list via manager
    # Return empty string for display, hide group, empty list for state
    return "", hide_clear_confirmation(), [] # Match outputs