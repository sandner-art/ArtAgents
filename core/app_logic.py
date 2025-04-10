# ArtAgent/core/app_logic.py
import gradio as gr
import json
import os
from PIL import Image
import numpy as np
import time

# Import necessary functions/classes from sibling modules or agents
from .utils import load_json, get_absolute_path
from . import history_manager as history
from agents.roles_config import load_all_roles, get_role_display_name, get_actual_role_name
from agents.ollama_agent import get_llm_response
# Import ollama_manager to call release_model
from . import ollama_manager

# --- Constants used by logic functions ---
SETTINGS_FILE = 'settings.json'
MODELS_FILE = 'models.json'
LIMITERS_FILE = 'limiters.json'

# --- Chat Logic ---
def chat_logic(
    # UI Inputs
    
    folder_path, role_display_name, user_input, model_with_vision, max_tokens_ui,
    file_handling_option, limiter_handling_option, single_image_np,
    use_ollama_api_options, release_model_on_change,
    # State Inputs
    current_settings, # Settings dict from state
    models_data_state, # models.json content from state
    limiters_data_state, # limiters.json content from state
    selected_model_tracker_value, # Previous model name from state
    file_agents_dict, # Agents loaded from file from state
    history_list_state, # Full history list from state
    session_history_list_state # Current session history list from state
    ):
    """Handles the core chat logic, calling the Ollama agent."""
    print(f"Chat Logic Triggered: Role='{role_display_name}', Model='{model_with_vision}'")

    # Use copies of mutable state lists
    history_list = list(history_list_state)
    current_session_history = list(session_history_list_state) # This is the list we modify

    # Get Actual Role Name
    actual_role_name = get_actual_role_name(role_display_name)

    # Reload roles_data to include file agents for *this* call
    roles_data_current = load_all_roles(current_settings, file_agents=file_agents_dict)

    # 1. Find Model Name and Info from state
    model_name = None
    model_info = None
    if not models_data_state or not isinstance(models_data_state, list):
         return "Error: Models data not loaded.", "\n---\n".join(current_session_history), None
    for m in models_data_state:
         m_name = m.get("name")
         if not m_name: continue
         if m_name == model_with_vision or f"{m_name} (VISION)" == model_with_vision:
              model_name = m_name
              model_info = m
              break
    if not model_name or not model_info: return "Error: Selected model info not found.", "\n---\n".join(current_session_history), None

    # 2. Handle Model Release (Call function from ollama_manager)
    if release_model_on_change and selected_model_tracker_value and selected_model_tracker_value != model_name:
        print(f"Requesting release of previous model: {selected_model_tracker_value}")
        ollama_manager.release_model(selected_model_tracker_value, current_settings.get("ollama_url"))

    # 3. Prepare Prompt and Options
    limiter_settings = limiters_data_state.get(limiter_handling_option, {})
    limiter_prompt_format = limiter_settings.get("limiter_prompt_format", "")
    limiter_token_slider = limiter_settings.get("limiter_token_slider")
    effective_max_tokens = min(max_tokens_ui, limiter_token_slider) if limiter_handling_option != "Off" and limiter_token_slider is not None else max_tokens_ui

    role_description = roles_data_current.get(actual_role_name, {}).get("description", "Unknown Role")
    prompt = f"Role: {actual_role_name} - {role_description}\n{limiter_prompt_format}\nUser Input: {user_input}\n"

    agent_ollama_options = {}
    if use_ollama_api_options: pass # Agent function handles merging

    # 4. Handle Image Input
    pil_images_list = []
    is_single_image_mode = False
    image_source_info = "[None - Text Only]"
    if single_image_np is not None and isinstance(single_image_np, np.ndarray):
        if model_info.get("vision"):
            try:
                pil_image = Image.fromarray(single_image_np.astype('uint8'))
                pil_images_list = [pil_image]; is_single_image_mode = True; image_source_info = "[Single Upload]"
            except Exception as e: return f"Error processing single image: {e}", "\n---\n".join(current_session_history), model_name
        else: print("Warning: Single image provided, but model lacks vision.")

    # 5. Determine Mode and Call Agent
    final_response = "Processing..."
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    entry_prefix = f"Timestamp: {timestamp}\nRole: {actual_role_name}\nModel: {model_name}\nInput: {user_input}"

    if is_single_image_mode or (not folder_path or not os.path.isdir(folder_path)):
        print(f"Calling agent (Single/Text): model={model_name}, role={actual_role_name}")
        final_response = get_llm_response(role=actual_role_name, prompt=prompt, model=model_name, settings=current_settings, roles_data=roles_data_current, images=pil_images_list, max_tokens=effective_max_tokens, ollama_api_options=agent_ollama_options)
        entry = f"{entry_prefix}\nImage: {image_source_info}\nResponse:\n{final_response}\n---\n"
        history_list = history.add_to_history(history_list, entry)
        current_session_history.append(entry)

    elif folder_path and os.path.isdir(folder_path):
        if not model_info.get("vision"):
            print("Warning: Folder path provided, but model lacks vision. Processing text-only.")
            final_response = get_llm_response(role=actual_role_name, prompt=prompt, model=model_name, settings=current_settings, roles_data=roles_data_current, images=None, max_tokens=effective_max_tokens, ollama_api_options=agent_ollama_options)
            entry = f"{entry_prefix}\nImage: [None - Folder Ignored]\nResponse:\n{final_response}\n---\n"
            history_list = history.add_to_history(history_list, entry)
            current_session_history.append(entry)
        else:
            print(f"Processing image folder: {folder_path}")
            confirmation_messages = []
            processed_files = 0
            base_prompt = prompt
            files_in_folder = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])

            for file_name in files_in_folder:
                file_path = os.path.join(folder_path, file_name)
                try:
                    print(f"  Processing file: {file_name}")
                    image = Image.open(file_path)
                    image_prompt = f"{base_prompt}\nImage Context: Analyzing '{file_name}'\n"
                    img_response = get_llm_response(role=actual_role_name, prompt=image_prompt, model=model_name, settings=current_settings, roles_data=roles_data_current, images=[image], max_tokens=effective_max_tokens, ollama_api_options=agent_ollama_options)

                    # File Handling (Simplified logic example)
                    base_name = os.path.splitext(file_name)[0]
                    output_file = os.path.join(folder_path, f"{base_name}.txt")
                    action_taken = "Skipped"
                    # ... (Implement Overwrite, Skip, Append, Prepend logic) ...
                    confirmation_messages.append(f"  - {file_name}: {action_taken} -> {base_name}.txt")

                    # History Update per Image
                    entry = f"{entry_prefix}\nImage: {file_name}\nResponse:\n{img_response}\n---\n"
                    history_list = history.add_to_history(history_list, entry)
                    current_session_history.append(entry)
                    processed_files += 1
                except Exception as e:
                    error_msg = f"Error processing file {file_name}: {e}"; print(f"  {error_msg}"); confirmation_messages.append(f"  - {file_name}: Error - {e}")

            if processed_files == 0: final_response = "No valid image files found/processed."
            else: final_response = f"Folder processing complete ({processed_files} files):\n" + "\n".join(confirmation_messages)

    # Return response text, history text, and model name used
    # Must match outputs=[llm_response_display, current_session_history_display, model_state]
    return final_response, "\n---\n".join(current_session_history), model_name, current_session_history


# --- Comment Logic ---
def comment_logic(
    # UI Inputs
    llm_response_text, comment, max_tokens_ui,
    use_ollama_api_options,
    # State Inputs
    model_state_value, # Model used in last response
    current_settings,
    file_agents_dict, # Agents loaded from file
    history_list_state,
    session_history_list_state
    ):
    """Handles the commenting logic."""
    print(f"Comment Logic Triggered: Model='{model_state_value}'")

    # Use copies of mutable state lists
    history_list = list(history_list_state)
    current_session_history = list(session_history_list_state)

    if not comment or not model_state_value: return llm_response_text, "\n---\n".join(current_session_history)

    # Reload roles
    roles_data_current = load_all_roles(current_settings, file_agents=file_agents_dict)
    role = "User" # Fixed role for comment
    role_description = roles_data_current.get(role, {}).get("description", "Follow up based on comment.")
    prompt = f"Previous Response:\n{llm_response_text}\n\nUser Comment/Instruction:\n{comment}\n\nPlease revise or continue based on the comment."
    agent_ollama_options = {}
    if use_ollama_api_options: pass

    response = get_llm_response(
        role=role, prompt=prompt, model=model_state_value, settings=current_settings,
        roles_data=roles_data_current, images=None, max_tokens=max_tokens_ui,
        ollama_api_options=agent_ollama_options
    )

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    entry = f"Timestamp: {timestamp}\nRole: {role} (Comment)\nModel: {model_state_value}\nInput: {comment}\nContext: Previous response\nResponse:\n{response}\n---\n"
    history_list = history.add_to_history(history_list, entry)
    current_session_history.append(entry)

    # Must match outputs=[llm_response_display, current_session_history_display]
    return response, "\n---\n".join(current_session_history), current_session_history

# --- UI Callbacks ---

def update_max_tokens_on_limiter_change(limiter_choice, current_max_tokens_value):
    """Updates the max token slider value based on limiter selection."""
    # Reload limiters data in case it changed
    current_limiters_data = load_json(LIMITERS_FILE, is_relative=True)
    if limiter_choice == "Off": return gr.update() # No change

    limiter_settings = current_limiters_data.get(limiter_choice, {})
    limiter_token_val = limiter_settings.get("limiter_token_slider")
    if limiter_token_val is not None and int(limiter_token_val) != int(current_max_tokens_value):
        print(f"Limiter '{limiter_choice}' updating max_tokens slider to: {limiter_token_val}")
        return gr.update(value=limiter_token_val)
    return gr.update()

def clear_session_history_callback(session_history_list_state):
    """Clears the session history list."""
    print("Session history cleared.")
    # Returns empty string for display, and empty list for the state
    return "", [] # Match outputs = [display, state]

def update_role_dropdown_callback(use_default, use_custom, file_agents_dict):
    """Reloads roles based on settings flags and file agents, updates dropdown."""
    print("Updating role dropdown...")
    current_settings = load_json(SETTINGS_FILE, is_relative=True) # Reload settings
    current_settings["using_default_agents"] = use_default
    current_settings["using_custom_agents"] = use_custom
    combined_roles = load_all_roles(current_settings, file_agents=file_agents_dict)
    file_agent_keys = list(file_agents_dict.keys()) if file_agents_dict else []
    display_choices = sorted([get_role_display_name(name, file_agent_keys) for name in combined_roles.keys()])
    new_value = display_choices[0] if display_choices else None
    print(f" Role choices updated: {len(display_choices)} total")
    return gr.Dropdown.update(choices=display_choices, value=new_value)

def handle_agent_file_upload(uploaded_file):
    """Processes uploaded agent JSON file. Returns dict, filename, update obj."""
    if uploaded_file is None:
        print("Agent file upload cleared.")
        return {}, None, gr.Dropdown.update() # Need update for dropdown trigger

    try:
        file_path = uploaded_file.name
        print(f"Processing uploaded agent file: {os.path.basename(file_path)}")
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read().strip()
        if not content: raise ValueError("Uploaded agent file is empty.")
        loaded_agents = json.loads(content)
        if not isinstance(loaded_agents, dict): raise ValueError("Uploaded file is not a dict.")
        # Basic validation (optional)
        # ...
        print(f"Successfully loaded {len(loaded_agents)} agent(s) from file.")
        return loaded_agents, os.path.basename(file_path), gr.Dropdown.update()
    except Exception as e:
        error_msg = f"Invalid Agent File: {e}"; print(error_msg); gr.Warning(error_msg)
        return {}, None, gr.Dropdown.update()

def load_profile_options_callback(profile_name_to_load, all_profiles_data_state, ordered_option_keys_state):
    """Loads profile options onto UI components. Returns list of updates."""
    print(f"Loading profile: {profile_name_to_load}")
    ordered_keys = ordered_option_keys_state # Keys in UI order
    if not profile_name_to_load or not all_profiles_data_state or not ordered_keys:
        print("Cannot load profile: Missing profile name, data, or ordered keys.")
        return [gr.update() for _ in ordered_keys]

    profile_options = all_profiles_data_state.get(profile_name_to_load)
    if not profile_options or not isinstance(profile_options, dict):
        print(f"Profile '{profile_name_to_load}' not found or invalid.")
        return [gr.update() for _ in ordered_keys]

    updates_list = []
    for key in ordered_keys:
        if key in profile_options:
            new_value = profile_options[key]
            print(f"  Updating '{key}' UI to: {new_value}")
            updates_list.append(gr.update(value=new_value))
        else:
            updates_list.append(gr.update()) # No change if key not in profile
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
        current_settings = load_json(settings_path, is_relative=False)
        previous_theme = current_settings.get("gradio_theme", "Default")

        # Update general values
        current_settings["ollama_url"] = ollama_url_in; current_settings["max_tokens_slider"] = max_tokens_slider_range_in; current_settings["ollama_api_prompt_to_console"] = api_to_console_in; current_settings["using_default_agents"] = use_default_in; current_settings["using_custom_agents"] = use_custom_in; current_settings["use_ollama_api_options"] = use_ollama_opts_default_in; current_settings["release_model_on_change"] = release_model_default_in; current_settings["gradio_theme"] = theme_select_in

        # Update ollama_api_options
        if "ollama_api_options" not in current_settings: current_settings["ollama_api_options"] = {}
        ordered_keys = ordered_option_keys_state # Use keys from state

        num_expected = len(ordered_keys)
        num_received = len(api_option_values)
        if num_expected == num_received:
            for key, value in zip(ordered_keys, api_option_values):
                current_settings["ollama_api_options"][key] = value # Save directly for simplicity now
            print(f" Ollama API options updated ({num_expected} values).")
        else: print(f"Warning: API Options count mismatch. Expected {num_expected}, received {num_received}. Options NOT saved.")

        with open(settings_path, 'w', encoding='utf-8') as file: json.dump(current_settings, file, indent=4)
        print(f"Settings saved successfully to {settings_path}")
        save_msg = "Settings saved successfully."
        if theme_select_in != previous_theme: save_msg += " Restart required to apply theme change."
        return save_msg
    except Exception as e: error_msg = f"Error saving settings: {e}"; print(error_msg); return error_msg

# --- History Callbacks ---
def show_clear_confirmation(): return gr.update(visible=True)
def hide_clear_confirmation(): return gr.update(visible=False)
def clear_full_history_callback(history_list_state):
    """Clears the persistent history file and returns empty list/string."""
    print("Full history file cleared.")
    history.save_history([]) # Save empty list
    # Must match outputs=[full_history_display, confirm_clear_group, history_list_state]
    return "", hide_clear_confirmation(), [] # Return empty string, hide group, return empty list for state