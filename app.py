# ArtAgent/app.py

# Part 1: Imports
import gradio as gr
import json
import os
from PIL import Image
import numpy as np
import requests
import tempfile
import shutil
import atexit
import time

# Import from refactored structure
from core.utils import load_json, format_json_to_html_table, get_theme_object, AVAILABLE_THEMES, get_absolute_path
from core.ollama_checker import OllamaStatusChecker
from core import history_manager as history
from agents.roles_config import load_all_roles, get_role_display_name, get_actual_role_name
from agents.ollama_agent import get_llm_response

# Import UI creation functions
from ui.chat_tab import create_chat_tab
from ui.app_settings_tab import create_app_settings_tab
from ui.roles_tab import create_roles_tabs
from ui.history_tab import create_history_tab
from ui.common_ui_elements import create_footer

# --- Constants ---
SETTINGS_FILE = 'settings.json'
MODELS_FILE = 'models.json'
LIMITERS_FILE = 'limiters.json'
PROFILES_FILE = 'ollama_profiles.json'
DEFAULT_ROLES_FILE = 'agents/agent_roles.json'
CUSTOM_ROLES_FILE = 'agents/custom_agent_roles.json'

# --- Utility Functions (App Specific or Loading) ---
# Use load_json from core.utils for consistency
def load_settings(settings_file=SETTINGS_FILE):
    return load_json(settings_file, is_relative=True)

def load_models(model_file=MODELS_FILE):
    models_list = load_json(model_file, is_relative=True)
    return models_list if isinstance(models_list, list) else [] # Ensure it's a list

def load_limiters(limiter_file=LIMITERS_FILE):
    return load_json(limiter_file, is_relative=True)

def load_profiles(profile_file=PROFILES_FILE):
    return load_json(profile_file, is_relative=True)

# --- Load Initial Data ---
settings = load_settings()
OLLAMA_API_URL = settings.get("ollama_url", "http://localhost:11434/api/generate")

models_data = load_models() # Now returns a list
model_names_with_vision = []
if models_data: # Check if models_data is not empty
    model_names_with_vision = [f"{m['name']} (VISION)" if m.get('vision') else m.get('name', 'Unknown') for m in models_data]
else:
    print("Warning: No models loaded from models.json")

limiters_data = load_limiters()
limiters_names = list(limiters_data.keys())
profiles_data = load_profiles()
profile_names = list(profiles_data.keys())

# Load initial roles based on startup settings
roles_data = load_all_roles(settings) # Initial load
initial_role_names = list(roles_data.keys())
# Generate initial display names
initial_role_display_names = sorted([get_role_display_name(name) for name in initial_role_names])


# Load initial history
history_list = history.load_history()
current_session_history = [] # Session history resets each run

# --- Perform Initial Startup Check (Console Feedback Only) ---
startup_checker = OllamaStatusChecker(OLLAMA_API_URL) # Pass full URL
if not startup_checker.check():
    print(startup_checker.get_console_message())
else:
    print(f"[Startup Check] Ollama appears responsive at base URL {startup_checker.base_url}. Application starting.")

# --- Temp Dir Setup ---
temp_dir = None
def cleanup_temp_dir():
    global temp_dir
    if temp_dir and os.path.exists(temp_dir):
        print(f"Cleaning up temp directory: {temp_dir}")
        try: shutil.rmtree(temp_dir)
        except OSError as e: print(f"Error removing temp directory {temp_dir}: {e}")
atexit.register(cleanup_temp_dir)

# --- Model Release Functions ---
def release_model(model_name, ollama_api_url):
     """Sends request to Ollama to release (unload) a model."""
     if not model_name or not ollama_api_url: return "Missing model name or URL for release."
     payload = {"model": model_name, "keep_alive": 0}
     try:
         print(f"Sending release request for model: {model_name} to {ollama_api_url}")
         # Use the generate endpoint as per Ollama docs for unloading
         response = requests.post(ollama_api_url, json=payload, timeout=20)
         response.raise_for_status()
         msg = f"Model '{model_name}' release request sent successfully."
         print(msg)
         return msg
     except requests.exceptions.Timeout: msg = f"Error releasing model '{model_name}': Request timed out."; print(msg); return msg
     except requests.exceptions.RequestException as e:
         error_detail = str(e)
         try: # Try to get more info from response on error
             if e.response is not None: error_detail += f" | Status: {e.response.status_code} | Body: {e.response.text[:200]}"
         except Exception: pass
         msg = f"Error releasing model '{model_name}': {error_detail}"; print(msg); return msg

def release_all_models_callback():
    """Callback for the 'Release All' button."""
    print("Attempting to release all models specified in models.json...")
    # Reload settings in case URL changed
    current_settings = load_settings()
    ollama_url = current_settings.get("ollama_url")
    if not ollama_url: return "Cannot release models: Ollama URL not found in settings."

    models_to_release = load_models() # Reload models list
    if not models_to_release: return "No models found or invalid format in models.json."

    results = []
    for model_info in models_to_release:
        model_name = model_info.get("name")
        if model_name: results.append(release_model(model_name, ollama_url))

    summary = "\n".join(results) if results else "No valid model names found to release."
    print(f"Model release process finished.\nSummary:\n{summary}")
    return summary

# --- Callback Function Definitions ---

def chat_logic(folder_path, role_display_name, user_input, model_with_vision, max_tokens_ui,
               file_handling_option, limiter_handling_option, single_image_np,
               current_settings, # From state
               use_ollama_api_options, release_model_on_change,
               selected_model_tracker_value, # Value from state (previous model)
               file_agents_dict # From state
               ):
    """Handles the core chat logic, calling the Ollama agent."""
    global history_list, current_session_history # Access globals

    # Get Actual Role Name from display name
    actual_role_name = get_actual_role_name(role_display_name)
    print(f"Chat Logic: Role Display='{role_display_name}', Actual='{actual_role_name}'")

    # Reload roles_data to include file agents used in THIS call
    roles_data_current = load_all_roles(current_settings, file_agents=file_agents_dict)

    # 1. Find Model Name and Info
    model_name = None
    model_info = None
    current_models_data = load_models() # Reload in case models.json changed
    for m in current_models_data:
         m_name = m.get("name")
         if not m_name: continue
         if m_name == model_with_vision or f"{m_name} (VISION)" == model_with_vision:
              model_name = m_name
              model_info = m
              break
    if not model_name or not model_info: return "Error: Selected model not found or invalid.", "\n".join(current_session_history), None

    # 2. Handle Model Release (if applicable)
    if release_model_on_change and selected_model_tracker_value and selected_model_tracker_value != model_name:
        print(f"Requesting release of previous model: {selected_model_tracker_value}")
        release_model(selected_model_tracker_value, current_settings.get("ollama_url"))

    # 3. Prepare Prompt and Options
    current_limiters_data = load_limiters() # Reload limiters
    limiter_settings = current_limiters_data.get(limiter_handling_option, {})
    limiter_prompt_format = limiter_settings.get("limiter_prompt_format", "")
    limiter_token_slider = limiter_settings.get("limiter_token_slider") # Can be None

    # Effective max tokens logic
    if limiter_handling_option != "Off" and limiter_token_slider is not None:
         effective_max_tokens = min(max_tokens_ui, limiter_token_slider)
    else:
         effective_max_tokens = max_tokens_ui

    role_description = roles_data_current.get(actual_role_name, {}).get("description", "Unknown Role")
    prompt = f"Role: {actual_role_name} - {role_description}\n{limiter_prompt_format}\nUser Input: {user_input}\n"

    agent_ollama_options = {} # Start fresh for agent call specific overrides
    if use_ollama_api_options:
        # No overrides needed here, agent function merges based on settings/role
        pass

    # 4. Handle Image Input
    pil_images_list = []
    is_single_image_mode = False
    image_source_info = "[None - Text Only]"

    if single_image_np is not None and isinstance(single_image_np, np.ndarray):
        if model_info.get("vision"):
            try:
                pil_image = Image.fromarray(single_image_np.astype('uint8'))
                pil_images_list = [pil_image]
                is_single_image_mode = True
                image_source_info = "[Single Upload]"
                print("Processing single image input.")
            except Exception as e: return f"Error processing single image: {e}", "\n".join(current_session_history), model_name
        else: print("Warning: Single image provided, but selected model does not support vision.")

    # 5. Determine Mode and Call Agent
    final_response = "Processing..."
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    # --- Call Logic ---
    if is_single_image_mode or (not folder_path or not os.path.isdir(folder_path)):
        # Single Image or Text-Only
        print(f"Calling agent (Single/Text): model={model_name}, role={actual_role_name}")
        final_response = get_llm_response(
             role=actual_role_name, prompt=prompt, model=model_name, settings=current_settings,
             roles_data=roles_data_current, images=pil_images_list, max_tokens=effective_max_tokens,
             ollama_api_options=agent_ollama_options
         )
        entry = f"Timestamp: {timestamp}\nRole: {actual_role_name}\nModel: {model_name}\nInput: {user_input}\nImage: {image_source_info}\nResponse:\n{final_response}\n---\n"
        history_list = history.add_to_history(history_list, entry)
        current_session_history.append(entry)

    elif folder_path and os.path.isdir(folder_path):
        # Folder Processing
        if not model_info.get("vision"):
            print("Warning: Folder path provided, but selected model does not support vision. Processing as text-only.")
            final_response = get_llm_response(role=actual_role_name, prompt=prompt, model=model_name, settings=current_settings, roles_data=roles_data_current, images=None, max_tokens=effective_max_tokens, ollama_api_options=agent_ollama_options)
            entry = f"Timestamp: {timestamp}\nRole: {actual_role_name}\nModel: {model_name}\nInput: {user_input}\nImage: {image_source_info}\nResponse:\n{final_response}\n---\n"
            history_list = history.add_to_history(history_list, entry)
            current_session_history.append(entry)
        else:
            # Process folder with vision model (image by image)
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
                    img_response = get_llm_response(
                         role=actual_role_name, prompt=image_prompt, model=model_name, settings=current_settings,
                         roles_data=roles_data_current, images=[image], max_tokens=effective_max_tokens,
                         ollama_api_options=agent_ollama_options
                    )

                    # --- File Handling ---
                    base_name = os.path.splitext(file_name)[0]
                    output_file = os.path.join(folder_path, f"{base_name}.txt")
                    action_taken = "Skipped"
                    file_exists = os.path.exists(output_file)

                    if not file_exists or file_handling_option == "Overwrite":
                         with open(output_file, 'w', encoding='utf-8') as f: f.write(img_response)
                         action_taken = "Written" if not file_exists else "Overwritten"
                    elif file_handling_option == "Skip":
                         pass # Default action_taken is "Skipped"
                    elif file_handling_option == "Append":
                         with open(output_file, 'a', encoding='utf-8') as f: f.write("\n\n---\n\n" + img_response)
                         action_taken = "Appended"
                    elif file_handling_option == "Prepend":
                         try:
                            original_content = ""
                            if file_exists:
                                with open(output_file, 'r', encoding='utf-8') as f: original_content = f.read()
                            with open(output_file, 'w', encoding='utf-8') as f: f.write(img_response + "\n\n---\n\n" + original_content)
                            action_taken = "Prepended"
                         except Exception as prepend_e:
                            action_taken = f"Prepend Error: {prepend_e}"

                    confirmation_messages.append(f"  - {file_name}: {action_taken} -> {base_name}.txt")

                    # --- History Update per Image ---
                    entry = f"Timestamp: {timestamp}\nRole: {actual_role_name}\nModel: {model_name}\nInput: {user_input}\nImage: {file_name}\nResponse:\n{img_response}\n---\n"
                    history_list = history.add_to_history(history_list, entry)
                    current_session_history.append(entry)
                    processed_files += 1

                except Exception as e:
                    error_msg = f"Error processing file {file_name}: {e}"
                    print(f"  {error_msg}")
                    confirmation_messages.append(f"  - {file_name}: Error - {e}")

            if processed_files == 0: final_response = "No valid image files found or processed in the directory."
            else: final_response = f"Folder processing complete ({processed_files} files):\n" + "\n".join(confirmation_messages)

    # Return response text, history text, and model name used
    return final_response, "\n---\n".join(current_session_history), model_name


def comment_logic(llm_response_text, comment, model_state_value, current_settings, use_ollama_api_options, max_tokens_ui, file_agents_dict):
    """Handles the commenting logic."""
    global history_list, current_session_history # Access globals

    if not comment or not model_state_value:
        print("Comment ignored: No comment text or model state.")
        return llm_response_text, "\n---\n".join(current_session_history)

    # Reload roles to ensure consistency
    roles_data_current = load_all_roles(current_settings, file_agents=file_agents_dict)

    print(f"Processing comment on model {model_state_value}...")
    role = "User" # Fixed role for comment context
    role_description = roles_data_current.get(role, {}).get("description", "Follow up based on comment.")
    prompt = f"Previous Response:\n{llm_response_text}\n\nUser Comment/Instruction:\n{comment}\n\nPlease revise or continue based on the comment."

    agent_ollama_options = {} # Agent handles merging
    if use_ollama_api_options: pass

    # Call the agent
    response = get_llm_response(
        role=role, prompt=prompt, model=model_state_value, settings=current_settings,
        roles_data=roles_data_current, images=None, max_tokens=max_tokens_ui,
        ollama_api_options=agent_ollama_options
    )

    # Update history
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    entry = f"Timestamp: {timestamp}\nRole: {role} (Comment)\nModel: {model_state_value}\nInput: {comment}\nContext: Previous response\nResponse:\n{response}\n---\n"
    history_list = history.add_to_history(history_list, entry)
    current_session_history.append(entry)

    return response, "\n---\n".join(current_session_history)


def update_max_tokens_on_limiter_change(limiter_choice, current_max_tokens_value):
    """Updates the max token slider value based on limiter selection."""
    if limiter_choice == "Off":
        # No change when turned off, keep user's current slider value
        return gr.update()

    current_limiters_data = load_limiters() # Reload in case file changed
    limiter_settings = current_limiters_data.get(limiter_choice, {})
    limiter_token_val = limiter_settings.get("limiter_token_slider")

    if limiter_token_val is not None:
        # Check if the limiter value differs from current slider value
        if int(limiter_token_val) != int(current_max_tokens_value):
             print(f"Limiter '{limiter_choice}' updating max_tokens slider to: {limiter_token_val}")
             return gr.update(value=limiter_token_val)
        else:
             # Value already matches, no need for UI update flicker
             return gr.update()
    # No value defined in limiter, no change
    return gr.update()


def clear_session_history_callback():
    """Clears the session history list."""
    global current_session_history
    current_session_history = []
    print("Session history cleared.")
    return "" # Return empty string to clear the display textbox


def update_role_dropdown_callback(use_default, use_custom, file_agents_dict):
    """Reloads roles based on settings flags and file agents, updates dropdown."""
    print("Updating role dropdown...")
    # Reload settings to get latest flags
    current_settings = load_settings()
    current_settings["using_default_agents"] = use_default
    current_settings["using_custom_agents"] = use_custom

    # Reload all roles including potential file agents
    combined_roles = load_all_roles(current_settings, file_agents=file_agents_dict)

    # Generate display names (add prefix for file agents)
    file_agent_keys = list(file_agents_dict.keys()) if file_agents_dict else []
    display_choices = sorted([get_role_display_name(name, file_agent_keys) for name in combined_roles.keys()])

    new_value = display_choices[0] if display_choices else None
    print(f" Role choices updated: {len(display_choices)} total")
    # Return a Gradio update object for the dropdown
    return gr.Dropdown.update(choices=display_choices, value=new_value)


def handle_agent_file_upload(uploaded_file):
    """Processes uploaded agent JSON file."""
    if uploaded_file is None:
        print("Agent file upload cancelled or cleared.")
        # Return empty dict for state, None for filename display
        # Need to trigger role dropdown update
        return {}, None, gr.Dropdown.update() # Send update object for dropdown

    try:
        file_path = uploaded_file.name # Gradio provides temp path
        print(f"Processing uploaded agent file: {os.path.basename(file_path)}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content: raise ValueError("Uploaded agent file is empty.")

        loaded_agents = json.loads(content)

        if not isinstance(loaded_agents, dict): raise ValueError("Uploaded file is not a JSON dictionary.")
        # Basic validation
        for key, value in loaded_agents.items():
            if not isinstance(value, dict) or 'description' not in value:
                print(f"Warning: Agent '{key}' in file has invalid format.")
                # Continue loading other agents

        print(f"Successfully loaded {len(loaded_agents)} agent(s) from file.")
        # Return loaded dict, filename, and trigger dropdown update
        return loaded_agents, os.path.basename(file_path), gr.Dropdown.update()

    except (json.JSONDecodeError, ValueError) as e:
        error_msg = f"Invalid Agent File: {e}"
        print(error_msg)
        gr.Warning(error_msg)
        return {}, None, gr.Dropdown.update()
    except Exception as e:
        error_msg = f"Error processing agent file: {e}"
        print(error_msg)
        gr.Warning(error_msg)
        return {}, None, gr.Dropdown.update()


def load_profile_options_callback(profile_name_to_load, all_profiles_data_state):
    """Loads profile options onto UI components."""
    print(f"Loading profile: {profile_name_to_load}")
    if not profile_name_to_load or not all_profiles_data_state:
        print("No profile selected or profiles data missing.")
        # Need to return an update for EACH component listed in 'outputs'
        return [gr.update() for _ in initial_ollama_options_ordered_keys] # Use the known order

    profile_options = all_profiles_data_state.get(profile_name_to_load)
    if not profile_options or not isinstance(profile_options, dict):
        print(f"Profile '{profile_name_to_load}' not found or invalid.")
        return [gr.update() for _ in initial_ollama_options_ordered_keys]

    updates_list = []
    # Iterate through the keys in the order the components were listed in 'outputs'
    for key in initial_ollama_options_ordered_keys:
        if key in profile_options:
            new_value = profile_options[key]
            print(f"  Updating '{key}' UI to: {new_value}")
            updates_list.append(gr.update(value=new_value))
        else:
            # If profile doesn't define a key, send no change update
            print(f"  Key '{key}' not in profile, UI value unchanged.")
            updates_list.append(gr.update())

    # Ensure the number of updates matches the number of output components
    if len(updates_list) != len(initial_ollama_options_ordered_keys):
        print("Error: Mismatch in generating updates for profile load.")
        # Return no-change updates for safety
        return [gr.update() for _ in initial_ollama_options_ordered_keys]

    return updates_list


def save_settings_callback(
    # General Settings Inputs
    ollama_url_in, max_tokens_slider_range_in, api_to_console_in,
    use_default_in, use_custom_in, use_ollama_opts_default_in,
    release_model_default_in, theme_select_in,
    # API Options Inputs (passed via *args)
    *api_option_values):
    """Saves all settings back to settings.json."""
    print("Saving application settings...")
    try:
        settings_path = get_absolute_path(SETTINGS_FILE) # Use absolute path
        # Load current settings to preserve structure/unhandled keys
        current_settings = load_json(settings_path, is_relative=False)

        # Store previous theme to check if restart needed
        previous_theme = current_settings.get("gradio_theme", "Default")

        # Update general values
        current_settings["ollama_url"] = ollama_url_in
        current_settings["max_tokens_slider"] = max_tokens_slider_range_in
        current_settings["ollama_api_prompt_to_console"] = api_to_console_in
        current_settings["using_default_agents"] = use_default_in
        current_settings["using_custom_agents"] = use_custom_in
        current_settings["use_ollama_api_options"] = use_ollama_opts_default_in
        current_settings["release_model_on_change"] = release_model_default_in
        current_settings["gradio_theme"] = theme_select_in

        # Update ollama_api_options
        if "ollama_api_options" not in current_settings: current_settings["ollama_api_options"] = {}
        # Get keys in the consistent order they were displayed
        initial_options = load_json(settings_path, is_relative=False).get("ollama_api_options", {})
        ordered_keys = sorted(initial_options.keys())

        num_expected = len(ordered_keys)
        num_received = len(api_option_values)

        if num_expected == num_received:
            for key, value in zip(ordered_keys, api_option_values):
                # Attempt type conversion based on original type if possible
                original_value = initial_options.get(key)
                original_type = type(original_value)
                try:
                    if original_type == bool and isinstance(value, bool): current_settings["ollama_api_options"][key] = value
                    elif original_type == int: current_settings["ollama_api_options"][key] = int(value)
                    elif original_type == float: current_settings["ollama_api_options"][key] = float(value)
                    else: current_settings["ollama_api_options"][key] = value # Save as string or original type if no conversion needed
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert saved value '{value}' for option '{key}'. Saving as is.")
                    current_settings["ollama_api_options"][key] = value
            print(f" Ollama API options updated ({num_expected} values).")
        else:
             print(f"Warning: Mismatch in Ollama API options count. Expected {num_expected}, received {num_received}. API Options NOT saved.")
             # Optionally return error early: return "Error: API Options count mismatch."

        # Save the updated dictionary back to file
        with open(settings_path, 'w', encoding='utf-8') as file:
            json.dump(current_settings, file, indent=4)

        # --- Reload global settings if needed by other parts of the app immediately ---
        global settings
        settings = current_settings
        print(f"Settings saved successfully to {settings_path}")

        save_msg = "Settings saved successfully."
        if theme_select_in != previous_theme:
             save_msg += " Restart application to apply theme change."
        return save_msg
    except Exception as e:
        error_msg = f"Error saving settings: {e}"
        print(error_msg)
        return error_msg


# --- History Callbacks ---
def show_clear_confirmation(): return gr.update(visible=True)
def hide_clear_confirmation(): return gr.update(visible=False)
def clear_full_history_callback():
    global history_list
    history_list = []
    history.save_history(history_list) # Save empty list via manager
    print("Full history file cleared.")
    # Return update for the display textbox and hide the confirmation group
    return gr.update(value=""), hide_clear_confirmation()


# --- Build UI ---
selected_theme_name = settings.get("gradio_theme", "Default")
theme_object = get_theme_object(selected_theme_name)
print(f"Applying theme: {selected_theme_name}")

# Store initial options keys order globally for callbacks
initial_ollama_options = settings.get("ollama_api_options", {})
initial_ollama_options_ordered_keys = sorted(initial_ollama_options.keys())


with gr.Blocks(title="ArtAgents", theme=theme_object) as demo:

    # --- Create UI Tabs ---
    chat_comps = create_chat_tab(initial_role_display_names, model_names_with_vision, limiters_names, settings)
    settings_comps = create_app_settings_tab(settings)
    roles_comps = create_roles_tabs(
         load_json(DEFAULT_ROLES_FILE, is_relative=True),
         load_json(CUSTOM_ROLES_FILE, is_relative=True)
    )
    history_comps = create_history_tab(history_list)
    # caption_comps = create_captions_tab(...) # Add later

    create_footer()

    # --- Define Main App States ---
    settings_state = gr.State(value=settings) # Pass current settings to callbacks

    # --- Wire Event Handlers ---

    # -- Chat Tab --
    chat_comps['model_with_vision'].change(
        lambda x: x, inputs=[chat_comps['model_with_vision']],
        outputs=[chat_comps['selected_model_tracker']]
    )

    update_roles_trigger_inputs = [
        settings_comps['settings_use_default'],
        settings_comps['settings_use_custom'],
        chat_comps['loaded_file_agents_state'] # Include file agent state
    ]
    settings_comps['settings_use_default'].change(
        fn=update_role_dropdown_callback, inputs=update_roles_trigger_inputs,
        outputs=[chat_comps['role_dropdown']]
    )
    settings_comps['settings_use_custom'].change(
        fn=update_role_dropdown_callback, inputs=update_roles_trigger_inputs,
        outputs=[chat_comps['role_dropdown']]
    )

    chat_comps['agent_file_upload'].upload(
         fn=handle_agent_file_upload,
         inputs=[chat_comps['agent_file_upload']],
         outputs=[
              chat_comps['loaded_file_agents_state'],
              chat_comps['loaded_agent_file_display'],
              chat_comps['role_dropdown']
         ]
    )

    chat_comps['limiter_handling_option'].change(
        fn=update_max_tokens_on_limiter_change,
        inputs=[chat_comps['limiter_handling_option'], chat_comps['max_tokens_slider']],
        outputs=[chat_comps['max_tokens_slider']]
    )

    submit_inputs = [
        chat_comps['folder_path'], chat_comps['role_dropdown'], chat_comps['user_input'],
        chat_comps['model_with_vision'], chat_comps['max_tokens_slider'],
        chat_comps['file_handling_option'], chat_comps['limiter_handling_option'],
        chat_comps['single_image_display'], settings_state,
        chat_comps['use_ollama_api_options'], chat_comps['release_model_on_change'],
        chat_comps['selected_model_tracker'], chat_comps['loaded_file_agents_state'] # Pass file agents
    ]
    submit_outputs = [
        chat_comps['llm_response_display'], chat_comps['current_session_history_display'],
        chat_comps['model_state']
    ]
    chat_comps['submit_button'].click(fn=chat_logic, inputs=submit_inputs, outputs=submit_outputs)

    comment_inputs = [
        chat_comps['llm_response_display'], chat_comps['comment_input'],
        chat_comps['model_state'], settings_state,
        chat_comps['use_ollama_api_options'], chat_comps['max_tokens_slider'],
        chat_comps['loaded_file_agents_state'] # Pass file agents
    ]
    comment_outputs = [
        chat_comps['llm_response_display'], chat_comps['current_session_history_display']
    ]
    chat_comps['comment_button'].click(fn=comment_logic, inputs=comment_inputs, outputs=comment_outputs)

    chat_comps['clear_session_button'].click(
        fn=clear_session_history_callback, inputs=[],
        outputs=[chat_comps['current_session_history_display']]
    )

    # -- App Settings Tab --
    # Get the list of UI components in the correct order
    api_option_components_list = [settings_comps['ollama_options_ui_elements'][key] for key in initial_ollama_options_ordered_keys]

    settings_comps['load_profile_button'].click(
         fn=load_profile_options_callback,
         inputs=[
              settings_comps['profile_select'],
              gr.State(profiles_data), # Pass loaded profiles DATA via state - THIS IS OK
              # --- REMOVE THIS problematic gr.State line ---
              # gr.State(settings_comps['ollama_options_ui_elements']),
              # --- Instead, pass the dictionary of components directly if needed, ---
              # --- but the callback actually receives components from the 'outputs' list ---
              # --- Let's restructure the callback slightly ---
         ],
         # The outputs list correctly tells Gradio which components to update
         outputs=api_option_components_list # Pass the LIST of components to be updated
    )


    settings_save_inputs = [
         settings_comps['settings_ollama_url'], settings_comps['settings_max_tokens_slider_range'],
         settings_comps['settings_api_to_console'], settings_comps['settings_use_default'],
         settings_comps['settings_use_custom'], settings_comps['settings_use_ollama_opts_default'],
         settings_comps['settings_release_model_default'], settings_comps['settings_theme_select']
    ] + api_option_components_list

    settings_comps['settings_save_button'].click(
        fn=save_settings_callback,
        inputs=settings_save_inputs,
        outputs=[settings_comps['save_status_display']]
    )

    settings_comps['release_models_button'].click(
         fn=release_all_models_callback, inputs=[],
         outputs=[settings_comps['release_status_display']]
    )

    # -- History Tab --
    history_comps['clear_history_button'].click(
         fn=show_clear_confirmation, inputs=[], outputs=[history_comps['confirm_clear_group']]
    )
    history_comps['yes_clear_button'].click(
         fn=clear_full_history_callback, inputs=[],
         outputs=[history_comps['full_history_display'], history_comps['confirm_clear_group']]
    )
    history_comps['no_clear_button'].click(
         fn=hide_clear_confirmation, inputs=[], outputs=[history_comps['confirm_clear_group']]
    )


# --- atexit handler ---
def on_exit():
    """Function called on script exit."""
    print("Application exiting...")
    cleanup_temp_dir()
    print("Cleanup finished.")
atexit.register(on_exit)

# --- Launch the Application ---
if __name__ == "__main__":
    print("Initializing ArtAgents...")
    demo.launch(
        # share=True # For public link
        # server_name="0.0.0.0" # For network access
    )
    print("Gradio App shutdown.")