# ArtAgent/core/app_logic.py
import copy
import gradio as gr
import json
import os
from PIL import Image # Ensure Image is imported
import numpy as np
import time
# Removed re import as cleaning is now in utils
# import re

# Import necessary functions/classes from sibling modules or agents
from .utils import load_json, get_absolute_path, clean_agent_artifacts # Import cleaner
from .image_utils import open_image, is_image_filename, to_pil_image
from . import history_manager as history
from agents.roles_config import load_all_roles, get_role_display_name, get_actual_role_name
from agents.ollama_agent import get_llm_response, stream_llm_response, is_error_response
# Import ollama_manager to call release_model and agent_manager for workflows
from . import ollama_manager
from . import agent_manager # Import the agent manager

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
DIRECT_CALL_LABEL = "(Direct Agent Call)"
TEAM_PREFIX = "[Team] "
REFINER_SYSTEM_PROMPT = (
    "You are an AI assistant specialized in refining and modifying existing text based on user instructions. "
    "Keep the core essence of the text unless asked otherwise, and output only the revised text."
)


def resolve_model(models_data_state, model_with_vision):
    """Finds the model entry for a dropdown value ('name' or 'name (VISION)'). Returns (name, info) or (None, None)."""
    if not models_data_state or not model_with_vision:
        return None, None
    for m in models_data_state:
        m_name = m.get("name") if isinstance(m, dict) else None
        if m_name and (m_name == model_with_vision or f"{m_name} (VISION)" == model_with_vision):
            return m_name, m
    return None, None


def build_agent_system_prompt(role_name, role_description, limiter_prompt_format=""):
    """System prompt for a single agent: its persona plus any prompt-style limiter."""
    parts = []
    if role_name:
        parts.append(f"You are acting as the agent '{role_name}'.")
        if role_description:
            parts.append(role_description)
    if limiter_prompt_format:
        parts.append(limiter_prompt_format)
    return "\n\n".join(parts) if parts else None


def _join_session(session_list):
    return "\n---\n".join(session_list)


def iter_execute_chat_or_team(
    # UI Inputs (Common)
    folder_path, user_input, model_with_vision, max_tokens_ui,
    file_handling_option, limiter_handling_option, single_image_input,
    use_ollama_api_options, release_model_on_change,
    # UI Inputs (Specific)
    selected_role_or_team, # A role, a "[Team] name", or "(Direct Agent Call)"
    clean_artifacts_flag: bool,
    # State Inputs
    current_settings,
    models_data_state,
    limiters_data_state,
    teams_data_state,
    selected_model_tracker_value,
    file_agents_dict,
    history_list_state,
    session_history_list_state,
    ):
    """
    Streaming router: runs a team workflow or a single agent / direct model call.

    Yields (response_text, session_history_text, model_name_state, session_history_list, conversation)
    repeatedly while the answer streams in; the last item is final. `conversation` is the
    chat state used by Comment/Refine (None while streaming).
    """
    print(f"\nExecuting chat/team with selection: '{selected_role_or_team}'")
    print(f"  Clean Artifacts Flag: {clean_artifacts_flag}")
    session_list = list(session_history_list_state or [])
    last = ("Error: Processing failed.", _join_session(session_list), None, session_list, None)

    if selected_role_or_team and selected_role_or_team.startswith(TEAM_PREFIX):
        team_name = selected_role_or_team[len(TEAM_PREFIX):]
        if not teams_data_state or team_name not in teams_data_state:
            print(f"Warning: Selected team '{team_name}' not found in loaded teams data.")
            yield f"Error: Selected Agent Team '{team_name}' definition not found.", _join_session(session_list), None, session_list, None
            return
        stream = _iter_team(team_name, teams_data_state[team_name], user_input, model_with_vision, single_image_input,
                            current_settings, models_data_state, file_agents_dict, history_list_state, session_list)
    else:
        if not selected_role_or_team or selected_role_or_team == DIRECT_CALL_LABEL:
            print("Running Direct Model Call (no agent persona).")
        else:
            print(f"Running Single Agent: '{selected_role_or_team}'")
        stream = iter_chat_logic(
            folder_path=folder_path, role_display_name=selected_role_or_team, user_input=user_input,
            model_with_vision=model_with_vision, max_tokens_ui=max_tokens_ui,
            file_handling_option=file_handling_option, limiter_handling_option=limiter_handling_option,
            single_image_input=single_image_input, current_settings=current_settings,
            use_ollama_api_options=use_ollama_api_options, release_model_on_change=release_model_on_change,
            models_data_state=models_data_state, limiters_data_state=limiters_data_state,
            selected_model_tracker_value=selected_model_tracker_value, file_agents_dict=file_agents_dict,
            history_list_state=history_list_state, session_history_list_state=session_list,
        )

    for last in stream:
        yield last

    # --- Artifact cleaning on the final answer ---
    response_text, session_text, model_state, final_session_list, conversation = last
    if clean_artifacts_flag and isinstance(response_text, str) and not response_text.startswith("Error:") and not response_text.startswith("⚠️ Error:"):
        cleaned_text = clean_agent_artifacts(response_text)
        if cleaned_text != response_text:
            print("  Applied artifact cleaning to final response.")
            if conversation and conversation.get("messages"):
                conversation["messages"][-1]["content"] = cleaned_text
            yield cleaned_text, session_text, model_state, final_session_list, conversation


def _iter_team(team_name, team_definition, user_input, model_with_vision, single_image_input,
               current_settings, models_data_state, file_agents_dict, history_list_state, session_list):
    """Runs a team workflow, streaming step output into the response box."""
    session_list = list(session_list)
    print(f"Running Agent Team Workflow: '{team_name}'")
    if not models_data_state:
        yield "Error: Models data state is missing.", _join_session(session_list), None, session_list, None
        return
    worker_model_name, model_info = resolve_model(models_data_state, model_with_vision)
    if not worker_model_name:
        yield f"Error: Could not determine worker model name for team from input '{model_with_vision}'.", _join_session(session_list), None, session_list, None
        return
    try:
        image = to_pil_image(single_image_input)
    except Exception as e:
        yield f"Error processing input image: {e}", _join_session(session_list), None, session_list, None
        return

    all_roles_data = load_all_roles(current_settings, file_agents=file_agents_dict)
    final_output = "Error: Workflow ended without a result."
    for event in agent_manager.iter_team_workflow(
        team_name=team_name,
        team_definition=team_definition,
        user_input=user_input,
        initial_settings=current_settings,
        all_roles_data=all_roles_data,
        history_list=list(history_list_state or []),
        worker_model_name=worker_model_name,
        single_image_input=image,
        worker_model_has_vision=bool(model_info.get("vision")),
    ):
        if isinstance(event, agent_manager.TeamProgress):
            header = f"⏳ Team '{team_name}' · {event.message}"
            body = f"\n\n{event.partial_output}" if event.partial_output else ""
            yield header + body, _join_session(session_list), None, session_list, None
        elif isinstance(event, agent_manager.TeamResult):
            final_output = event.final_output

    summary_entry = (f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\nWorkflow Run: '{team_name}' (Model: {worker_model_name})\n"
                     f"Final Output Length: {len(final_output)}\n---\n(Full steps in Persistent History)")
    session_list.append(summary_entry)
    conversation = {
        "model": worker_model_name, "role": "Refinement Assistant", "system": REFINER_SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": f"Create the final output for this request: {user_input}"},
            {"role": "assistant", "content": final_output},
        ],
    }
    yield final_output, _join_session(session_list), None, session_list, conversation


def execute_chat_or_team(*args, **kwargs) -> tuple[str, str, str | None, list]:
    """
    Blocking router (used by captioning): same inputs as iter_execute_chat_or_team.

    Returns:
        tuple: (response_text, session_history_text, model_name_used_state, new_session_history_list)
    """
    last = None
    for last in iter_execute_chat_or_team(*args, **kwargs):
        pass
    return tuple(last[:4]) if last else ("Error: Processing failed.", "", None, [])


# --- Single Agent / Direct Model Logic ---
def iter_chat_logic(
    # UI Inputs
    folder_path, role_display_name, user_input, model_with_vision, max_tokens_ui,
    file_handling_option, limiter_handling_option, single_image_input,
    use_ollama_api_options, release_model_on_change,
    # State Inputs
    current_settings, models_data_state, limiters_data_state,
    selected_model_tracker_value, file_agents_dict,
    history_list_state, session_history_list_state
    ):
    """
    Single agent (or direct model) call, streaming.

    Yields (response, session_history_text, model_name, session_history_list, conversation).
    Folder mode yields a progress line per image instead of tokens.
    """
    history_list = list(history_list_state or [])
    current_session_history = list(session_history_list_state or [])

    def snapshot(response, model=None, conversation=None):
        return response, _join_session(current_session_history), model, list(current_session_history), conversation

    direct_call = not role_display_name or role_display_name == DIRECT_CALL_LABEL
    actual_role_name = None if direct_call else get_actual_role_name(role_display_name)
    print(f"Single Agent Logic: Role='{actual_role_name or '(direct)'}'")
    roles_data_current = load_all_roles(current_settings, file_agents=file_agents_dict)

    # 1. Find Model Name and Info
    if not models_data_state:
        yield snapshot("Error: Models data not loaded."); return
    if not model_with_vision:
        yield snapshot("Error: No model specified for agent execution."); return
    model_name, model_info = resolve_model(models_data_state, model_with_vision)
    if not model_name:
        yield snapshot(f"Error: Selected model info not found for '{model_with_vision}'."); return
    print(f"  Using model: {model_name}")

    # 2. Handle Model Release
    if release_model_on_change and selected_model_tracker_value and selected_model_tracker_value != model_name:
        previous = selected_model_tracker_value.replace(" (VISION)", "")
        if previous != model_name:
            print(f"Requesting release of previous model: {previous}")
            ollama_manager.release_model(previous, current_settings)

    # 3. Prepare prompts and options
    limiter_settings = (limiters_data_state or {}).get(limiter_handling_option, {})
    limiter_prompt_format = limiter_settings.get("limiter_prompt_format", "")
    limiter_token_slider = limiter_settings.get("limiter_token_slider")
    effective_max_tokens = min(max_tokens_ui, limiter_token_slider) if limiter_handling_option != "Off" and limiter_token_slider is not None else max_tokens_ui
    role_description = roles_data_current.get(actual_role_name, {}).get("description", "Unknown Role") if actual_role_name else ""
    system_prompt = build_agent_system_prompt(actual_role_name, role_description, limiter_prompt_format)
    llm_kwargs = dict(
        role=actual_role_name or "", model=model_name, settings=current_settings, roles_data=roles_data_current,
        max_tokens=effective_max_tokens, system=system_prompt, use_global_options=use_ollama_api_options,
    )

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    entry_prefix = f"Timestamp: {timestamp}\nRole: {actual_role_name or '(Direct Model Call)'}\nModel: {model_name}\nInput: {user_input}"

    def conversation_for(answer):
        return {"model": model_name, "role": actual_role_name, "system": system_prompt,
                "messages": [{"role": "user", "content": user_input or ""}, {"role": "assistant", "content": answer}]}

    # 4. Image input (numpy from the UI or PIL from captioning)
    try:
        pil_image = to_pil_image(single_image_input)
    except Exception as e:
        yield snapshot(f"Error processing single numpy image: {e}", model_name); return
    image_source_info = "[None - Text Only]"
    if pil_image is not None and not model_info.get("vision"):
        print("Warning: Single image provided, but selected model lacks vision. Sending text only.")
        image_source_info = "[Single Image Ignored - No Vision]"
        pil_image = None

    # 5a. Single image call (streamed)
    if pil_image is not None:
        print(f"Calling agent (Single Image): model={model_name}, role={actual_role_name}")
        final_response = ""
        for final_response in stream_llm_response(prompt=user_input or "", images=[pil_image], **llm_kwargs):
            yield snapshot(final_response, model_name)
        entry = f"{entry_prefix}\nImage: [Image Data Sent]\nResponse:\n{final_response}\n---\n"
        history_list = history.add_to_history(history_list, entry)
        current_session_history.append(entry)
        yield snapshot(final_response, model_name, None if is_error_response(final_response) else conversation_for(final_response))
        return

    # 5b. Folder processing (one call per image, progress per file)
    if folder_path and os.path.isdir(folder_path) and model_info.get("vision"):
        yield from _iter_folder(folder_path, user_input, file_handling_option, llm_kwargs, entry_prefix, timestamp,
                                actual_role_name, model_name, history_list, current_session_history, snapshot)
        return
    if folder_path and os.path.isdir(folder_path):
        print("Warning: Folder path provided, but model lacks vision. Processing text-only.")
        image_source_info = "[None - Folder Ignored]"

    # 5c. Text-only call (streamed)
    print(f"Calling agent (Text Only): model={model_name}, role={actual_role_name}")
    final_response = ""
    for final_response in stream_llm_response(prompt=user_input or "", images=None, **llm_kwargs):
        yield snapshot(final_response, model_name)
    entry = f"{entry_prefix}\nImage: {image_source_info}\nResponse:\n{final_response}\n---\n"
    history_list = history.add_to_history(history_list, entry)
    current_session_history.append(entry)
    yield snapshot(final_response, model_name, None if is_error_response(final_response) else conversation_for(final_response))


def _iter_folder(folder_path, user_input, file_handling_option, llm_kwargs, entry_prefix, timestamp,
                 actual_role_name, model_name, history_list, current_session_history, snapshot):
    """Captions every image in a folder with a vision model, yielding a progress line per file."""
    print(f"Processing image folder: {folder_path}")
    confirmation_messages = []; processed_files = 0; skipped_files = 0
    try:
        files_in_folder = sorted([
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and is_image_filename(f)
        ])
    except Exception as list_e:
        yield snapshot(f"Error listing folder: {list_e}", model_name); return

    total = len(files_in_folder)
    for index, file_name in enumerate(files_in_folder, start=1):
        yield snapshot(f"⏳ Processing image {index}/{total}: {file_name}\n" + "\n".join(confirmation_messages), model_name)
        file_path = os.path.join(folder_path, file_name)
        loop_img = None
        try:
            if file_handling_option == "Skip" and os.path.exists(os.path.join(folder_path, f"{os.path.splitext(file_name)[0]}.txt")):
                # Don't spend a model call on an image whose caption would be discarded anyway
                confirmation_messages.append(f"  - {file_name}: Skipped (caption file exists)")
                skipped_files += 1
                continue
            print(f"  Processing file: {file_name}")
            loop_img = open_image(file_path) # Restricted to PNG/JPEG/BMP/GIF/WEBP decoders
            image_prompt = f"{user_input or ''}\nImage Context: Analyzing '{file_name}'\n"
            img_response = get_llm_response(prompt=image_prompt, images=[loop_img], **llm_kwargs)

            # --- File Handling Logic ---
            base_name = os.path.splitext(file_name)[0]
            output_file = os.path.join(folder_path, f"{base_name}.txt")
            action_taken = "Skipped"
            file_exists = os.path.exists(output_file)
            original_content = ""
            usable = bool(img_response) and not is_error_response(img_response)

            if file_exists and file_handling_option in ("Append", "Prepend"):
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        original_content = f.read().strip()
                except Exception as read_e:
                    print(f"  Warning: Could not read existing file {output_file}: {read_e}")

            if file_handling_option == "Overwrite" or not file_exists:
                if usable:
                    with open(output_file, 'w', encoding='utf-8') as f: f.write(img_response)
                    action_taken = "Written" if not file_exists else "Overwritten"
                else: action_taken = "Skipped (Empty/Error Response)"
            elif file_handling_option == "Append":
                if usable:
                    separator = "\n\n---\n\n" if original_content else ""
                    with open(output_file, 'w', encoding='utf-8') as f: f.write(original_content + separator + img_response)
                    action_taken = "Appended"
                else: action_taken = "Skipped (Empty/Error Response)"
            elif file_handling_option == "Prepend":
                if usable:
                    separator = "\n\n---\n\n" if original_content else ""
                    with open(output_file, 'w', encoding='utf-8') as f: f.write(img_response + separator + original_content)
                    action_taken = "Prepended"
                else: action_taken = "Skipped (Empty/Error Response)"

            confirmation_messages.append(f"  - {file_name}: {action_taken} -> {base_name}.txt")
            entry = f"{entry_prefix}\nImage: {file_name} [Data Sent]\nResponse:\n{img_response}\n---\n"
            history_list = history.add_to_history(history_list, entry); current_session_history.append(entry); processed_files += 1

        except Exception as e:
            print(f"  Error processing file '{file_name}': {e}")
            confirmation_messages.append(f"  - {file_name}: Error - {e}")
            error_entry = f"Timestamp: {timestamp}\nRole: {actual_role_name}\nModel: {model_name}\nInput: {user_input}\nImage: {file_name}\nERROR: {e}\n---\n"
            history_list = history.add_to_history(history_list, error_entry); current_session_history.append(error_entry)
        finally:
            if loop_img is not None:
                try: loop_img.close()
                except Exception as e_close: print(f"  Warning: Error closing loop image {file_name}: {e_close}")

    if not files_in_folder: final_response = "No supported image files found in the directory."
    elif processed_files == 0 and skipped_files == 0: final_response = "No images could be processed:\n" + "\n".join(confirmation_messages)
    else: final_response = f"Folder processing complete ({processed_files} processed, {skipped_files} skipped):\n" + "\n".join(confirmation_messages)
    yield snapshot(final_response, model_name)


def chat_logic(*args, **kwargs) -> tuple[str, str, str | None, list]:
    """Blocking single-agent call; same inputs as iter_chat_logic, returns the first four outputs."""
    last = None
    for last in iter_chat_logic(*args, **kwargs):
        pass
    return tuple(last[:4]) if last else ("Error: Processing failed.", "", None, [])


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
             try: current_max_tokens_int = int(float(current_max_tokens_value))
             except (ValueError, TypeError):
                 print(f"Warning: Could not convert current max_tokens slider value '{current_max_tokens_value}' to int.")
                 return gr.update()
             if limiter_token_int != current_max_tokens_int:
                  print(f"Limiter '{limiter_choice}' updating max_tokens slider to: {limiter_token_int}")
                  return gr.update(value=limiter_token_int)
         except (ValueError, TypeError): print(f"Warning: Invalid numeric value for limiter token value '{limiter_token_val}'.")
    return gr.update()


def clear_session_history_callback(session_history_list_state):
    """Clears the session history list state."""
    print("Session history cleared.")
    return "", [] # Match outputs = [display, state]


MAX_AGENT_FILE_BYTES = 1_000_000 # Agent definitions are small; refuse anything bigger


def handle_agent_file_upload(uploaded_file):
    """
    Processes an uploaded agent JSON file.

    Returns:
        tuple: (valid_agents_dict, loaded_filename_or_None) matching the two
               outputs wired in app.py. Dropdowns are refreshed by the chained .then().
    """
    if uploaded_file is None:
        print("Agent file upload cleared.")
        return {}, None

    try:
        # Gradio 4+ passes a filepath string; older versions passed a tempfile wrapper with .name
        file_path = uploaded_file if isinstance(uploaded_file, (str, os.PathLike)) else uploaded_file.name
        file_basename = os.path.basename(file_path)
        print(f"Processing uploaded agent file: {file_basename}")
        if os.path.getsize(file_path) > MAX_AGENT_FILE_BYTES:
            raise ValueError(f"File is larger than {MAX_AGENT_FILE_BYTES // 1000} KB.")
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read().strip()
        if not content: raise ValueError("Uploaded agent file is empty.")
        loaded_agents = json.loads(content)
        if not isinstance(loaded_agents, dict): raise ValueError("Uploaded file is not a dict.")
        # Basic validation
        valid_agents = {}
        for key, value in loaded_agents.items():
            if isinstance(value, dict) and isinstance(value.get('description'), str): valid_agents[str(key)] = value
            else: print(f"Warning: Agent '{key}' in file '{file_basename}' skipped (invalid format).")
        if not valid_agents: raise ValueError("No valid agent definitions found.")
        print(f"Successfully loaded {len(valid_agents)} agent(s) from {file_basename}.")
        return valid_agents, file_basename
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
        error_msg = f"Invalid Agent File: {e}"; print(error_msg); gr.Warning(error_msg); return {}, None
    except Exception as e:
        error_msg = f"Error processing agent file: {e}"; print(error_msg); gr.Warning(error_msg); return {}, None


def load_profile_options_callback(profile_name_to_load, all_profiles_data_state, ordered_option_keys_state):
    """Loads profile options onto UI components. Returns list of updates."""
    print(f"Loading profile: {profile_name_to_load}")
    ordered_keys = ordered_option_keys_state
    num_outputs = len(ordered_keys) if ordered_keys else 0
    no_change_updates = [gr.update() for _ in range(num_outputs)]

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
            # print(f"  Updating '{key}' UI to: {new_value}") # Reduce verbosity
            updates_list.append(gr.update(value=new_value))
        else:
            updates_list.append(gr.update()) # Send no-change update

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
        current_settings["ollama_api_options"] = current_settings.get("ollama_api_options", {})
        ordered_keys = ordered_option_keys_state
        num_expected = len(ordered_keys); num_received = len(api_option_values)
        initial_options_for_types = load_json(settings_path, is_relative=False).get("ollama_api_options", {})

        if ordered_keys and num_expected == num_received:
            api_options_updates = current_settings["ollama_api_options"].copy()
            for key, value in zip(ordered_keys, api_option_values):
                original_value = initial_options_for_types.get(key)
                try:
                    if isinstance(original_value, bool): api_options_updates[key] = bool(value)
                    elif isinstance(original_value, int): api_options_updates[key] = int(value)
                    elif isinstance(original_value, float): api_options_updates[key] = float(value)
                    else: api_options_updates[key] = value
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert saved value '{value}' for option '{key}' based on original type {type(original_value)}. Saving as received. Error: {e}")
                    api_options_updates[key] = value
            current_settings["ollama_api_options"] = api_options_updates
            # print(f" Ollama API options updated ({num_expected} values).") # Reduce verbosity
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
        return "", "", [], "concatenate", empty_state

    team_data = all_teams_data_state.get(team_name_to_load)
    if not team_data or not isinstance(team_data, dict):
        print(f"Error: Team data not found or invalid for '{team_name_to_load}'.")
        error_state = {"name": team_name_to_load, "description": "Error: Team data not found.", "steps": [], "assembly_strategy": "concatenate"}
        return team_name_to_load, "Error: Team data not found.", [], "concatenate", error_state

    steps = copy.deepcopy(team_data.get("steps", [])) # Editor gets its own copy of the steps
    if not isinstance(steps, list): steps = []

    editor_state = {
        "name": team_name_to_load,
        "description": team_data.get("description", ""),
        "steps": steps,
        "assembly_strategy": team_data.get("assembly_strategy", "concatenate")
    }
    # print(f" Loaded: {editor_state}") # Reduce verbosity
    return ( editor_state["name"], editor_state["description"], editor_state["steps"], editor_state["assembly_strategy"], editor_state )


def clear_team_editor():
    """Clears the editor fields for creating a new team."""
    print("Clearing team editor.")
    empty_state = {"name": "", "description": "", "steps": [], "assembly_strategy": "concatenate"}
    return "", "", [], "concatenate", empty_state


def add_step_to_editor(agent_role_display_name_to_add, current_editor_state):
    """Adds the selected agent as a new step to the current editor state."""
    if not agent_role_display_name_to_add:
        print("Add Step: No agent selected."); gr.Warning("Please select an agent role to add.")
        return current_editor_state, gr.update()

    # Deep copy: the steps list must not be shared with the loaded team until "Save" is pressed
    editor_state = copy.deepcopy(current_editor_state) if isinstance(current_editor_state, dict) else {"name": "", "description": "", "steps": [], "assembly_strategy": "concatenate"}
    if "steps" not in editor_state or not isinstance(editor_state["steps"], list): editor_state["steps"] = []

    actual_role_name = get_actual_role_name(agent_role_display_name_to_add)
    new_step = {"role": actual_role_name}
    editor_state["steps"].append(new_step)
    print(f"Add Step: Added '{actual_role_name}'. Steps: {len(editor_state['steps'])}")
    return editor_state, editor_state["steps"]


def remove_step_from_editor(step_index_to_remove, current_editor_state):
    """Removes a step at the specified index (1-based) from the editor state."""
    # Deep copy: the steps list must not be shared with the loaded team until "Save" is pressed
    editor_state = copy.deepcopy(current_editor_state) if isinstance(current_editor_state, dict) else {"name": "", "description": "", "steps": [], "assembly_strategy": "concatenate"}
    steps = editor_state.get("steps", [])
    if not isinstance(steps, list): steps = []

    try:
        index_0_based = int(step_index_to_remove) - 1
        if not (0 <= index_0_based < len(steps)): raise IndexError
    except (ValueError, TypeError, IndexError):
        msg = f"Invalid step number '{step_index_to_remove}' (must be 1 to {len(steps)})."
        print(f"Remove Step: {msg}"); gr.Warning(msg)
        return editor_state, gr.update()

    removed_step = steps.pop(index_0_based);
    editor_state["steps"] = steps
    print(f"Remove Step: Removed step {step_index_to_remove} ('{removed_step.get('role')}'). Steps: {len(steps)}")
    return editor_state, editor_state["steps"]


def save_team_from_editor(
    team_name_in, description_in, assembly_strategy_in,
    current_editor_state, all_teams_data_state,
    current_settings_state, file_agents_dict_state
    ) -> tuple[dict, dict, dict, dict, str]:
    """Saves the team currently defined in the editor fields."""
    print("Attempting to save team...")
    team_name = team_name_in.strip()
    default_outputs = [all_teams_data_state, gr.update(), gr.update(), gr.update(), "Save failed."]

    if not team_name: msg = "Team Name cannot be empty."; print(msg); gr.Warning(msg); return default_outputs[0:4] + [msg]

    steps = current_editor_state.get("steps", [])
    if not steps: msg = f"Save Warning: Team '{team_name}' has no steps."; print(msg); gr.Warning(msg)

    all_teams_data = all_teams_data_state.copy() if isinstance(all_teams_data_state, dict) else {}
    team_data = {"description": description_in.strip(), "steps": steps, "assembly_strategy": assembly_strategy_in}
    all_teams_data[team_name] = team_data

    if save_teams_to_file(all_teams_data):
        msg = f"Team '{team_name}' saved successfully."
        print(msg)
        # Reload roles/teams for dropdown updates
        combined_roles = load_all_roles(current_settings_state, file_agents=file_agents_dict_state)
        file_agent_keys = list(file_agents_dict_state.keys()) if file_agents_dict_state else []
        role_display_choices = sorted([get_role_display_name(name, file_agent_keys) for name in combined_roles.keys()])
        new_team_choices = sorted(list(all_teams_data.keys()))
        new_chat_choices = ["(Direct Agent Call)"] + sorted([f"[Team] {name}" for name in new_team_choices]) + role_display_choices
        return (
            all_teams_data,
            gr.update(choices=new_team_choices, value=team_name),
            gr.update(choices=new_chat_choices, value=f"[Team] {team_name}"),
            gr.update(choices=new_chat_choices, value=f"[Team] {team_name}"),
            msg
        )
    else:
        msg = f"Error: Failed to save team data."; print(msg); gr.Warning(msg)
        return default_outputs[0:4] + [msg]


def delete_team_logic(
    team_name_to_delete, all_teams_data_state,
    current_settings_state, file_agents_dict_state
    ) -> tuple[dict, dict, dict, dict, str, str, str, list, str, dict]:
    """Deletes the selected team."""
    print(f"Attempting to delete team: '{team_name_to_delete}'")
    clear_name, clear_desc, clear_steps, clear_strat, clear_state = clear_team_editor()
    no_change_outputs = [
        all_teams_data_state, gr.update(), gr.update(), gr.update(),
        "Delete failed.",
        clear_name, clear_desc, clear_steps, clear_strat, clear_state
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
        return (
            all_teams_data,
            gr.update(choices=new_team_choices, value=None),
            gr.update(choices=new_chat_choices, value="(Direct Agent Call)"),
            gr.update(choices=new_chat_choices, value="(Direct Agent Call)"),
            msg,
            clear_name, clear_desc, clear_steps, clear_strat, clear_state
        )
    else:
        msg = f"Error: Failed to save teams after deletion."; print(msg); gr.Warning(msg)
        return no_change_outputs[0:4] + [msg] + no_change_outputs[5:]


# --- History Callbacks ---
def show_clear_confirmation(): return gr.update(visible=True)
def hide_clear_confirmation(): return gr.update(visible=False)
def clear_full_history_callback(history_list_state):
    """Clears the persistent history file and returns updates."""
    print("Full history file cleared.")
    history.save_history([]) # Save empty list via manager
    return "", hide_clear_confirmation(), [] # Match outputs