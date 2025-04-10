# ArtAgent/app.py

# Part 1: Imports
import gradio as gr
# Standard lib imports
import json
import os
import atexit
# No direct need for PIL, numpy, requests, tempfile, shutil, time in this file anymore

# Import from project structure
from core.utils import load_json, get_theme_object, get_absolute_path # Removed format_json_to_html_table
from core.ollama_checker import OllamaStatusChecker
from core import history_manager as history # Use alias for clarity
from core.sweep_manager import run_sweep # Import sweep logic
from ui.sweep_tab import create_sweep_tab # Import sweep UI tab creator
# Import logic functions that will be used as callbacks
from core.app_logic import (
    execute_chat_or_team, # Use the router function for submit
    comment_logic,
    update_max_tokens_on_limiter_change,
    clear_session_history_callback,
    # update_role_dropdown_callback, # Logic now in refresh_agent_team_dropdowns_wrapper
    handle_agent_file_upload,
    load_profile_options_callback,
    save_settings_callback,
    show_clear_confirmation,
    hide_clear_confirmation,
    clear_full_history_callback,
    # Import team editor logic functions
    load_team_for_editing, clear_team_editor, add_step_to_editor,
    remove_step_from_editor, save_team_from_editor, delete_team_logic
)
# Import the logic function for releasing models (not the callback itself)
from core.ollama_manager import release_all_models_logic
# Import functions needed for initial UI setup from roles_config
from agents.roles_config import load_all_roles, get_role_display_name # Import get_role_display_name

# Import UI creation functions
from ui.chat_tab import create_chat_tab
from ui.app_settings_tab import create_app_settings_tab
from ui.team_editor_tab import create_team_editor_tab # Import new UI tab function
from ui.roles_tab import create_roles_tabs
from ui.history_tab import create_history_tab
from ui.common_ui_elements import create_footer
from ui.captions_tab import create_captions_tab # Import new UI tab
from core.captioning_logic import ( # Import captioning logic functions
    load_images_and_captions,
    update_caption_display,
    save_caption,
    batch_edit_captions,
    # --- NEW: Add placeholder imports for generation logic ---
    generate_captions_for_selected, # Will be implemented later
    generate_captions_for_all       # Will be implemented later
)
# --- Constants ---
SETTINGS_FILE = 'settings.json'
MODELS_FILE = 'models.json'
LIMITERS_FILE = 'limiters.json'
PROFILES_FILE = 'ollama_profiles.json'
AGENT_TEAMS_FILE = 'agent_teams.json' # Define team file constant
DEFAULT_ROLES_FILE = 'agents/agent_roles.json' # Used indirectly by load_all_roles
CUSTOM_ROLES_FILE = 'agents/custom_agent_roles.json' # Used indirectly by load_all_roles

# --- Utility Functions (App Specific or Loading) ---
# Use load_json from core.utils for consistency
def load_settings(settings_file=SETTINGS_FILE):
    return load_json(settings_file, is_relative=True)

def load_models(model_file=MODELS_FILE):
    models_list = load_json(model_file, is_relative=True)
    return models_list if isinstance(models_list, list) else []

def load_limiters(limiter_file=LIMITERS_FILE):
    limiter_dict = load_json(limiter_file, is_relative=True)
    return limiter_dict if isinstance(limiter_dict, dict) else {}

def load_profiles(profile_file=PROFILES_FILE):
    profile_dict = load_json(profile_file, is_relative=True)
    return profile_dict if isinstance(profile_dict, dict) else {}

def load_agent_teams(team_file=AGENT_TEAMS_FILE): # Function to load teams
    team_dict = load_json(team_file, is_relative=True)
    return team_dict if isinstance(team_dict, dict) else {}

# --- Load Initial Data ---
settings = load_settings()
OLLAMA_API_URL = settings.get("ollama_url", "http://localhost:11434/api/generate")

models_data = load_models()
limiters_data = load_limiters()
profiles_data = load_profiles()
history_list = history.load_history() # Persistent history list
teams_data = load_agent_teams() # Load teams

# Generate initial lists for UI dropdowns
model_names_with_vision = [f"{m['name']} (VISION)" if m.get('vision') else m.get('name', 'Unknown') for m in models_data] if models_data else []
limiters_names = list(limiters_data.keys())
profile_names = list(profiles_data.keys())
team_names = list(teams_data.keys())

# Load initial roles based on startup settings flags
initial_roles_data = load_all_roles(settings) # Pass settings to respect flags
initial_role_names = list(initial_roles_data.keys())
initial_role_display_names = sorted([get_role_display_name(name) for name in initial_role_names])

# Create combined list for initial agent/team dropdowns (Used by Chat AND Captions tab) # MODIFIED
initial_agent_team_choices = ["(Direct Agent Call)"] + sorted([f"[Team] {name}" for name in team_names]) + initial_role_display_names
# Generate initial list of ALL available agents for editor's dropdown
all_available_agents_initial = load_all_roles(settings, file_agents={}) # Load all non-file agents initially
all_available_agent_display_names_initial = sorted([get_role_display_name(name) for name in all_available_agents_initial.keys()])
all_initial_worker_model_choices = sorted(list(set([name.replace(" (VISION)","") for name in model_names_with_vision])))

# --- Perform Initial Startup Check (Console Feedback Only) ---
startup_checker = OllamaStatusChecker(OLLAMA_API_URL) # Pass full URL
if not startup_checker.check(): print(startup_checker.get_console_message())
else: print(f"[Startup Check] Ollama appears responsive at base URL {startup_checker.base_url}. Application starting.")

# --- Build UI ---
selected_theme_name = settings.get("gradio_theme", "Default")
theme_object = get_theme_object(selected_theme_name)
print(f"Applying theme: {selected_theme_name}")

# Get initial options keys order for callbacks that need it
initial_ollama_options = settings.get("ollama_api_options", {})
initial_ollama_options_ordered_keys = sorted(initial_ollama_options.keys())

# --- Gradio Interface Definition ---
with gr.Blocks(title="ArtAgents", theme=theme_object) as demo:

    # --- Define Main App States ---
    settings_state = gr.State(value=settings)
    models_data_state = gr.State(value=models_data)
    limiters_data_state = gr.State(value=limiters_data)
    profiles_data_state = gr.State(value=profiles_data)
    teams_data_state = gr.State(value=teams_data) # Add state for teams data
    history_list_state = gr.State(value=history_list) # Persistent history list
    session_history_state = gr.State(value=[]) # Current Session history (list of strings)
    api_option_keys_state = gr.State(value=initial_ollama_options_ordered_keys)
    # State for the list of all agent display names available for the editor's dropdown
    all_available_agents_display_state = gr.State(value=all_available_agent_display_names_initial)
    # Add states for captioning tab
    caption_image_paths_state = gr.State({}) # filename: absolute_path
    caption_data_state = gr.State({})        # filename: caption_text
    caption_selected_item_state = gr.State(None) # filename of selected image

    # --- Create UI Tabs using functions from ui/ ---
    # Pass initial data needed to build the UI components correctly
    chat_comps = create_chat_tab(initial_agent_team_choices, model_names_with_vision, limiters_names, settings)
    settings_comps = create_app_settings_tab(settings)
    editor_comps = create_team_editor_tab(
        initial_team_names=sorted(team_names),
        initial_available_agent_names=all_available_agent_display_names_initial # Pass initial list
    )
    sweep_comps = create_sweep_tab(
        initial_team_names=sorted(team_names), # Use loaded team names
        initial_model_names=all_initial_worker_model_choices # Use filtered model names
    )
    roles_comps = create_roles_tabs(
         load_json(DEFAULT_ROLES_FILE, is_relative=True),
         load_json(CUSTOM_ROLES_FILE, is_relative=True)
    )
    history_comps = create_history_tab(history_list)
    # Create the Captions tab instance, passing initial choices # <--- FIXED ---
    caption_comps = create_captions_tab(
        initial_agent_team_choices=initial_agent_team_choices
    )

    create_footer()

    # --- Define Wrapper/Helper Callbacks specific to app.py ---
    # This function refreshes the combined Agent/Team dropdown in the CHAT tab AND CAPTIONS TAB
    def refresh_agent_team_dropdowns_wrapper(use_default, use_custom, file_agents_dict, teams_dict):
         print("Refreshing Chat & Captions tab agent/team dropdowns...") # Updated print
         current_settings = load_settings()
         current_settings["using_default_agents"] = use_default; current_settings["using_custom_agents"] = use_custom
         combined_roles = load_all_roles(current_settings, file_agents=file_agents_dict)
         file_agent_keys = list(file_agents_dict.keys()) if file_agents_dict else []
         role_display_choices = sorted([get_role_display_name(name, file_agent_keys) for name in combined_roles.keys()])
         team_display_choices = sorted([f"[Team] {name}" for name in teams_dict.keys()])
         all_choices = ["(Direct Agent Call)"] + team_display_choices + role_display_choices
         new_value = all_choices[0] if all_choices else None
         print(f" Chat/Captions Agent/Team choices updated: {len(all_choices)} total")
         # Return updates for BOTH dropdowns
         chat_update = gr.Dropdown.update(choices=all_choices, value=new_value)
         caption_update = gr.Dropdown.update(choices=all_choices, value=new_value) # Add update for captions dropdown
         return chat_update, caption_update # Return tuple of updates

    # This function refreshes the AGENT dropdown in the TEAM EDITOR tab
    def refresh_available_agents_for_editor_wrapper(use_default, use_custom, file_agents_dict):
        print("Refreshing available agent list for team editor...")
        current_settings = load_settings()
        current_settings["using_default_agents"] = use_default
        current_settings["using_custom_agents"] = use_custom
        combined_roles = load_all_roles(current_settings, file_agents=file_agents_dict)
        file_agent_keys = list(file_agents_dict.keys()) if file_agents_dict else []
        display_choices = sorted([get_role_display_name(name, file_agent_keys) for name in combined_roles.keys()])
        print(f" Available agents for editor updated: {len(display_choices)} total")
        # Return update for the dropdown component AND the state holding the list
        return gr.Dropdown.update(choices=display_choices), display_choices

    # Wrapper for releasing all models using state for settings
    def release_all_models_ui_callback(current_settings_state):
         return release_all_models_logic(current_settings_state)


    # --- Wire Event Handlers ---

    # -- Chat Tab Wiring --
    chat_comps['model_with_vision'].change(
        fn=lambda x: x, inputs=[chat_comps['model_with_vision']],
        outputs=[chat_comps['selected_model_tracker']]
    )

    # Agent file upload handling (chained events to update state AND dropdowns)
    agent_file_upload_outputs = [
        chat_comps['loaded_file_agents_state'], # 1. Update file agent state
        chat_comps['loaded_agent_file_display'],# 2. Update filename display
    ]
    file_upload_event = chat_comps['agent_file_upload'].upload(
         fn=handle_agent_file_upload, # Logic from core/app_logic.py
         inputs=[chat_comps['agent_file_upload']],
         outputs=agent_file_upload_outputs
    )
    # After file upload, refresh Chat, Captions, and Editor dropdowns # <--- FIXED ---
    refresh_trigger_inputs = [
        settings_comps['settings_use_default'], settings_comps['settings_use_custom'],
        chat_comps['loaded_file_agents_state'], teams_data_state # Needs teams too for chat/captions dropdown
    ]
    refresh_editor_inputs = [
        settings_comps['settings_use_default'], settings_comps['settings_use_custom'],
        chat_comps['loaded_file_agents_state']
    ]
    file_upload_event.then(
        # Update Chat & Captions agent/team dropdowns
        fn=refresh_agent_team_dropdowns_wrapper, inputs=refresh_trigger_inputs,
        outputs=[chat_comps['role_dropdown'], caption_comps['caption_agent_selector']]
    ).then(
         fn=refresh_available_agents_for_editor_wrapper, # Update Editor's agent pool
         inputs=refresh_editor_inputs,
         outputs=[editor_comps['agent_to_add_dropdown'], all_available_agents_display_state] # Update dropdown & state
    )

    # Checkbox changes trigger dropdown refreshes (Chat, Captions, Editor) # <--- FIXED ---
    checkbox_refresh_trigger_inputs = [
        settings_comps['settings_use_default'], settings_comps['settings_use_custom'],
        chat_comps['loaded_file_agents_state'], teams_data_state # Include teams for chat/captions dropdown
    ]
    checkbox_refresh_editor_inputs = [
        settings_comps['settings_use_default'], settings_comps['settings_use_custom'],
        chat_comps['loaded_file_agents_state']
    ]
    settings_comps['settings_use_default'].change(
        fn=refresh_agent_team_dropdowns_wrapper, inputs=checkbox_refresh_trigger_inputs,
        outputs=[chat_comps['role_dropdown'], caption_comps['caption_agent_selector']] # Update Chat & Captions dropdowns
    ).then(
        fn=refresh_available_agents_for_editor_wrapper, inputs=checkbox_refresh_editor_inputs,
        outputs=[editor_comps['agent_to_add_dropdown'], all_available_agents_display_state] # Update Editor dropdown
    )
    settings_comps['settings_use_custom'].change(
        fn=refresh_agent_team_dropdowns_wrapper, inputs=checkbox_refresh_trigger_inputs,
        outputs=[chat_comps['role_dropdown'], caption_comps['caption_agent_selector']] # Update Chat & Captions dropdowns
    ).then(
        fn=refresh_available_agents_for_editor_wrapper, inputs=checkbox_refresh_editor_inputs,
        outputs=[editor_comps['agent_to_add_dropdown'], all_available_agents_display_state] # Update Editor dropdown
    )

    # Update max tokens slider based on limiter choice
    chat_comps['limiter_handling_option'].change(
        fn=update_max_tokens_on_limiter_change, # Logic from core/app_logic.py
        inputs=[chat_comps['limiter_handling_option'], chat_comps['max_tokens_slider']],
        outputs=[chat_comps['max_tokens_slider']]
    )

    # Submit Action - Calls the main router function
    submit_inputs = [
        # UI Component values needed by logic
        chat_comps['folder_path'], chat_comps['user_input'], chat_comps['model_with_vision'],
        chat_comps['max_tokens_slider'], chat_comps['file_handling_option'],
        chat_comps['limiter_handling_option'], chat_comps['single_image_display'],
        chat_comps['use_ollama_api_options'], chat_comps['release_model_on_change'],
        chat_comps['role_dropdown'], # This now holds role OR team name display value
        # State values needed by logic
        settings_state, models_data_state, limiters_data_state, teams_data_state,
        chat_comps['selected_model_tracker'], chat_comps['loaded_file_agents_state'],
        history_list_state, session_history_state
    ]
    # execute_chat_or_team returns: response_text, session_history_text, model_name_used, new_session_history_list
    submit_outputs = [
        chat_comps['llm_response_display'],          # 1. Response Text -> Textbox
        chat_comps['current_session_history_display'], # 2. Session History Text -> Textbox
        chat_comps['model_state'],                  # 3. Model Name Used -> State
        session_history_state                       # 4. Updated Session History List -> State
    ]
    chat_comps['submit_button'].click(
        fn=execute_chat_or_team, # Call the router function from core/app_logic.py
        inputs=submit_inputs,
        outputs=submit_outputs
        )

    # Comment Action
    comment_inputs = [
        # UI Component values
        chat_comps['llm_response_display'], chat_comps['comment_input'],
        chat_comps['max_tokens_slider'], chat_comps['use_ollama_api_options'],
        # State values needed by comment_logic
        chat_comps['model_state'], # Use model from last run state
        settings_state, chat_comps['loaded_file_agents_state'],
        history_list_state, session_history_state
    ]
    # comment_logic returns: new_response_text, new_session_history_list
    comment_outputs = [
        chat_comps['llm_response_display'],           # 1. New Response Text -> Textbox
        chat_comps['current_session_history_display'], # 2a. Session History Text -> Textbox
        session_history_state                       # 2b. Update Session History List -> State
    ]
    chat_comps['comment_button'].click(
        fn=comment_logic, # Call the logic function
        inputs=comment_inputs,
        outputs=comment_outputs
        )

    # Clear Session History Action
    chat_comps['clear_session_button'].click(
        fn=clear_session_history_callback, # Logic from core/app_logic.py
        inputs=[session_history_state], # Pass state to clear
        outputs=[chat_comps['current_session_history_display'], session_history_state] # Clear display and state
    )

    # -- App Settings Tab Wiring --
    # Get list of UI components for API options in correct order
    api_option_components_list = [settings_comps['ollama_options_ui_elements'][key] for key in initial_ollama_options_ordered_keys]

    # Load Profile Action
    settings_comps['load_profile_button'].click(
         fn=load_profile_options_callback, # Logic from core/app_logic.py
         inputs=[
              settings_comps['profile_select'], # Selected profile name
              profiles_data_state,           # Full profile data from state
              api_option_keys_state         # Ordered keys list from state
         ],
         # outputs is the list of UI components themselves
         outputs=api_option_components_list
    )

    # Save Settings Action
    settings_save_inputs = [
         # General settings components' values will be passed
         settings_comps['settings_ollama_url'], settings_comps['settings_max_tokens_slider_range'],
         settings_comps['settings_api_to_console'], settings_comps['settings_use_default'],
         settings_comps['settings_use_custom'], settings_comps['settings_use_ollama_opts_default'],
         settings_comps['settings_release_model_default'], settings_comps['settings_theme_select'],
         # State inputs needed by save logic
         api_option_keys_state,
    ] + api_option_components_list # Add API option components values dynamically as *args

    settings_comps['settings_save_button'].click(
        fn=save_settings_callback, # Logic from core/app_logic.py
        inputs=settings_save_inputs,
        outputs=[settings_comps['save_status_display']]
        # This saves to settings.json. Consider updating settings_state if immediate reflection needed.
    )

    # Release Models Action needs the current settings (for URL), pass via state
    settings_comps['release_models_button'].click(
         fn=release_all_models_ui_callback, # Use the wrapper defined above
         inputs=[settings_state], # Pass settings dict from state
         outputs=[settings_comps['release_status_display']]
    )

    # -- Agent Team Editor Tab Wiring --
    editor_comps['load_team_button'].click(
        fn=load_team_for_editing, # Logic from core/app_logic.py
        inputs=[
            editor_comps['team_select_dropdown'],
            teams_data_state # Need all teams data
        ],
        outputs=[ # Update all editor fields + state
            editor_comps['team_name_textbox'],
            editor_comps['team_description_textbox'],
            editor_comps['steps_display_json'],
            editor_comps['assembly_strategy_radio'],
            editor_comps['current_team_editor_state']
        ]
    )

    editor_comps['clear_editor_button'].click(
        fn=clear_team_editor, # Logic from core/app_logic.py
        inputs=[],
        outputs=[ # Clear all editor fields + state
            editor_comps['team_name_textbox'],
            editor_comps['team_description_textbox'],
            editor_comps['steps_display_json'],
            editor_comps['assembly_strategy_radio'],
            editor_comps['current_team_editor_state']
        ]
    )

    editor_comps['add_step_button'].click(
        fn=add_step_to_editor, # Logic from core/app_logic.py
        inputs=[
            editor_comps['agent_to_add_dropdown'], # Agent role display name to add
            editor_comps['current_team_editor_state'] # Current state being edited
        ],
        outputs=[ # Update editor state and the JSON display
            editor_comps['current_team_editor_state'],
            editor_comps['steps_display_json']
        ]
    )

    editor_comps['remove_step_button'].click(
        fn=remove_step_from_editor, # Logic from core/app_logic.py
        inputs=[
            editor_comps['step_index_to_remove'], # 1-based index from UI
            editor_comps['current_team_editor_state'] # Current state being edited
        ],
        outputs=[ # Update editor state and the JSON display
            editor_comps['current_team_editor_state'],
            editor_comps['steps_display_json']
        ]
    )

    # Save Team Button needs to update team state and dropdowns (Chat & Captions) # <--- FIXED ---
    save_team_inputs = [
        editor_comps['team_name_textbox'],
        editor_comps['team_description_textbox'],
        editor_comps['assembly_strategy_radio'],
        editor_comps['current_team_editor_state'], # Pass internal state with steps list
        teams_data_state, # Pass state with all teams to update
        settings_state, # Needed by logic to update dropdowns
        chat_comps['loaded_file_agents_state'] # Needed by logic to update dropdowns
    ]
    save_team_outputs = [
        teams_data_state,                       # 1. Update the main state holding all teams
        editor_comps['team_select_dropdown'],   # 2. Update the editor's team selection dropdown
        chat_comps['role_dropdown'],            # 3. Update the chat tab's agent/team dropdown
        caption_comps['caption_agent_selector'],# 4. Update the captions tab's agent/team dropdown # <-- ADDED
        editor_comps['save_status_textbox']     # 5. Show status message
    ]
    editor_comps['save_team_button'].click(
        fn=save_team_from_editor, # Logic in core/app_logic.py needs corresponding output adjustment
        inputs=save_team_inputs,
        outputs=save_team_outputs # Pass the updated list of outputs
    )

    # Delete Team Button needs to update team state, dropdowns (Chat & Captions), and clear editor # <--- FIXED ---
    delete_team_inputs = [
        editor_comps['team_select_dropdown'], # Name to delete
        teams_data_state,                     # All teams data state
        settings_state,                       # Needed by logic to update dropdowns
        chat_comps['loaded_file_agents_state'] # Needed by logic to update dropdowns
    ]
    delete_outputs = [
         teams_data_state,                      # 1. Update main teams state
         editor_comps['team_select_dropdown'],  # 2. Update editor dropdown
         chat_comps['role_dropdown'],           # 3. Update chat dropdown
         caption_comps['caption_agent_selector'],# 4. Update the captions tab's agent/team dropdown # <-- ADDED
         editor_comps['save_status_textbox'],   # 5. Status message
         # 6-10: Outputs to clear editor fields after delete (from clear_team_editor return)
         editor_comps['team_name_textbox'],
         editor_comps['team_description_textbox'],
         editor_comps['steps_display_json'],
         editor_comps['assembly_strategy_radio'],
         editor_comps['current_team_editor_state']
    ]
    editor_comps['delete_team_button'].click(
        fn=delete_team_logic, # Logic in core/app_logic.py needs corresponding output adjustment
        inputs=delete_team_inputs,
        outputs=delete_outputs # Pass the updated list of outputs
    )


    # -- History Tab Wiring --
    history_comps['clear_history_button'].click(
         fn=show_clear_confirmation, inputs=[], outputs=[history_comps['confirm_clear_group']]
    )
    history_comps['yes_clear_button'].click(
         fn=clear_full_history_callback, # Logic from core/app_logic.py
         inputs=[history_list_state], # Pass persistent history state
         outputs=[
              history_comps['full_history_display'], # Update display
              history_comps['confirm_clear_group'],  # Hide confirmation
              history_list_state                     # Update persistent history state
         ]
    )
    history_comps['no_clear_button'].click(
         fn=hide_clear_confirmation, inputs=[], outputs=[history_comps['confirm_clear_group']]
    )

    # -- Experiment Sweep Tab Wiring --
    sweep_comps['sweep_start_button'].click(
        fn=run_sweep, # Call the modified run_sweep
        inputs=[ # Inputs remain the same
            sweep_comps['sweep_prompts_input'],
            sweep_comps['sweep_teams_select'],
            sweep_comps['sweep_models_select'],
            sweep_comps['sweep_output_folder_input'],
            sweep_comps['sweep_log_intermediate_checkbox'],
            settings_state,
            teams_data_state,
        ],
        outputs=[sweep_comps['sweep_status_display']]
    )

    # -- Captions Tab Wiring --
    caption_comps['captions_load_button'].click(
        fn=load_images_and_captions,
        inputs=[caption_comps['captions_folder_path']],
        # Outputs: image_selector_choices, image_paths_state, caption_data_state, status_display, selected_item_state, caption_display
        outputs=[
            caption_comps['captions_image_selector'], # Update choices
            caption_image_paths_state,          # Update state
            caption_data_state,                 # Update state
            caption_comps['captions_status_display'], # Show status
            caption_selected_item_state,        # Update state with first item
            caption_comps['captions_caption_display'], # Show first caption
            caption_comps['caption_selected_filename_display'] # Show first filename
        ]
    )

    # Update caption display AND preview when selection changes in CheckboxGroup
    caption_comps['captions_image_selector'].change(
        fn=update_caption_display,
        inputs=[
            caption_comps['captions_image_selector'], # List of selected filenames
            caption_data_state,                     # Dict of captions
            caption_image_paths_state               # Dict of image paths (ADDED)
        ],
        outputs=[
            caption_comps['captions_caption_display'],    # Update caption text
            caption_selected_item_state,                # Update selected item state
            caption_comps['caption_selected_filename_display'], # Update filename display
            caption_comps['caption_image_preview']      # Update image preview (ADDED)
        ]
    )
    # Save current caption button
    caption_comps['captions_save_button'].click(
        fn=save_caption,
        inputs=[
            caption_selected_item_state, # Filename of image being edited
            caption_comps['captions_caption_display'], # The edited text
            caption_image_paths_state, # Dict mapping filename -> path
            caption_data_state        # Dict mapping filename -> caption (to update state)
        ],
        # Outputs: status_display, caption_data_state
        outputs=[
            caption_comps['captions_status_display'],
            caption_data_state # Update the state with the saved caption
        ]
    )

    # Batch Append Button
    caption_comps['captions_append_button'].click(
        fn=lambda *args: batch_edit_captions(*args, mode="Append"), # Use lambda to pass mode
        inputs=[
            caption_comps['captions_image_selector'], # List of selected filenames
            caption_comps['captions_batch_text'],
            caption_image_paths_state,
            caption_data_state
        ],
        # Outputs: status_display, caption_data_state
        outputs=[
            caption_comps['captions_status_display'],
            caption_data_state # Update state with modified captions
        ]
    )

    # Batch Prepend Button
    caption_comps['captions_prepend_button'].click(
        fn=lambda *args: batch_edit_captions(*args, mode="Prepend"), # Use lambda to pass mode
        inputs=[
            caption_comps['captions_image_selector'], # List of selected filenames
            caption_comps['captions_batch_text'],
            caption_image_paths_state,
            caption_data_state
        ],
        # Outputs: status_display, caption_data_state
        outputs=[
            caption_comps['captions_status_display'],
            caption_data_state # Update state with modified captions
        ]
    )

    # --- NEW: Wire Generate Caption Buttons ---
    # Generate for Selected
    caption_comps['caption_generate_selected_button'].click(
        fn=generate_captions_for_selected, # Function to be created in captioning_logic.py
        inputs=[ # Inputs needed by the generation logic
            caption_comps['captions_image_selector'], # Selected image filenames (list)
            caption_comps['caption_agent_selector'], # Selected Agent/Team name
            caption_comps['caption_generate_mode'], # Overwrite/Skip/etc.
            caption_image_paths_state,      # State: filename -> path
            caption_data_state,             # State: filename -> caption (to potentially update)
            settings_state,                 # State: App settings (for Ollama URL, options)
            models_data_state,              # State: Available models
            limiters_data_state,            # State: Limiters
            teams_data_state,               # State: Team definitions
            chat_comps['loaded_file_agents_state'], # State: File agents
            history_list_state,             # State: Persistent history (for logging)
            session_history_state,          # State: Session history (for logging/display)
        ],
        outputs=[ # Outputs to update
            caption_comps['captions_status_display'], # Display status messages
            caption_data_state,                       # Update caption state if captions changed
            # Optionally update the caption display if only one image was selected and processed
            caption_comps['captions_caption_display'], # Update display with generated caption
            session_history_state # Pass back updated session history list
        ]
    )

    # Generate for All
    caption_comps['caption_generate_all_button'].click(
        fn=generate_captions_for_all, # Function to be created in captioning_logic.py
        inputs=[ # Inputs needed by the generation logic
            caption_image_paths_state,      # State: filename -> path (all loaded images)
            caption_data_state,             # State: filename -> caption (all loaded captions)
            caption_comps['caption_agent_selector'], # Selected Agent/Team name
            caption_comps['caption_generate_mode'], # Overwrite/Skip/etc.
            settings_state,                 # State: App settings
            models_data_state,              # State: Available models
            limiters_data_state,            # State: Limiters
            teams_data_state,               # State: Team definitions
            chat_comps['loaded_file_agents_state'], # State: File agents
            history_list_state,             # State: Persistent history
            session_history_state,          # State: Session history
        ],
        outputs=[ # Outputs to update
            caption_comps['captions_status_display'], # Display status messages
            caption_data_state,                       # Update caption state with all generated captions
            # Update caption display with the caption of the *last* processed image in the batch
            caption_comps['captions_caption_display'],
            session_history_state # Pass back updated session history list
        ]
    )

# --- atexit handler ---
def on_exit():
    """Function called on script exit."""
    print("Application exiting...")
    # cleanup_temp_dir() # Add back if needed
    print("Cleanup finished.")
atexit.register(on_exit)

# --- Launch the Application ---
if __name__ == "__main__":
    print("Initializing ArtAgents...")
    # Set launch parameters
    demo.launch(
        # share=True # For public link
        # server_name="0.0.0.0" # For network access
        # debug=True # Enable Gradio debug mode if needed
    )
    print("Gradio App shutdown.")