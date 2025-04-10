# ArtAgent/app.py

# Part 1: Imports
import gradio as gr
# Standard lib imports
import json
import os
import atexit
# No direct need for numpy, requests, tempfile, shutil, time in this file anymore

# Import from project structure
from core.utils import load_json, get_theme_object, get_absolute_path
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

# --- Corrected: Moved captioning logic imports to where they are used or defined ---
from core.captioning_logic import (
    load_images_and_captions,
    # update_caption_display, # Replaced by update_caption_display_from_gallery
    update_caption_display_from_gallery, # NEW function for Gallery
    save_caption,
    batch_edit_captions, # Needs redesign for Gallery multi-select
    generate_captions_for_selected, # Uses single selected item state
    generate_captions_for_all
)
# --- End Correction ---

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
# Create list specifically for vision models for Captions tab
vision_model_names = [f"{m['name']} (VISION)" for m in models_data if m.get('vision')] if models_data else []
limiters_names = list(limiters_data.keys())
profile_names = list(profiles_data.keys())
team_names = list(teams_data.keys())

# Load initial roles based on startup settings flags
initial_roles_data = load_all_roles(settings) # Pass settings to respect flags
initial_role_names = list(initial_roles_data.keys())
initial_role_display_names = sorted([get_role_display_name(name) for name in initial_role_names])

# Create combined list for initial agent/team dropdowns (Used by Chat AND Captions tab)
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
    chat_comps = create_chat_tab(initial_agent_team_choices, model_names_with_vision, limiters_names, settings)
    settings_comps = create_app_settings_tab(settings)
    editor_comps = create_team_editor_tab(
        initial_team_names=sorted(team_names),
        initial_available_agent_names=all_available_agent_display_names_initial
    )
    sweep_comps = create_sweep_tab(
        initial_team_names=sorted(team_names),
        initial_model_names=all_initial_worker_model_choices
    )
    roles_comps = create_roles_tabs(
         load_json(DEFAULT_ROLES_FILE, is_relative=True),
         load_json(CUSTOM_ROLES_FILE, is_relative=True)
    )
    history_comps = create_history_tab(history_list)
    # Pass vision models list to captions tab
    caption_comps = create_captions_tab(
        initial_agent_team_choices=initial_agent_team_choices,
        initial_vision_models=vision_model_names # Pass the vision model list
    )

    create_footer()

    # --- Define Wrapper/Helper Callbacks specific to app.py ---
    # Refreshes Chat & Captions Agent/Team dropdowns
    def refresh_agent_team_dropdowns_wrapper(use_default, use_custom, file_agents_dict, teams_dict):
         print("Refreshing Chat & Captions tab agent/team dropdowns...")
         current_settings = load_settings()
         current_settings["using_default_agents"] = use_default; current_settings["using_custom_agents"] = use_custom
         combined_roles = load_all_roles(current_settings, file_agents=file_agents_dict)
         file_agent_keys = list(file_agents_dict.keys()) if file_agents_dict else []
         role_display_choices = sorted([get_role_display_name(name, file_agent_keys) for name in combined_roles.keys()])
         team_display_choices = sorted([f"[Team] {name}" for name in teams_dict.keys()])
         all_choices = ["(Direct Agent Call)"] + team_display_choices + role_display_choices
         new_value = all_choices[0] if all_choices else None
         print(f" Chat/Captions Agent/Team choices updated: {len(all_choices)} total")
         chat_update = gr.Dropdown.update(choices=all_choices, value=new_value)
         caption_update = gr.Dropdown.update(choices=all_choices, value=new_value)
         return chat_update, caption_update

    # Refreshes Team Editor Agent dropdown
    def refresh_available_agents_for_editor_wrapper(use_default, use_custom, file_agents_dict):
        print("Refreshing available agent list for team editor...")
        current_settings = load_settings()
        current_settings["using_default_agents"] = use_default
        current_settings["using_custom_agents"] = use_custom
        combined_roles = load_all_roles(current_settings, file_agents=file_agents_dict)
        file_agent_keys = list(file_agents_dict.keys()) if file_agents_dict else []
        display_choices = sorted([get_role_display_name(name, file_agent_keys) for name in combined_roles.keys()])
        print(f" Available agents for editor updated: {len(display_choices)} total")
        return gr.Dropdown.update(choices=display_choices), display_choices

    # Releases all models
    def release_all_models_ui_callback(current_settings_state):
         return release_all_models_logic(current_settings_state)


    # --- Wire Event Handlers ---

    # -- Chat Tab Wiring --
    chat_comps['model_with_vision'].change(
        fn=lambda x: x, inputs=[chat_comps['model_with_vision']],
        outputs=[chat_comps['selected_model_tracker']]
    )

    # Agent file upload handling
    agent_file_upload_outputs = [
        chat_comps['loaded_file_agents_state'],
        chat_comps['loaded_agent_file_display'],
    ]
    file_upload_event = chat_comps['agent_file_upload'].upload(
         fn=handle_agent_file_upload,
         inputs=[chat_comps['agent_file_upload']],
         outputs=agent_file_upload_outputs
    )
    refresh_trigger_inputs = [
        settings_comps['settings_use_default'], settings_comps['settings_use_custom'],
        chat_comps['loaded_file_agents_state'], teams_data_state
    ]
    refresh_editor_inputs = [
        settings_comps['settings_use_default'], settings_comps['settings_use_custom'],
        chat_comps['loaded_file_agents_state']
    ]
    file_upload_event.then(
        fn=refresh_agent_team_dropdowns_wrapper, inputs=refresh_trigger_inputs,
        outputs=[chat_comps['role_dropdown'], caption_comps['caption_agent_selector']] # Update Chat & Captions
    ).then(
         fn=refresh_available_agents_for_editor_wrapper, inputs=refresh_editor_inputs,
         outputs=[editor_comps['agent_to_add_dropdown'], all_available_agents_display_state]
    )

    # Checkbox changes trigger dropdown refreshes
    checkbox_refresh_trigger_inputs = [
        settings_comps['settings_use_default'], settings_comps['settings_use_custom'],
        chat_comps['loaded_file_agents_state'], teams_data_state
    ]
    checkbox_refresh_editor_inputs = [
        settings_comps['settings_use_default'], settings_comps['settings_use_custom'],
        chat_comps['loaded_file_agents_state']
    ]
    settings_comps['settings_use_default'].change(
        fn=refresh_agent_team_dropdowns_wrapper, inputs=checkbox_refresh_trigger_inputs,
        outputs=[chat_comps['role_dropdown'], caption_comps['caption_agent_selector']]
    ).then(
        fn=refresh_available_agents_for_editor_wrapper, inputs=checkbox_refresh_editor_inputs,
        outputs=[editor_comps['agent_to_add_dropdown'], all_available_agents_display_state]
    )
    settings_comps['settings_use_custom'].change(
        fn=refresh_agent_team_dropdowns_wrapper, inputs=checkbox_refresh_trigger_inputs,
        outputs=[chat_comps['role_dropdown'], caption_comps['caption_agent_selector']]
    ).then(
        fn=refresh_available_agents_for_editor_wrapper, inputs=checkbox_refresh_editor_inputs,
        outputs=[editor_comps['agent_to_add_dropdown'], all_available_agents_display_state]
    )

    # Update max tokens slider based on limiter choice
    chat_comps['limiter_handling_option'].change(
        fn=update_max_tokens_on_limiter_change,
        inputs=[chat_comps['limiter_handling_option'], chat_comps['max_tokens_slider']],
        outputs=[chat_comps['max_tokens_slider']]
    )

    # Submit Action
    submit_inputs = [
        chat_comps['folder_path'], chat_comps['user_input'], chat_comps['model_with_vision'],
        chat_comps['max_tokens_slider'], chat_comps['file_handling_option'],
        chat_comps['limiter_handling_option'], chat_comps['single_image_display'],
        chat_comps['use_ollama_api_options'], chat_comps['release_model_on_change'],
        chat_comps['role_dropdown'],
        settings_state, models_data_state, limiters_data_state, teams_data_state,
        chat_comps['selected_model_tracker'], chat_comps['loaded_file_agents_state'],
        history_list_state, session_history_state
    ]
    submit_outputs = [
        chat_comps['llm_response_display'],
        chat_comps['current_session_history_display'],
        chat_comps['model_state'],
        session_history_state
    ]
    chat_comps['submit_button'].click(fn=execute_chat_or_team, inputs=submit_inputs, outputs=submit_outputs)

    # Comment Action
    comment_inputs = [
        chat_comps['llm_response_display'], chat_comps['comment_input'],
        chat_comps['max_tokens_slider'], chat_comps['use_ollama_api_options'],
        chat_comps['model_state'],
        settings_state, chat_comps['loaded_file_agents_state'],
        history_list_state, session_history_state
    ]
    comment_outputs = [
        chat_comps['llm_response_display'],
        chat_comps['current_session_history_display'],
        session_history_state
    ]
    chat_comps['comment_button'].click(fn=comment_logic, inputs=comment_inputs, outputs=comment_outputs)

    # Clear Session History Action
    chat_comps['clear_session_button'].click(
        fn=clear_session_history_callback,
        inputs=[session_history_state],
        outputs=[chat_comps['current_session_history_display'], session_history_state]
    )

    # -- App Settings Tab Wiring --
    api_option_components_list = [settings_comps['ollama_options_ui_elements'][key] for key in initial_ollama_options_ordered_keys]
    settings_comps['load_profile_button'].click(
         fn=load_profile_options_callback,
         inputs=[settings_comps['profile_select'], profiles_data_state, api_option_keys_state],
         outputs=api_option_components_list
    )
    settings_save_inputs = [
         settings_comps['settings_ollama_url'], settings_comps['settings_max_tokens_slider_range'],
         settings_comps['settings_api_to_console'], settings_comps['settings_use_default'],
         settings_comps['settings_use_custom'], settings_comps['settings_use_ollama_opts_default'],
         settings_comps['settings_release_model_default'], settings_comps['settings_theme_select'],
         api_option_keys_state,
    ] + api_option_components_list
    settings_comps['settings_save_button'].click(
        fn=save_settings_callback, inputs=settings_save_inputs, outputs=[settings_comps['save_status_display']]
    )
    settings_comps['release_models_button'].click(
         fn=release_all_models_ui_callback, inputs=[settings_state], outputs=[settings_comps['release_status_display']]
    )

    # -- Agent Team Editor Tab Wiring --
    # (Wiring adjusted previously to update caption dropdown on save/delete)
    editor_comps['load_team_button'].click(
        fn=load_team_for_editing,
        inputs=[editor_comps['team_select_dropdown'], teams_data_state],
        outputs=[ editor_comps['team_name_textbox'], editor_comps['team_description_textbox'], editor_comps['steps_display_json'], editor_comps['assembly_strategy_radio'], editor_comps['current_team_editor_state'] ]
    )
    editor_comps['clear_editor_button'].click(
        fn=clear_team_editor, inputs=[],
        outputs=[ editor_comps['team_name_textbox'], editor_comps['team_description_textbox'], editor_comps['steps_display_json'], editor_comps['assembly_strategy_radio'], editor_comps['current_team_editor_state'] ]
    )
    editor_comps['add_step_button'].click(
        fn=add_step_to_editor,
        inputs=[editor_comps['agent_to_add_dropdown'], editor_comps['current_team_editor_state']],
        outputs=[editor_comps['current_team_editor_state'], editor_comps['steps_display_json']]
    )
    editor_comps['remove_step_button'].click(
        fn=remove_step_from_editor,
        inputs=[editor_comps['step_index_to_remove'], editor_comps['current_team_editor_state']],
        outputs=[editor_comps['current_team_editor_state'], editor_comps['steps_display_json']]
    )
    save_team_inputs = [ editor_comps['team_name_textbox'], editor_comps['team_description_textbox'], editor_comps['assembly_strategy_radio'], editor_comps['current_team_editor_state'], teams_data_state, settings_state, chat_comps['loaded_file_agents_state'] ]
    save_team_outputs = [ teams_data_state, editor_comps['team_select_dropdown'], chat_comps['role_dropdown'], caption_comps['caption_agent_selector'], editor_comps['save_status_textbox'] ]
    editor_comps['save_team_button'].click( fn=save_team_from_editor, inputs=save_team_inputs, outputs=save_team_outputs )
    delete_team_inputs = [ editor_comps['team_select_dropdown'], teams_data_state, settings_state, chat_comps['loaded_file_agents_state'] ]
    delete_outputs = [ teams_data_state, editor_comps['team_select_dropdown'], chat_comps['role_dropdown'], caption_comps['caption_agent_selector'], editor_comps['save_status_textbox'], editor_comps['team_name_textbox'], editor_comps['team_description_textbox'], editor_comps['steps_display_json'], editor_comps['assembly_strategy_radio'], editor_comps['current_team_editor_state'] ]
    editor_comps['delete_team_button'].click( fn=delete_team_logic, inputs=delete_team_inputs, outputs=delete_outputs )


    # -- History Tab Wiring --
    history_comps['clear_history_button'].click( fn=show_clear_confirmation, inputs=[], outputs=[history_comps['confirm_clear_group']] )
    history_comps['yes_clear_button'].click( fn=clear_full_history_callback, inputs=[history_list_state], outputs=[ history_comps['full_history_display'], history_comps['confirm_clear_group'], history_list_state ] )
    history_comps['no_clear_button'].click( fn=hide_clear_confirmation, inputs=[], outputs=[history_comps['confirm_clear_group']] )

    # -- Experiment Sweep Tab Wiring --
    sweep_comps['sweep_start_button'].click( fn=run_sweep, inputs=[ sweep_comps['sweep_prompts_input'], sweep_comps['sweep_teams_select'], sweep_comps['sweep_models_select'], sweep_comps['sweep_output_folder_input'], sweep_comps['sweep_log_intermediate_checkbox'], settings_state, teams_data_state, ], outputs=[sweep_comps['sweep_status_display']] )

    # -- Captions Tab Wiring --
    caption_comps['captions_load_button'].click(
        fn=load_images_and_captions,
        inputs=[caption_comps['captions_folder_path']],
        outputs=[
            caption_comps['captions_image_gallery'], # Target Gallery
            caption_image_paths_state,
            caption_data_state,
            caption_comps['captions_status_display'],
            caption_selected_item_state,
            caption_comps['captions_caption_display'],
            caption_comps['caption_selected_filename_display'],
            # No 8th output needed for Gallery initial value
        ]
    )

    # --- Handle Gallery Selection ---
    caption_comps['captions_image_gallery'].select(
        fn=update_caption_display_from_gallery, # Use the new callback
        # Input is the event data itself, plus states needed for lookup
        inputs=[caption_data_state, caption_image_paths_state],
        # Outputs: update caption text, update selected filename state, update filename display
        outputs=[
            caption_comps['captions_caption_display'],
            caption_selected_item_state,
            caption_comps['caption_selected_filename_display'],
        ]
        # The evt: gr.SelectData is implicitly passed as the first argument to the callback
    )

    # Save current caption button
    caption_comps['captions_save_button'].click(
        fn=save_caption,
        inputs=[
            caption_selected_item_state, # Use the state holding the selected filename
            caption_comps['captions_caption_display'],
            caption_image_paths_state,
            caption_data_state
        ],
        outputs=[ caption_comps['captions_status_display'], caption_data_state ]
    )

    # Batch Append/Prepend Buttons (Currently Disabled in UI - wiring kept for future)
    # Needs update if multi-select is implemented for Gallery
    caption_comps['captions_append_button'].click(
        fn=lambda *args: batch_edit_captions(*args, mode="Append"),
        # Inputs need redesign based on multi-select method
        inputs=[ caption_selected_item_state, # Placeholder - needs actual multi-selection
                 caption_comps['captions_batch_text'], caption_image_paths_state, caption_data_state ],
        outputs=[ caption_comps['captions_status_display'], caption_data_state ]
    )
    caption_comps['captions_prepend_button'].click(
        fn=lambda *args: batch_edit_captions(*args, mode="Prepend"),
        # Inputs need redesign
        inputs=[ caption_selected_item_state, # Placeholder
                 caption_comps['captions_batch_text'], caption_image_paths_state, caption_data_state ],
        outputs=[ caption_comps['captions_status_display'], caption_data_state ]
    )

    # Generate Caption for SELECTED image
    caption_comps['caption_generate_selected_button'].click(
        fn=generate_captions_for_selected,
        inputs=[
            caption_selected_item_state, # Pass the selected filename state
            caption_comps['caption_agent_selector'],
            caption_comps['caption_model_selector'],
            caption_comps['caption_generate_mode'],
            caption_image_paths_state,
            caption_data_state,
            settings_state,
            models_data_state,
            limiters_data_state,
            teams_data_state,
            chat_comps['loaded_file_agents_state'],
            history_list_state,
            session_history_state,
        ],
        outputs=[
            caption_comps['captions_status_display'],
            caption_data_state,
            caption_comps['captions_caption_display'], # Show generated caption
            session_history_state
        ]
    )

    # Generate for All
    caption_comps['caption_generate_all_button'].click(
        fn=generate_captions_for_all,
        inputs=[
            caption_image_paths_state,
            caption_data_state,
            caption_comps['caption_agent_selector'],
            caption_comps['caption_model_selector'],
            caption_comps['caption_generate_mode'],
            settings_state,
            models_data_state,
            limiters_data_state,
            teams_data_state,
            chat_comps['loaded_file_agents_state'],
            history_list_state,
            session_history_state,
        ],
        outputs=[
            caption_comps['captions_status_display'],
            caption_data_state,
            caption_comps['captions_caption_display'], # Show last generated caption
            session_history_state
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