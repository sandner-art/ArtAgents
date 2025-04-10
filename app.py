# ArtAgent/app.py

# Part 1: Imports
import gradio as gr
import json
import os
# Only import modules needed directly in this file
# from PIL import Image # Not needed directly
# import numpy as np # Not needed directly
# import requests # Not needed directly
# import tempfile # Not needed directly
# import shutil # Not needed directly
import atexit
# import time # Not needed directly

# Import from refactored structure
from core.utils import load_json, format_json_to_html_table, get_theme_object, AVAILABLE_THEMES, get_absolute_path
from core.ollama_checker import OllamaStatusChecker
from core import history_manager as history
# --- Import get_role_display_name here ---
from agents.roles_config import load_all_roles, get_role_display_name, get_actual_role_name
# from agents.ollama_agent import get_llm_response # Not called directly from app.py anymore

# Import logic functions that will be used as callbacks
from core.app_logic import (
    chat_logic, comment_logic, update_max_tokens_on_limiter_change,
    clear_session_history_callback, update_role_dropdown_callback,
    handle_agent_file_upload, load_profile_options_callback, save_settings_callback,
    show_clear_confirmation, hide_clear_confirmation, clear_full_history_callback
)
# Import the logic function for releasing models (not the callback itself)
from core.ollama_manager import release_all_models_logic

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
DEFAULT_ROLES_FILE = 'agents/agent_roles.json' # Used indirectly by load_all_roles
CUSTOM_ROLES_FILE = 'agents/custom_agent_roles.json' # Used indirectly by load_all_roles

# --- Utility Functions (App Specific or Loading) ---
# Use load_json from core.utils for consistency
def load_settings(settings_file=SETTINGS_FILE):
    return load_json(settings_file, is_relative=True)

def load_models(model_file=MODELS_FILE):
    models_list = load_json(model_file, is_relative=True)
    return models_list if isinstance(models_list, list) else [] # Ensure it's a list

def load_limiters(limiter_file=LIMITERS_FILE):
    limiter_dict = load_json(limiter_file, is_relative=True)
    return limiter_dict if isinstance(limiter_dict, dict) else {} # Ensure dict

def load_profiles(profile_file=PROFILES_FILE):
    profile_dict = load_json(profile_file, is_relative=True)
    return profile_dict if isinstance(profile_dict, dict) else {} # Ensure dict

# --- Load Initial Data ---
settings = load_settings()
OLLAMA_API_URL = settings.get("ollama_url", "http://localhost:11434/api/generate")

models_data = load_models()
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
# Use load_all_roles which handles settings flags internally
initial_roles_data = load_all_roles(settings)
initial_role_names = list(initial_roles_data.keys())
# Generate initial display names using the imported function
initial_role_display_names = sorted([get_role_display_name(name) for name in initial_role_names])

# Load initial history
history_list = history.load_history() # This is the persistent history list
# current_session_history list is managed via state within Gradio now

# --- Perform Initial Startup Check (Console Feedback Only) ---
startup_checker = OllamaStatusChecker(OLLAMA_API_URL) # Pass full URL
if not startup_checker.check():
    print(startup_checker.get_console_message())
else:
    print(f"[Startup Check] Ollama appears responsive at base URL {startup_checker.base_url}. Application starting.")

# --- Temp Dir Cleanup (if needed by any component later) ---
# Currently no global temp dir needed
def cleanup_temp_dir(): pass # Placeholder if needed later
# atexit.register(cleanup_temp_dir)

# --- Build UI ---
selected_theme_name = settings.get("gradio_theme", "Default")
theme_object = get_theme_object(selected_theme_name)
print(f"Applying theme: {selected_theme_name}")

# Store initial options keys order globally for callbacks that need it
initial_ollama_options = settings.get("ollama_api_options", {})
initial_ollama_options_ordered_keys = sorted(initial_ollama_options.keys())

with gr.Blocks(title="ArtAgents", theme=theme_object) as demo:

    # --- Define Main App States ---
    # These hold data needed across different callback executions
    settings_state = gr.State(value=settings)
    models_data_state = gr.State(value=models_data)
    limiters_data_state = gr.State(value=limiters_data)
    profiles_data_state = gr.State(value=profiles_data)
    history_list_state = gr.State(value=history_list) # Persistent history list
    session_history_state = gr.State(value=[]) # Session history (list of strings)
    # State to hold the ordered keys for API options (used by save/load profile)
    api_option_keys_state = gr.State(value=initial_ollama_options_ordered_keys)

    # --- Create UI Tabs using functions from ui/ ---
    # Pass initial data needed to build the UI elements
    # Note: initial_role_display_names is used for the initial dropdown population
    chat_comps = create_chat_tab(initial_role_display_names, model_names_with_vision, limiters_names, settings)
    settings_comps = create_app_settings_tab(settings) # Pass initial settings for default values
    # Pass loaded initial role data for display
    roles_comps = create_roles_tabs(
         load_json(DEFAULT_ROLES_FILE, is_relative=True),
         load_json(CUSTOM_ROLES_FILE, is_relative=True)
    )
    history_comps = create_history_tab(history_list) # Pass initial history list for display

    create_footer() # Add common footer

    # --- Wire Event Handlers ---
    # Connect UI component events (clicks, changes) to logic functions

    # -- Chat Tab Wiring --
    # Track selected model for release logic
    chat_comps['model_with_vision'].change(
        fn=lambda x: x, inputs=[chat_comps['model_with_vision']],
        outputs=[chat_comps['selected_model_tracker']]
    )

    # Agent file upload triggers state update -> triggers dropdown update via .then()
    agent_file_upload_outputs = [
        chat_comps['loaded_file_agents_state'],
        chat_comps['loaded_agent_file_display'],
        # No dropdown output here, handled by .then()
    ]
    file_upload_event = chat_comps['agent_file_upload'].upload(
         fn=handle_agent_file_upload,
         inputs=[chat_comps['agent_file_upload']],
         outputs=agent_file_upload_outputs
    ) # Assign the event

    # Chain the dropdown update after file upload finishes
    file_upload_event.then(
        fn=update_role_dropdown_callback,
        inputs=[
            settings_comps['settings_use_default'], # Need current checkbox state
            settings_comps['settings_use_custom'],
            chat_comps['loaded_file_agents_state'] # Use the state updated by the first fn
        ],
        outputs=[chat_comps['role_dropdown']] # Target only the dropdown
    )

    # Checkbox changes trigger dropdown update
    update_roles_trigger_inputs = [
        settings_comps['settings_use_default'],
        settings_comps['settings_use_custom'],
        chat_comps['loaded_file_agents_state'] # Also depends on file state
    ]
    settings_comps['settings_use_default'].change(
        fn=update_role_dropdown_callback, inputs=update_roles_trigger_inputs,
        outputs=[chat_comps['role_dropdown']]
    )
    settings_comps['settings_use_custom'].change(
        fn=update_role_dropdown_callback, inputs=update_roles_trigger_inputs,
        outputs=[chat_comps['role_dropdown']]
    )

    # Update max tokens slider based on limiter choice
    chat_comps['limiter_handling_option'].change(
        fn=update_max_tokens_on_limiter_change,
        inputs=[chat_comps['limiter_handling_option'], chat_comps['max_tokens_slider']],
        outputs=[chat_comps['max_tokens_slider']]
    )

    # Submit Action - Gather all necessary inputs including state
    submit_inputs = [
        # UI Component values
        chat_comps['folder_path'], chat_comps['role_dropdown'], chat_comps['user_input'],
        chat_comps['model_with_vision'], chat_comps['max_tokens_slider'],
        chat_comps['file_handling_option'], chat_comps['limiter_handling_option'],
        chat_comps['single_image_display'],
        chat_comps['use_ollama_api_options'], chat_comps['release_model_on_change'],
        # State values
        settings_state, models_data_state, limiters_data_state,
        chat_comps['selected_model_tracker'], chat_comps['loaded_file_agents_state'],
        history_list_state, session_history_state
    ]
    # chat_logic expected returns: final_response, new_session_history_list, model_name
    submit_outputs = [
        chat_comps['llm_response_display'],  # Output 1: Response text
        chat_comps['current_session_history_display'], # Output 2a: Session text display
        chat_comps['model_state'],          # Output 3: Model name used state
        session_history_state               # Output 2b: Update session list state
        # history_list_state is updated implicitly by save_history call within add_to_history
    ]
    chat_comps['submit_button'].click(
        fn=chat_logic, # Call the logic function
        inputs=submit_inputs,
        outputs=submit_outputs
        )

    # Comment Action
    comment_inputs = [
        # UI Component values
        chat_comps['llm_response_display'], chat_comps['comment_input'],
        chat_comps['max_tokens_slider'], chat_comps['use_ollama_api_options'],
        # State values
        chat_comps['model_state'], # Use model from last run state
        settings_state, chat_comps['loaded_file_agents_state'],
        history_list_state, session_history_state
    ]
    # comment_logic expected returns: new_response_text, new_session_history_list
    comment_outputs = [
        chat_comps['llm_response_display'], # Output 1: New response text
        chat_comps['current_session_history_display'], # Output 2a: Session text display
        session_history_state               # Output 2b: Update session list state
    ]
    chat_comps['comment_button'].click(
        fn=comment_logic, # Call the logic function
        inputs=comment_inputs,
        outputs=comment_outputs
        )

    # Clear Session History Action
    chat_comps['clear_session_button'].click(
        fn=clear_session_history_callback,
        inputs=[session_history_state], # Pass state if needed by logic
        outputs=[chat_comps['current_session_history_display'], session_history_state] # Clear display and state
    )

    # -- App Settings Tab Wiring --
    # Get list of UI components for API options in correct order for callbacks
    api_option_components_list = [settings_comps['ollama_options_ui_elements'][key] for key in initial_ollama_options_ordered_keys]

    # Load Profile Action
    settings_comps['load_profile_button'].click(
         fn=load_profile_options_callback,
         inputs=[
              settings_comps['profile_select'], # Selected profile name
              profiles_data_state,           # Full profile data from state
              api_option_keys_state         # Ordered keys list from state
         ],
         # outputs is the list of components themselves to be updated
         outputs=api_option_components_list
    )

    # Save Settings Action
    settings_save_inputs = [
         # General settings components
         settings_comps['settings_ollama_url'], settings_comps['settings_max_tokens_slider_range'],
         settings_comps['settings_api_to_console'], settings_comps['settings_use_default'],
         settings_comps['settings_use_custom'], settings_comps['settings_use_ollama_opts_default'],
         settings_comps['settings_release_model_default'], settings_comps['settings_theme_select'],
         # State inputs needed by save logic
         api_option_keys_state,
    ] + api_option_components_list # Add API option component values dynamically as *args

    settings_comps['settings_save_button'].click(
        fn=save_settings_callback,
        inputs=settings_save_inputs,
        outputs=[settings_comps['save_status_display']]
        # Note: This saves settings to file. The settings_state is NOT automatically
        # updated here unless save_settings_callback returns the new settings dict
        # and we add settings_state to the outputs list. For now, settings_state
        # only updates on app restart or if explicitly updated elsewhere.
    )

    # Release Models Action needs the current settings (for URL)
    settings_comps['release_models_button'].click(
         fn=lambda s: release_all_models_logic(s), # Wrap in lambda to pass state
         inputs=[settings_state],
         outputs=[settings_comps['release_status_display']]
    )

    # -- History Tab Wiring --
    history_comps['clear_history_button'].click(
         fn=show_clear_confirmation, inputs=[], outputs=[history_comps['confirm_clear_group']]
    )
    history_comps['yes_clear_button'].click(
         fn=clear_full_history_callback, inputs=[history_list_state], # Pass persistent history state
         outputs=[
              history_comps['full_history_display'], # Update display
              history_comps['confirm_clear_group'],  # Hide confirmation
              history_list_state                     # Update persistent history state
         ]
    )
    history_comps['no_clear_button'].click(
         fn=hide_clear_confirmation, inputs=[], outputs=[history_comps['confirm_clear_group']]
    )

# --- atexit handler ---
def on_exit():
    """Function called on script exit."""
    print("Application exiting...")
    # cleanup_temp_dir() # Add back if temp dir is used
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