# ArtAgent/app.py

# Part 1: Imports
import gradio as gr
import json
import os
import atexit
# No direct need for numpy, requests, tempfile, shutil, time in this file anymore

# Import from project structure
# --- Updated Imports ---
from core.utils import load_json, get_theme_object, get_absolute_path
from core.ollama_checker import OllamaStatusChecker
from core import history_manager as history # Use alias for clarity
from core.sweep_manager import iter_sweep # Streaming sweep (Stop button can end it between runs)
# Import logic functions that will be used as callbacks
from core.app_logic import (
    iter_execute_chat_or_team, # Streaming router used for submit
    # comment_logic, # <-- REMOVE from app_logic import
    update_max_tokens_on_limiter_change,
    clear_session_history_callback,
    handle_agent_file_upload,
    load_profile_options_callback,
    save_settings_callback,
    show_clear_confirmation,
    hide_clear_confirmation,
    clear_full_history_callback,
    load_team_for_editing, clear_team_editor, add_step_to_editor,
    remove_step_from_editor, save_team_from_editor, delete_team_logic,
)
from core.refinement_logic import iter_comment_logic # Streaming, multi-turn refine
# --- <<< END UPDATED IMPORT >>> ---
# Import the logic function for releasing models
from core.ollama_manager import (
    release_all_models_logic, discover_models, model_choices, vision_model_choices, worker_model_choices,
)
from core.ollama_client import base_url_from_settings, VALID_OPTIONS as VALID_OLLAMA_OPTIONS
from core.image_gen_settings import save_connection_settings_callback
from core import image_gen_ui
# Import functions needed for initial UI setup from roles_config
from agents.roles_config import load_all_roles, get_role_display_name
# Import UI creation functions
from ui.chat_tab import create_chat_tab
from ui.app_settings_tab import create_app_settings_tab
from ui.team_editor_tab import create_team_editor_tab
# from ui.roles_tab import create_roles_tabs # <-- Remove this import
from ui.info_tab import create_info_tab # <-- Add this import
from ui.history_tab import create_history_tab
from ui.sweep_tab import create_sweep_tab
from ui.captions_tab import create_captions_tab
from ui.image_gen_tab import create_image_gen_tab
from ui.common_ui_elements import create_footer
from core.image_utils import pillow_security_warning
# Import captioning logic
from core.captioning_logic import (
    load_images_and_captions,
    # update_caption_display, # Replaced by update_caption_display_from_gallery
    update_caption_display_from_gallery, # NEW function for Gallery
    save_caption,
    batch_edit_captions,
    export_caption_dataset,
    generate_captions_for_selected, # Uses single selected item state
    iter_generate_captions_for_all, # Streaming batch (Stop button)
)
# --- End Updated Imports ---

# --- Constants ---
SETTINGS_FILE = 'settings.json'
MODELS_FILE = 'models.json'
LIMITERS_FILE = 'limiters.json'
PROFILES_FILE = 'ollama_profiles.json'
AGENT_TEAMS_FILE = 'agent_teams.json'
DEFAULT_ROLES_FILE = 'agents/agent_roles.json' # Needed for Info tab data
CUSTOM_ROLES_FILE = 'agents/custom_agent_roles.json' # Needed for Info tab data

# --- Utility Functions (App Specific or Loading) ---
def load_settings(settings_file=SETTINGS_FILE): return load_json(settings_file, is_relative=True)
def load_models(model_file=MODELS_FILE): models_list = load_json(model_file, is_relative=True); return models_list if isinstance(models_list, list) else []
def load_limiters(limiter_file=LIMITERS_FILE): limiter_dict = load_json(limiter_file, is_relative=True); return limiter_dict if isinstance(limiter_dict, dict) else {}
def load_profiles(profile_file=PROFILES_FILE): profile_dict = load_json(profile_file, is_relative=True); return profile_dict if isinstance(profile_dict, dict) else {}
def load_agent_teams(team_file=AGENT_TEAMS_FILE): team_dict = load_json(team_file, is_relative=True); return team_dict if isinstance(team_dict, dict) else {}
# --- Load Raw Role Data for Info Tab ---
default_roles_data_for_info = load_json(DEFAULT_ROLES_FILE, is_relative=True)
custom_roles_data_for_info = load_json(CUSTOM_ROLES_FILE, is_relative=True)
# --- End Role Data Loading ---


# --- Load Initial Data ---
settings = load_settings()
OLLAMA_BASE_URL = base_url_from_settings(settings)

# Installed Ollama models (with vision flags from /api/show); falls back to models.json
models_data, models_status_message = discover_models(settings, load_models())
print(f"[Models] {models_status_message}")
limiters_data = load_limiters()
profiles_data = load_profiles()
history_list = history.load_history() # Persistent history list
teams_data = load_agent_teams() # Load teams

# Generate initial lists for UI dropdowns
model_names_with_vision = model_choices(models_data)
vision_model_names = vision_model_choices(models_data)
limiters_names = list(limiters_data.keys())
profile_names = list(profiles_data.keys())
team_names = list(teams_data.keys())

# Load initial roles based on startup settings flags
initial_roles_data = load_all_roles(settings) # Pass settings to respect flags
initial_role_names = list(initial_roles_data.keys())
initial_role_display_names = sorted([get_role_display_name(name) for name in initial_role_names])
initial_agent_team_choices = ["(Direct Agent Call)"] + sorted([f"[Team] {name}" for name in team_names]) + initial_role_display_names
all_available_agents_initial = load_all_roles(settings, file_agents={}) # Load all non-file agents initially
all_available_agent_display_names_initial = sorted([get_role_display_name(name) for name in all_available_agents_initial.keys()])
all_initial_worker_model_choices = worker_model_choices(models_data)


# --- Dependency Security Check (Console Feedback Only) ---
_pillow_warning = pillow_security_warning()
if _pillow_warning: print(_pillow_warning)

# --- Perform Initial Startup Check (Console Feedback Only) ---
startup_checker = OllamaStatusChecker(OLLAMA_BASE_URL)
if not startup_checker.check(): print(startup_checker.get_console_message())
else: print(f"[Startup Check] Ollama appears responsive at base URL {startup_checker.base_url}. Application starting.")

# --- Build UI ---
selected_theme_name = settings.get("gradio_theme", "Default")
theme_object = get_theme_object(selected_theme_name)
print(f"Applying theme: {selected_theme_name}")

initial_ollama_options = settings.get("ollama_api_options", {})
# Only options current Ollama understands get UI controls (obsolete keys stay in settings.json, unused)
initial_ollama_options_ordered_keys = sorted(k for k in initial_ollama_options.keys() if k in VALID_OLLAMA_OPTIONS)

# --- Gradio Interface Definition ---
# delete_cache: every hour, remove cached uploads/gallery thumbnails older than a day
with gr.Blocks(title="ArtAgents", delete_cache=(3600, 86400)) as demo:

    # --- Define Main App States ---
    settings_state = gr.State(value=settings)
    models_data_state = gr.State(value=models_data)
    limiters_data_state = gr.State(value=limiters_data)
    profiles_data_state = gr.State(value=profiles_data)
    teams_data_state = gr.State(value=teams_data)
    history_list_state = gr.State(value=history_list)
    session_history_state = gr.State(value=[])
    api_option_keys_state = gr.State(value=initial_ollama_options_ordered_keys)
    all_available_agents_display_state = gr.State(value=all_available_agent_display_names_initial)
    caption_image_paths_state = gr.State({})
    caption_data_state = gr.State({})
    caption_selected_item_state = gr.State(None)


    # --- Create UI Tabs using functions from ui/ ---
    # --- Updated Tab Creation Order/Calls ---
    with gr.Tabs() as main_tabs:
        chat_comps = create_chat_tab(initial_agent_team_choices, model_names_with_vision, limiters_names, settings)
        caption_comps = create_captions_tab(initial_agent_team_choices, vision_model_names)
        image_comps = create_image_gen_tab(settings)
        editor_comps = create_team_editor_tab(initial_team_names=sorted(team_names), initial_available_agent_names=all_available_agent_display_names_initial)
        sweep_comps = create_sweep_tab(initial_team_names=sorted(team_names), initial_model_names=all_initial_worker_model_choices)
        history_comps = create_history_tab(history_list)
        info_comps = create_info_tab(default_roles_data_for_info, custom_roles_data_for_info)
        settings_comps = create_app_settings_tab(settings)

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
         chat_update = gr.update(choices=all_choices, value=new_value)
         caption_update = gr.update(choices=all_choices, value=new_value)
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
        return gr.update(choices=display_choices), display_choices

    # Releases all models
    def release_all_models_ui_callback(current_settings_state):
         return release_all_models_logic(current_settings_state)

    # Re-reads installed models from Ollama and updates every model dropdown
    def refresh_models_callback(current_settings_state, current_chat_model, current_caption_model):
        models, status = discover_models(current_settings_state, load_models())
        chat_choices, vision_choices = model_choices(models), vision_model_choices(models)
        keep = lambda value, choices: value if value in choices else (choices[0] if choices else None)
        return (
            models,
            gr.update(choices=chat_choices, value=keep(current_chat_model, chat_choices)),
            gr.update(choices=vision_choices, value=keep(current_caption_model, vision_choices)),
            gr.update(choices=worker_model_choices(models)),
            status, status,
        )


    # --- Wire Event Handlers ---

    # -- Chat Tab Wiring --
    refresh_models_inputs = [settings_state, chat_comps['model_with_vision'], caption_comps['caption_model_selector']]
    refresh_models_outputs = [
        models_data_state, chat_comps['model_with_vision'], caption_comps['caption_model_selector'],
        sweep_comps['sweep_models_select'], chat_comps['models_status_display'], settings_comps['settings_models_status'],
    ]
    chat_comps['refresh_models_button'].click(fn=refresh_models_callback, inputs=refresh_models_inputs, outputs=refresh_models_outputs)

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
    # Chain dropdown updates after file upload
    file_upload_event.then(
        fn=refresh_agent_team_dropdowns_wrapper, inputs=refresh_trigger_inputs,
        outputs=[chat_comps['role_dropdown'], caption_comps['caption_agent_selector']] # Update Chat & Captions
    ).then(
         fn=refresh_available_agents_for_editor_wrapper, inputs=refresh_editor_inputs,
         outputs=[editor_comps['agent_to_add_dropdown'], all_available_agents_display_state] # Update Editor Agent list
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

    # Submit Action - Includes clean_artifacts flag now
    submit_inputs = [
        chat_comps['folder_path'], chat_comps['user_input'], chat_comps['model_with_vision'],
        chat_comps['max_tokens_slider'], chat_comps['file_handling_option'],
        chat_comps['limiter_handling_option'], chat_comps['single_image_display'],
        chat_comps['use_ollama_api_options'], chat_comps['release_model_on_change'],
        chat_comps['role_dropdown'],
        chat_comps['clean_prompt_artifacts'], # Flag for cleaner
        # States
        settings_state, models_data_state, limiters_data_state, teams_data_state,
        chat_comps['model_state'], # Model used in the previous run (for "Unload Previous Model")
        chat_comps['loaded_file_agents_state'],
        history_list_state, session_history_state
    ]
    submit_outputs = [
        chat_comps['llm_response_display'],
        chat_comps['current_session_history_display'],
        chat_comps['model_state'],
        session_history_state,
        chat_comps['conversation_state'],
    ]
    # Streams tokens into the response box; Stop closes the stream (Ollama stops generating)
    submit_event = chat_comps['submit_button'].click(fn=iter_execute_chat_or_team, inputs=submit_inputs, outputs=submit_outputs)

    # Comment Action
    comment_inputs = [
        chat_comps['llm_response_display'],
        chat_comps['comment_input'],
        chat_comps['max_tokens_slider'],
        chat_comps['use_ollama_api_options'],
        # --- MODIFIED: Pass the currently selected model dropdown value ---
        chat_comps['model_with_vision'], # <-- Pass the dropdown value directly
        # --- End Modification ---
        settings_state,
        chat_comps['loaded_file_agents_state'],
        history_list_state,
        session_history_state,
        chat_comps['conversation_state'],
    ]
    comment_outputs = [
        chat_comps['llm_response_display'],
        chat_comps['current_session_history_display'],
        session_history_state,
        chat_comps['conversation_state'],
    ]
    comment_event = chat_comps['comment_button'].click(fn=iter_comment_logic, inputs=comment_inputs, outputs=comment_outputs)
    chat_comps['stop_button'].click(fn=None, cancels=[submit_event, comment_event])
    chat_comps['send_to_image_button'].click(
        fn=image_gen_ui.send_response_to_image_tab,
        inputs=[chat_comps['llm_response_display']],
        outputs=[image_comps['image_prompt'], main_tabs],
    )
    # --- <<< END Comment Wiring >>> ---

    # Clear Session History Action
    chat_comps['clear_session_button'].click(
        fn=clear_session_history_callback,
        inputs=[session_history_state],
        outputs=[chat_comps['current_session_history_display'], session_history_state]
    )

    # Copy: handled client-side by the response Textbox's built-in copy button (buttons=["copy"])


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
    ).then(
        # Connection + image-server settings; also reloads settings_state so changes apply without restart
        fn=save_connection_settings_callback,
        inputs=[
            settings_comps['save_status_display'], settings_comps['settings_ollama_url'], settings_comps['settings_think_mode'],
            settings_comps['settings_comfyui_url'], settings_comps['settings_sdwebui_url'],
            settings_comps['settings_ollama_images_url'], settings_comps['settings_image_output_dir'],
            settings_comps['settings_save_prompt_txt'],
        ],
        outputs=[settings_comps['save_status_display'], settings_state],
    )
    settings_comps['settings_refresh_models_button'].click(fn=refresh_models_callback, inputs=refresh_models_inputs, outputs=refresh_models_outputs)
    settings_comps['release_models_button'].click(
         fn=release_all_models_ui_callback, inputs=[settings_state], outputs=[settings_comps['release_status_display']]
    )


    # -- Agent Team Editor Tab Wiring --
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
    # Save/Delete wiring includes updating caption dropdown now
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
    sweep_event = sweep_comps['sweep_start_button'].click(
        fn=iter_sweep,
        inputs=[ sweep_comps['sweep_prompts_input'], sweep_comps['sweep_teams_select'], sweep_comps['sweep_models_select'], sweep_comps['sweep_output_folder_input'], sweep_comps['sweep_log_intermediate_checkbox'], settings_state, teams_data_state, ],
        outputs=[sweep_comps['sweep_status_display']]
    )
    sweep_comps['sweep_stop_button'].click(fn=None, cancels=[sweep_event])


    # -- Captions Tab Wiring --
    caption_comps['captions_load_button'].click(
        fn=load_images_and_captions,
        inputs=[caption_comps['captions_folder_path']],
        outputs=[ caption_comps['captions_image_gallery'], caption_image_paths_state, caption_data_state, caption_comps['captions_status_display'], caption_selected_item_state, caption_comps['captions_caption_display'], caption_comps['caption_selected_filename_display'], ]
    )
    caption_comps['captions_image_gallery'].select(
        fn=update_caption_display_from_gallery,
        inputs=[caption_data_state, caption_image_paths_state],
        outputs=[ caption_comps['captions_caption_display'], caption_selected_item_state, caption_comps['caption_selected_filename_display'], ]
    )
    caption_comps['captions_save_button'].click(
        fn=save_caption,
        inputs=[ caption_selected_item_state, caption_comps['captions_caption_display'], caption_image_paths_state, caption_data_state ],
        outputs=[ caption_comps['captions_status_display'], caption_data_state ]
    )
    # Batch Append/Prepend (all loaded images or the selected one)
    batch_inputs = [caption_comps['captions_batch_scope'], caption_selected_item_state, caption_comps['captions_batch_text'], caption_image_paths_state, caption_data_state]
    batch_outputs = [caption_comps['captions_status_display'], caption_data_state]
    caption_comps['captions_append_button'].click(
        fn=lambda scope, sel, text, paths, caps: batch_edit_captions(scope, sel, text, "Append", paths, caps),
        inputs=batch_inputs, outputs=batch_outputs
    ).then(
        fn=lambda selected, captions: (captions or {}).get(selected, "") if selected else "",
        inputs=[caption_selected_item_state, caption_data_state], outputs=[caption_comps['captions_caption_display']]
    )
    caption_comps['captions_prepend_button'].click(
        fn=lambda scope, sel, text, paths, caps: batch_edit_captions(scope, sel, text, "Prepend", paths, caps),
        inputs=batch_inputs, outputs=batch_outputs
    ).then(
        fn=lambda selected, captions: (captions or {}).get(selected, "") if selected else "",
        inputs=[caption_selected_item_state, caption_data_state], outputs=[caption_comps['captions_caption_display']]
    )
    caption_comps['captions_export_button'].click(
        fn=export_caption_dataset,
        inputs=[caption_image_paths_state, caption_data_state, caption_comps['captions_export_format']],
        outputs=[caption_comps['captions_status_display']]
    )
    caption_options = [caption_comps['caption_instruction'], caption_comps['caption_prefix'], caption_comps['caption_suffix']]
    # Generate Caption for Selected
    caption_comps['caption_generate_selected_button'].click(
        fn=generate_captions_for_selected,
        inputs=[ caption_selected_item_state, caption_comps['caption_agent_selector'], caption_comps['caption_model_selector'], caption_comps['caption_generate_mode'], caption_image_paths_state, caption_data_state, settings_state, models_data_state, limiters_data_state, teams_data_state, chat_comps['loaded_file_agents_state'], history_list_state, session_history_state, ] + caption_options,
        outputs=[ caption_comps['captions_status_display'], caption_data_state, caption_comps['captions_caption_display'], session_history_state ]
    )
    # Generate for All (streams progress; Stop ends the batch between images)
    caption_all_event = caption_comps['caption_generate_all_button'].click(
        fn=iter_generate_captions_for_all,
        inputs=[ caption_image_paths_state, caption_data_state, caption_comps['caption_agent_selector'], caption_comps['caption_model_selector'], caption_comps['caption_generate_mode'], settings_state, models_data_state, limiters_data_state, teams_data_state, chat_comps['loaded_file_agents_state'], history_list_state, session_history_state, ] + caption_options,
        outputs=[ caption_comps['captions_status_display'], caption_data_state, caption_comps['captions_caption_display'], session_history_state ]
    )
    # The batch shows the LAST image's caption; switch back to the selected image's caption,
    # so "Save THIS Caption" can't write it into the wrong .txt file (also after Stop).
    sync_selected_caption = lambda selected, captions: (captions or {}).get(selected, "") if selected else ""
    caption_all_event.then(
        fn=sync_selected_caption,
        inputs=[caption_selected_item_state, caption_data_state],
        outputs=[caption_comps['captions_caption_display']]
    )
    caption_comps['caption_stop_button'].click(fn=None, cancels=[caption_all_event]).then(
        fn=sync_selected_caption,
        inputs=[caption_selected_item_state, caption_data_state],
        outputs=[caption_comps['captions_caption_display']]
    )


    # -- Image Generation Tab Wiring --
    backend_option_outputs = [
        image_comps['image_model'], image_comps['image_workflow'], image_comps['image_sampler'],
        image_comps['image_negative_prompt'], image_comps['image_steps'], image_comps['image_cfg'],
        image_comps['image_status'],
    ]
    image_comps['image_backend'].change(fn=image_gen_ui.refresh_backend_options, inputs=[image_comps['image_backend'], settings_state], outputs=backend_option_outputs)
    image_comps['image_refresh_button'].click(fn=image_gen_ui.refresh_backend_options, inputs=[image_comps['image_backend'], settings_state], outputs=backend_option_outputs)
    image_comps['use_last_response_button'].click(
        fn=image_gen_ui.clean_prompt_text, inputs=[chat_comps['llm_response_display']], outputs=[image_comps['image_prompt']]
    )
    image_generate_event = image_comps['image_generate_button'].click(
        fn=image_gen_ui.iter_generate_images,
        inputs=[
            image_comps['image_prompt'], image_comps['image_negative_prompt'], image_comps['image_backend'],
            image_comps['image_model'], image_comps['image_workflow'], image_comps['image_sampler'],
            image_comps['image_width'], image_comps['image_height'], image_comps['image_steps'],
            image_comps['image_cfg'], image_comps['image_seed'], image_comps['image_batch'],
            settings_state, image_comps['gallery_items_state'],
        ],
        outputs=[image_comps['image_status'], image_comps['image_gallery'], image_comps['gallery_items_state']],
    )
    image_comps['image_stop_button'].click(fn=None, cancels=[image_generate_event])
    image_comps['image_clear_button'].click(
        fn=image_gen_ui.clear_gallery, inputs=[],
        outputs=[image_comps['image_gallery'], image_comps['gallery_items_state'], image_comps['image_status']]
    )
    # Fill the backend's model/workflow lists when the page opens
    demo.load(fn=image_gen_ui.refresh_backend_options, inputs=[image_comps['image_backend'], settings_state], outputs=backend_option_outputs)


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
    # Security notes:
    # - No `allowed_paths`: gallery images are served as cached thumbnails, so
    #   Gradio never needs direct access to your folders (the previous
    #   allowed_paths=["/"] let anyone who could reach the app read any file).
    # - Keep the default localhost binding. Only use share=True or
    #   server_name="0.0.0.0" together with `auth=("user", "password")`, because
    #   the app reads and writes files in folders typed into the UI.
    demo.launch(
        theme=theme_object, # Gradio 6: theme moved from gr.Blocks() to launch()
        max_file_size="25mb", # Agent .json files and single images are small
        # share=True # For public link (add auth!)
        # server_name="0.0.0.0" # For network access (add auth!)
        # debug=True # Enable Gradio debug mode if needed
    )
    print("Gradio App shutdown.")