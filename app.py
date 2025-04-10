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
import time # For status updates

# Import from refactored structure
from core.utils import load_json, format_json_to_html_table
from core.ollama_checker import OllamaStatusChecker
from core import history_manager as history # Use alias
from agents.roles_config import load_all_roles
from agents.ollama_agent import get_llm_response

# --- Constants ---
SETTINGS_FILE = 'settings.json'
MODELS_FILE = 'models.json'
LIMITERS_FILE = 'limiters.json'

# --- Utility Functions (specific to app.py if any, or rely on core.utils) ---
def load_settings(settings_file=SETTINGS_FILE):
    return load_json(settings_file, is_relative=True) # Use core util

def load_models(model_file=MODELS_FILE):
    return load_json(model_file, is_relative=True)

def load_limiters(limiter_file=LIMITERS_FILE):
    return load_json(limiter_file, is_relative=True)

# --- Load Settings ---
settings = load_settings()
OLLAMA_API_URL = settings.get("ollama_url", "http://localhost:11434/api/generate")

# --- Perform Initial Startup Check (Console Feedback Only) ---
startup_checker = OllamaStatusChecker(OLLAMA_API_URL) # Pass full URL
if not startup_checker.check():
    print(startup_checker.get_console_message())
else:
    print(f"[Startup Check] Ollama appears responsive at base URL {startup_checker.base_url}. Application starting.")

# --- Load initial data for UI ---
models_data = load_models()
model_names_with_vision = [f"{m['name']} (VISION)" if m.get('vision') else m['name'] for m in models_data]
limiters_data = load_limiters()
# Use history manager functions
history_list = history.load_history()
current_session_history = [] # Session history resets each run
# Load roles using the dedicated function and current settings
roles_data = load_all_roles(settings)
role_names = list(roles_data.keys())

# --- Temp Dir Setup ---
temp_dir = None
def cleanup_temp_dir():
    global temp_dir
    if temp_dir and os.path.exists(temp_dir):
        print(f"Cleaning up temp directory: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
        except OSError as e:
            print(f"Error removing temp directory {temp_dir}: {e}")
atexit.register(cleanup_temp_dir)

# --- Core Application Logic / Gradio Callbacks ---

def chat_logic(folder_path, role, user_input, model_with_vision, max_tokens_ui,
               file_handling_option, limiter_handling_option, single_image_np,
               current_settings, # From state
               use_ollama_api_options, release_model_on_change,
               selected_model_tracker_value, # Value from state (previous model)
               ):
    """Handles the core chat logic, calling the Ollama agent."""
    global history_list, current_session_history # Use state if implemented later

    # 1. Find Model Name and Info
    model_name = None
    model_info = None
    for m in models_data:
         m_name = m.get("name")
         if m_name == model_with_vision or f"{m_name} (VISION)" == model_with_vision:
              model_name = m_name
              model_info = m
              break

    if not model_name or not model_info:
         return "Error: Selected model not found or invalid.", "\n".join(current_session_history), None

    # 2. Handle Model Release (if applicable)
    if release_model_on_change and selected_model_tracker_value and selected_model_tracker_value != model_name:
        print(f"Requesting release of previous model: {selected_model_tracker_value}")
        # We need a release function (could be in ollama_agent.py or utils)
        release_model(selected_model_tracker_value, current_settings.get("ollama_url"))

    # 3. Prepare Prompt and Options
    limiter_settings = limiters_data.get(limiter_handling_option, {})
    limiter_prompt_format = limiter_settings.get("limiter_prompt_format", "")
    limiter_token_slider = limiter_settings.get("limiter_token_slider", max_tokens_ui)
    # Effective max tokens is the lower of the slider and the limiter
    effective_max_tokens = min(max_tokens_ui, limiter_token_slider) if limiter_handling_option != "Off" else max_tokens_ui

    role_description = roles_data.get(role, {}).get("description", "Unknown Role")
    # Construct base prompt
    prompt = f"Role: {role} - {role_description}\n{limiter_prompt_format}\nUser Input: {user_input}\n"

    agent_ollama_options = {} # Options specific to this agent call
    if use_ollama_api_options:
        # These will be merged with global/role options inside get_llm_response
        pass # No specific overrides here, relying on agent internal merge for now

    # 4. Handle Image Input
    pil_images_list = []
    processed_single_image = None # Track if single image was used
    is_single_image_mode = False

    if single_image_np is not None and isinstance(single_image_np, np.ndarray):
        if model_info.get("vision"):
            try:
                pil_image = Image.fromarray(single_image_np.astype('uint8'))
                pil_images_list = [pil_image]
                processed_single_image = pil_image # Store PIL image if needed later
                is_single_image_mode = True
                print("Processing single image input.")
            except Exception as e:
                 return f"Error processing single image: {e}", "\n".join(current_session_history), model_name
        else:
             print("Warning: Single image provided, but selected model does not support vision.")

    # 5. Determine Mode (Single Image, Folder, or Text-Only)
    final_response = "Processing..."
    if is_single_image_mode:
         print(f"DEBUG: About to call get_llm_response with model_name='{model_name}'")
        # Call agent with single image
        
         final_response = get_llm_response(
             role=role, prompt=prompt, model=model_name, settings=current_settings,
             roles_data=roles_data, images=pil_images_list, max_tokens=effective_max_tokens,
             ollama_api_options=agent_ollama_options
         )
         # Update history for single image call
         entry = f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\nRole: {role}\nModel: {model_name}\nInput: {user_input}\nImage: [Single Upload]\nResponse:\n{final_response}\n---\n"
         history_list = history.add_to_history(history_list, entry)
         current_session_history.append(entry)

    elif folder_path and os.path.isdir(folder_path):
        if not model_info.get("vision"):
            # Folder path given, but model has no vision - treat as text-only
            print("Warning: Folder path provided, but selected model does not support vision. Processing as text-only.")
            final_response = get_llm_response(
                role=role, prompt=prompt, model=model_name, settings=current_settings,
                roles_data=roles_data, images=None, max_tokens=effective_max_tokens,
                ollama_api_options=agent_ollama_options
            )
            entry = f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\nRole: {role}\nModel: {model_name}\nInput: {user_input}\nImage: [None - Text Only]\nResponse:\n{final_response}\n---\n"
            history_list = history.add_to_history(history_list, entry)
            current_session_history.append(entry)
        else:
            # Process folder with vision model (image by image)
            print(f"Processing image folder: {folder_path}")
            confirmation_messages = []
            processed_files = 0
            base_prompt = prompt # Keep base prompt separate

            for file_name in sorted(os.listdir(folder_path)): # Sort for consistent order
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    try:
                        print(f"  Processing file: {file_name}")
                        image = Image.open(file_path)
                        # Optional: Add filename context to prompt?
                        image_prompt = f"{base_prompt}\nImage Context: Analyzing '{file_name}'\n"
                        # Call agent for each image
                        img_response = get_llm_response(
                             role=role, prompt=image_prompt, model=model_name, settings=current_settings,
                             roles_data=roles_data, images=[image], max_tokens=effective_max_tokens,
                             ollama_api_options=agent_ollama_options
                        )

                        # --- File Handling ---
                        base_name = os.path.splitext(file_name)[0]
                        output_file = os.path.join(folder_path, f"{base_name}.txt")
                        action_taken = "Skipped"
                        if not os.path.exists(output_file) or file_handling_option == "Overwrite":
                             with open(output_file, 'w', encoding='utf-8') as f: f.write(img_response)
                             action_taken = "Written" if not os.path.exists(output_file) else "Overwritten"
                        elif file_handling_option == "Skip":
                             pass # Default action_taken is "Skipped"
                        elif file_handling_option == "Append":
                             with open(output_file, 'a', encoding='utf-8') as f: f.write("\n\n---\n\n" + img_response)
                             action_taken = "Appended"
                        elif file_handling_option == "Prepend":
                            try:
                                with open(output_file, 'r+', encoding='utf-8') as f:
                                    content = f.read()
                                    f.seek(0, 0)
                                    f.write(img_response + "\n\n---\n\n" + content)
                                action_taken = "Prepended"
                            except FileNotFoundError: # If file deleted between check and open
                                 with open(output_file, 'w', encoding='utf-8') as f: f.write(img_response)
                                 action_taken = "Written"
                        confirmation_messages.append(f"  - {file_name}: {action_taken} -> {base_name}.txt")

                        # --- History Update per Image ---
                        entry = f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\nRole: {role}\nModel: {model_name}\nInput: {user_input}\nImage: {file_name}\nResponse:\n{img_response}\n---\n"
                        history_list = history.add_to_history(history_list, entry)
                        current_session_history.append(entry)
                        processed_files += 1

                    except Exception as e:
                        error_msg = f"Error processing file {file_name}: {e}"
                        print(f"  {error_msg}")
                        confirmation_messages.append(f"  - {file_name}: Error - {e}")

            if processed_files == 0:
                final_response = "No valid image files found or processed in the directory."
            else:
                final_response = f"Folder processing complete ({processed_files} files):\n" + "\n".join(confirmation_messages)

    else:
        # Text-only scenario (no single image, no folder path)
        print("Processing text-only request.")
        final_response = get_llm_response(
             role=role, prompt=prompt, model=model_name, settings=current_settings,
             roles_data=roles_data, images=None, max_tokens=effective_max_tokens,
             ollama_api_options=agent_ollama_options
         )
         # Update history for text only call
        entry = f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\nRole: {role}\nModel: {model_name}\nInput: {user_input}\nImage: [None - Text Only]\nResponse:\n{final_response}\n---\n"
        history_list = history.add_to_history(history_list, entry)
        current_session_history.append(entry)

    # Return response text, history text, and model name used
    return final_response, "\n---\n".join(current_session_history), model_name


def comment_logic(llm_response_text, comment, model_state_value, current_settings, use_ollama_api_options, max_tokens_ui):
    """Handles the commenting logic."""
    global history_list, current_session_history # Use state if implemented later

    if not comment or not model_state_value:
        print("Comment ignored: No comment text or model state.")
        return llm_response_text, "\n---\n".join(current_session_history)

    print(f"Processing comment on model {model_state_value}...")
    role = "User" # Assuming comment comes from user perspective
    role_description = roles_data.get(role, {}).get("description", "Follow up based on comment.")
    # Construct prompt for comment
    prompt = f"Previous Response:\n{llm_response_text}\n\nUser Comment/Instruction:\n{comment}\n\nPlease revise or continue based on the comment."

    agent_ollama_options = {}
    if use_ollama_api_options:
        pass # Rely on internal merge for now

    # Call the agent
    response = get_llm_response(
        role=role, prompt=prompt, model=model_state_value, settings=current_settings,
        roles_data=roles_data, images=None, max_tokens=max_tokens_ui, # Use UI max tokens for comment
        ollama_api_options=agent_ollama_options
    )

    # Update history
    entry = f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\nRole: {role} (Comment)\nModel: {model_state_value}\nInput: {comment}\nContext: Previous response\nResponse:\n{response}\n---\n"
    history_list = history.add_to_history(history_list, entry)
    current_session_history.append(entry)

    return response, "\n---\n".join(current_session_history)

# --- Other Callbacks ---

def update_max_tokens_on_limiter_change(limiter_choice, current_max_tokens):
    """Updates the max token slider value based on limiter selection."""
    if limiter_choice == "Off":
        # Keep user value or default if limiter turned off
        # Maybe return gr.update() to leave it unchanged? Or a default?
        return current_max_tokens # Or return settings.get("max_tokens_slider", 1500) // 2 ?

    limiter_settings = limiters_data.get(limiter_choice, {})
    limiter_token_val = limiter_settings.get("limiter_token_slider")
    if limiter_token_val is not None:
         # Return the limiter's value, potentially clamping the slider
         print(f"Limiter '{limiter_choice}' suggests max_tokens: {limiter_token_val}")
         return limiter_token_val
    return current_max_tokens # No change if limiter doesn't define a value


def update_role_dropdown(use_default, use_custom):
    """Reloads roles and updates the dropdown choices."""
    # This requires reloading settings *within the callback* if they could have changed
    current_settings = load_settings()
    current_settings["using_default_agents"] = use_default # Update based on checkbox state
    current_settings["using_custom_agents"] = use_custom
    # Reload roles based on potentially changed settings
    global roles_data, role_names # Update global state used by UI
    roles_data = load_all_roles(current_settings)
    role_names = list(roles_data.keys())
    # Update the dropdown component
    return gr.update(choices=role_names, value=role_names[0] if role_names else None)


def clear_session_history_callback():
    """Clears the session history list."""
    global current_session_history
    current_session_history = []
    print("Session history cleared.")
    return "" # Return empty string to clear the display textbox

def save_settings_callback(ollama_url_in, max_tokens_slider_in, api_to_console_in,
                           use_default_in, use_custom_in, use_ollama_opts_in, release_model_in,
                           *ollama_api_options_values):
    """Saves all settings back to settings.json."""
    # Load current settings to preserve structure/unhandled keys
    current_settings = load_settings()

    # Update values from UI inputs
    current_settings["ollama_url"] = ollama_url_in
    current_settings["max_tokens_slider"] = max_tokens_slider_in
    current_settings["ollama_api_prompt_to_console"] = api_to_console_in
    current_settings["using_default_agents"] = use_default_in
    current_settings["using_custom_agents"] = use_custom_in
    current_settings["use_ollama_api_options"] = use_ollama_opts_in
    current_settings["release_model_on_change"] = release_model_in

    # Update ollama_api_options (assuming order matches the UI generation)
    if "ollama_api_options" not in current_settings:
         current_settings["ollama_api_options"] = {}

    option_keys = list(current_settings.get("ollama_api_options", {}).keys())
    # Make sure the number of keys matches the number of values received
    num_expected_options = len(option_keys)
    num_received_options = len(ollama_api_options_values)

    if num_expected_options == num_received_options:
        for key, value in zip(option_keys, ollama_api_options_values):
             # Attempt type conversion based on original type (or just save)
             original_type = type(current_settings["ollama_api_options"].get(key))
             try:
                 if original_type == bool:
                     # Checkbox gives bool directly
                     current_settings["ollama_api_options"][key] = value
                 elif original_type == int:
                     current_settings["ollama_api_options"][key] = int(value)
                 elif original_type == float:
                     current_settings["ollama_api_options"][key] = float(value)
                 else: # Assume string or other JSON serializable
                     current_settings["ollama_api_options"][key] = value
             except (ValueError, TypeError) as e:
                 print(f"Warning: Could not convert value '{value}' for option '{key}'. Saving as is. Error: {e}")
                 current_settings["ollama_api_options"][key] = value # Save raw value on error
    else:
         print(f"Warning: Mismatch in Ollama API options count. Expected {num_expected_options}, received {num_received_options}. Options not saved.")


    # Save the updated dictionary back to file
    try:
        settings_path = os.path.join(os.path.dirname(__file__), SETTINGS_FILE) # Ensure correct path
        with open(settings_path, 'w', encoding='utf-8') as file:
            json.dump(current_settings, file, indent=4)
        print(f"Settings saved successfully to {settings_path}")
        # Optionally reload settings globally? Depends if needed immediately
        global settings # Declare intent to modify global
        settings = current_settings
        return "Settings saved successfully."
    except Exception as e:
        print(f"Error saving settings to {settings_path}: {e}")
        return f"Error saving settings: {e}"

# --- Model Release Function ---
def release_model(model_name, ollama_api_url):
     """Sends request to Ollama to release (unload) a model."""
     # Uses the generate endpoint with keep_alive: 0
     payload = {"model": model_name, "keep_alive": 0}
     try:
         print(f"Sending release request for model: {model_name} to {ollama_api_url}")
         response = requests.post(ollama_api_url, json=payload, timeout=20) # Short timeout for release
         response.raise_for_status()
         msg = f"Model '{model_name}' release request sent successfully."
         print(msg)
         return msg
     except requests.exceptions.Timeout:
        msg = f"Error releasing model '{model_name}': Request timed out."
        print(msg)
        return msg
     except requests.exceptions.RequestException as e:
         msg = f"Error releasing model '{model_name}': {e}"
         print(msg)
         return msg

def release_all_models():
    """Iterates through models.json and attempts to release each model."""
    print("Attempting to release all models specified in models.json...")
    current_settings = load_settings() # Get current URL
    ollama_url = current_settings.get("ollama_url")
    if not ollama_url:
         msg = "Cannot release models: Ollama URL not found in settings."
         print(msg)
         return msg

    models_to_release = load_models()
    if not models_to_release:
        msg = "No models found in models.json to release."
        print(msg)
        return msg

    results = []
    for model_info in models_to_release:
        model_name = model_info.get("name")
        if model_name:
           results.append(release_model(model_name, ollama_url))

    summary = "\n".join(results)
    print(f"Model release process finished.\nSummary:\n{summary}")
    return summary

# --- Gradio UI Definition ---
with gr.Blocks(title="ArtAgents", theme=gr.themes.Soft()) as demo: # Added theme

    # --- Add Status Indicator (optional but nice) ---
    # status_text = "Application Loaded. Ollama status checked at startup." if startup_checker.is_available else f"Application Loaded. {startup_checker.status_message}"
    # status_indicator = gr.Markdown(status_text)

    with gr.Tab("Chat"):
        # Define states needed within this tab's callbacks
        selected_model_tracker = gr.State(None) # Tracks dropdown value before submit
        model_state = gr.State(None) # Stores model name used in last run
        settings_state = gr.State(value=settings) # Pass current settings to callbacks

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Source")
                folder_path = gr.Textbox(label="Image Folder Path (Optional)", info="Process all images in this folder.")
                single_image_display = gr.Image(label="Single Image Input (Optional)", type="numpy", sources=["upload", "clipboard"])
                file_handling_option = gr.Radio(["Overwrite", "Skip", "Append", "Prepend"], label="Folder File Handling", value="Skip", info="How to handle existing .txt files in the folder.")

            with gr.Column(scale=2):
                gr.Markdown("### Agent & Model Configuration")
                role = gr.Dropdown(role_names, label="Select Agent", value=role_names[0] if role_names else None)
                model_with_vision = gr.Dropdown(model_names_with_vision, label="Select Model")
                user_input = gr.Textbox(label="User Input / Prompt Instructions", lines=3, placeholder="Enter your main prompt or instructions here...")
                with gr.Accordion("Advanced Options", open=False):
                     limiter_handling_option = gr.Radio(["Off"] + list(limiters_data.keys()), label="Prompt Style Limiter", value="Off", info="Apply style constraints.")
                     max_tokens_slider = gr.Slider(
                         minimum=50, maximum=settings.get("max_tokens_slider", 4096), step=10, # Use setting for max
                         value=settings.get("max_tokens_slider", 1500) // 2, label="Max Tokens (Approx)",
                         info="Adjust max response length. May be overridden by limiter."
                     )
                     # Checkboxes moved inside Accordion
                     use_ollama_api_options = gr.Checkbox(label="Use Advanced Ollama API Options", value=settings.get("use_ollama_api_options", False), info="Apply settings from the 'App' tab.")
                     release_model_on_change = gr.Checkbox(label="Unload Previous Model on Change", value=settings.get("release_model_on_change", False), info="Free up VRAM by unloading the last used model when selecting a new one.")


        with gr.Row():
             submit_button = gr.Button("✨ Generate Response", variant="primary", scale=2)
             comment_button = gr.Button("💬 Comment/Refine", scale=1)
             clear_session_button = gr.Button("🧹 Clear Session History", scale=1)

        with gr.Row():
             with gr.Column(scale=2):
                 gr.Markdown("### LLM Response")
                 llm_response_display = gr.Textbox(label="Response Output", lines=15, interactive=False)
                 comment_input = gr.Textbox(label="Enter Comment / Refinement", lines=2, placeholder="Type your follow-up instruction here...")
             with gr.Column(scale=1):
                 gr.Markdown("### Session History")
                 current_session_history_display = gr.Textbox(label="Current Session Log", lines=20, interactive=False)


        # --- Event Handlers for Chat Tab ---
        # Track selected model for release logic
        model_with_vision.change(lambda x: x, inputs=[model_with_vision], outputs=[selected_model_tracker])

        # Update max tokens display based on limiter
        limiter_handling_option.change(
            fn=update_max_tokens_on_limiter_change,
            inputs=[limiter_handling_option, max_tokens_slider],
            outputs=[max_tokens_slider]
            # Note: This only changes the slider *value*. The actual max_tokens used
            # is calculated inside chat_logic based on both slider and limiter.
        )

        # Submit Action
        submit_button.click(
            fn=chat_logic,
            inputs=[folder_path, role, user_input, model_with_vision, max_tokens_slider,
                    file_handling_option, limiter_handling_option, single_image_display,
                    settings_state, # Pass settings state
                    use_ollama_api_options,
                    release_model_on_change, # Pass checkbox value directly
                    selected_model_tracker, # Pass the tracked state
                   ],
            # chat_logic returns: response_text, history_text, model_name_used
            outputs=[llm_response_display, current_session_history_display, model_state]
        )

        # Comment Action
        comment_button.click(
            fn=comment_logic,
            inputs=[llm_response_display, comment_input, model_state, # Use model from last run
                    settings_state, use_ollama_api_options, max_tokens_slider,
                   ],
            # comment_logic returns: response_text, history_text
            outputs=[llm_response_display, current_session_history_display]
        )

        # Clear Session History Action
        clear_session_button.click(
            fn=clear_session_history_callback,
            inputs=[],
            outputs=[current_session_history_display]
        )

    # --- App Settings Tab ---
    with gr.Tab("App Settings"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### General Settings")
                settings_ollama_url = gr.Textbox(label="Ollama URL (e.g., http://host:port/api/generate)", value=settings.get("ollama_url", ""))
                settings_max_tokens_slider = gr.Slider(label="Max Tokens Slider Range", minimum=512, maximum=16384, step=256, value=settings.get("max_tokens_slider", 4096))
                settings_api_to_console = gr.Checkbox(label="Log API Request Details to Console", value=settings.get("ollama_api_prompt_to_console", True))
                settings_use_default = gr.Checkbox(label="Load Default Agents (agents/agent_roles.json)", value=settings.get("using_default_agents", True))
                settings_use_custom = gr.Checkbox(label="Load Custom Agents (agents/custom_agent_roles.json)", value=settings.get("using_custom_agents", False))
                settings_use_ollama_opts = gr.Checkbox(label="Enable 'Use Advanced Ollama API Options' Checkbox by Default", value=settings.get("use_ollama_api_options", False))
                settings_release_model = gr.Checkbox(label="Enable 'Unload Previous Model' Checkbox by Default", value=settings.get("release_model_on_change", False))

                release_models_button = gr.Button("Release All Ollama Models Now", variant="stop", info="Attempts to unload models listed in models.json.")
                release_status_display = gr.Textbox(label="Release Status", interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("### Default Ollama API Options")
                gr.Markdown("Configure default parameters sent via the API 'options'. These can be overridden by agent roles or the 'Use Advanced...' checkbox.")
                # Dynamically create components for Ollama options
                ollama_options_components = []
                initial_ollama_options = settings.get("ollama_api_options", {})
                with gr.Group():
                    for key, value in initial_ollama_options.items():
                        comp_args = {"label": key, "value": value}
                        if isinstance(value, bool):
                            comp = gr.Checkbox(**comp_args)
                        elif key == "seed" and isinstance(value, int):
                             comp = gr.Number(**comp_args, precision=0)
                        elif isinstance(value, int):
                            # Define reasonable ranges based on parameter meaning
                            max_val = 32768 if "ctx" in key else (100 if key == "top_k" else (128 if "repeat" in key else 1000))
                            min_val = 0 if key != "repeat_last_n" else -1 # Allow -1 for repeat_last_n
                            step = 64 if "ctx" in key else 1
                            comp = gr.Slider(minimum=min_val, maximum=max_val, step=step, **comp_args)
                        elif isinstance(value, float):
                            comp = gr.Slider(minimum=0.0, maximum=2.0, step=0.05, **comp_args) # Wider range for temp/penalty
                        else:
                            # Fallback for strings or other types
                            comp = gr.Textbox(**comp_args)
                        ollama_options_components.append(comp)

        settings_save_button = gr.Button("Save All App Settings", variant="primary")
        save_status_display = gr.Textbox(label="Save Status", interactive=False)

        # --- Event Handlers for App Settings Tab ---
        settings_inputs = [
             settings_ollama_url, settings_max_tokens_slider, settings_api_to_console,
             settings_use_default, settings_use_custom, settings_use_ollama_opts, settings_release_model
        ] + ollama_options_components

        settings_save_button.click(
            fn=save_settings_callback,
            inputs=settings_inputs,
            outputs=[save_status_display]
        )

        release_models_button.click(
             fn=release_all_models,
             inputs=[],
             outputs=[release_status_display]
        )
         # Reload roles dropdown if agent usage flags change
        settings_use_default.change(fn=update_role_dropdown, inputs=[settings_use_default, settings_use_custom], outputs=[role])
        settings_use_custom.change(fn=update_role_dropdown, inputs=[settings_use_default, settings_use_custom], outputs=[role])


    # --- Agent Roles Tabs ---
    with gr.Tab("Default Agent Roles"):
        gr.Markdown("### `agents/agent_roles.json` (Read Only)")
        # Display default roles using HTML table utility
        default_roles_data = load_json('agents/agent_roles.json', is_relative=True)
        default_roles_display = gr.HTML(format_json_to_html_table(default_roles_data))

    with gr.Tab("Custom Agent Roles"):
        gr.Markdown("### `agents/custom_agent_roles.json` (Read Only)")
        # Display custom roles using HTML table utility
        custom_roles_data = load_json('agents/custom_agent_roles.json', is_relative=True)
        custom_roles_display = gr.HTML(format_json_to_html_table(custom_roles_data))
        # TODO: Add editing interface here later if desired

    # --- History Tab ---
    with gr.Tab("Full History"):
        gr.Markdown("### Interaction History (`core/history.json`)")
        full_history_display = gr.Textbox(label="Full History Log", lines=25, value="\n---\n".join(history_list), interactive=False)
        # Confirmation components for clearing history
        confirm_clear_group = gr.Group(visible=False)
        with confirm_clear_group:
             confirm_msg = gr.Markdown("❓ **Are you sure you want to permanently clear the entire history file?**")
             yes_clear_button = gr.Button("Yes, Clear History", variant="stop")
             no_clear_button = gr.Button("No, Cancel")
        clear_history_button = gr.Button("Clear Full History File...")

        # --- Event Handlers for History Tab ---
        def show_confirmation(): return gr.update(visible=True)
        def hide_confirmation(): return gr.update(visible=False)
        def clear_full_history():
            global history_list
            history_list = []
            history.save_history(history_list) # Save empty list
            print("Full history file cleared.")
            return gr.update(value=""), hide_confirmation() # Update display and hide buttons

        clear_history_button.click(fn=show_confirmation, inputs=[], outputs=[confirm_clear_group])
        yes_clear_button.click(fn=clear_full_history, inputs=[], outputs=[full_history_display, confirm_clear_group])
        no_clear_button.click(fn=hide_confirmation, inputs=[], outputs=[confirm_clear_group])

    # --- Captions Tab ---
    # TODO: Add captioning tab components and logic if needed, potentially moving from app_v1

    # Add footer/branding
    gr.Markdown("---")
    gr.Markdown("ArtAgents | sandner.art | [Creative AI/ML Research](https://github.com/sandner-art)")


# --- atexit handler ---
def on_exit():
    """Function called on script exit."""
    print("Application exiting...")
    cleanup_temp_dir() # Clean up temp image dir if used
    # Optional: Try to release models if enabled in settings?
    # current_settings = load_settings()
    # if current_settings.get("release_models_on_exit", False): # Example setting
    #     release_all_models()
    print("Cleanup finished.")

atexit.register(on_exit)

# --- Launch the Application ---
if __name__ == "__main__":
    print("Initializing ArtAgents...")
    demo.launch()
    print("Gradio App shutdown.")