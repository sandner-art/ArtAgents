# ArtAgent/ui/chat_tab.py
import gradio as gr
# Import help content functions/data
from core.help_content import get_tooltip, get_markdown

def create_chat_tab(initial_roles_list, initial_models_list, initial_limiters_list, initial_settings):
    """Creates the Gradio components for the Chat Tab."""

    with gr.Tab("Chat"):
        # --- Define States needed only within this tab's scope/callbacks ---
        # Passed back to app.py in the return dictionary
        selected_model_tracker = gr.State(None) # Tracks dropdown value before submit
        model_state = gr.State(None) # Stores model name used in last run
        loaded_file_agents_state = gr.State({}) # Store file agents temporarily

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Source")
                with gr.Group():
                     folder_path = gr.Textbox(
                         label="Image Folder Path (Optional)",
                         info=get_tooltip("folder_path") # Add tooltip
                     )
                     single_image_display = gr.Image(
                          label="Single Image Input (Optional)",
                          type="numpy", # For compatibility with logic assuming numpy
                          height=256,
                          # sources=["upload", "clipboard"], # Removed for Gradio 3.x
                          info=get_tooltip("single_image") # Add tooltip
                     )
                     file_handling_option = gr.Radio(
                          ["Overwrite", "Skip", "Append", "Prepend"],
                          label="Folder .txt File Handling", value="Skip", scale=2, min_width=80,
                          info=get_tooltip("file_handling") # Add tooltip
                     )

            with gr.Column(scale=2):
                gr.Markdown("### Agent & Model Configuration")
                with gr.Group(): # Group main controls
                    role_dropdown = gr.Dropdown(
                        initial_roles_list, label="Select Agent",
                        value=initial_roles_list[0] if initial_roles_list else None,
                        elem_id="role_dropdown",
                        info=get_tooltip("role_dropdown") # Add tooltip
                    )
                    model_with_vision = gr.Dropdown(
                        initial_models_list, label="Select Model", elem_id="model_dropdown",
                        info=get_tooltip("model_dropdown") # Add tooltip
                    )
                    user_input = gr.Textbox(
                        label="User Input / Prompt Instructions", lines=3,
                        placeholder="Enter your main prompt or instructions here...", elem_id="user_input",
                        info=get_tooltip("user_input") # Add tooltip
                    )

                with gr.Accordion("Advanced & Experimental Options", open=False):
                     with gr.Row():
                          limiter_handling_option = gr.Radio(
                               ["Off"] + initial_limiters_list, label="Prompt Style Limiter",
                               value="Off",
                               info=get_tooltip("limiter") # Add tooltip
                          )
                          max_tokens_slider = gr.Slider(
                              minimum=50, maximum=initial_settings.get("max_tokens_slider", 4096), step=10,
                              value=initial_settings.get("max_tokens_slider", 1500) // 2, label="Max Tokens (Approx)",
                              info=get_tooltip("max_tokens") # Add tooltip
                          )
                     with gr.Row():
                          use_ollama_api_options = gr.Checkbox(
                               label="Use Advanced Ollama API Options (from App Settings)",
                               value=initial_settings.get("use_ollama_api_options", False),
                               info=get_tooltip("use_advanced_options") # Add tooltip
                          )
                          release_model_on_change = gr.Checkbox(
                               label="Unload Previous Model on Change",
                               value=initial_settings.get("release_model_on_change", False),
                               info=get_tooltip("release_model") # Add tooltip
                          )
                          # --- Cleaner Checkbox ---
                          clean_prompt_artifacts = gr.Checkbox(
                              label="Clean Prompt Artifacts (e.g., '--- Output from...') [Experimental]",
                              value=False, # Default to False
                              info="If checked, attempts to remove agent output headers from the final response."
                          )
                          # --- End Cleaner Checkbox ---
                     with gr.Row():
                          agent_file_upload = gr.File(
                               label="Load Agents from .json File (Session Only)",
                               file_types=['.json'], scale=2,
                               info=get_tooltip("agent_file_upload") # Add tooltip
                          )
                          loaded_agent_file_display = gr.Textbox(
                               label="Loaded File", interactive=False, scale=1
                          )
                     gr.Markdown(get_markdown("agent_file_format"))


        with gr.Row():
             submit_button = gr.Button("✨ Generate Response", variant="primary", scale=2, info=get_tooltip("submit_button"))
             comment_button = gr.Button("💬 Comment/Refine", scale=1, info=get_tooltip("comment_button"))
             clear_session_button = gr.Button("🧹 Clear Session History", scale=1, info=get_tooltip("clear_session_button"))

        with gr.Row():
             with gr.Column(scale=2):
                 # --- Row for Response Label + Copy Button ---
                 with gr.Row():
                     gr.Markdown("### LLM Response", scale=10) # Give label more space
                     copy_response_button = gr.Button("📋 Copy", variant="tool", scale=1) # Use tool variant, small scale
                 # --- End Row ---
                 llm_response_display = gr.Textbox(
                     # Label removed as it's now in the Markdown header above
                     lines=15, interactive=False,
                     elem_id="llm_response" # Assign elem_id for JS access
                 )
                 comment_input = gr.Textbox(
                      label="Enter Comment / Refinement", lines=2,
                      placeholder="Type your follow-up instruction here...", elem_id="comment_input",
                      info=get_tooltip("comment_input") # Add tooltip
                 )
             with gr.Column(scale=1):
                 gr.Markdown("### Session History")
                 current_session_history_display = gr.Textbox(
                      label="Current Session Log", lines=20, interactive=False, elem_id="session_history"
                 )

        # --- Dummy HTML for JS Execution ---
        js_trigger_output = gr.HTML(visible=False, value="<!-- JS Trigger -->")
        # --- End Dummy HTML ---


    # Return dictionary of key components and states needed by app.py
    return {
        # UI Components needed for input/output binding
        "role_dropdown": role_dropdown,
        "model_with_vision": model_with_vision,
        "user_input": user_input,
        "folder_path": folder_path,
        "single_image_display": single_image_display,
        "file_handling_option": file_handling_option,
        "limiter_handling_option": limiter_handling_option,
        "max_tokens_slider": max_tokens_slider,
        "use_ollama_api_options": use_ollama_api_options,
        "release_model_on_change": release_model_on_change,
        "clean_prompt_artifacts": clean_prompt_artifacts, # <-- Added checkbox
        "agent_file_upload": agent_file_upload,
        "loaded_agent_file_display": loaded_agent_file_display,
        "submit_button": submit_button,
        "comment_button": comment_button,
        "clear_session_button": clear_session_button,
        "llm_response_display": llm_response_display,
        "copy_response_button": copy_response_button, # <-- Added button
        "comment_input": comment_input,
        "current_session_history_display": current_session_history_display,
        "js_trigger_output": js_trigger_output, # <-- Added dummy HTML output
        # States used by this tab's logic or passed between callbacks
        "selected_model_tracker": selected_model_tracker,
        "model_state": model_state,
        "loaded_file_agents_state": loaded_file_agents_state,
    }