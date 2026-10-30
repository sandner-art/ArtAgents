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
        conversation_state = gr.State(None) # Chat history of the last answer, used by Comment/Refine

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Source")
                with gr.Group():
                     folder_path = gr.Textbox(
                         label="Image Folder Path (Optional)"
                     )
                     single_image_display = gr.Image(
                          label="Single Image Input (Optional)",
                          type="numpy", # For compatibility with logic assuming numpy
                          height=256
                     )
                     file_handling_option = gr.Radio(
                          ["Overwrite", "Skip", "Append", "Prepend"],
                          label="Folder .txt File Handling", value="Skip"
                     )

            with gr.Column(scale=2):
                gr.Markdown("### Agent & Model Configuration")
                with gr.Group(): # Group main controls
                    role_dropdown = gr.Dropdown(
                        initial_roles_list, label="Select Agent",
                        value=initial_roles_list[0] if initial_roles_list else None,
                        elem_id="role_dropdown"
                    )
                    with gr.Row():
                        model_with_vision = gr.Dropdown(
                            initial_models_list, label="Select Model", elem_id="model_dropdown",
                            value=initial_models_list[0] if initial_models_list else None, scale=5
                        )
                        refresh_models_button = gr.Button("🔄", scale=0, min_width=50, elem_id="refresh_models")
                    models_status_display = gr.Markdown("", elem_id="models_status")
                    user_input = gr.Textbox(
                        label="User Input / Prompt Instructions", lines=3,
                        placeholder="Enter your main prompt or instructions here...", elem_id="user_input"
                    )

                with gr.Accordion("Advanced & Experimental Options", open=False):
                     with gr.Row():
                          limiter_handling_option = gr.Radio(
                               ["Off"] + initial_limiters_list, label="Prompt Style Limiter",
                               value="Off"
                          )
                          max_tokens_slider = gr.Slider(
                              minimum=50, maximum=initial_settings.get("max_tokens_slider", 4096), step=10,
                              value=initial_settings.get("max_tokens_slider", 1500) // 2, label="Max Tokens (Approx)"
                          )
                     with gr.Row():
                          use_ollama_api_options = gr.Checkbox(
                               label="Use Advanced Ollama API Options (from App Settings)",
                               value=initial_settings.get("use_ollama_api_options", False)
                          )
                          release_model_on_change = gr.Checkbox(
                               label="Unload Previous Model on Change",
                               value=initial_settings.get("release_model_on_change", False)
                          )
                          clean_prompt_artifacts = gr.Checkbox(
                              label="Clean Prompt Artifacts (e.g., '--- Output from...') [Experimental]",
                              value=False
                          )
                     with gr.Row():
                          agent_file_upload = gr.File(
                               label="Load Agents from .json File (Session Only)",
                               file_count="single",
                               file_types=[".json"],
                               type="filepath",
                               scale=2
                          )
                          loaded_agent_file_display = gr.Textbox(
                               label="Loaded File", interactive=False, scale=1
                          )
                     gr.Markdown(get_markdown("agent_file_format"))


        with gr.Row():
             submit_button = gr.Button("✨ Generate Response", variant="primary", scale=2)
             stop_button = gr.Button("⏹ Stop", variant="stop", scale=1, elem_id="chat_stop")
             comment_button = gr.Button("💬 Comment/Refine", scale=1)
             send_to_image_button = gr.Button("🎨 Send to Image Generation", scale=1)
             clear_session_button = gr.Button("🧹 Clear Session History", scale=1)

        with gr.Row():
             with gr.Column(scale=2):
                 llm_response_display = gr.Textbox(
                     label="LLM Response", lines=15, interactive=False,
                     buttons=["copy"], # Built-in copy-to-clipboard button (shown in the label bar)
                     elem_id="llm_response"
                 )
                 comment_input = gr.Textbox(
                      label="Enter Comment / Refinement", lines=2,
                      placeholder="Type your follow-up instruction here...", elem_id="comment_input"
                 )
             with gr.Column(scale=1):
                 gr.Markdown("### Session History")
                 current_session_history_display = gr.Textbox(
                      label="Current Session Log", lines=20, interactive=False,
                      buttons=["copy"], elem_id="session_history"
                 )


    return {
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
        "clean_prompt_artifacts": clean_prompt_artifacts,
        "agent_file_upload": agent_file_upload,
        "loaded_agent_file_display": loaded_agent_file_display,
        "submit_button": submit_button,
        "stop_button": stop_button,
        "send_to_image_button": send_to_image_button,
        "refresh_models_button": refresh_models_button,
        "models_status_display": models_status_display,
        "conversation_state": conversation_state,
        "comment_button": comment_button,
        "clear_session_button": clear_session_button,
        "llm_response_display": llm_response_display,
        "comment_input": comment_input,
        "current_session_history_display": current_session_history_display,
        "selected_model_tracker": selected_model_tracker,
        "model_state": model_state,
        "loaded_file_agents_state": loaded_file_agents_state,
    }