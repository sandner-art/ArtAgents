# ArtAgent/ui/chat_tab.py
import gradio as gr

def create_chat_tab(initial_roles_list, initial_models_list, initial_limiters_list, initial_settings):
    """Creates the Gradio components for the Chat Tab."""

    with gr.Tab("Chat"):
        # Define states needed within this tab's callbacks or for carrying values
        selected_model_tracker = gr.State(None) # Tracks dropdown value before submit
        model_state = gr.State(None) # Stores model name used in last run
        # Store loaded file agents temporarily for this session
        loaded_file_agents_state = gr.State({})

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Source")
                with gr.Group():
                     folder_path = gr.Textbox(label="Image Folder Path (Optional)", info="Process all images in this folder.")
                     single_image_display = gr.Image(
                          label="Single Image Input (Optional)",
                          type="numpy", # Keep numpy if needed by chat_logic
                          sources=["upload", "clipboard"],
                          height=256 # Limit initial height
                     )
                     file_handling_option = gr.Radio(
                          ["Overwrite", "Skip", "Append", "Prepend"],
                          label="Folder .txt File Handling", value="Skip", scale=2, min_width=80
                     )


            with gr.Column(scale=2):
                gr.Markdown("### Agent & Model Configuration")
                with gr.Group(): # Group main controls
                    # Dropdowns populated with initial data
                    role_dropdown = gr.Dropdown(
                        initial_roles_list, label="Select Agent",
                        value=initial_roles_list[0] if initial_roles_list else None,
                        elem_id="role_dropdown" # Add elem_id for potential JS/CSS later
                    )
                    model_with_vision = gr.Dropdown(
                        initial_models_list, label="Select Model", elem_id="model_dropdown"
                    )
                    user_input = gr.Textbox(
                        label="User Input / Prompt Instructions", lines=3,
                        placeholder="Enter your main prompt or instructions here...", elem_id="user_input"
                    )

                with gr.Accordion("Advanced & Experimental Options", open=False):
                     with gr.Row():
                          limiter_handling_option = gr.Radio(
                               ["Off"] + initial_limiters_list, label="Prompt Style Limiter",
                               value="Off", info="Apply style constraints."
                          )
                          max_tokens_slider = gr.Slider(
                              minimum=50, maximum=initial_settings.get("max_tokens_slider", 4096), step=10,
                              value=initial_settings.get("max_tokens_slider", 1500) // 2, label="Max Tokens (Approx)",
                              info="Adjust max response length. May be overridden by limiter."
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
                     with gr.Row():
                          agent_file_upload = gr.File(
                               label="Load Agents from .json File (Session Only)",
                               file_types=['.json'], scale=2
                          )
                          loaded_agent_file_display = gr.Textbox(
                               label="Loaded File", interactive=False, scale=1
                          )


        with gr.Row():
             submit_button = gr.Button("✨ Generate Response", variant="primary", scale=2)
             comment_button = gr.Button("💬 Comment/Refine", scale=1)
             clear_session_button = gr.Button("🧹 Clear Session History", scale=1)

        with gr.Row():
             with gr.Column(scale=2):
                 gr.Markdown("### LLM Response")
                 llm_response_display = gr.Textbox(label="Response Output", lines=15, interactive=False, elem_id="llm_response")
                 comment_input = gr.Textbox(
                      label="Enter Comment / Refinement", lines=2,
                      placeholder="Type your follow-up instruction here...", elem_id="comment_input"
                 )
             with gr.Column(scale=1):
                 gr.Markdown("### Session History")
                 current_session_history_display = gr.Textbox(
                      label="Current Session Log", lines=20, interactive=False, elem_id="session_history"
                 )

    # Return dictionary of key components needed by app.py for wiring events
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
        "agent_file_upload": agent_file_upload,
        "loaded_agent_file_display": loaded_agent_file_display,
        "submit_button": submit_button,
        "comment_button": comment_button,
        "clear_session_button": clear_session_button,
        "llm_response_display": llm_response_display,
        "comment_input": comment_input,
        "current_session_history_display": current_session_history_display,
        # States used by this tab's logic
        "selected_model_tracker": selected_model_tracker,
        "model_state": model_state,
        "loaded_file_agents_state": loaded_file_agents_state,
    }