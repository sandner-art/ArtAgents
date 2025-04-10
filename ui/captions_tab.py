# ArtAgent/ui/captions_tab.py
import gradio as gr
from core.help_content import get_tooltip # Assuming help content will be added

# Modified function signature to accept initial agent/team choices
def create_captions_tab(initial_agent_team_choices):
    """Creates the Gradio components for the Captions Editor Tab."""

    with gr.Tab("Image Captions Editor"):
        gr.Markdown("## View, Edit, Save & Generate Image Captions") # Updated title
        gr.Markdown(
            "Load a folder of images. Select image(s) to view/edit captions, or use an AI Agent/Team to generate new captions." # Updated description
            )

        # State specific to this tab's operation (defined in app.py, referenced here via components)
        # caption_image_paths_state = gr.State({}) # filename: absolute_path
        # caption_data_state = gr.State({})        # filename: caption_text
        # caption_selected_item_state = gr.State(None) # filename of selected image

        with gr.Row():
            captions_folder_path = gr.Textbox(
                label="Image Folder Path",
                placeholder="Enter the full path to the folder containing images and .txt captions",
                info=get_tooltip("captions_folder_path"),
                scale=3
            )
            captions_load_button = gr.Button("Load Folder", variant="secondary", scale=1)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Images")
                captions_image_selector = gr.CheckboxGroup(
                    label="Select Image File(s)",
                    info=get_tooltip("captions_image_selector"),
                    # Choices will be populated by the load callback
                )
                caption_image_preview = gr.Image(
                    label="Selected Image Preview",
                    type="filepath",
                    interactive=False,
                    height=350,
                    show_download_button=False
                 )

            with gr.Column(scale=2):
                gr.Markdown("### Manual Caption Editing")
                caption_selected_filename_display = gr.Textbox(
                    label="Selected File", interactive=False,
                    info="Filename of the image currently being previewed/edited."
                )
                captions_caption_display = gr.Textbox(
                    label="Caption / Text Content", lines=10, # Reduced lines slightly
                    interactive=True,
                    placeholder="Select an image to view/edit its caption...",
                    info=get_tooltip("captions_caption_display")
                )
                captions_save_button = gr.Button(
                    "Save THIS Caption", variant="primary",
                    info=get_tooltip("captions_save_button")
                )

                # --- NEW: Agent Caption Generation Section ---
                gr.Markdown("---")
                gr.Markdown("### AI Caption Generation")
                with gr.Group():
                    caption_agent_selector = gr.Dropdown(
                        choices=initial_agent_team_choices, # Use passed-in choices
                        label="Select Agent/Team for Captioning",
                        value=initial_agent_team_choices[0] if initial_agent_team_choices else None, # Default to first option
                        # Add tooltip key (define in help_content.py)
                        info=get_tooltip("caption_agent_selector")
                    )
                    # File handling specific to GENERATED captions
                    caption_generate_mode = gr.Radio(
                        ["Overwrite", "Skip", "Append", "Prepend"],
                        label="Generated Caption File Handling", value="Overwrite",
                        # Add tooltip key (define in help_content.py)
                        info=get_tooltip("caption_generate_mode")
                    )
                    with gr.Row():
                         caption_generate_selected_button = gr.Button(
                              "Generate Caption(s) for SELECTED Image(s)",
                              variant="secondary",
                              # Add tooltip key (define in help_content.py)
                              info=get_tooltip("caption_generate_selected_button")
                         )
                         caption_generate_all_button = gr.Button(
                              "Generate Captions for ALL Loaded Images",
                              variant="secondary",
                              # Add tooltip key (define in help_content.py)
                              info=get_tooltip("caption_generate_all_button")
                         )
                # --- End NEW Section ---

                # Status Display (used by manual and generated actions)
                captions_status_display = gr.Textbox(label="Status", interactive=False, lines=3) # Increased lines

                # Batch operations section (Manual Append/Prepend)
                with gr.Accordion("Manual Batch Append/Prepend (for Selected Images)", open=False):
                    gr.Markdown("Manually add text to the beginning or end of captions for **all currently checked** images.")
                    captions_batch_text = gr.Textbox(
                        label="Text to Add",
                        info=get_tooltip("captions_batch_text")
                    )
                    with gr.Row():
                        captions_prepend_button = gr.Button("Prepend Text", info=get_tooltip("captions_prepend_button"))
                        captions_append_button = gr.Button("Append Text", info=get_tooltip("captions_append_button"))

    # Return dictionary of components and states defined within this function scope
    return {
        # Existing Components
        "captions_folder_path": captions_folder_path,
        "captions_load_button": captions_load_button,
        "captions_image_selector": captions_image_selector,
        "caption_image_preview": caption_image_preview,
        "caption_selected_filename_display": caption_selected_filename_display,
        "captions_caption_display": captions_caption_display,
        "captions_save_button": captions_save_button,
        "captions_batch_text": captions_batch_text,
        "captions_prepend_button": captions_prepend_button,
        "captions_append_button": captions_append_button,
        "captions_status_display": captions_status_display,

        # NEW Components for Agent Captioning
        "caption_agent_selector": caption_agent_selector,
        "caption_generate_mode": caption_generate_mode,
        "caption_generate_selected_button": caption_generate_selected_button,
        "caption_generate_all_button": caption_generate_all_button,

        # State keys (defined in app.py) referenced by this tab's logic
        "caption_image_paths_state_key": "caption_image_paths_state",
        "caption_data_state_key": "caption_data_state",
        "caption_selected_item_state_key": "caption_selected_item_state",
    }