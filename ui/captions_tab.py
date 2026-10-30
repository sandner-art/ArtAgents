# ArtAgent/ui/captions_tab.py
import gradio as gr
from core.help_content import get_tooltip

# Signature remains the same (accepts models list)
def create_captions_tab(initial_agent_team_choices, initial_vision_models):
    """Creates the Gradio components for the Captions Editor Tab."""

    with gr.Tab("Image Captions Editor"):
        gr.Markdown("## View, Edit, Save & Generate Image Captions")
        gr.Markdown(
            "Load a folder. Click an image in the gallery below to select it for viewing, editing, or AI caption generation." # Updated description
            )

        with gr.Row():
            captions_folder_path = gr.Textbox(
                label="Image Folder Path",
                placeholder="Enter the full path to the folder containing images and .txt captions",
                scale=3
            )
            captions_load_button = gr.Button("Load Folder", variant="secondary", scale=1)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Images")
                captions_image_gallery = gr.Gallery(
                    label="Loaded Images",
                    show_label=True,
                    object_fit="contain",
                    height=550, # Adjust height
                    columns=5, # Adjust columns
                    preview=True, # Allows clicking for selection event
                    elem_id="caption_gallery" # Add elem_id if needed
                )

            with gr.Column(scale=2):
                gr.Markdown("### Selected Image Caption") # Renamed section
                caption_selected_filename_display = gr.Textbox(
                    label="Selected Image Filename", interactive=False
                )
                captions_caption_display = gr.Textbox(
                    label="Caption / Text Content", lines=10,
                    interactive=True,
                    placeholder="Click an image in the gallery to view/edit its caption..."
                )
                captions_save_button = gr.Button(
                    "Save THIS Caption", variant="primary"
                )

                gr.Markdown("---")
                gr.Markdown("### AI Caption Generation (for Selected Image)") # Clarified scope
                with gr.Group():
                    caption_model_selector = gr.Dropdown(
                        choices=initial_vision_models,
                        label="Select Vision Model",
                        value=initial_vision_models[0] if initial_vision_models else None
                    )
                    caption_agent_selector = gr.Dropdown(
                        choices=initial_agent_team_choices,
                        label="Select Agent/Team",
                        value=initial_agent_team_choices[0] if initial_agent_team_choices else None
                    )
                    caption_generate_mode = gr.Radio(
                        ["Overwrite", "Skip", "Append", "Prepend"],
                        label="Generated Caption File Handling", value="Overwrite"
                    )
                    caption_generate_selected_button = gr.Button(
                         "Generate Caption for SELECTED Image",
                         variant="secondary"
                    )
                    caption_generate_all_button = gr.Button(
                         "Generate Captions for ALL Loaded Images",
                         variant="secondary"
                    )

                captions_status_display = gr.Textbox(label="Status", interactive=False, lines=3)

                with gr.Accordion("Manual Batch Append/Prepend (Requires Multi-Select - TBD)", open=False):
                    gr.Markdown("Manually add text to captions for **multiple selected images** (Multi-select UI to be determined).")
                    captions_batch_text = gr.Textbox(
                        label="Text to Add"
                    )
                    with gr.Row():
                        # Disabled until multi-select is implemented (setting .interactive after creation has no effect)
                        captions_prepend_button = gr.Button("Prepend Text", interactive=False)
                        captions_append_button = gr.Button("Append Text", interactive=False)


    return {
        "captions_folder_path": captions_folder_path,
        "captions_load_button": captions_load_button,
        "captions_image_gallery": captions_image_gallery,
        "caption_selected_filename_display": caption_selected_filename_display,
        "captions_caption_display": captions_caption_display,
        "captions_save_button": captions_save_button,
        "captions_batch_text": captions_batch_text,
        "captions_prepend_button": captions_prepend_button,
        "captions_append_button": captions_append_button,
        "captions_status_display": captions_status_display,
        "caption_model_selector": caption_model_selector,
        "caption_agent_selector": caption_agent_selector,
        "caption_generate_mode": caption_generate_mode,
        "caption_generate_selected_button": caption_generate_selected_button,
        "caption_generate_all_button": caption_generate_all_button,
        "caption_image_paths_state_key": "caption_image_paths_state",
        "caption_data_state_key": "caption_data_state",
        "caption_selected_item_state_key": "caption_selected_item_state",
    }