# ArtAgent/ui/captions_tab.py
import gradio as gr
from core.help_content import get_tooltip
from core.captioning_logic import DEFAULT_CAPTION_INSTRUCTION, BATCH_SCOPE_ALL, BATCH_SCOPE_SELECTED

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
                    caption_instruction = gr.Textbox(
                        label="Caption Instruction (sent with each image)",
                        value=DEFAULT_CAPTION_INSTRUCTION, lines=2
                    )
                    with gr.Row():
                        caption_prefix = gr.Textbox(
                            label="Prefix (e.g. trigger word)", placeholder="e.g. 'ohwx style, '", scale=1
                        )
                        caption_suffix = gr.Textbox(
                            label="Suffix", placeholder="e.g. ', film grain'", scale=1
                        )
                    caption_generate_mode = gr.Radio(
                        ["Overwrite", "Skip", "Append", "Prepend"],
                        label="Generated Caption File Handling", value="Overwrite"
                    )
                    caption_generate_selected_button = gr.Button(
                         "Generate Caption for SELECTED Image",
                         variant="secondary"
                    )
                    with gr.Row():
                        caption_generate_all_button = gr.Button(
                             "Generate Captions for ALL Loaded Images",
                             variant="secondary", scale=3
                        )
                        caption_stop_button = gr.Button("⏹ Stop", variant="stop", scale=1, elem_id="caption_stop")

                captions_status_display = gr.Textbox(label="Status", interactive=False, lines=3)

                with gr.Accordion("Batch Edit & Dataset Export", open=False):
                    gr.Markdown("Add text (e.g. a LoRA trigger word or style tag) to existing captions. "
                                "Text is added exactly as typed, so include your own separator.")
                    captions_batch_scope = gr.Radio(
                        [BATCH_SCOPE_ALL, BATCH_SCOPE_SELECTED], value=BATCH_SCOPE_ALL, label="Apply to"
                    )
                    captions_batch_text = gr.Textbox(label="Text to Add")
                    with gr.Row():
                        captions_prepend_button = gr.Button("Prepend Text")
                        captions_append_button = gr.Button("Append Text")
                    gr.Markdown("---")
                    with gr.Row():
                        captions_export_format = gr.Dropdown(
                            ["metadata.jsonl", "metadata.csv"], value="metadata.jsonl",
                            label="Export captions for training (written into the image folder)", scale=3
                        )
                        captions_export_button = gr.Button("Export", scale=1)


    return {
        "captions_folder_path": captions_folder_path,
        "captions_load_button": captions_load_button,
        "captions_image_gallery": captions_image_gallery,
        "caption_selected_filename_display": caption_selected_filename_display,
        "captions_caption_display": captions_caption_display,
        "captions_save_button": captions_save_button,
        "captions_batch_text": captions_batch_text,
        "captions_batch_scope": captions_batch_scope,
        "captions_export_format": captions_export_format,
        "captions_export_button": captions_export_button,
        "caption_instruction": caption_instruction,
        "caption_prefix": caption_prefix,
        "caption_suffix": caption_suffix,
        "caption_stop_button": caption_stop_button,
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