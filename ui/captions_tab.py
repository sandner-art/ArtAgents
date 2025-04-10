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
                info=get_tooltip("captions_folder_path"),
                scale=3
            )
            captions_load_button = gr.Button("Load Folder", variant="secondary", scale=1)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Images")
                # --- Replace CheckboxGroup with Gallery ---
                captions_image_gallery = gr.Gallery(
                    label="Loaded Images",
                    show_label=True,
                    object_fit="contain",
                    height=550, # Adjust height
                    columns=5, # Adjust columns
                    preview=True, # Allows clicking for selection event
                    # Tooltip isn't directly supported, add info elsewhere if needed
                    elem_id="caption_gallery" # Add elem_id if needed
                )
                # --- End Gallery ---
                # Remove the separate Image Preview, Gallery handles preview
                # caption_image_preview = gr.Image(...)

            with gr.Column(scale=2):
                gr.Markdown("### Selected Image Caption") # Renamed section
                caption_selected_filename_display = gr.Textbox(
                    label="Selected Image Filename", interactive=False, # Renamed label
                    info="Filename of the image currently selected in the gallery."
                )
                captions_caption_display = gr.Textbox(
                    label="Caption / Text Content", lines=10,
                    interactive=True,
                    placeholder="Click an image in the gallery to view/edit its caption...",
                    info=get_tooltip("captions_caption_display")
                )
                captions_save_button = gr.Button(
                    "Save THIS Caption", variant="primary",
                    info=get_tooltip("captions_save_button")
                )

                gr.Markdown("---")
                gr.Markdown("### AI Caption Generation (for Selected Image)") # Clarified scope
                with gr.Group():
                    caption_model_selector = gr.Dropdown(
                        choices=initial_vision_models,
                        label="Select Vision Model", # Simplified label
                        value=initial_vision_models[0] if initial_vision_models else None,
                        info=get_tooltip("caption_model_selector")
                    )
                    caption_agent_selector = gr.Dropdown(
                        choices=initial_agent_team_choices,
                        label="Select Agent/Team", # Simplified label
                        value=initial_agent_team_choices[0] if initial_agent_team_choices else None,
                        info=get_tooltip("caption_agent_selector")
                    )
                    caption_generate_mode = gr.Radio(
                        ["Overwrite", "Skip", "Append", "Prepend"],
                        label="Generated Caption File Handling", value="Overwrite",
                        info=get_tooltip("caption_generate_mode")
                    )
                    # --- Modify Buttons ---
                    caption_generate_selected_button = gr.Button(
                         "Generate Caption for SELECTED Image", # Renamed Button
                         variant="secondary",
                         info=get_tooltip("caption_generate_selected_button")
                    )
                    # Keep batch generate, but maybe disable initially until multi-select is refined?
                    caption_generate_all_button = gr.Button(
                         "Generate Captions for ALL Loaded Images",
                         variant="secondary",
                         # interactive=False, # Optionally disable initially
                         info=get_tooltip("caption_generate_all_button")
                    )
                    # --- End Button Modify ---

                captions_status_display = gr.Textbox(label="Status", interactive=False, lines=3)

                with gr.Accordion("Manual Batch Append/Prepend (Requires Multi-Select - TBD)", open=False):
                    gr.Markdown("Manually add text to captions for **multiple selected images** (Multi-select UI to be determined).")
                    captions_batch_text = gr.Textbox(
                        label="Text to Add",
                        info=get_tooltip("captions_batch_text")
                    )
                    with gr.Row():
                        captions_prepend_button = gr.Button("Prepend Text", info=get_tooltip("captions_prepend_button"))
                        captions_append_button = gr.Button("Append Text", info=get_tooltip("captions_append_button"))
                        # Disable batch buttons until multi-select is clear
                        captions_prepend_button.interactive = False
                        captions_append_button.interactive = False


    return {
        # Existing Components (some removed/renamed)
        "captions_folder_path": captions_folder_path,
        "captions_load_button": captions_load_button,
        # "captions_image_selector": captions_image_selector, # Replaced by gallery
        "captions_image_gallery": captions_image_gallery, # NEW Gallery component
        # "caption_image_preview": caption_image_preview, # Removed, gallery has preview
        "caption_selected_filename_display": caption_selected_filename_display,
        "captions_caption_display": captions_caption_display,
        "captions_save_button": captions_save_button,
        "captions_batch_text": captions_batch_text,
        "captions_prepend_button": captions_prepend_button,
        "captions_append_button": captions_append_button,
        "captions_status_display": captions_status_display,

        # Agent Captioning Components
        "caption_model_selector": caption_model_selector,
        "caption_agent_selector": caption_agent_selector,
        "caption_generate_mode": caption_generate_mode,
        "caption_generate_selected_button": caption_generate_selected_button, # Renamed
        "caption_generate_all_button": caption_generate_all_button,

        # State keys remain the same conceptually
        "caption_image_paths_state_key": "caption_image_paths_state",
        "caption_data_state_key": "caption_data_state",
        "caption_selected_item_state_key": "caption_selected_item_state",
    }