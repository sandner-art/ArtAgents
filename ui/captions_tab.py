# ArtAgent/ui/captions_tab.py
import gradio as gr
from core.help_content import get_tooltip # Assuming help content will be added

def create_captions_tab():
    """Creates the Gradio components for the Captions Editor Tab."""

    with gr.Tab("Image Captions Editor"):
        gr.Markdown("## View, Edit, and Save Image Captions (.txt files)")
        gr.Markdown(
            "Load a folder of images. Select an image by its filename in the list below "
            "to view/edit its caption (text content from the corresponding `.txt` file). "
            "A preview of the selected image will appear."
            )

        # State specific to this tab's operation (defined in app.py, referenced here via components)
        # caption_image_paths_state = gr.State({}) # filename: absolute_path
        # caption_data_state = gr.State({})        # filename: caption_text
        # caption_selected_item_state = gr.State(None) # filename of selected image

        with gr.Row():
            captions_folder_path = gr.Textbox(
                label="Image Folder Path",
                placeholder="Enter the full path to the folder containing images and .txt captions",
                info=get_tooltip("captions_folder_path"), # Added tooltip key (define in help_content.py)
                scale=3
            )
            captions_load_button = gr.Button("Load Folder", variant="secondary", scale=1)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Images")
                # Using CheckboxGroup with filenames for simplicity and multi-select
                captions_image_selector = gr.CheckboxGroup(
                    label="Select Image File(s)",
                    info=get_tooltip("captions_image_selector"), # Added tooltip key
                    # Choices will be populated by the load callback
                )
                # ADDED Image Preview Component
                caption_image_preview = gr.Image(
                    label="Selected Image Preview",
                    type="filepath", # Expects a file path string as input
                    interactive=False, # Preview only, not for upload
                    height=350, # Increased height a bit
                    show_download_button=False # Hide download button for preview
                 )

            with gr.Column(scale=2):
                gr.Markdown("### Caption Text")
                # Display filename for context
                caption_selected_filename_display = gr.Textbox(
                    label="Selected File", interactive=False,
                    info="Filename of the image currently being previewed/edited."
                )
                captions_caption_display = gr.Textbox(
                    label="Caption / Text Content", lines=13, # Increased lines slightly
                    interactive=True,
                    placeholder="Select an image to view/edit its caption...",
                    info=get_tooltip("captions_caption_display") # Added tooltip key
                )
                captions_save_button = gr.Button(
                    "Save THIS Caption", variant="primary",
                    info=get_tooltip("captions_save_button") # Added tooltip key
                )
                captions_status_display = gr.Textbox(label="Status", interactive=False, lines=2)

                # Batch operations section
                with gr.Accordion("Batch Append/Prepend (for Selected Images)", open=False):
                    gr.Markdown("Add text to the beginning or end of captions for **all currently checked** images in the list.")
                    captions_batch_text = gr.Textbox(
                        label="Text to Add",
                        info=get_tooltip("captions_batch_text") # Added tooltip key
                    )
                    with gr.Row():
                        captions_prepend_button = gr.Button("Prepend Text", info=get_tooltip("captions_prepend_button"))
                        captions_append_button = gr.Button("Append Text", info=get_tooltip("captions_append_button"))

    # Return dictionary of components and states defined within this function scope
    # Note: State variables themselves are usually defined in app.py,
    # this dict just returns the UI components linked to them.
    return {
        # UI Components
        "captions_folder_path": captions_folder_path,
        "captions_load_button": captions_load_button,
        "captions_image_selector": captions_image_selector,
        "caption_image_preview": caption_image_preview, # Added preview component
        "caption_selected_filename_display": caption_selected_filename_display,
        "captions_caption_display": captions_caption_display,
        "captions_save_button": captions_save_button,
        "captions_status_display": captions_status_display,
        "captions_batch_text": captions_batch_text,
        "captions_prepend_button": captions_prepend_button,
        "captions_append_button": captions_append_button,
        # States are defined in app.py, but keys are listed here for clarity
        # about what states this tab's logic will interact with via app.py wiring
        "caption_image_paths_state_key": "caption_image_paths_state",
        "caption_data_state_key": "caption_data_state",
        "caption_selected_item_state_key": "caption_selected_item_state",
    }