# ArtAgent/ui/image_gen_tab.py
import gradio as gr

from core.image_gen_ui import backend_labels, default_backend


def create_image_gen_tab(initial_settings):
    """Creates the Image Generation tab (ComfyUI, Forge/A1111/SD.Next, Ollama image models)."""
    with gr.Tab("Image Generation", id="image_gen"):
        gallery_items_state = gr.State([])
        gr.Markdown("## Generate Images from Your Prompts")
        gr.Markdown(
            "Write a prompt or send one from the Chat tab (🎨 button). Generated images are saved as PNG with the "
            "prompt and settings embedded, in the output folder set under **App Settings → Image Generation Servers**."
        )
        with gr.Row():
            with gr.Column(scale=2):
                image_prompt = gr.Textbox(label="Prompt", lines=6, elem_id="image_prompt")
                use_last_response_button = gr.Button("⬅ Use last LLM response", size="sm")
                image_negative_prompt = gr.Textbox(label="Negative Prompt", lines=2, elem_id="image_negative")
                with gr.Group():
                    with gr.Row():
                        image_backend = gr.Dropdown(
                            backend_labels(), value=default_backend(initial_settings),
                            label="Backend", scale=4, elem_id="image_backend"
                        )
                        image_refresh_button = gr.Button("🔄", scale=0, min_width=50, elem_id="image_refresh")
                    image_model = gr.Dropdown(
                        [], label="Model / Checkpoint", allow_custom_value=True, elem_id="image_model"
                    )
                    image_workflow = gr.Dropdown(
                        [], label="ComfyUI Workflow (image_workflows/comfyui, API format)", elem_id="image_workflow"
                    )
                    image_sampler = gr.Dropdown([""], value="", label="Sampler (empty = default)", allow_custom_value=True)
                with gr.Row():
                    image_width = gr.Slider(256, 2048, value=1024, step=64, label="Width")
                    image_height = gr.Slider(256, 2048, value=1024, step=64, label="Height")
                with gr.Row():
                    image_steps = gr.Number(value=0, precision=0, minimum=0, maximum=200, label="Steps (0 = default)")
                    image_cfg = gr.Number(value=0, minimum=0, maximum=30, step=0.5, label="CFG (0 = default)")
                    image_seed = gr.Number(value=-1, precision=0, minimum=-1, label="Seed (-1 = random)")
                    image_batch = gr.Slider(1, 8, value=1, step=1, label="Batch Size")
                with gr.Row():
                    image_generate_button = gr.Button("🎨 Generate", variant="primary", scale=3, elem_id="image_generate")
                    image_stop_button = gr.Button("⏹ Stop", variant="stop", scale=1, elem_id="image_stop")
                image_status = gr.Textbox(label="Status", interactive=False, lines=2, elem_id="image_status")
            with gr.Column(scale=3):
                image_gallery = gr.Gallery(
                    label="Generated Images (this session)", columns=2, height=720,
                    object_fit="contain", format="png", elem_id="image_gallery"
                )
                image_clear_button = gr.Button("Clear Gallery", size="sm")

    return {
        "gallery_items_state": gallery_items_state,
        "image_prompt": image_prompt,
        "use_last_response_button": use_last_response_button,
        "image_negative_prompt": image_negative_prompt,
        "image_backend": image_backend,
        "image_refresh_button": image_refresh_button,
        "image_model": image_model,
        "image_workflow": image_workflow,
        "image_sampler": image_sampler,
        "image_width": image_width,
        "image_height": image_height,
        "image_steps": image_steps,
        "image_cfg": image_cfg,
        "image_seed": image_seed,
        "image_batch": image_batch,
        "image_generate_button": image_generate_button,
        "image_stop_button": image_stop_button,
        "image_status": image_status,
        "image_gallery": image_gallery,
        "image_clear_button": image_clear_button,
    }
