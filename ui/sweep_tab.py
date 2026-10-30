# ArtAgent/ui/sweep_tab.py
import gradio as gr
from core.help_content import get_tooltip # Assuming help content is added later

def create_sweep_tab(initial_team_names, initial_model_names):
    """Creates the Gradio components for the Experiment Sweep Tab."""

    with gr.Tab("Experiment Sweep"):
        gr.Markdown("## Run Experiments with Multiple Configurations")
        gr.Markdown("Define a base prompt and select multiple Agent Teams and/or Worker Models to run. Results will be saved as structured JSON protocol files.")

        with gr.Row():
            with gr.Column(scale=2):
                sweep_prompts_input = gr.Textbox(
                    label="Base User Prompt(s)", lines=5,
                    placeholder="Example:\ndesign a minimalist chair using bent plywood\na photorealistic portrait of an astronaut on Mars"
                )
            with gr.Column(scale=1):
                sweep_output_folder_input = gr.Textbox(
                    label="Output Subfolder Name", value="sweep_results"
                )
                sweep_log_intermediate_checkbox = gr.Checkbox(
                    label="Log Intermediate Agent Steps?", value=False
                )

        with gr.Row():
            sweep_teams_select = gr.CheckboxGroup(
                label="Select Agent Teams/Workflows to Test",
                choices=initial_team_names
            )
            worker_model_choices = sorted({name.replace(" (VISION)", "") for name in initial_model_names})
            sweep_models_select = gr.CheckboxGroup(
                label="Select Worker Models to Test",
                choices=worker_model_choices
            )

        with gr.Row():
            sweep_start_button = gr.Button("🚀 Start Sweep Run", variant="primary", scale=3)
            sweep_stop_button = gr.Button("⏹ Stop", variant="stop", scale=1, elem_id="sweep_stop")

        with gr.Row():
            gr.Markdown("### Sweep Progress & Status")
            sweep_status_display = gr.Textbox(
                label="Status Log", lines=15, interactive=False
            )

    # Return dictionary of components needed by app.py
    return {
        "sweep_prompts_input": sweep_prompts_input,
        "sweep_teams_select": sweep_teams_select,
        "sweep_models_select": sweep_models_select,
        "sweep_output_folder_input": sweep_output_folder_input,
        "sweep_log_intermediate_checkbox": sweep_log_intermediate_checkbox,
        "sweep_start_button": sweep_start_button,
        "sweep_stop_button": sweep_stop_button,
        "sweep_status_display": sweep_status_display,
    }