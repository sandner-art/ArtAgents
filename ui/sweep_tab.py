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
                    info="Enter one prompt per line. Each prompt will be run against all selected configurations.",
                    placeholder="Example:\ndesign a minimalist chair using bent plywood\na photorealistic portrait of an astronaut on Mars"
                )
            with gr.Column(scale=1):
                sweep_output_folder_input = gr.Textbox(
                    label="Output Subfolder Name", value="sweep_results",
                    info="A folder named 'sweep_runs/[TIMESTAMP]_[NAME]' will be created in the project root."
                )
                sweep_log_intermediate_checkbox = gr.Checkbox(
                    label="Log Intermediate Agent Steps?", value=False,
                    info="Include the output of each agent step within the team workflow in the protocol file."
                )

        with gr.Row():
            sweep_teams_select = gr.CheckboxGroup(
                label="Select Agent Teams/Workflows to Test",
                choices=initial_team_names,
                info="Check one or more predefined teams."
            )
            # Filter out the (VISION) suffix for model selection for workers
            worker_model_choices = sorted(list(set([name.replace(" (VISION)","") for name in initial_model_names])))
            sweep_models_select = gr.CheckboxGroup(
                label="Select Worker Models to Test",
                choices=worker_model_choices,
                info="Check one or more models to use for the agent steps."
            )

        with gr.Row():
            sweep_start_button = gr.Button("🚀 Start Sweep Run", variant="primary")

        with gr.Row():
            gr.Markdown("### Sweep Progress & Status")
            sweep_status_display = gr.Textbox(
                label="Status Log", lines=15, interactive=False,
                info="Progress updates and final summary will appear here."
            )

    # Return dictionary of components needed by app.py
    return {
        "sweep_prompts_input": sweep_prompts_input,
        "sweep_teams_select": sweep_teams_select,
        "sweep_models_select": sweep_models_select,
        "sweep_output_folder_input": sweep_output_folder_input,
        "sweep_log_intermediate_checkbox": sweep_log_intermediate_checkbox,
        "sweep_start_button": sweep_start_button,
        "sweep_status_display": sweep_status_display,
    }