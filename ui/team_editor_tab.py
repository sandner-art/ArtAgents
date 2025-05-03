# ArtAgent/ui/team_editor_tab.py
import gradio as gr
from core.help_content import get_tooltip # Assuming help content is added later

def create_team_editor_tab(initial_team_names, initial_available_agent_names):
    """Creates the Gradio components for the Agent Team Editor tab."""

    with gr.Tab("Agent Team Editor"):
        gr.Markdown("## Create and Edit Agent Workflows (Teams)")
        gr.Markdown("Define sequences of agents to perform complex tasks. Saved teams appear in the Chat tab dropdown.")

        # State to hold the data for the team currently being edited
        current_team_editor_state = gr.State(value={
            "name": "", "description": "", "steps": [], "assembly_strategy": "concatenate"
        })

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Load / Manage Teams")
                team_select_dropdown = gr.Dropdown(
                    choices=initial_team_names, label="Select Team to Load/Edit",
                    info="Select a team to view or modify its steps below."
                )
                load_team_button = gr.Button("Load Selected Team")
                delete_team_button = gr.Button("Delete Selected Team", variant="stop")
                clear_editor_button = gr.Button("Clear Editor / New Team")

            with gr.Column(scale=2):
                gr.Markdown("### Team Details")
                team_name_textbox = gr.Textbox(label="Team Name", info="Unique name for this workflow.")
                team_description_textbox = gr.Textbox(label="Team Description", lines=2, info="Briefly explain what this team does.")
                assembly_strategy_radio = gr.Radio(
                    # ***** UPDATED CHOICES HERE *****
                    choices=["concatenate", "refine_last", "summarize_all", "structured_concatenate"],
                    value="concatenate",
                    label="Final Output Strategy",
                    # ***** UPDATED INFO HERE *****
                    info=("'concatenate': Joins all step outputs.\n"
                          "'refine_last': Uses only the last step's output.\n"
                          "'summarize_all': Calls an LLM to synthesize all outputs.\n"
                          "'structured_concatenate': Joins outputs with agent/step labels.")
                )

        gr.Markdown("---")
        gr.Markdown("### Define Workflow Steps")

        with gr.Row():
            with gr.Column(scale=2):
                steps_display_json = gr.JSON(label="Current Steps (Read-Only View)", scale=2)

            with gr.Column(scale=1):
                gr.Markdown("#### Add/Remove Steps")
                agent_to_add_dropdown = gr.Dropdown(
                    choices=initial_available_agent_names, label="Select Agent Role for New Step",
                    info="Choose an agent (Default, Custom, or [File]) to add."
                )
                add_step_button = gr.Button("Add Selected Agent as Step", variant="secondary")
                gr.Markdown("---")
                step_index_to_remove = gr.Number(label="Step Number to Remove", minimum=1, precision=0, value=1)
                remove_step_button = gr.Button("Remove Step #", variant="secondary")

        gr.Markdown("---")
        with gr.Row():
             save_team_button = gr.Button("Save Current Team Definition", variant="primary")
             save_status_textbox = gr.Textbox(label="Status", interactive=False)

    # Return dictionary of components needed for wiring
    return {
        "team_select_dropdown": team_select_dropdown,
        "load_team_button": load_team_button,
        "delete_team_button": delete_team_button,
        "clear_editor_button": clear_editor_button,
        "team_name_textbox": team_name_textbox,
        "team_description_textbox": team_description_textbox,
        "assembly_strategy_radio": assembly_strategy_radio,
        "steps_display_json": steps_display_json,
        "agent_to_add_dropdown": agent_to_add_dropdown,
        "add_step_button": add_step_button,
        "step_index_to_remove": step_index_to_remove,
        "remove_step_button": remove_step_button,
        "save_team_button": save_team_button,
        "save_status_textbox": save_status_textbox,
        "current_team_editor_state": current_team_editor_state,
    }