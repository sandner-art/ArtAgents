# ArtAgent/ui/team_editor_tab.py
import gradio as gr
from core.help_content import get_tooltip # Assuming help content is added later

def create_team_editor_tab(initial_team_names, initial_available_agent_names):
    """Creates the Gradio components for the Agent Team Editor tab."""

    with gr.Tab("Agent Team Editor"):
        gr.Markdown("## Create and Edit Agent Workflows (Teams)")
        gr.Markdown("Define sequences of agents to perform complex tasks. Saved teams appear in the Chat tab dropdown.")

        # State to hold the data for the team currently being edited
        # Structure: {'name': str, 'description': str, 'steps': list[dict], 'assembly_strategy': str}
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
                    choices=["concatenate", "refine_last"], value="concatenate", label="Final Output Strategy",
                    info="'concatenate' joins all step outputs, 'refine_last' uses only the output of the last step (assumed to be a refiner)."
                )

        gr.Markdown("---")
        gr.Markdown("### Define Workflow Steps")

        with gr.Row():
            with gr.Column(scale=2):
                # Display current steps (using JSON for simplicity in V1)
                steps_display_json = gr.JSON(label="Current Steps (Read-Only View)", scale=2)
                # Alternative display (less ideal for complex steps):
                # steps_display_df = gr.DataFrame(headers=["Step", "Role", "Goal"], interactive=False)

            with gr.Column(scale=1):
                gr.Markdown("#### Add/Remove Steps")
                agent_to_add_dropdown = gr.Dropdown(
                    choices=initial_available_agent_names, label="Select Agent Role for New Step",
                    info="Choose an agent (Default, Custom, or [File]) to add."
                )
                # Goal textbox is optional for now, can be added later
                # step_goal_textbox = gr.Textbox(label="Optional Goal for Step")
                add_step_button = gr.Button("Add Selected Agent as Step", variant="secondary")

                gr.Markdown("---") # Separator

                step_index_to_remove = gr.Number(label="Step Number to Remove", minimum=1, precision=0, value=1)
                remove_step_button = gr.Button("Remove Step #", variant="secondary")
                # Reordering buttons omitted for V1 simplicity
                # move_up_button = gr.Button("Move Up")
                # move_down_button = gr.Button("Move Down")


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
        # State for this tab
        "current_team_editor_state": current_team_editor_state,
    }