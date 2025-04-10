# ArtAgent/ui/roles_tab.py
import gradio as gr
# Import help content if adding any explanations here
from core.utils import format_json_to_html_table

def create_roles_tabs(default_roles_data, custom_roles_data):
    """Creates the Gradio tabs for displaying agent roles."""

    with gr.Tab("Default Agent Roles"):
        gr.Markdown("### `agents/agent_roles.json` (Read Only)")
        gr.Markdown("These are the standard agents available when 'Load Default Agents' is checked in App Settings.")
        default_roles_display = gr.HTML(format_json_to_html_table(default_roles_data))

    with gr.Tab("Custom Agent Roles"):
        gr.Markdown("### `agents/custom_agent_roles.json` (Read Only)")
        gr.Markdown("These are your custom agents, loaded when 'Load Custom Agents' is checked. They override default agents with the same name. Edit the JSON file directly to make persistent changes.")
        # Note: Editing UI could be added here later, referencing logic similar to original settingsapp.py
        custom_roles_display = gr.HTML(format_json_to_html_table(custom_roles_data))

    # Return components if needed later for updates (e.g., refresh button)
    return {
        "default_roles_display": default_roles_display,
        "custom_roles_display": custom_roles_display,
    }