# ArtAgent/ui/roles_tab.py
import gradio as gr
from core.utils import load_json, format_json_to_html_table

def create_roles_tabs(default_roles_data, custom_roles_data):
    """Creates the Gradio tabs for displaying agent roles."""

    with gr.Tab("Default Agent Roles"):
        gr.Markdown("### `agents/agent_roles.json` (Read Only)")
        default_roles_display = gr.HTML(format_json_to_html_table(default_roles_data))

    with gr.Tab("Custom Agent Roles"):
        gr.Markdown("### `agents/custom_agent_roles.json` (Read Only)")
        gr.Markdown("*(Edit this file directly or implement an editor UI)*") # Placeholder note
        custom_roles_display = gr.HTML(format_json_to_html_table(custom_roles_data))

    # Return components if needed later for updates (e.g., refresh button)
    return {
        "default_roles_display": default_roles_display,
        "custom_roles_display": custom_roles_display,
    }