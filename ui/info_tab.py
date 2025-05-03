# ArtAgent/ui/info_tab.py
import gradio as gr
from core.utils import format_json_to_html_table # Utility to display roles as tables
# from core.help_content import get_markdown # Optional: If using markdown help here

# --- Placeholder for Version (Could be loaded from a file later) ---
APP_VERSION = "0.9.2-alpha" # Example version

def create_info_tab(default_roles_data, custom_roles_data):
    """Creates the Gradio components for the consolidated Info Tab."""

    with gr.Tab("Info") as info_tab_main:
        gr.Markdown(f"""
        ## ArtAgents - Agent-Based Creative Tool
        **Version:** {APP_VERSION}

        ArtAgents is a framework for experimenting with multi-agent systems, prompt engineering,
        and content generation using local LLMs via Ollama.

        **Key Features:**
        *   Utilize specialized AI agents and build custom workflows (Teams).
        *   Interact via Chat, generate/edit Image Captions, run experiment Sweeps.
        *   Configure Ollama settings and agent loading preferences.

        **Developed by:** [Daniel Sandner (sandner.art)](https://sandner.art/)

        ---
        Select the sub-tabs below for more specific details.
        """)

        with gr.Tabs() as info_sub_tabs:
            with gr.TabItem("Default Roles") as default_roles_tab:
                gr.Markdown("### `agents/agent_roles.json` (Read Only)")
                gr.Markdown("These are the standard agents available when 'Load Default Agents' is checked in App Settings.")
                default_roles_display = gr.HTML(format_json_to_html_table(default_roles_data))

            with gr.TabItem("Custom Roles") as custom_roles_tab:
                gr.Markdown("### `agents/custom_agent_roles.json` (Read Only)")
                gr.Markdown("These are your custom agents, loaded when 'Load Custom Agents' is checked. They override default agents with the same name. Edit the JSON file directly to make persistent changes.")
                custom_roles_display = gr.HTML(format_json_to_html_table(custom_roles_data))

            # Add more TabItems here later if needed (e.g., About Ollama, Libraries Used)

    # Return dictionary of components (useful if they need dynamic updates later)
    return {
        "info_tab_main": info_tab_main,
        "info_sub_tabs": info_sub_tabs,
        "default_roles_tab": default_roles_tab,
        "custom_roles_tab": custom_roles_tab,
        "default_roles_display": default_roles_display,
        "custom_roles_display": custom_roles_display,
    }