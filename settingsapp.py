import gradio as gr
import json
import os

# Load JSON files
def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return {}

settings = load_json('settings.json')
agent_roles = load_json('agent_roles.json')
custom_agent_roles = load_json('custom_agent_roles.json')

# Save settings.json
def save_settings(ollama_url, max_tokens_slider, ollama_api_prompt_to_console, using_default_agents, using_custom_agents):
    settings = {
        "ollama_url": ollama_url,
        "max_tokens_slider": max_tokens_slider,
        "ollama_api_prompt_to_console": ollama_api_prompt_to_console,
        "using_default_agents": using_default_agents,
        "using_custom_agents": using_custom_agents
    }
    with open('settings.json', 'w') as file:
        json.dump(settings, file, indent=4)
    return "Settings saved successfully."

# Save custom_agent_roles.json
def save_custom_agent_roles_to_file(custom_agent_roles):
    with open('custom_agent_roles.json', 'w') as file:
        json.dump(custom_agent_roles, file, indent=4)
    return "Custom agent roles saved successfully."

# Function to format JSON into an HTML table
def format_json_to_html_table(data):
    html = "<table style='width:100%; border-collapse: collapse;'>"
    html += "<tr><th style='border: 1px solid #ddd; padding: 8px;'>Agent</th><th style='border: 1px solid #ddd; padding: 8px;'>Role</th></tr>"
    for agent, role in data.items():
        html += f"<tr><td style='border: 1px solid #ddd; padding: 8px;'>{agent}</td><td style='border: 1px solid #ddd; padding: 8px; word-wrap: break-word;'>{role}</td></tr>"
    html += "</table>"
    return html

# Gradio Interface
def settings_tab():
    with gr.Blocks() as demo:
        gr.Markdown("## Settings Tab")

        with gr.Tab("Settings"):
            gr.Markdown("### General Settings")
            ollama_url = gr.Textbox(label="Ollama URL", value=settings.get("ollama_url", ""))
            max_tokens_slider = gr.Slider(label="Max Tokens", minimum=1, maximum=3000, step=1, value=settings.get("max_tokens_slider", 1500))
            ollama_api_prompt_to_console = gr.Checkbox(label="Ollama API Prompt to Console", value=settings.get("ollama_api_prompt_to_console", False))
            using_default_agents = gr.Checkbox(label="Using Default Agents", value=settings.get("using_default_agents", False))
            using_custom_agents = gr.Checkbox(label="Using Custom Agents", value=settings.get("using_custom_agents", False))
            save_settings_button = gr.Button("Save Settings")

            save_settings_button.click(
                fn=save_settings,
                inputs=[ollama_url, max_tokens_slider, ollama_api_prompt_to_console, using_default_agents, using_custom_agents],
                outputs=[gr.Textbox(label="Status", lines=1)]
            )

        with gr.Tab("Agent Roles"):
            gr.Markdown("### agent_roles.json")
            agent_roles_html = format_json_to_html_table(agent_roles)
            agent_roles_display = gr.HTML(agent_roles_html)

        with gr.Tab("Custom Agent Roles"):
            gr.Markdown("### Edit Custom Agent Roles")
            custom_agent_roles_display = gr.JSON(value=custom_agent_roles, label="Custom Agent Roles JSON", visible=False)
            unsaved_changes_indicator = gr.Markdown("**Unsaved Changes**", visible=False)

            with gr.Blocks() as custom_agent_roles_block:
                with gr.Row():
                    agent_label = gr.Markdown("**Agent**")
                    role_label = gr.Markdown("**Role**")
                agent_role_rows = []

                def create_row(agent, role, custom_agent_roles_display, unsaved_changes_indicator):
                    with gr.Row() as row:
                        agent_textbox = gr.Textbox(label="", value=agent, lines=1, scale=1, interactive=True)
                        role_textbox = gr.Textbox(label="", value=role, lines=3, scale=4, interactive=True)
                        remove_button = gr.Button("Remove", scale=1, variant="stop")

                        remove_button.click(
                            fn=remove_agent_role,
                            inputs=[agent_textbox, role_textbox, custom_agent_roles_display],
                            outputs=[custom_agent_roles_display, unsaved_changes_indicator]
                        )
                    return row

                for agent, role in custom_agent_roles.items():
                    agent_role_rows.append(create_row(agent, role, custom_agent_roles_display, unsaved_changes_indicator))

                with gr.Row():
                    new_agent_textbox = gr.Textbox(label="New Agent", lines=1, scale=1, interactive=True)
                    new_role_textbox = gr.Textbox(label="New Role", lines=3, scale=4, interactive=True)
                    add_button = gr.Button("Add New Agent Role", scale=1)

                add_button.click(
                    fn=add_agent_role,
                    inputs=[new_agent_textbox, new_role_textbox, custom_agent_roles_display, unsaved_changes_indicator],
                    outputs=[custom_agent_roles_display, new_agent_textbox, new_role_textbox, unsaved_changes_indicator]
                )

                save_button = gr.Button("Save Custom Agent Roles")

                save_button.click(
                    fn=save_custom_agent_roles,
                    inputs=[custom_agent_roles_display, unsaved_changes_indicator],
                    outputs=[gr.Textbox(label="Status", lines=1), unsaved_changes_indicator]
                )

    return demo

# Function to remove an agent role
def remove_agent_role(agent_textbox, role_textbox, custom_agent_roles_display):
    agent = agent_textbox
    custom_agent_roles = custom_agent_roles_display.value
    if agent in custom_agent_roles:
        del custom_agent_roles[agent]
    return gr.update(value=custom_agent_roles), gr.update(visible=True)

# Function to add a new agent role
def add_agent_role(agent_textbox, role_textbox, custom_agent_roles_display, unsaved_changes_indicator):
    agent = agent_textbox
    role = role_textbox
    custom_agent_roles = custom_agent_roles_display.value
    if agent and role:
        custom_agent_roles[agent] = role
    return gr.update(value=custom_agent_roles), "", "", gr.update(visible=True)

# Function to save custom agent roles
def save_custom_agent_roles(custom_agent_roles_display, unsaved_changes_indicator):
    custom_agent_roles = custom_agent_roles_display.value
    status_message = save_custom_agent_roles_to_file(custom_agent_roles)
    return status_message, gr.update(visible=False)

# Launch the demo
demo = settings_tab()
demo.launch()