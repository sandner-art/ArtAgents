import json
import os

# Load JSON files
def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return {}

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

def save_custom_agent_roles_to_file(custom_agent_roles):
    with open('custom_agent_roles.json', 'w') as file:
        json.dump(custom_agent_roles, file, indent=4)
    return "Custom agent roles saved successfully."

def format_json_to_html_table(data):
    html = "<table style='width:100%; border-collapse: collapse;'>"
    html += "<tr><th style='border: 1px solid #ddd; padding: 8px;'>Agent</th><th style='border: 1px solid #ddd; padding: 8px;'>Role</th></tr>"
    for agent, role in data.items():
        html += f"<tr><td style='border: 1px solid #ddd; padding: 8px;'>{agent}</td><td style='border: 1px solid #ddd; padding: 8px; word-wrap: break-word;'>{role}</td></tr>"
    html += "</table>"
    return html

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