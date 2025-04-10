# ArtAgent/core/utils.py
import json
import os

# --- Determine Project Root ---
# This assumes utils.py is in core/, which is one level down from the root
# Adjust if structure changes
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_absolute_path(relative_path):
    """Constructs an absolute path from a path relative to the project root."""
    return os.path.join(PROJECT_ROOT, relative_path)

def load_json(file_path, is_relative=True):
    """
    Loads a JSON file. Handles relative paths from project root.

    Args:
        file_path (str): The path to the JSON file.
        is_relative (bool): If True, file_path is treated as relative to project root.

    Returns:
        dict or list: The loaded JSON data, or empty dict/list on error/empty file.
    """
    full_path = get_absolute_path(file_path) if is_relative else file_path

    if not os.path.exists(full_path):
        print(f"Warning: File not found {full_path}. Returning empty dict.")
        return {}

    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            if not content:
                # print(f"Warning: {full_path} is empty. Returning empty dict.") # Less verbose
                return {}
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {full_path}: {e}. Returning empty dict.")
        return {}
    except Exception as e:
        print(f"Error reading {full_path}: {e}. Returning empty dict.")
        return {}


def format_json_to_html_table(data):
    """Formats a dictionary (like agent roles) into an HTML table."""
    if not isinstance(data, dict):
        return "<p>Invalid data format for table generation.</p>"

    html = "<table style='width:100%; border-collapse: collapse; border: 1px solid #ddd;'>"
    html += "<thead><tr>"
    # Assuming simple key-value or dict structure like roles {agent: {description: ...}}
    # Adjust headers if data structure is different
    html += "<th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Agent/Key</th>"
    html += "<th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Description/Value</th>"
    html += "</tr></thead>"
    html += "<tbody>"

    for key, value in data.items():
        html += f"<tr><td style='border: 1px solid #ddd; padding: 8px; vertical-align: top;'><strong>{key}</strong></td>"
        display_value = ""
        if isinstance(value, dict):
            # Handle nested dict, common for roles
            desc = value.get('description', 'N/A')
            opts = value.get('ollama_api_options')
            display_value = f"<p><i>Description:</i> {desc}</p>"
            if opts:
                 display_value += "<p><i>Options:</i><pre><code>" + json.dumps(opts, indent=2) + "</code></pre></p>"

        elif isinstance(value, str):
             display_value = value
        else:
             try:
                 display_value = json.dumps(value, indent=2)
             except TypeError:
                 display_value = str(value)

        html += f"<td style='border: 1px solid #ddd; padding: 8px; word-wrap: break-word;'>{display_value}</td></tr>"

    html += "</tbody></table>"
    return html