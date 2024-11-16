import gradio as gr
import json
import os
from PIL import Image
from agent import get_llm_response, load_roles
import numpy as np
from settings import load_json, save_settings, format_json_to_html_table, remove_agent_role, add_agent_role, save_custom_agent_roles

def load_models(model_file):
    return load_json(model_file)

def load_limiters(limiter_file):
    return load_json(limiter_file)

def load_settings(settings_file):
    return load_json(settings_file)

def update_max_tokens(limiter_handling_option, user_set_max_tokens, is_user_adjusted):
    limiters = load_limiters('limiters.json')
    limiter_settings = limiters.get(limiter_handling_option, {})
    limiter_token_slider = limiter_settings.get("limiter_token_slider", user_set_max_tokens)
    
    if is_user_adjusted:
        return user_set_max_tokens
    
    return limiter_token_slider

def chat(folder_path, role, user_input, model_with_vision, max_tokens, file_handling_option, limiters_handling_option, single_image, settings):
    model = next((m["name"] for m in models if f"{m['name']} (VISION)" == model_with_vision or m["name"] == model_with_vision), None)

    if model is None:
        return "Error: Selected model not found."

    model_info = next((m for m in models if m["name"] == model), None)
    if model_info is None:
        return "Error: Model information not found."

    limiters = load_limiters('limiters.json')
    limiter_settings = limiters.get(limiters_handling_option, {})
    limiter_prompt_format = limiter_settings.get("limiter_prompt_format", "")
    limiter_token_slider = limiter_settings.get("limiter_token_slider", max_tokens)

    max_tokens = min(max_tokens, limiter_token_slider)

    confirmation_messages = []

    def process_image(image, file_path):
        roles = load_roles('agent_roles.json')
        role_description = roles.get(role, "Unknown Role")

        prompt = f"User Input: {user_input}\n\nRole: {role}\nDescription: {role_description}\n{limiter_prompt_format}"

        print(f"Processing image: {file_path}")

        response = get_llm_response(role, prompt, model, [image], max_tokens, file_path, user_input, model_with_vision, max_tokens, single_image, limiters_handling_option)

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(os.path.dirname(file_path), f"{base_name}.txt")

        if os.path.exists(output_file):
            if file_handling_option == "Overwrite":
                with open(output_file, 'w') as f:
                    f.write(response)
                confirmation_messages.append(f"Overwrote existing file: {output_file}\n")
            elif file_handling_option == "Skip":
                confirmation_messages.append(f"Skipped existing file: {output_file}\n")
            elif file_handling_option == "Append":
                with open(output_file, 'a') as f:
                    f.write("\n" + response)
                confirmation_messages.append(f"Appended to existing file: {output_file}\n")
            elif file_handling_option == "Prepend":
                with open(output_file, 'r') as f:
                    existing_content = f.read()
                with open(output_file, 'w') as f:
                    f.write(response + "\n" + existing_content)
                confirmation_messages.append(f"Prepended to existing file: {output_file}\n")
        else:
            with open(output_file, 'w') as f:
                f.write(response)
            confirmation_messages.append(f"Created new file: {output_file}\n")

    if not folder_path.strip() or not os.path.isdir(folder_path):
        roles = load_roles('agent_roles.json')
        role_description = roles.get(role, "Unknown Role")

        prompt = f"User Input: {user_input}\n\nRole: {role}\nDescription: {role_description}\n{limiter_prompt_format}"

        if single_image is not None and model_info["vision"]:
            image = Image.fromarray(single_image.astype('uint8'))
            response = get_llm_response(role, prompt, model, [image], max_tokens, None, user_input, model_with_vision, max_tokens, single_image, limiters_handling_option)
        else:
            response = get_llm_response(role, prompt, model, [], max_tokens, None, user_input, model_with_vision, max_tokens, None, limiters_handling_option)

        return response

    if not model_info["vision"]:
        roles = load_roles('agent_roles.json')
        role_description = roles.get(role, "Unknown Role")

        prompt = f"User Input: {user_input}\n\nRole: {role}\nDescription: {role_description}\n{limiter_prompt_format}"

        response = get_llm_response(role, prompt, model, [], max_tokens, None, user_input, model_with_vision, max_tokens, None, limiters_handling_option)

        return response

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image = Image.open(file_path)
            process_image(image, file_path)

    if not confirmation_messages:
        return "No valid image files found in the directory."

    return "\n".join(confirmation_messages)

# Load roles for dropdown
def load_roles(default_roles_file, custom_roles_file, settings):
    roles = {}
    if settings.get("using_default_agents", False):
        roles.update(load_json(default_roles_file))
    if settings.get("using_custom_agents", False):
        roles.update(load_json(custom_roles_file))
    return roles

# Load models for dropdown
models = load_models('models.json')
model_names_with_vision = [f"{m['name']} (VISION)" if m['vision'] else m['name'] for m in models]

# Load settings
settings = load_settings('settings.json')
ollama_url = settings.get("ollama_url", "http://localhost:11434/api/generate")
max_tokens_slider = settings.get("max_tokens_slider", 1500)
ollama_api_prompt_to_console = settings.get("ollama_api_prompt_to_console", True)

def update_role_dropdown(using_default_agents, using_custom_agents):
    roles = load_roles('agent_roles.json', 'custom_agent_roles.json', {"using_default_agents": using_default_agents, "using_custom_agents": using_custom_agents})
    role_names = list(roles.keys())
    return gr.update(choices=role_names, value=role_names[0] if role_names else None)

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# ArtAgents | Agent-Based Chat with Ollama")
    gr.Markdown("Select an agent, model, and provide input to get a response from Ollama. You can provide a folder path of images for multimodal input.")

    with gr.Tab("Chat"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Folder Input")
                folder_path = gr.Textbox(label="Folder Path")
                file_handling_option = gr.Radio(["Overwrite", "Skip", "Append", "Prepend"], label="File Handling", value="Skip")
                with gr.Row():
                    with gr.Column(scale=1):
                        single_image_display = gr.Image(label="Single Image Input")
                    with gr.Column(scale=1):
                        limiters_handling_option = gr.Radio(["Off", "Flux", "XL", "SD3.5"], label="Limiters", value="Off")
                        max_tokens = gr.Slider(50, max_tokens_slider, step=10, value=max_tokens_slider // 2, label="Max Tokens")
                        using_default_agents = gr.Checkbox(label="Using Default Agents", value=settings.get("using_default_agents", False))
                        using_custom_agents = gr.Checkbox(label="Using Custom Agents", value=settings.get("using_custom_agents", False))
            with gr.Column(scale=1):
                gr.Markdown("### Common Inputs")
                role_names = list(load_roles('agent_roles.json', 'custom_agent_roles.json', settings).keys())
                role = gr.Dropdown(role_names, label="Select Agent", value=role_names[0] if role_names else None)
                user_input = gr.Textbox(label="User Input", lines=2)
                model_with_vision = gr.Dropdown(model_names_with_vision, label="Select Model")
                submit_button = gr.Button("Submit")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Output")
                llm_response = gr.Textbox(label="LLM Response", lines=10)
                gr.Markdown("sandner.art | [Creative AI/ML Research](https://github.com/sandner-art)")
        is_user_adjusted = gr.State(value=False)

        def on_limiter_change(limiter_handling_option, user_set_max_tokens, is_user_adjusted):
            limiters = load_limiters('limiters.json')
            limiter_settings = limiters.get(limiter_handling_option, {})
            limiter_token_slider = limiter_settings.get("limiter_token_slider", user_set_max_tokens)
            return limiter_token_slider, False

        limiters_handling_option.change(
            fn=on_limiter_change,
            inputs=[limiters_handling_option, max_tokens, is_user_adjusted],
            outputs=[max_tokens, is_user_adjusted]
        )

        def on_max_tokens_change(max_tokens, is_user_adjusted):
            return max_tokens, True

        max_tokens.change(
            fn=on_max_tokens_change,
            inputs=[max_tokens, is_user_adjusted],
            outputs=[max_tokens, is_user_adjusted]
        )

        
        using_default_agents.change(
            fn=update_role_dropdown,
            inputs=[using_default_agents, using_custom_agents],
            outputs=[role]
        )

        using_custom_agents.change(
            fn=update_role_dropdown,
            inputs=[using_default_agents, using_custom_agents],
            outputs=[role]
        )

        submit_button.click(
            chat,
            inputs=[folder_path, role, user_input, model_with_vision, max_tokens, file_handling_option, limiters_handling_option, single_image_display, gr.State(settings)],
            outputs=llm_response
        )

    with gr.Tab("Settings"):
        
        with gr.Tab("App"):
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
            agent_roles_html = format_json_to_html_table(load_roles('agent_roles.json', 'custom_agent_roles.json', settings))
            agent_roles_display = gr.HTML(agent_roles_html)

        with gr.Tab("Custom Agent Roles"):
            gr.Markdown("### Edit Custom Agent Roles")
            custom_agent_roles = load_json('custom_agent_roles.json')
            custom_agent_roles_display = gr.JSON(value=custom_agent_roles, label="Custom Agent Roles JSON", visible=False)
            unsaved_changes_indicator = gr.Markdown("**Unsaved Changes**", visible=False)

            with gr.Blocks() as custom_agent_roles_block:
                with gr.Row():
                    agent_label = gr.Markdown("**Agent/Role**")
                    #role_label = gr.Markdown("**Role**")
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

if __name__ == "__main__":
    demo.launch()