# app.py

# Part 1: Imports and Utility Functions
import gradio as gr
import json
import os
from PIL import Image
from agent import get_llm_response, load_roles
import numpy as np
import history  # Import the history module
from settings import format_json_to_html_table  # Import the format_json_to_html_table function

# Load JSON files
def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return {}

# Define a function to load models
def load_models(model_file):
    return load_json(model_file)

# Define a function to load limiters
def load_limiters(limiter_file):
    return load_json(limiter_file)

# Define a function to load settings
def load_settings(settings_file):
    return load_json(settings_file)

# Define a function to update the max tokens based on limiters
def update_max_tokens(limiter_handling_option, user_set_max_tokens, is_user_adjusted):
    limiters = load_limiters('limiters.json')
    limiter_settings = limiters.get(limiter_handling_option, {})
    limiter_token_slider = limiter_settings.get("limiter_token_slider", user_set_max_tokens)

    if is_user_adjusted:
        return user_set_max_tokens

    return limiter_token_slider

# Part 2: Chat Functionality
def chat(folder_path, role, user_input, model_with_vision, max_tokens, file_handling_option, limiters_handling_option, single_image, settings, use_ollama_api_options):
    global history_list  # Declare history_list as a global variable
    global current_session_history  # Declare current_session_history as a global variable

    model = next((m["name"] for m in models if f"{m['name']} (VISION)" == model_with_vision or m["name"] == model_with_vision), None)

    if model is None:
        return "Error: Selected model not found.", "\n".join(current_session_history), model

    model_info = next((m for m in models if m["name"] == model), None)
    if model_info is None:
        return "Error: Model information not found.", "\n".join(current_session_history), model

    limiter_settings = limiters.get(limiters_handling_option, {})
    limiter_prompt_format = limiter_settings.get("limiter_prompt_format", "")
    limiter_token_slider = limiter_settings.get("limiter_token_slider", max_tokens)

    max_tokens = min(max_tokens, limiter_token_slider)

    roles = load_roles('agent_roles.json', 'custom_agent_roles.json', settings)
    role_description = roles.get(role, {}).get("description", "Unknown Role")
    role_settings = roles.get(role, {}).get("ollama_api_options", {})

    prompt = f"User Input: {user_input}\n\nRole: {role}\nDescription: {role_description}\n{limiter_prompt_format}"

    if use_ollama_api_options:
        ollama_api_options = settings.get("ollama_api_options", {})
        ollama_api_options.update(role_settings)
    else:
        ollama_api_options = {}

    confirmation_messages = []

    def process_image(image, file_path):
        response = get_llm_response(role, prompt, model, [image], max_tokens, file_path, user_input, model_with_vision, max_tokens, single_image, limiters_handling_option, ollama_api_options)

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

        history_list.append(f"User Input: {user_input}\nRole: {role}\nResponse: {response}\n")
        current_session_history.append(f"User Input: {user_input}\nRole: {role}\nResponse: {response}\n")
        history_list = history.add_to_history(history_list, f"User Input: {user_input}\nRole: {role}\nResponse: {response}\n")

    if not folder_path.strip() or not os.path.isdir(folder_path):
        if single_image is not None and model_info["vision"]:
            image = Image.fromarray(single_image.astype('uint8'))
            response = get_llm_response(role, prompt, model, [image], max_tokens, None, user_input, model_with_vision, max_tokens, single_image, limiters_handling_option, ollama_api_options)
        else:
            response = get_llm_response(role, prompt, model, [], max_tokens, None, user_input, model_with_vision, max_tokens, None, limiters_handling_option, ollama_api_options)

        history_list.append(f"User Input: {user_input}\nRole: {role}\nResponse: {response}\n")
        current_session_history.append(f"User Input: {user_input}\nRole: {role}\nResponse: {response}\n")
        history_list = history.add_to_history(history_list, f"User Input: {user_input}\nRole: {role}\nResponse: {response}\n")
        return response, "\n".join(current_session_history), model

    if not model_info["vision"]:
        response = get_llm_response(role, prompt, model, [], max_tokens, None, user_input, model_with_vision, max_tokens, None, limiters_handling_option, ollama_api_options)
        history_list.append(f"User Input: {user_input}\nRole: {role}\nResponse: {response}\n")
        current_session_history.append(f"User Input: {user_input}\nRole: {role}\nResponse: {response}\n")
        history_list = history.add_to_history(history_list, f"User Input: {user_input}\nRole: {role}\nResponse: {response}\n")
        return response, "\n".join(current_session_history), model

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image = Image.open(file_path)
            process_image(image, file_path)

    if not confirmation_messages:
        return "No valid image files found in the directory.", "\n".join(current_session_history), model

    return "\n".join(confirmation_messages), "\n".join(current_session_history), model

def handle_comment(llm_response, comment, model, settings, use_ollama_api_options, max_tokens_slider_value):
    global history_list  # Declare history_list as a global variable
    global current_session_history  # Declare current_session_history as a global variable

    if not comment:
        return llm_response, "\n".join(current_session_history)

    roles = load_roles('agent_roles.json', 'custom_agent_roles.json', settings)
    role = "User"  # Assuming user role for comment
    role_description = roles.get(role, {}).get("description", "Unknown Role")
    role_settings = roles.get(role, {}).get("ollama_api_options", {})

    prompt = f"LLM Response: {llm_response}\n\nUser Comment: {comment}\n\nRole: {role}\nDescription: {role_description}\n"

    if use_ollama_api_options:
        ollama_api_options = settings.get("ollama_api_options", {})
        ollama_api_options.update(role_settings)
    else:
        ollama_api_options = {}

    response = get_llm_response(role, prompt, model, [], max_tokens_slider_value, None, comment, None, max_tokens_slider_value, None, "Off", ollama_api_options)
    history_list.append(f"User Comment: {comment}\nResponse: {response}\n")
    current_session_history.append(f"User Comment: {comment}\nResponse: {response}\n")
    history_list = history.add_to_history(history_list, f"User Comment: {comment}\nResponse: {response}\n")
    return response, "\n".join(current_session_history)

# Part 3: Load Settings and Initialize Components
# Load models, limiters, and settings
models = load_models('models.json')
model_names_with_vision = [f"{m['name']} (VISION)" if m['vision'] else m['name'] for m in models]
limiters = load_limiters('limiters.json')
settings = load_settings('settings.json')

# Initialize history
history_list = history.load_history()  # Load history from file
current_session_history = []  # Initialize current session history as empty

# Load default and custom agent roles
roles = load_roles('agent_roles.json', 'custom_agent_roles.json', settings)

# Part 4: Gradio Interface Setup
with gr.Blocks(title="ArtAgents") as demo:
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
                        max_tokens = gr.Slider(50, settings.get("max_tokens_slider", 1500), step=10, value=settings.get("max_tokens_slider", 1500) // 2, label="Max Tokens")
                        using_default_agents = gr.Checkbox(label="Using Default Agents", value=settings.get("using_default_agents", False))
                        using_custom_agents = gr.Checkbox(label="Using Custom Agents", value=settings.get("using_custom_agents", False))
                        use_ollama_api_options = gr.Checkbox(label="Use Ollama API Options", value=settings.get("use_ollama_api_options", False))  # Single instance of checkbox

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
                comment_input = gr.Textbox(label="Comment", lines=2)
                comment_button = gr.Button("Comment")
                gr.Markdown("sandner.art | [Creative AI/ML Research](https://github.com/sandner-art)")
                current_session_history_display = gr.Textbox(label="History", lines=15, value="")  # Initialize as empty
                clear_button = gr.Button("Clear")  # Add Clear button
        is_user_adjusted = gr.State(value=False)
        model_state = gr.State(value=None)  # Add a state to store the model

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

        def update_role_dropdown(using_default_agents, using_custom_agents):
            roles = {}
            if using_default_agents:
                with open('agent_roles.json', 'r') as file:
                    roles.update(json.load(file))
            if using_custom_agents:
                with open('custom_agent_roles.json', 'r') as file:
                    roles.update(json.load(file))
            role_names = list(roles.keys())
            return gr.update(choices=role_names, value=role_names[0] if role_names else None)

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

        def chat_with_model(folder_path, role, user_input, model_with_vision, max_tokens, file_handling_option, limiters_handling_option, single_image, settings, use_ollama_api_options):
            response, hist, model = chat(folder_path, role, user_input, model_with_vision, max_tokens, file_handling_option, limiters_handling_option, single_image, settings, use_ollama_api_options)
            current_session_history_display.value = hist  # Update history display
            return response, hist, model

        submit_button.click(
            fn=chat_with_model,
            inputs=[folder_path, role, user_input, model_with_vision, max_tokens, file_handling_option, limiters_handling_option, single_image_display, gr.State(settings), use_ollama_api_options],
            outputs=[llm_response, current_session_history_display, model_state]
        )

        def handle_comment(llm_response, comment, model, settings, use_ollama_api_options, max_tokens_slider_value):
            global history_list  # Declare history_list as a global variable
            global current_session_history  # Declare current_session_history as a global variable

            if not comment:
                return llm_response, "\n".join(current_session_history)

            roles = load_roles('agent_roles.json', 'custom_agent_roles.json', settings)
            role = "User"  # Assuming user role for comment
            role_description = roles.get(role, {}).get("description", "Unknown Role")
            role_settings = roles.get(role, {}).get("ollama_api_options", {})

            prompt = f"LLM Response: {llm_response}\n\nUser Comment: {comment}\n\nRole: {role}\nDescription: {role_description}\n"

            if use_ollama_api_options:
                ollama_api_options = settings.get("ollama_api_options", {})
                ollama_api_options.update(role_settings)
            else:
                ollama_api_options = {}

            response = get_llm_response(role, prompt, model, [], max_tokens_slider_value, None, comment, None, max_tokens_slider_value, None, "Off", ollama_api_options)
            history_list.append(f"User Comment: {comment}\nResponse: {response}\n")
            current_session_history.append(f"User Comment: {comment}\nResponse: {response}\n")
            history_list = history.add_to_history(history_list, f"User Comment: {comment}\nResponse: {response}\n")
            return response, "\n".join(current_session_history)

        comment_button.click(
            fn=handle_comment,
            inputs=[llm_response, comment_input, model_state, gr.State(settings), use_ollama_api_options, max_tokens],
            outputs=[llm_response, current_session_history_display]
        )

        def clear_history():
            global current_session_history  # Declare current_session_history as a global variable
            current_session_history = []
            return ""

        clear_button.click(
            fn=clear_history,
            outputs=[current_session_history_display]
        )

    with gr.Tab("App"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### General Settings")
                ollama_url = gr.Textbox(label="Ollama URL", value=settings.get("ollama_url", ""))
                max_tokens_slider = gr.Slider(label="Max Tokens", minimum=1, maximum=3000, step=1, value=settings.get("max_tokens_slider", 1500))
                ollama_api_prompt_to_console = gr.Checkbox(label="Ollama API Prompt to Console", value=settings.get("ollama_api_prompt_to_console", True))
                using_default_agents = gr.Checkbox(label="Using Default Agents", value=settings.get("using_default_agents", False))
                using_custom_agents = gr.Checkbox(label="Using Custom Agents", value=settings.get("using_custom_agents", False))
                use_ollama_api_options = gr.Checkbox(label="Use Ollama API Options", value=settings.get("use_ollama_api_options", False))
            with gr.Column(scale=1):
                gr.Markdown("### Ollama API Options")
                ollama_api_options_group = gr.Group()
                ollama_api_options_components = []
                with ollama_api_options_group:
                    for key, value in settings.get("ollama_api_options", {}).items():
                        if isinstance(value, bool):
                            component = gr.Checkbox(label=key, value=value)
                        elif isinstance(value, int):
                            component = gr.Slider(label=key, minimum=0, maximum=10000, step=1, value=value)
                        elif isinstance(value, float):
                            component = gr.Slider(label=key, minimum=0.0, maximum=1.0, step=0.01, value=value)
                        else:
                            component = gr.Textbox(label=key, value=value)
                        ollama_api_options_components.append(component)

                save_settings_button = gr.Button("Save Settings")

                def save_settings(ollama_url, max_tokens_slider, ollama_api_prompt_to_console, using_default_agents, using_custom_agents, use_ollama_api_options, *ollama_api_options_values):
                    updated_settings = {
                        "ollama_url": ollama_url,
                        "max_tokens_slider": max_tokens_slider,
                        "ollama_api_prompt_to_console": ollama_api_prompt_to_console,
                        "using_default_agents": using_default_agents,
                        "using_custom_agents": using_custom_agents,
                        "ollama_api_options": {},
                        "use_ollama_api_options": use_ollama_api_options
                    }

                    for key, value in zip(settings.get("ollama_api_options", {}).keys(), ollama_api_options_values):
                        updated_settings["ollama_api_options"][key] = value

                    with open('settings.json', 'w') as file:
                        json.dump(updated_settings, file, indent=4)

                    return "Settings saved successfully."

                save_settings_button.click(
                    fn=save_settings,
                    inputs=[ollama_url, max_tokens_slider, ollama_api_prompt_to_console, using_default_agents, using_custom_agents, use_ollama_api_options] + ollama_api_options_components,
                    outputs=[gr.Textbox(label="Status", lines=1)]
                )

    with gr.Tab("Agent Roles"):
        gr.Markdown("### agent_roles.json")
        agent_roles_html = format_json_to_html_table(load_roles('agent_roles.json', 'custom_agent_roles.json', settings))
        agent_roles_display = gr.HTML(agent_roles_html)

    with gr.Tab("Custom Agent Roles"):
        gr.Markdown("### custom_agent_roles.json")
        custom_agent_roles = load_json('custom_agent_roles.json')
        custom_agent_roles_html = format_json_to_html_table(custom_agent_roles)
        custom_agent_roles_display = gr.HTML(custom_agent_roles_html)

    with gr.Tab("History"):
        gr.Markdown("### Interaction History")
        history_display = gr.Textbox(label="Session History", lines=15, value="\n".join(history_list))  # Load initial history
        confirmation_message = gr.Textbox(label="Confirmation Message", lines=1, value="Do you really want to clear the Session History?", visible=False)
        yes_button = gr.Button("Yes", visible=False)
        no_button = gr.Button("No", visible=False)
        clear_history_button = gr.Button("Clear History")  # Add Clear History button

        def show_confirmation():
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

        def hide_confirmation():
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        def clear_session_history():
            global history_list  # Declare history_list as a global variable
            history_list = []
            history.save_history(history_list)  # Save the cleared history to history.json
            return gr.update(value="\n".join(history_list)), *hide_confirmation()

        clear_history_button.click(
            fn=show_confirmation,
            inputs=[],
            outputs=[confirmation_message, yes_button, no_button]
        )

        yes_button.click(
            fn=clear_session_history,
            inputs=[],
            outputs=[history_display, confirmation_message, yes_button, no_button]
        )

        no_button.click(
            fn=hide_confirmation,
            inputs=[],
            outputs=[confirmation_message, yes_button, no_button]
        )

# Launch the Gradio App
if __name__ == "__main__":
    demo.launch()
