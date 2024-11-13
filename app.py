import gradio as gr
import json
import os
from PIL import Image
from agent import get_llm_response, load_roles

def load_models(model_file):
    with open(model_file, 'r') as file:
        return json.load(file)

def chat(folder_path, role, user_input, model_with_vision, max_tokens, file_handling_option):
    # Extract the original model name from the dropdown selection
    model = next((m["name"] for m in models if f"{m['name']} (VISION)" == model_with_vision or m["name"] == model_with_vision), None)

    if model is None:
        return "Error: Selected model not found."

    # Check if the model supports vision
    model_info = next((m for m in models if m["name"] == model), None)
    if model_info is None:
        return "Error: Model information not found."

    # Initialize a list to store confirmation messages
    confirmation_messages = []

    def process_image(image, file_path):
        roles = load_roles('agent_roles.json')
        role_description = roles.get(role, "Unknown Role")

        # Construct the prompt for the LLM
        prompt = f"User Input: {user_input}\n\nRole: {role}\nDescription: {role_description}"

        response = get_llm_response(role, prompt, model, [image], max_tokens)

        # Create or update the .txt file based on the selected options
        base_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract the base name without extension
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

    # Check if the folder path is empty or not a valid directory
    if not folder_path.strip() or not os.path.isdir(folder_path):
        roles = load_roles('agent_roles.json')
        role_description = roles.get(role, "Unknown Role")

        # Construct the prompt for the LLM without images
        prompt = f"User Input: {user_input}\n\nRole: {role}\nDescription: {role_description}"

        response = get_llm_response(role, prompt, model, [], max_tokens)

        return response

    # If the model does not support vision, process only text inputs
    if not model_info["vision"]:
        roles = load_roles('agent_roles.json')
        role_description = roles.get(role, "Unknown Role")

        # Construct the prompt for the LLM without images
        prompt = f"User Input: {user_input}\n\nRole: {role}\nDescription: {role_description}"

        response = get_llm_response(role, prompt, model, [], max_tokens)

        return response

    # Iterate through all files in the directory
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if the file is an image
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Open the image
            image = Image.open(file_path)
            process_image(image, file_path)

    if not confirmation_messages:
        return "No valid image files found in the directory."

    return "\n".join(confirmation_messages)

# Load roles for dropdown
with open('agent_roles.json', 'r') as file:
    roles = json.load(file)
    role_names = list(roles.keys())

# Load models for dropdown
models = load_models('models.json')
model_names_with_vision = [f"{m['name']} (VISION)" if m['vision'] else m['name'] for m in models]

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# sandner.art | Agent-Based Chat with Ollama")
    gr.Markdown("Select an agent, model, and provide input to get a response from Ollama. You can provide a folder path of images for multimodal input.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Folder Input")
            folder_path = gr.Textbox(label="Folder Path")
            file_handling_option = gr.Radio(["Overwrite", "Skip", "Append", "Prepend"], label="File Handling", value="Skip")

        with gr.Column(scale=1):
            gr.Markdown("### Common Inputs")
            role = gr.Dropdown(role_names, label="Select Agent")
            user_input = gr.Textbox(label="User Input", lines=2)
            model_with_vision = gr.Dropdown(model_names_with_vision, label="Select Model")
            max_tokens = gr.Slider(50, 1500, step=10, value=1500, label="Max Tokens")
            submit_button = gr.Button("Submit")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### LLM Response")
            llm_response = gr.Textbox(label="LLM Response", lines=10)

    submit_button.click(
        chat,
        inputs=[folder_path, role, user_input, model_with_vision, max_tokens, file_handling_option],
        outputs=llm_response
    )

if __name__ == "__main__":
    demo.launch()