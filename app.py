import gradio as gr
import json
import os
from PIL import Image
from agent import get_llm_response

def load_models(model_file):
    with open(model_file, 'r') as file:
        return json.load(file)

def chat(role, user_input, model_with_vision, file_paths):
    # Extract the original model name from the dropdown selection
    model = next((m["name"] for m in models if f"{m['name']} (VISION)" == model_with_vision or m["name"] == model_with_vision), None)

    if model is None:
        return "Error: Selected model not found."

    # Check if the model supports vision
    model_info = next((m for m in models if m["name"] == model), None)
    if model_info and not model_info["vision"]:
        return "Error: Selected model does not support vision."

    # Ensure file_paths is a list
    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    # Get the directory of the first file
    image_dir = os.path.dirname(file_paths[0])

    # Initialize a list to store confirmation messages
    confirmation_messages = []

    for file_path in file_paths:
        if not os.path.isfile(file_path):
            return f"Error: {file_path} is not a valid file."

        # Check if the file is an image
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            return f"Error: {file_path} is not a valid image file."

        # Open the image
        image = Image.open(file_path)
        response = get_llm_response(role, user_input, model, [image])

        # Create a separate .txt file for each image in the same directory
        base_name = os.path.splitext(os.path.basename(file_path))[0]  # Extract the base name without extension
        output_file = os.path.join(image_dir, f"{base_name}.txt")
        with open(output_file, 'w') as f:
            f.write(response)

        confirmation_messages.append(f"Caption for {os.path.basename(file_path)} saved to {output_file}\n")

    return "\n".join(confirmation_messages)

# Load roles for dropdown
with open('agent_roles.json', 'r') as file:
    roles = json.load(file)
    role_names = list(roles.keys())

# Load models for dropdown
models = load_models('models.json')
model_names_with_vision = [f"{m['name']} (VISION)" if m['vision'] else m['name'] for m in models]

# Create Gradio interface
iface = gr.Interface(
    fn=chat,
    inputs=[
        gr.Dropdown(role_names, label="Select Agent"),
        gr.Textbox(label="User Input", lines=2), # Set the height of the user input Textbox
        gr.Dropdown(model_names_with_vision, label="Select Model"),
        gr.File(label="Upload Images", file_count="directory")  # Changed to folder input
    ],
    outputs=gr.Textbox(label="LLM Response", lines=3), # Set the height of the LLM Response
    title="sandner.art | Agent-Based Chat with Ollama",
    description="Select an agent, model, and provide input to get a response from Ollama. You can also upload a folder of images for multimodal input.",
    flagging_mode="never"  # Disable flagging
)

if __name__ == "__main__":
    iface.launch()