import gradio as gr
import json
import os
from PIL import Image
from agent import get_llm_response, load_roles

def load_models(model_file):
    with open(model_file, 'r') as file:
        return json.load(file)

def chat(role, user_input, model_with_vision, input_type, folder_path, file_path, max_tokens, overwrite, skip, append, prepend):
    # Extract the original model name from the dropdown selection
    model = next((m["name"] for m in models if f"{m['name']} (VISION)" == model_with_vision or m["name"] == model_with_vision), None)

    if model is None:
        return "Error: Selected model not found."

    # Check if the model supports vision
    model_info = next((m for m in models if m["name"] == model), None)
    if model_info and not model_info["vision"]:
        return "Error: Selected model does not support vision."

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
            if skip:
                confirmation_messages.append(f"Skipped existing file: {output_file}\n")
                return
            elif overwrite:
                with open(output_file, 'w') as f:
                    f.write(response)
                confirmation_messages.append(f"Overwrote existing file: {output_file}\n")
            elif append:
                with open(output_file, 'a') as f:
                    f.write("\n" + response)
                confirmation_messages.append(f"Appended to existing file: {output_file}\n")
            elif prepend:
                with open(output_file, 'r') as f:
                    existing_content = f.read()
                with open(output_file, 'w') as f:
                    f.write(response + "\n" + existing_content)
                confirmation_messages.append(f"Prepended to existing file: {output_file}\n")
        else:
            with open(output_file, 'w') as f:
                f.write(response)
            confirmation_messages.append(f"Created new file: {output_file}\n")

    if input_type == "Folder":
        # Ensure folder_path is a valid directory
        if not os.path.isdir(folder_path):
            return f"Error: {folder_path} is not a valid directory."

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

    elif input_type == "Single Image":
        # Ensure file_path is a valid file
        if not os.path.isfile(file_path):
            return f"Error: {file_path} is not a valid file."

        # Check if the file is an image
        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            return f"Error: {file_path} is not a valid image file."

        # Open the image
        image = Image.open(file_path)
        process_image(image, file_path)

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
    gr.Markdown("Select an agent, model, and provide input to get a response from Ollama. You can also provide a folder path of images or a single image file for multimodal input.")
    
    with gr.Row():
        with gr.Column():
            input_type = gr.Radio(["Folder", "Single Image"], label="Input Type", value="Folder")
            with gr.Group(visible=True) as folder_options:
                folder_path = gr.Textbox(label="Folder Path")
                overwrite = gr.Checkbox(label="Overwrite Existing .txt Files", value=False)
                skip = gr.Checkbox(label="Skip Existing .txt Files", value=True)
                append = gr.Checkbox(label="Append to Existing .txt Files", value=False)
                prepend = gr.Checkbox(label="Prepend to Existing .txt Files", value=False)
            with gr.Group(visible=False) as single_image_options:
                file_path = gr.File(label="Upload Image", file_count="single")
                image_display = gr.Image(label="Uploaded Image")
            role = gr.Dropdown(role_names, label="Select Agent")
            user_input = gr.Textbox(label="User Input", lines=2)
            model_with_vision = gr.Dropdown(model_names_with_vision, label="Select Model")
            max_tokens = gr.Slider(50, 1500, step=10, value=1500, label="Max Tokens")
        
        with gr.Column():
            llm_response = gr.Textbox(label="LLM Response", lines=3)
    
    def update_input_fields(input_type):
        folder_options.visible = input_type == "Folder"
        single_image_options.visible = input_type == "Single Image"
        return folder_options, single_image_options
    
    input_type.change(update_input_fields, inputs=input_type, outputs=[folder_options, single_image_options])
    
    def update_image_display(file_path):
        if file_path:
            return gr.Image.update(value=file_path, visible=True)
        else:
            return gr.Image.update(value=None, visible=False)
    
    file_path.change(update_image_display, inputs=file_path, outputs=image_display)
    
    submit_button = gr.Button("Submit")
    submit_button.click(chat, inputs=[role, user_input, model_with_vision, input_type, folder_path, file_path, max_tokens, overwrite, skip, append, prepend], outputs=llm_response)

if __name__ == "__main__":
    demo.launch()