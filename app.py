import gradio as gr
import json
from agent import get_llm_response

def load_models(model_file):
    with open(model_file, 'r') as file:
        return json.load(file)

def chat(role, user_input, model_with_vision, images):
    # Extract the original model name from the dropdown selection
    model = next((m["name"] for m in models if f"{m['name']} (VISION)" == model_with_vision or m["name"] == model_with_vision), None)
    
    if model is None:
        return "Error: Selected model not found."
    
    # Ensure images is always a list
    if images is None:
        images = []
    elif isinstance(images, dict):  # Gradio returns a dictionary for single image uploads
        images = [images['image']]
    elif not isinstance(images, list):  # Ensure images is a list
        images = [images]
    
    # Check if the model supports vision
    model_info = next((m for m in models if m["name"] == model), None)
    if model_info and not model_info["vision"]:
        images = []  # Disable image sending if the model does not support vision
    
    response = get_llm_response(role, user_input, model, images)
    return response

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
        gr.Textbox(label="User Input"),
        gr.Dropdown(model_names_with_vision, label="Select Model"),
        gr.Image(type="pil", label="Upload Image")  # Removed 'optional' here
    ],
    outputs=gr.Textbox(label="LLM Response"),
    title="Agent-Based Chat with Ollama",
    description="Select an agent, model, and provide input to get a response from Ollama. You can also upload an image for multimodal input."
)

if __name__ == "__main__":
    iface.launch()