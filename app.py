import gradio as gr
import json
from agent import get_llm_response

def chat(role, user_input, model, images):
    # Check if images are uploaded
    if images is None:
        images = []
    elif isinstance(images, dict):  # Gradio returns a dictionary for single image uploads
        images = [images['image']]
    elif isinstance(images, list):  # Gradio returns a list for multiple image uploads
        images = [image['image'] for image in images]
    
    response = get_llm_response(role, user_input, model, images)
    return response

# Load roles for dropdown
with open('agent_roles.json', 'r') as file:
    roles = json.load(file)
    role_names = list(roles.keys())

# Define available models
model_names = [
    "impactframes/llama3_ifai_sd_prompt_mkr_q4km:latest",
    "llava"
]

# Create Gradio interface
iface = gr.Interface(
    fn=chat,
    inputs=[
        gr.Dropdown(role_names, label="Select Agent"),
        gr.Textbox(label="User Input"),
        gr.Dropdown(model_names, label="Select Model"),
        gr.Image(type="pil", label="Upload Image")  # Removed 'optional' here
    ],
    outputs=gr.Textbox(label="LLM Response"),
    title="Agent-Based Chat with Ollama",
    description="Select an agent, model, and provide input to get a response from Ollama. You can also upload an image for multimodal input."
)

if __name__ == "__main__":
    iface.launch()