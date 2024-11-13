import gradio as gr
import json
from agent import get_llm_response

def chat(role, user_input, images):
    # Check if images are uploaded
    if images is None:
        images = []
    elif isinstance(images, dict):  # Gradio returns a dictionary for single image uploads
        images = [images['name']]
    elif isinstance(images, list):  # Gradio returns a list for multiple image uploads
        images = [image['name'] for image in images]
    
    response = get_llm_response(role, user_input, images)
    return response

# Load roles for dropdown
with open('agent_roles.json', 'r') as file:
    roles = json.load(file)
    role_names = list(roles.keys())

# Create Gradio interface
iface = gr.Interface(
    fn=chat,
    inputs=[
        gr.Dropdown(role_names, label="Select Agent"),
        gr.Textbox(label="User Input"),
        gr.Image(type="pil", label="Upload Image")  # Removed 'optional' here
    ],
    outputs=gr.Textbox(label="LLM Response"),
    title="Agent-Based Chat with LLM",
    description="Select an agent and provide input to get a response from the LLM. You can also upload an image for multimodal input."
)

if __name__ == "__main__":
    iface.launch()