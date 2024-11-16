import json
import requests
from PIL import Image
import io
import base64
import numpy as np

def load_roles(default_role_file, custom_role_file, settings):
    roles = {}
    if settings.get("using_custom_agents", False):
        with open(custom_role_file, 'r') as file:
            roles.update(json.load(file))
    if settings.get("using_default_agents", False):
        with open(default_role_file, 'r') as file:
            roles.update(json.load(file))
    return roles

def load_settings(settings_file):
    with open(settings_file, 'r') as file:
        return json.load(file)

def get_llm_response(role, prompt, model, images=None, max_tokens=1500, file_path=None, user_input=None, model_with_vision=None, num_predict=None, single_image=None, limiters_handling_option=None):
    settings = load_settings('settings.json')
    roles = load_roles('agent_roles.json', 'custom_agent_roles.json', settings)
    role_description = roles.get(role, "Unknown Role")
    
    # Construct the prompt for the LLM
    if images:
        # Convert images to base64
        image_captions = []
        image_data = []
        for i, image in enumerate(images):
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_data.append(img_str)
            image_captions.append(f"Caption for image {i+1}: This is a description of the uploaded image {i+1}.")
        
        prompt += f"\nImage Captions: {', '.join(image_captions)}"
    
    # Load settings
    ollama_url = settings.get("ollama_url", "http://localhost:11434/api/generate")
    ollama_api_prompt_to_console = settings.get("ollama_api_prompt_to_console", True)
    ollama_api_options = settings.get("ollama_api_options", {})
    
    # Log the full prompt and other details to the console
    if ollama_api_prompt_to_console:
        log_details = {
            "model": model,
            "temperature": 0.7,  # Assuming a default temperature value
            "num_predict": max_tokens,  # Use max_tokens as num_predict
            "image_file": "single_image" if isinstance(single_image, np.ndarray) else None,
            "text_caption": file_path if file_path else None,
            "prompt": prompt
        }
        print(json.dumps(log_details, indent=4))
    
    # Define the payload for the API request
    payload = {
        "model": model,  # Use the selected model
        "prompt": prompt,
        "max_tokens": max_tokens,  # Use the specified max tokens
        "options": ollama_api_options
    }
    
    if images:
        payload["images"] = image_data
    
    try:
        # Send the request to the Ollama API with stream=True
        response = requests.post(ollama_url, json=payload, stream=True)
        response.raise_for_status()  # Raise an error for bad responses
        
        # Initialize an empty string to store the complete response
        complete_response = ""
        
        # Process each chunk in the response
        for chunk in response.iter_lines(decode_unicode=True):
            if chunk:
                try:
                    # Parse the JSON chunk
                    chunk_data = json.loads(chunk)
                    # Append the response text from the chunk to the complete response
                    complete_response += chunk_data.get('response', "")
                except json.JSONDecodeError as e:
                    return f"Error decoding JSON chunk: {e}\nChunk: {chunk}"
        
        llm_response = complete_response if complete_response else "No response from LLM"
    
    except requests.RequestException as e:
        llm_response = f"Error communicating with Ollama: {e}"
    
    return llm_response