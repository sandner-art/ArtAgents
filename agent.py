import json
import requests
from PIL import Image
import io
import base64

def load_roles(role_file):
    with open(role_file, 'r') as file:
        return json.load(file)

def get_llm_response(role, prompt, model, images=None, max_tokens=200):
    roles = load_roles('agent_roles.json')
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
    
    # Define the Ollama API endpoint
    ollama_url = "http://localhost:11434/api/generate"  # Ensure this is the correct endpoint
    
    # Define the payload for the API request
    payload = {
        "model": model,  # Use the selected model
        "prompt": prompt,
        "max_tokens": max_tokens  # Use the specified max tokens
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