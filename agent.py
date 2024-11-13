import json
import requests

def load_roles(role_file):
    with open(role_file, 'r') as file:
        return json.load(file)

def get_llm_response(role, user_input, images=None):
    roles = load_roles('agent_roles.json')
    role_description = roles.get(role, "Unknown Role")
    
    # Construct the prompt for the LLM
    prompt = f"Role: {role}\nDescription: {role_description}\nUser Input: {user_input}"
    
    if images:
        # Simulate multimodal LLM response
        image_captions = [f"Caption for image {i+1}: This is a description of the uploaded image {i+1}." for i in range(len(images))]
        prompt += f"\nImage Captions: {', '.join(image_captions)}"
    
    # Define the Ollama API endpoint
    ollama_url = "http://localhost:11434/api/generate"  # Adjust the URL and port as needed
    
    # Define the payload for the API request
    payload = {
        "model": "impactframes/llama3_ifai_sd_prompt_mkr_q4km:latest",  # Replace with your actual model name
        "prompt": prompt,
        "max_tokens": 300  # Adjust the number of tokens as needed
    }
    
    try:
        # Send the request to the Ollama API
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()  # Raise an error for bad responses
        response_data = response.json()
        llm_response = response_data.get('response', "No response from LLM")
    except requests.RequestException as e:
        llm_response = f"Error communicating with Ollama: {e}"
    
    return llm_response