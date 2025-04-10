# ArtAgent/agents/ollama_agent.py

import json
import requests
from PIL import Image # Keep if image processing happens here
import io             # Keep if image processing happens here
import base64         # Keep if image processing happens here
import numpy as np    # Keep if image processing happens here (less likely now)

# Removed internal load_settings/load_roles - Assume these are passed in
# from core.utils import load_json # Not needed if settings/roles passed

def get_llm_response(
    role: str,
    prompt: str,
    model: str,
    settings: dict,         # Pass settings dict
    roles_data: dict,       # Pass loaded roles dict
    images: list = None,    # List of PIL Image objects
    max_tokens: int = 1500, # Still useful as fallback for num_predict
    # Optional args removed for clarity, add back if needed by specific logic:
    # file_path=None, user_input=None, model_with_vision=None, num_predict=None,
    # single_image=None, limiters_handling_option=None,
    ollama_api_options: dict = None # Allow direct override
    ) -> str:
    """
    Sends a request to the Ollama API and returns the response.

    Args:
        role (str): The selected agent role name.
        prompt (str): The constructed prompt for the LLM.
        model (str): The name of the Ollama model to use.
        settings (dict): The application's settings dictionary.
        roles_data (dict): The loaded dictionary of all available agent roles.
        images (list, optional): A list of PIL Image objects. Defaults to None.
        max_tokens (int, optional): Fallback for num_predict if not specified elsewhere. Defaults to 1500.
        ollama_api_options (dict, optional): Options to directly override/merge settings. Defaults to None.

    Returns:
        str: The LLM response string, or an error message string.
    """

    ollama_url = settings.get("ollama_url", "http://localhost:11434/api/generate")
    ollama_api_prompt_to_console = settings.get("ollama_api_prompt_to_console", True)

    # --- Get Role Info ---
    # Role description might be included in the prompt *before* calling this function now
    # role_description = roles_data.get(role, {}).get("description", "Unknown Role")

    # --- Merge Ollama API Options ---
    # Priority: Direct call > Role-specific > Global Settings
    effective_options = settings.get("ollama_api_options", {}).copy() # Start with global
    role_settings = roles_data.get(role, {}).get("ollama_api_options", {})
    effective_options.update(role_settings) # Apply role settings
    if ollama_api_options: # Apply direct overrides
        effective_options.update(ollama_api_options)

    # Ensure num_predict is set (using max_tokens as fallback)
    if "num_predict" not in effective_options:
        effective_options["num_predict"] = max_tokens

    # --- Log details if enabled ---
    if ollama_api_prompt_to_console:
        try:
            log_details = {
                "model": model,
                "prompt_length": len(prompt),
                # Avoid logging full prompt if too long or sensitive
                "prompt_start": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "images_count": len(images) if images else 0,
                "effective_options": effective_options,
            }
            print("--- Ollama Request ---")
            print(json.dumps(log_details, indent=2))
            print("----------------------")
        except Exception as log_e:
            print(f"Error logging request details: {log_e}")


    # --- Define Payload ---
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True, # Use streaming
        "options": effective_options
    }

    # --- Handle Images ---
    if images:
        image_data = []
        try:
            print(f"Processing {len(images)} image(s) for payload...")
            for i, img_object in enumerate(images):
                if isinstance(img_object, Image.Image): # Check if it's a PIL Image
                    buffered = io.BytesIO()
                    # Choose format (JPEG is often smaller, PNG supports transparency)
                    save_format = "JPEG" if img_object.mode != "RGBA" else "PNG"
                    img_object.save(buffered, format=save_format)
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    image_data.append(img_str)
                    print(f"  Processed image {i+1} ({save_format}, size: {len(img_str)} bytes)")
                else:
                    print(f"Warning: Item {i+1} in images list is not a PIL Image object, skipping.")
            if image_data:
                payload["images"] = image_data
            else:
                 print("Warning: Image list provided but no valid PIL Images found/processed.")
        except Exception as img_e:
            print(f"Error processing image for Ollama payload: {img_e}")
            # Return error immediately if image processing fails critically
            return f"⚠️ Error: Failed to process image data. Details: {img_e}"

    print(f"DEBUG: Sending Payload to Ollama:\n{json.dumps(payload, indent=2)}") # Print the actual payload

    # --- Perform Request ---
    llm_response = ""
    try:
        print(f"Sending request to Ollama model '{model}' at {ollama_url}...")
        response = requests.post(ollama_url, json=payload, stream=True, timeout=180) # Longer timeout for generation
        response.raise_for_status() # Raise HTTP errors (4xx, 5xx)

        print("Connection successful. Receiving stream...")
        complete_response = ""
        chunk_count = 0
        for chunk in response.iter_lines(decode_unicode=True):
            chunk_count += 1
            if chunk:
                # print(f"Chunk {chunk_count}: {chunk[:80]}...") # Verbose chunk logging
                try:
                    chunk_data = json.loads(chunk)
                    complete_response += chunk_data.get('response', "")
                    if chunk_data.get("done"):
                        print(f"Stream finished (done=true received after {chunk_count} chunks).")
                        # Optional: log context length, eval duration etc. if present
                        final_context = chunk_data.get('context')
                        if final_context:
                            print(f"  Final context length: {len(final_context)}")
                        break # Exit loop once done signal is received
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON chunk {chunk_count}: {e}\nChunk: {chunk}")
                    complete_response += f"\n\n[Error decoding stream chunk: {e}]"
                    break # Stop processing after error

        llm_response = complete_response if complete_response else "No response text received from LLM stream."

    except requests.exceptions.Timeout:
         print(f"Error: Ollama request timed out connecting to or streaming from {ollama_url}")
         llm_response = f"⚠️ Error: Request to Ollama timed out. The server might be busy, unresponsive, or the generation took too long."
    except requests.exceptions.ConnectionError:
         print(f"Error: Could not connect to Ollama at {ollama_url}. Is it running?")
         llm_response = f"⚠️ Error: Could not connect to Ollama at {ollama_url}. Please ensure the Ollama service is running."
    except requests.exceptions.RequestException as e:
         print(f"Error communicating with Ollama: {e}")
         # Attempt to get more detail from the response if possible
         error_detail = str(e)
         try:
              if e.response is not None:
                   error_detail += f" | Response Status: {e.response.status_code} | Response Text: {e.response.text[:500]}"
         except Exception:
              pass # Ignore errors trying to get more detail
         llm_response = f"⚠️ Error communicating with Ollama: {error_detail}"

    # print(f"LLM Response length: {len(llm_response)}")
    return llm_response