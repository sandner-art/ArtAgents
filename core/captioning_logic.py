# ArtAgent/core/captioning_logic.py
import os
import gradio as gr
from PIL import Image # Keep PIL needed for checking image files
import time
import numpy as np # Import numpy if conversion is needed

# Import utilities and core components
from .utils import get_absolute_path, load_json
from .app_logic import execute_chat_or_team # Import the router function
from . import history_manager as history # For logging

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')

# --- Existing functions: load_images_and_captions, update_caption_display, save_caption, batch_edit_captions ---
def load_images_and_captions(folder_path: str):
    """
    Loads image filenames and their corresponding caption file contents from a folder.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        tuple: Contains:
            - list[str]: Sorted list of image filenames (for CheckboxGroup choices).
            - dict[str, str]: Mapping of image filename to its absolute path.
            - dict[str, str]: Mapping of image filename to its caption text.
            - str: Status message.
            - str | None: First image filename (for selected item state & display).
            - str | None: Caption of the first image.
            - str | None: First image filename again (for filename display textbox).
    """
    captions = {}
    image_paths = {}
    image_filenames = []
    status = ""

    # Define default return values (7 items) for error cases
    empty_return = [], {}, {}, "Error: No folder specified.", None, None, None

    if not folder_path:
        return empty_return
    if not os.path.isdir(folder_path):
        status = f"Error: Invalid folder path provided: '{folder_path}'."
        print(status)
        return [], {}, {}, status, None, None, None

    print(f"Loading captions from folder: {folder_path}")
    found_count = 0
    try:
        # Iterate through sorted directory listing
        for filename in sorted(os.listdir(folder_path)):
            name, ext = os.path.splitext(filename)
            if ext.lower() in IMAGE_EXTENSIONS:
                image_abs_path = os.path.join(folder_path, filename)
                # Ensure it's actually a file, not a directory with an image extension name
                if os.path.isfile(image_abs_path):
                    image_filenames.append(filename) # Use filename as identifier
                    image_paths[filename] = image_abs_path

                    # Look for corresponding .txt file
                    text_filename = name + ".txt"
                    text_path = os.path.join(folder_path, text_filename)
                    caption_text = "" # Default to empty if no file or read error
                    if os.path.exists(text_path):
                        try:
                            with open(text_path, 'r', encoding='utf-8') as f:
                                caption_text = f.read()
                        except Exception as e:
                            print(f"Warning: Could not read caption file {text_filename}: {e}")
                            # Optionally include error in caption? For now, just empty.
                            # caption_text = f"Error reading caption: {e}"
                    captions[filename] = caption_text
                    found_count += 1

        if found_count == 0:
            status = "No supported image files found in the specified folder."
        else:
            status = f"Loaded {found_count} image(s)."
        print(status)

    except Exception as e:
        status = f"Error reading folder contents: {e}"
        print(status)
        # Return empty values on error, matching the 7-item signature
        return [], {}, {}, status, None, None, None

    # Prepare initial selection data
    first_image = image_filenames[0] if image_filenames else None
    first_caption = captions.get(first_image, "") if first_image else ""

    # Return 7 values matching the expected outputs in app.py
    # Outputs: image_selector_choices, image_paths_state, caption_data_state, status_display, selected_item_state, caption_display, selected_filename_display
    return image_filenames, image_paths, captions, status, first_image, first_caption, first_image


def update_caption_display(
    selected_items: list | str | None, # Input from CheckboxGroup/Selector
    caption_data_dict: dict,         # Input from caption_data_state
    image_paths_dict: dict           # Input from image_paths_state
    ) -> tuple[str, str | None, str | None, str | None]: # Return 4 values
    """
    Retrieves caption text AND image path for the selected image filename.
    Handles input potentially being a list from CheckboxGroup.

    Returns:
        tuple: (caption_text, selected_filename_state, selected_filename_display, image_path_for_preview)
    """
    selected_filename = None
    # Handle list input from CheckboxGroup
    if isinstance(selected_items, list):
        selected_filename = selected_items[0] if selected_items else None
        if selected_items and len(selected_items) > 1:
             print(f"Info: Multiple images selected ({len(selected_items)}), displaying data for first: {selected_filename}")
    elif isinstance(selected_items, str):
         selected_filename = selected_items # Handle single string if needed

    image_path_for_preview = None
    caption_text = ""
    filename_display = selected_filename # Default display name

    if not selected_filename or not isinstance(caption_data_dict, dict) or not isinstance(image_paths_dict, dict):
        print("Update caption display: No valid selection or required data dictionaries missing.")
        # Return 4 values for clearing outputs
        return "", None, None, None
    else:
        # Use .get() for safer dictionary access
        caption_text = caption_data_dict.get(selected_filename, f"Caption data not found for '{selected_filename}'.")
        image_path_for_preview = image_paths_dict.get(selected_filename) # Get full path using the key
        if not image_path_for_preview or not os.path.isfile(image_path_for_preview):
             print(f"Warning: Image path not found or invalid for selected file: {selected_filename}")
             filename_display = f"{selected_filename} (Image Path Error)" # Indicate error
             image_path_for_preview = None # Don't try to show preview if path is bad

        print(f"Displaying caption and preview for: {selected_filename}")

    # Outputs: caption_display, selected_item_state, selected_filename_display, image_preview
    return caption_text, selected_filename, filename_display, image_path_for_preview


def save_caption(
    selected_filename: str | None, # Filename from state
    caption_text: str,
    image_paths_dict: dict, # Map filename -> full path
    caption_data_dict: dict # Map filename -> caption (to update state)
    ) -> tuple[str, dict]: # Return status string and updated captions dict
    """
    Saves the edited caption text to the corresponding .txt file.

    Returns:
        tuple: (status_message, updated_caption_data_dict)
    """
    if not selected_filename:
        return "Error: No image selected to save caption for.", caption_data_dict

    if not isinstance(image_paths_dict, dict):
        return "Error: Image path data is missing or invalid.", caption_data_dict

    image_path = image_paths_dict.get(selected_filename)
    if not image_path:
        return f"Error: Could not find original path for '{selected_filename}'.", caption_data_dict

    # Ensure caption_data_dict is a dict, create new if not
    updated_captions = caption_data_dict.copy() if isinstance(caption_data_dict, dict) else {}

    try:
        folder_path = os.path.dirname(image_path)
        base_name = os.path.splitext(selected_filename)[0]
        text_path = os.path.join(folder_path, base_name + ".txt")

        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(caption_text)
        status = f"Caption saved successfully to {base_name}.txt"
        print(status)
        # Update the caption in the state dictionary as well
        updated_captions[selected_filename] = caption_text
        return status, updated_captions # Return status and updated state dict
    except Exception as e:
        status = f"Error saving caption to '{text_path}': {e}"
        print(status)
        # Return error and original state dict (or the copy if created)
        return status, (caption_data_dict if isinstance(caption_data_dict, dict) else {})


def batch_edit_captions(
    selected_filenames: list | None, # List from CheckboxGroup (can be None)
    text_to_add: str,
    mode: str, # "Append" or "Prepend"
    image_paths_dict: dict,
    caption_data_dict: dict
    ) -> tuple[str, dict]: # Return status string and updated captions dict
    """
    Appends or prepends text to the captions of selected images.

    Returns:
        tuple: (status_message, updated_caption_data_dict)
    """
    if not selected_filenames: # Check if list is empty or None
        return "No images selected for batch operation.", caption_data_dict
    # Allow adding empty text? If not, uncomment below
    # if not text_to_add:
    #     return "No text provided to add.", caption_data_dict
    if mode not in ["Append", "Prepend"]:
        return f"Invalid batch mode: {mode}", caption_data_dict
    if not isinstance(image_paths_dict, dict) or not isinstance(caption_data_dict, dict):
        return "Error: Image path or caption data is missing or invalid.", caption_data_dict


    print(f"Batch {mode.lower()}ing text to {len(selected_filenames)} selected captions...")
    # Work on a copy of the captions dictionary
    updated_captions = caption_data_dict.copy()
    processed_count = 0
    error_count = 0
    skipped_count = 0
    messages = []

    for filename in selected_filenames:
        image_path = image_paths_dict.get(filename)
        if not image_path:
            messages.append(f"- Skipped {filename}: Path not found.")
            skipped_count += 1
            continue

        # Use caption from the updated_captions dict to handle intermediate changes within the batch
        current_caption = updated_captions.get(filename, "")

        if mode == "Append":
            new_caption = current_caption + text_to_add
        else: # Prepend
            new_caption = text_to_add + current_caption

        try:
            folder_path = os.path.dirname(image_path)
            base_name = os.path.splitext(filename)[0]
            text_path = os.path.join(folder_path, base_name + ".txt")

            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(new_caption)
            updated_captions[filename] = new_caption # Update in our state dict copy
            processed_count += 1
        except Exception as e:
            messages.append(f"- Error {mode.lower()}ing {filename}: {e}")
            error_count += 1

    status = f"Batch {mode.lower()} complete. Processed: {processed_count}, Errors: {error_count}, Skipped: {skipped_count}."
    if messages:
        status += "\nDetails:\n" + "\n".join(messages)
    print(status)

    # Return status and the fully updated captions dictionary
    return status, updated_captions

# --- NEW Functions for Agent Caption Generation ---

# CORRECTED Signature: Added selected_model_display_name
def generate_captions_for_selected(
    selected_filenames: list | None,
    agent_or_team_display_name: str,
    selected_model_display_name: str, # <<< ADDED
    generate_mode: str, # Overwrite, Skip, Append, Prepend
    image_paths: dict,
    current_captions: dict,
    settings: dict,
    models_data: list,
    limiters_data: dict,
    teams_data: dict,
    file_agents: dict,
    history_list: list,
    session_history: list
    ) -> tuple[str, dict, str, list]:
    """
    Generates captions for selected images using an agent/team and a specified model.

    Returns:
        tuple: (status_message, updated_captions_dict, last_caption_generated, updated_session_history)
    """
    print("\n--- Running: Generate Captions for Selected ---")
    start_time = time.time()

    if not selected_filenames:
        return "No images selected to generate captions for.", current_captions, "", session_history
    if not agent_or_team_display_name or agent_or_team_display_name == "(Direct Agent Call)":
         return "Please select a valid Agent or Team for captioning.", current_captions, "", session_history
    # --- ADDED: Check if a vision model was actually selected ---
    if not selected_model_display_name:
         return "Please select a Vision Model for captioning.", current_captions, "", session_history
    # --- End Added Check ---

    # --- Vision Capability Check (Simplified - relies on model being passed now) ---
    is_likely_vision_agent = (
        any(tag in agent_or_team_display_name.lower() for tag in ['llava', 'vision', 'photographer', 'captioner']) or # Added more keywords
        agent_or_team_display_name.startswith("[Team]")
    )
    if not is_likely_vision_agent:
        print(f"Warning: Agent/Team '{agent_or_team_display_name}' selected, but ensure the chosen model '{selected_model_display_name}' is appropriate for image captioning.")

    status_messages = []
    updated_captions = current_captions.copy()
    processed_count = 0
    error_count = 0
    skipped_count = 0
    last_caption_generated = "" # Store the last successful one for display

    current_session_history = list(session_history)
    fixed_prompt = "Describe this image concisely. Focus on the main subject, action, and setting. Do not add commentary."

    for filename in selected_filenames:
        print(f"  Processing: {filename}")
        image_path = image_paths.get(filename)
        if not image_path or not os.path.isfile(image_path):
            msg = f"- Skipped {filename}: Image path not found or invalid."
            print(f"    {msg}")
            status_messages.append(msg)
            skipped_count += 1
            continue

        base_name = os.path.splitext(filename)[0]
        text_filename = base_name + ".txt"
        text_path = os.path.join(os.path.dirname(image_path), text_filename)
        caption_exists = os.path.exists(text_path)
        original_caption = updated_captions.get(filename, "")

        if caption_exists and generate_mode == "Skip":
            msg = f"- Skipped {filename}: Caption file '{text_filename}' already exists and mode is Skip."
            print(f"    {msg}")
            status_messages.append(msg)
            skipped_count += 1
            continue

        img = None # Initialize img variable
        try:
            img = Image.open(image_path)
            # Optional: Convert image if needed
            # if img.mode == 'RGBA' or img.mode == 'P':
            #      img = img.convert('RGB')
            #      print(f"    Converted image {filename} to RGB mode.")

            print(f"    Calling agent/team '{agent_or_team_display_name}' with model '{selected_model_display_name}' for {filename}...")

            # --- PASS selected_model_display_name to execute_chat_or_team ---
            # Note: The execute_chat_or_team -> chat_logic path expects `single_image_np`.
            # However, the underlying ollama_agent expects PIL. We pass PIL here.
            # If this causes issues, chat_logic might need adjustment to accept PIL
            # or we convert to numpy here: img_np = np.array(img)
            response_text, _, _, updated_session_history_list = execute_chat_or_team(
                folder_path=None,
                user_input=fixed_prompt,
                model_with_vision=selected_model_display_name, # <<< USE THE SELECTED MODEL
                max_tokens_ui=150, # Reasonable limit for captions
                file_handling_option="Skip", # Not used for single image calls within the loop
                limiter_handling_option="Off", # Agent/Team style applies
                single_image_np=img, # <<< Pass PIL Image (needs testing with router)
                use_ollama_api_options=True, # Apply agent/team specific options
                release_model_on_change=False, # Avoid unloading during loop
                selected_role_or_team=agent_or_team_display_name,
                # Pass necessary states
                current_settings=settings,
                models_data_state=models_data,
                limiters_data_state=limiters_data,
                teams_data_state=teams_data,
                selected_model_tracker_value=None, # Not needed for this call
                file_agents_dict=file_agents,
                history_list_state=history_list, # For persistent logging within execute_...
                session_history_list_state=current_session_history # Pass current session list
            )
            current_session_history = updated_session_history_list # Update session history

            # Check for errors returned by the agent/team execution
            if response_text is None or isinstance(response_text, str) and (response_text.startswith("Error:") or response_text.startswith("⚠️ Error:")):
                 raise ValueError(f"Agent/Team returned an error: {response_text}")

            generated_caption = response_text.strip() if isinstance(response_text, str) else "Error: Invalid response type"
            if not generated_caption or generated_caption.startswith("Error:"): # Double check after strip
                 raise ValueError(f"Agent/Team returned empty or error response: '{generated_caption}'")

            last_caption_generated = generated_caption
            print(f"    Generated Caption: {generated_caption[:100]}...")

            # --- Save Generated Caption ---
            action_taken = ""
            final_caption_to_write = generated_caption

            # Handle Append/Prepend modes
            if generate_mode == "Append":
                # Append only if there was original content
                final_caption_to_write = f"{original_caption}\n\n---\n\n{generated_caption}" if original_caption else generated_caption
                action_taken = "Appended" if caption_exists else "Written" # Adjust action based on original existence
            elif generate_mode == "Prepend":
                 # Prepend only if there was original content
                final_caption_to_write = f"{generated_caption}\n\n---\n\n{original_caption}" if original_caption else generated_caption
                action_taken = "Prepended" if caption_exists else "Written" # Adjust action based on original existence
            elif caption_exists and generate_mode == "Overwrite":
                action_taken = "Overwritten"
            elif not caption_exists: # Handles Overwrite mode when file doesn't exist
                 action_taken = "Written"
            else: # Should not happen if Skip was handled correctly
                 action_taken = "Error: Logic flaw in save mode"

            # Perform the save
            try:
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(final_caption_to_write)
                updated_captions[filename] = final_caption_to_write # Update state dict
                msg = f"- Success {filename}: Caption generated and file {action_taken}."
                print(f"    {msg}")
                status_messages.append(msg)
                processed_count += 1
            except Exception as e_save:
                 msg = f"- Error saving caption for {filename} ({text_filename}): {e_save}"
                 print(f"    {msg}")
                 status_messages.append(msg)
                 error_count += 1 # Count save error

        except Exception as e_gen:
            # Catch errors from Image.open or the agent/team call
            msg = f"- Error processing {filename}: {e_gen}"
            print(f"    {msg}")
            status_messages.append(msg)
            error_count += 1
        finally:
             # Ensure image file is closed if it was opened
             if img:
                 try:
                     img.close()
                 except Exception as e_close:
                     print(f"    Warning: Error closing image file {filename}: {e_close}")


    end_time = time.time()
    duration = end_time - start_time
    final_status = (f"Caption generation finished in {duration:.2f}s. "
                    f"Processed: {processed_count}, Errors: {error_count}, Skipped: {skipped_count}.\n"
                    + "\n".join(status_messages))
    print(final_status)

    # Return: status_message, updated_captions_dict, last_caption_generated, updated_session_history
    return final_status, updated_captions, last_caption_generated, current_session_history


# CORRECTED Signature: Added selected_model_display_name
def generate_captions_for_all(
    image_paths: dict,
    current_captions: dict,
    agent_or_team_display_name: str,
    selected_model_display_name: str, # <<< ADDED
    generate_mode: str,
    settings: dict,
    models_data: list,
    limiters_data: dict,
    teams_data: dict,
    file_agents: dict,
    history_list: list,
    session_history: list
    ) -> tuple[str, dict, str, list]:
    """
    Generates captions for ALL loaded images using an agent/team by calling
    the generate_captions_for_selected logic repeatedly.

    Returns:
        tuple: (status_message, updated_captions_dict, last_caption_generated, updated_session_history)
    """
    print("\n--- Running: Generate Captions for ALL ---")
    start_time = time.time()

    if not image_paths:
        return "No images loaded to generate captions for.", current_captions, "", session_history

    all_filenames = sorted(list(image_paths.keys())) # Process in defined order

    # --- ADDED: Check if a vision model was actually selected ---
    if not selected_model_display_name:
         return "Please select a Vision Model for captioning.", current_captions, "", session_history
    # --- End Added Check ---

    # Reuse the core logic by calling it for the list of all files
    # Pass the selected_model_display_name through
    final_status, updated_captions, last_caption, updated_session_history = generate_captions_for_selected(
         selected_filenames=all_filenames,
         agent_or_team_display_name=agent_or_team_display_name,
         selected_model_display_name=selected_model_display_name, # <<< Pass model name
         generate_mode=generate_mode,
         image_paths=image_paths,
         current_captions=current_captions,
         settings=settings,
         models_data=models_data,
         limiters_data=limiters_data,
         teams_data=teams_data,
         file_agents=file_agents,
         history_list=history_list,
         session_history=session_history
    )

    end_time = time.time()
    print(f"Batch generation finished in {end_time - start_time:.2f}s.")

    # Modify the status slightly to indicate it was a batch run
    final_status = f"Batch Run Summary:\n{final_status}"

    return final_status, updated_captions, last_caption, updated_session_history