# ArtAgent/core/captioning_logic.py
import os
import gradio as gr # Only needed if using gr.update directly, which we avoid here
from PIL import Image # Keep PIL needed for checking image files
from .utils import get_absolute_path, load_json # Assuming load_json might be needed if captions are complex
# --- NEW: Import dependencies needed for agent calls ---
import time
from .app_logic import execute_chat_or_team # Import the router function
from . import history_manager as history # For logging

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')

# --- (Existing functions: load_images_and_captions, update_caption_display, save_caption, batch_edit_captions) ---
# ... Keep existing functions here ...

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

# --- NEW Placeholder Functions for Agent Caption Generation ---

def generate_captions_for_selected(
    selected_filenames: list | None,
    agent_or_team_display_name: str,
    generate_mode: str,
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
    Placeholder: Generates captions for selected images using an agent/team.

    Returns:
        tuple: (status_message, updated_captions_dict, last_caption_generated, updated_session_history)
    """
    print("\n--- Received Request: Generate Captions for Selected ---")
    print(f"  Selected Files: {selected_filenames}")
    print(f"  Agent/Team: {agent_or_team_display_name}")
    print(f"  Mode: {generate_mode}")

    if not selected_filenames:
        return "No images selected to generate captions for.", current_captions, "", session_history

    # Basic check for vision model requirement (can be refined)
    is_vision_agent = "(VISION)" in agent_or_team_display_name or "[Team]" in agent_or_team_display_name
    # More robust check would involve looking up the actual agent/team and base model
    if not is_vision_agent: # Simple placeholder check
         # Check if the selected agent likely supports vision
         # This is simplistic; real check might involve looking up model in models.json
         # based on agent/team config or assuming specific agents like 'Llava' are vision-capable.
         pass # Allow non-vision agents for now, they might just use filename/context
         # A better check might look up the selected agent/team's underlying model
         # return f"Error: Selected agent/team '{agent_or_team_display_name}' might not support vision.", current_captions, "", session_history


    status = f"Caption generation for {len(selected_filenames)} selected images started...\n(NOT YET IMPLEMENTED)"
    updated_captions = current_captions.copy()
    last_caption = ""

    # --- TODO: Implement Loop ---
    # 1. Iterate through selected_filenames
    # 2. For each filename:
    #    a. Check if file exists based on image_paths[filename]
    #    b. Check generate_mode (Overwrite, Skip etc.) against existing text file
    #    c. Load image using PIL: `img = Image.open(image_paths[filename])`
    #    d. Construct prompt (e.g., "Describe this image.")
    #    e. Call execute_chat_or_team:
    #       response, _, model_name_used, session_history = execute_chat_or_team(...)
    #       Pass the loaded image object in a list: `single_image_pil=[img]` (or appropriate arg name)
    #       Pass necessary state data (settings, models, teams, etc.)
    #       Pass agent_or_team_display_name
    #       Pass a dummy user_input="" or a fixed prompt like "Describe the image."
    #       Use reasonable max_tokens, file handling ('None'), limiter ('Off'), etc.
    #    f. Process the `response` (clean up, etc.) -> generated_caption
    #    g. Save the generated_caption to the corresponding .txt file based on generate_mode
    #    h. Update updated_captions[filename] = generated_caption
    #    i. Update status message string
    #    j. Store the last generated caption

    # Placeholder return matching the output signature in app.py
    # outputs=[status_display, caption_data_state, caption_display, session_history_state]
    return status, updated_captions, last_caption, session_history


def generate_captions_for_all(
    image_paths: dict,
    current_captions: dict,
    agent_or_team_display_name: str,
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
    Placeholder: Generates captions for ALL loaded images using an agent/team.

    Returns:
        tuple: (status_message, updated_captions_dict, last_caption_generated, updated_session_history)
    """
    print("\n--- Received Request: Generate Captions for ALL ---")
    print(f"  Agent/Team: {agent_or_team_display_name}")
    print(f"  Mode: {generate_mode}")

    if not image_paths:
        return "No images loaded to generate captions for.", current_captions, "", session_history

    all_filenames = list(image_paths.keys())
    status = f"Batch caption generation for {len(all_filenames)} images started...\n(NOT YET IMPLEMENTED)"
    updated_captions = current_captions.copy()
    last_caption = ""

    # --- TODO: Implement Loop (similar to generate_captions_for_selected) ---
    # Iterate through all_filenames instead of selected_filenames

    # Placeholder return matching the output signature in app.py
    # outputs=[status_display, caption_data_state, caption_display, session_history_state]
    return status, updated_captions, last_caption, session_history


# --- Original Placeholder ---
# Kept for reference, but replaced by the above
# def generate_caption_with_agent(*args, **kwargs):
#     print("Agent caption generation not yet implemented.")
#     return "Agent caption generation is not yet available.", {}, {}, {} # Match expected outputs placeholder

# def batch_generate_captions(*args, **kwargs):
#     print("Batch agent caption generation not yet implemented.")
#     return "Batch agent caption generation is not yet available."