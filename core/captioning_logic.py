# ArtAgent/core/captioning_logic.py
import os
import gradio as gr # Not strictly needed here now, but kept for potential future use
from PIL import Image # Keep PIL needed for checking image files
import time
import numpy as np # Import numpy if conversion is needed

# Import utilities and core components
from .utils import get_absolute_path, load_json
from .app_logic import execute_chat_or_team # Import the router function
from . import history_manager as history # For logging

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')

# --- Function to load images and prepare data for Gallery ---
def load_images_and_captions(folder_path: str):
    """
    Loads images and captions, returning data formatted for gr.Gallery.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        tuple: Contains:
            - list[tuple[str, str]]: List of (image_path, filename) tuples for Gallery.
            - dict[str, str]: Mapping of image filename to its absolute path.
            - dict[str, str]: Mapping of image filename to its caption text.
            - str: Status message.
            - str | None: Filename of the first image (for selected item state).
            - str | None: Caption of the first image.
            - str | None: Filename of the first image again (for filename display).
            # Removed 8th element, Gallery selection is via event data
    """
    captions = {}
    image_paths = {}
    gallery_data = [] # NEW: List for Gallery component [(path, filename), ...]
    image_filenames_sorted = [] # Keep track of sorted filenames for consistency
    status = ""

    # Define default return values (7 items now) for error cases
    empty_return = [], {}, {}, "Error: No folder specified.", None, None, None

    if not folder_path:
        return empty_return
    if not os.path.isdir(folder_path):
        status = f"Error: Invalid folder path provided: '{folder_path}'."
        print(status)
        return [], {}, {}, status, None, None, None # 7 items

    print(f"Loading captions from folder: {folder_path}")
    found_count = 0
    try:
        # Iterate through sorted directory listing
        listdir_sorted = sorted(os.listdir(folder_path))
        for filename in listdir_sorted:
            name, ext = os.path.splitext(filename)
            if ext.lower() in IMAGE_EXTENSIONS:
                image_abs_path = os.path.join(folder_path, filename)
                if os.path.isfile(image_abs_path):
                    image_filenames_sorted.append(filename) # Store sorted name
                    image_paths[filename] = image_abs_path
                    # Gallery expects list of (image_path, label) tuples
                    gallery_data.append((image_abs_path, filename)) # Use filename as label

                    # Look for corresponding .txt file
                    text_filename = name + ".txt"
                    text_path = os.path.join(folder_path, text_filename)
                    caption_text = ""
                    if os.path.exists(text_path):
                        try:
                            with open(text_path, 'r', encoding='utf-8') as f:
                                caption_text = f.read()
                        except Exception as e:
                            print(f"Warning: Could not read caption file {text_filename}: {e}")
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
        return [], {}, {}, status, None, None, None # 7 items

    # Prepare initial selection data
    first_image_filename = image_filenames_sorted[0] if image_filenames_sorted else None
    first_caption = captions.get(first_image_filename, "") if first_image_filename else ""

    # Return 7 values matching the UPDATED expected outputs in app.py
    # Outputs: gallery_update, image_paths_state, caption_data_state, status_display, selected_item_state, caption_display, selected_filename_display
    return (
        gallery_data,         # 1. Data for Gallery component
        image_paths,          # 2. State update
        captions,             # 3. State update
        status,               # 4. Status Textbox
        first_image_filename, # 5. Initial selected filename state
        first_caption,        # 6. Initial caption display
        first_image_filename, # 7. Initial filename display
    )


# --- Function to handle Gallery selection event ---
def update_caption_display_from_gallery(
    evt: gr.SelectData, # Event data from Gallery selection
    caption_data_dict: dict,
    image_paths_dict: dict
    ) -> tuple[str, str, str]: # Return: caption_text, filename_for_state, filename_display
    """
    Handles the select event from the Gallery component.
    Updates caption display and selected item state based on the clicked image.

    Args:
        evt (gr.SelectData): Event data containing info about the selected item.
                               evt.value should be the label (filename).
                               evt.index might also be useful.
        caption_data_dict (dict): Current dictionary mapping filename -> caption.
        image_paths_dict (dict): Current dictionary mapping filename -> image path.


    Returns:
        tuple: (caption_text, selected_filename_for_state, filename_display)
    """
    selected_filename = None
    caption_text = ""
    filename_display = None

    if evt:
        selected_filename = evt.value # The 'label' of the selected gallery item IS the filename
        print(f"Gallery selected: Filename='{selected_filename}', Index={evt.index}")

        if selected_filename and isinstance(caption_data_dict, dict):
            caption_text = caption_data_dict.get(selected_filename, f"Caption data not found for '{selected_filename}'.")
            filename_display = selected_filename # Display the selected filename

            # Optional: Verify path exists (though Gallery usually shows valid images)
            # if selected_filename not in image_paths_dict or not os.path.isfile(image_paths_dict[selected_filename]):
            #     print(f"Warning: Path issue for gallery selected file: {selected_filename}")
            #     filename_display = f"{selected_filename} (Path Issue)"

        else:
            print("Warning: Gallery selection event did not provide expected filename or caption data is invalid.")
            caption_text = "Error retrieving caption."
            selected_filename = None # Clear selection state on error
            filename_display = "Error"
    else:
        print("Warning: Received empty gallery selection event.")
        # Return empty values if event data is missing
        caption_text = ""
        selected_filename = None
        filename_display = None

    # Outputs: caption_display, selected_item_state, selected_filename_display
    return caption_text, selected_filename, filename_display


# --- Function to save manually edited caption ---
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


# --- Function for manual batch editing ---
# NOTE: This function needs adjustment as gr.CheckboxGroup is removed.
# How multi-select works with Gallery needs consideration (maybe select multiple via shift/ctrl click?)
# For now, this function might not be directly usable until multi-select UI is clear.
def batch_edit_captions(
    # selected_filenames: list | None, # This input is no longer directly available
    # Need a different way to get multiple selections if required
    selected_items_from_gallery: list | None, # Placeholder if gallery selection provides multiple items
    text_to_add: str,
    mode: str, # "Append" or "Prepend"
    image_paths_dict: dict,
    caption_data_dict: dict
    ) -> tuple[str, dict]: # Return status string and updated captions dict
    """
    Appends or prepends text to the captions of selected images.
    *** CURRENTLY INCOMPATIBLE WITH GALLERY - Needs redesign for multi-select ***

    Returns:
        tuple: (status_message, updated_caption_data_dict)
    """
    # --- Placeholder / Needs Redesign ---
    print("Warning: Batch edit caption logic needs redesign for Gallery component.")
    return ("Batch edit feature needs update for Gallery UI.", caption_data_dict)
    # --- End Placeholder ---

    # selected_filenames = [] # Logic to extract filenames from gallery selection needed here

    # if not selected_filenames:
    #     return "No images selected for batch operation.", caption_data_dict
    # # ... (rest of original logic) ...


# --- Functions for Agent Caption Generation ---

# MODIFIED: Accepts single selected_filename from state
def generate_captions_for_selected(
    selected_filename: str | None, # Changed from list to single string/None
    agent_or_team_display_name: str,
    selected_model_display_name: str,
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
    Generates caption for the currently selected image using an agent/team and model.

    Returns:
        tuple: (status_message, updated_captions_dict, last_caption_generated, updated_session_history)
    """
    print("\n--- Running: Generate Caption for Selected Image ---")
    start_time = time.time()

    # --- Updated input checks ---
    if not selected_filename: # Check the single filename from state
        return "No image selected in the gallery.", current_captions, "", session_history
    if not agent_or_team_display_name or agent_or_team_display_name == "(Direct Agent Call)":
         return "Please select a valid Agent or Team for captioning.", current_captions, "", session_history
    if not selected_model_display_name:
         return "Please select a Vision Model for captioning.", current_captions, "", session_history

    # Re-check path validity for the single selected file
    image_path = image_paths.get(selected_filename)
    if not image_path or not os.path.isfile(image_path):
        msg = f"Error: Image path not found or invalid for selected file '{selected_filename}'."
        print(msg)
        # Return current state and error message
        return msg, current_captions, "", session_history

    # Vision Capability Check (Simplified)
    is_likely_vision_agent = (
        any(tag in agent_or_team_display_name.lower() for tag in ['llava', 'vision', 'photographer', 'captioner']) or
        agent_or_team_display_name.startswith("[Team]")
    )
    if not is_likely_vision_agent:
        print(f"Warning: Agent/Team '{agent_or_team_display_name}' selected, but ensure the chosen model '{selected_model_display_name}' is appropriate for image captioning.")

    status_messages = []
    updated_captions = current_captions.copy()
    processed_count = 0
    error_count = 0
    skipped_count = 0
    last_caption_generated = "" # Store the caption for display

    current_session_history = list(session_history)
    fixed_prompt = "Describe this image concisely. Focus on the main subject, action, and setting. Do not add commentary."

    # --- No loop needed for single selection ---
    print(f"  Processing selected: {selected_filename}")

    base_name = os.path.splitext(selected_filename)[0]
    text_filename = base_name + ".txt"
    text_path = os.path.join(os.path.dirname(image_path), text_filename)
    caption_exists = os.path.exists(text_path)
    original_caption = updated_captions.get(selected_filename, "")

    if caption_exists and generate_mode == "Skip":
        msg = f"- Skipped {selected_filename}: Caption file '{text_filename}' already exists and mode is Skip."
        print(f"    {msg}")
        status_messages.append(msg)
        skipped_count += 1
        # Return early as nothing was processed
        final_status = "\n".join(status_messages)
        return final_status, updated_captions, "", current_session_history


    img = None
    try:
        img = Image.open(image_path)
        print(f"    Calling agent/team '{agent_or_team_display_name}' with model '{selected_model_display_name}' for {selected_filename}...")

        response_text, _, _, updated_session_history_list = execute_chat_or_team(
            folder_path=None,
            user_input=fixed_prompt,
            model_with_vision=selected_model_display_name,
            max_tokens_ui=150,
            file_handling_option="Skip",
            limiter_handling_option="Off",
            single_image_input=img, # Pass PIL Image
            use_ollama_api_options=True,
            release_model_on_change=False,
            selected_role_or_team=agent_or_team_display_name,
            current_settings=settings,
            models_data_state=models_data,
            limiters_data_state=limiters_data,
            teams_data_state=teams_data,
            selected_model_tracker_value=None,
            file_agents_dict=file_agents,
            history_list_state=history_list,
            session_history_list_state=current_session_history
        )
        current_session_history = updated_session_history_list

        print(f"\nDEBUG CAPTIONING: Filename: {selected_filename}")
        print(f"DEBUG CAPTIONING: Raw Response Type: {type(response_text)}")
        print(f"DEBUG CAPTIONING: Raw Response Text: >>>{response_text}<<<")

        if response_text is None or isinstance(response_text, str) and (response_text.startswith("Error:") or response_text.startswith("⚠️ Error:")):
             raise ValueError(f"Agent/Team returned an error: {response_text}")

        generated_caption = response_text.strip() if isinstance(response_text, str) else "Error: Invalid response type"
        if not generated_caption or generated_caption.startswith("Error:"):
             raise ValueError(f"Agent/Team returned empty or error response: '{generated_caption}'")

        last_caption_generated = generated_caption
        print(f"    Generated Caption (Stripped): {generated_caption[:100]}...")

        action_taken = ""
        final_caption_to_write = generated_caption

        if generate_mode == "Append":
            final_caption_to_write = f"{original_caption}\n\n---\n\n{generated_caption}" if original_caption else generated_caption
            action_taken = "Appended" if caption_exists else "Written"
        elif generate_mode == "Prepend":
            final_caption_to_write = f"{generated_caption}\n\n---\n\n{original_caption}" if original_caption else generated_caption
            action_taken = "Prepended" if caption_exists else "Written"
        elif caption_exists and generate_mode == "Overwrite":
            action_taken = "Overwritten"
        elif not caption_exists:
             action_taken = "Written"
        else:
             action_taken = "Error: Logic flaw"

        try:
            print(f"DEBUG CAPTIONING: Preparing to write to {text_path}")
            print(f"DEBUG CAPTIONING: Content to Write: >>>{final_caption_to_write}<<<")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(final_caption_to_write)
            updated_captions[selected_filename] = final_caption_to_write
            msg = f"- Success {selected_filename}: Caption generated and file {action_taken}."
            print(f"    {msg}")
            status_messages.append(msg)
            processed_count += 1
        except Exception as e_save:
             msg = f"- Error saving caption for {selected_filename} ({text_filename}): {e_save}"
             print(f"    {msg}")
             status_messages.append(msg)
             error_count += 1

    except Exception as e_gen:
        msg = f"- Error processing {selected_filename}: {e_gen}"
        print(f"    {msg}")
        status_messages.append(msg)
        error_count += 1
    finally:
         if img:
             try: img.close()
             except Exception as e_close: print(f"    Warning: Error closing image file {selected_filename}: {e_close}")

    # --- Compile final status for single image ---
    end_time = time.time()
    duration = end_time - start_time
    status_prefix = f"Caption generation for '{selected_filename}' finished in {duration:.2f}s. "
    if processed_count == 1: status_prefix += "Status: Success."
    elif error_count == 1: status_prefix += "Status: Error."
    elif skipped_count == 1: status_prefix += "Status: Skipped."
    final_status = status_prefix + "\n" + "\n".join(status_messages)
    print(final_status)

    # Return: status_message, updated_captions_dict, caption_generated (for display), updated_session_history
    return final_status, updated_captions, last_caption_generated, current_session_history


def generate_captions_for_all(
    image_paths: dict,
    current_captions: dict,
    agent_or_team_display_name: str,
    selected_model_display_name: str,
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
    *** NOTE: This iterates and calls the single-image generation logic. ***

    Returns:
        tuple: (status_message, updated_captions_dict, last_caption_generated, updated_session_history)
    """
    print("\n--- Running: Generate Captions for ALL ---")
    start_time = time.time()

    if not image_paths:
        return "No images loaded to generate captions for.", current_captions, "", session_history
    if not selected_model_display_name:
         return "Please select a Vision Model for captioning.", current_captions, "", session_history
    if not agent_or_team_display_name or agent_or_team_display_name == "(Direct Agent Call)":
         return "Please select a valid Agent or Team for captioning.", current_captions, "", session_history

    all_filenames = sorted(list(image_paths.keys()))
    status_messages = []
    # Start with copies of the state dicts/lists
    batch_updated_captions = current_captions.copy()
    batch_session_history = list(session_history)
    batch_history_list = list(history_list) # For persistent logs within calls

    overall_processed = 0
    overall_errors = 0
    overall_skipped = 0
    last_caption = "" # Store the very last generated caption

    # Loop through all filenames and call the single-generation logic
    for filename in all_filenames:
         # Call the single-image function
         # Pass the current state of captions and history for this iteration
         single_status, single_updated_captions, single_last_caption, single_updated_session = generate_captions_for_selected(
             selected_filenames=[filename], # Pass single filename in a list (required by current func)
             # OR modify generate_captions_for_selected to take single filename directly if preferred
             agent_or_team_display_name=agent_or_team_display_name,
             selected_model_display_name=selected_model_display_name,
             generate_mode=generate_mode,
             image_paths=image_paths,
             current_captions=batch_updated_captions, # Pass the running updated dict
             settings=settings,
             models_data=models_data,
             limiters_data=limiters_data,
             teams_data=teams_data,
             file_agents=file_agents,
             history_list=batch_history_list, # Pass the running list
             session_history=batch_session_history # Pass the running list
         )

         # Update running state for the next iteration
         batch_updated_captions = single_updated_captions
         batch_session_history = single_updated_session
         if single_last_caption: # Store the latest non-empty caption
             last_caption = single_last_caption

         # Parse the single_status to update overall counts (this is a bit fragile)
         # Alternative: Modify generate_captions_for_selected to return counts
         if "Status: Success" in single_status: overall_processed += 1
         elif "Status: Error" in single_status: overall_errors += 1
         elif "Status: Skipped" in single_status: overall_skipped += 1
         status_messages.append(f"--- {filename} ---")
         status_messages.append(single_status.split('\n', 1)[1] if '\n' in single_status else single_status) # Add details

    end_time = time.time()
    duration = end_time - start_time
    final_status = (f"Batch Caption generation finished in {duration:.2f}s.\n"
                    f"Overall: Processed={overall_processed}, Errors={overall_errors}, Skipped={overall_skipped}.\n\n"
                    + "--- Details ---\n"
                    + "\n".join(status_messages))
    print(final_status)

    # Return final state
    return final_status, batch_updated_captions, last_caption, batch_session_history