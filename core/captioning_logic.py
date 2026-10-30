# ArtAgent/core/captioning_logic.py
import os
import gradio as gr # Keep for gr.SelectData type hint if desired
import time

# Import utilities and core components
from .app_logic import execute_chat_or_team
from .image_utils import IMAGE_EXTENSIONS, make_thumbnail, open_image

DEFAULT_CAPTION_INSTRUCTION = "Describe this image concisely. Focus on the main subject, action, and setting. Do not add commentary."

# --- Function to load images and prepare data for Gallery ---
def load_images_and_captions(folder_path: str):
    """
    Loads images and captions, returning data formatted for gr.Gallery.

    The gallery receives in-memory thumbnails rather than file paths, so Gradio
    serves them from its own cache and the app does not need to expose the
    user's folders (previously `allowed_paths=["/"]`, i.e. the whole disk).

    Args:
        folder_path (str): The path to the folder.

    Returns:
        tuple: Contains:
            - list[tuple[PIL.Image.Image, str]]: List of (thumbnail, filename) tuples for Gallery.
            - dict[str, str]: Mapping of image filename to its absolute path.
            - dict[str, str]: Mapping of image filename to its caption text.
            - str: Status message.
            - str | None: Filename of the first image (for selected item state).
            - str | None: Caption of the first image.
            - str | None: Filename of the first image again (for filename display).
    """
    captions = {}
    image_paths = {}
    gallery_data = [] # List for Gallery component [(path, filename), ...]
    image_filenames_sorted = []
    status = ""
    empty_return = [], {}, {}, "Error: No folder specified.", None, None, None

    if not folder_path: return empty_return
    if not os.path.isdir(folder_path):
        status = f"Error: Invalid folder path provided: '{folder_path}'."
        print(status)
        return [], {}, {}, status, None, None, None # 7 items

    print(f"Loading captions from folder: {folder_path}")
    found_count = 0
    skipped_count = 0
    try:
        listdir_sorted = sorted(os.listdir(folder_path))
        for filename in listdir_sorted:
            name, ext = os.path.splitext(filename)
            if ext.lower() in IMAGE_EXTENSIONS:
                image_abs_path = os.path.join(folder_path, filename)
                if os.path.isfile(image_abs_path):
                    try:
                        thumbnail = make_thumbnail(image_abs_path)
                    except Exception as e:
                        # Not a real PNG/JPEG/BMP/GIF/WEBP (or corrupt/oversized): skip it
                        print(f"Warning: Skipping '{filename}': {e}")
                        skipped_count += 1
                        continue
                    image_filenames_sorted.append(filename)
                    image_paths[filename] = image_abs_path
                    # Gallery expects list of (image, label) tuples
                    gallery_data.append((thumbnail, filename)) # Use filename as label

                    text_filename = name + ".txt"
                    text_path = os.path.join(folder_path, text_filename)
                    caption_text = ""
                    if os.path.exists(text_path):
                        try:
                            with open(text_path, 'r', encoding='utf-8') as f: caption_text = f.read()
                        except Exception as e: print(f"Warning: Could not read caption file {text_filename}: {e}")
                    captions[filename] = caption_text
                    found_count += 1

        status = f"Loaded {found_count} image(s)." if found_count > 0 else "No supported image files found."
        if skipped_count:
            status += f" Skipped {skipped_count} unreadable or unsupported file(s) (see console)."
        print(status)

    except Exception as e:
        status = f"Error reading folder contents: {e}"
        print(status)
        return [], {}, {}, status, None, None, None # 7 items

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
# CORRECTED Signature: Takes 3 arguments now
def update_caption_display_from_gallery(
    evt: gr.SelectData,
    caption_data_dict: dict,
    image_paths_dict: dict
    ) -> tuple[str, str, str]:
    """
    Handles the select event from the Gallery component.
    Updates caption display and selected item state based on the clicked image.
    CORRECTED FOR GRADIO 4+ EVENT DATA.
    """
    selected_filename = None
    caption_text = ""
    filename_display = None

    if evt:
        # --- MAJOR CHANGE HERE ---
        # In Gradio 4+, evt.value is a dictionary: {'image': {...}, 'caption': 'filename.jpg'}
        # We need to extract the caption, which holds our original filename.
        if isinstance(evt.value, dict) and 'caption' in evt.value:
            selected_filename = evt.value['caption']
            print(f"Gallery selected: Filename='{selected_filename}', Index={evt.index}")

            if selected_filename and isinstance(caption_data_dict, dict):
                caption_text = caption_data_dict.get(selected_filename, f"Caption data not found for '{selected_filename}'.")
                filename_display = selected_filename
            else:
                print("Warning: Could not get caption for selected filename or caption data is invalid.")
                caption_text = "Error retrieving caption."
                selected_filename = None
                filename_display = "Error"
        else:
            print(f"Warning: Gallery selection event did not provide the expected dictionary format. Got: {evt.value}")
            caption_text = "Error: Invalid event data."
            selected_filename = None
            filename_display = "Error"

    else:
        print("Warning: Received empty gallery selection event.")
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
    """
    if not selected_filename:
        return "Error: No image selected to save caption for.", caption_data_dict
    if not isinstance(image_paths_dict, dict):
        return "Error: Image path data is missing or invalid.", caption_data_dict
    image_path = image_paths_dict.get(selected_filename)
    if not image_path:
        return f"Error: Could not find original path for '{selected_filename}'.", caption_data_dict

    updated_captions = caption_data_dict.copy() if isinstance(caption_data_dict, dict) else {}
    try:
        folder_path = os.path.dirname(image_path)
        base_name = os.path.splitext(selected_filename)[0]
        text_path = os.path.join(folder_path, base_name + ".txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(caption_text)
        status = f"Caption saved successfully to {base_name}.txt"
        print(status)
        updated_captions[selected_filename] = caption_text
        return status, updated_captions
    except Exception as e:
        status = f"Error saving caption to '{text_path}': {e}"
        print(status)
        return status, (caption_data_dict if isinstance(caption_data_dict, dict) else {})


# --- Manual batch editing ---
BATCH_SCOPE_ALL = "All loaded images"
BATCH_SCOPE_SELECTED = "Selected image only"


def _caption_path(image_path: str) -> str:
    return os.path.splitext(image_path)[0] + ".txt"


def batch_edit_captions(
    scope: str,
    selected_filename: str | None,
    text_to_add: str,
    mode: str, # "Append" or "Prepend"
    image_paths_dict: dict,
    caption_data_dict: dict,
    ) -> tuple[str, dict]:
    """
    Appends or prepends text (e.g. a LoRA trigger word or style tag) to captions
    and writes them to the .txt files. Images without a caption get the text alone.
    Text is added verbatim, so include your own separator (e.g. "ohwx, ").
    """
    captions = dict(caption_data_dict or {})
    if not text_to_add:
        return "Enter the text to add first.", captions
    if not image_paths_dict:
        return "No images loaded.", captions
    if scope == BATCH_SCOPE_SELECTED:
        if not selected_filename or selected_filename not in image_paths_dict:
            return "Select an image in the gallery first (or choose 'All loaded images').", captions
        targets = [selected_filename]
    else:
        targets = list(image_paths_dict.keys())

    changed, errors = 0, []
    for filename in targets:
        current = captions.get(filename, "")
        if mode == "Prepend":
            new_caption = text_to_add + current
        else:
            new_caption = current + text_to_add
        try:
            with open(_caption_path(image_paths_dict[filename]), "w", encoding="utf-8") as f:
                f.write(new_caption)
            captions[filename] = new_caption
            changed += 1
        except OSError as e:
            errors.append(f"{filename}: {e}")
    status = f"{mode}ed text to {changed} caption file(s)."
    if errors:
        status += " Errors:\n" + "\n".join(errors)
    print(status)
    return status, captions


def export_caption_dataset(image_paths_dict: dict, caption_data_dict: dict, export_format: str) -> str:
    """
    Writes metadata.jsonl ({"file_name", "text"} per line; Hugging Face imagefolder /
    diffusers training format) or metadata.csv (file_name,text) next to the images.
    """
    import csv
    import json as _json
    if not image_paths_dict:
        return "No images loaded."
    folder = os.path.dirname(next(iter(image_paths_dict.values())))
    rows = [(name, (caption_data_dict or {}).get(name, "")) for name in sorted(image_paths_dict)]
    missing = sum(1 for _, text in rows if not text.strip())
    try:
        if export_format == "metadata.csv":
            out_path = os.path.join(folder, "metadata.csv")
            with open(out_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["file_name", "text"])
                writer.writerows(rows)
        else:
            out_path = os.path.join(folder, "metadata.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for name, text in rows:
                    f.write(_json.dumps({"file_name": name, "text": text}, ensure_ascii=False) + "\n")
    except OSError as e:
        return f"Error writing dataset file: {e}"
    status = f"Exported {len(rows)} caption(s) to {out_path}."
    if missing:
        status += f" Warning: {missing} image(s) have an empty caption."
    return status


# --- Functions for Agent Caption Generation ---

# MODIFIED: Accepts single selected_filename from state
def generate_captions_for_selected(
    selected_filename: str | None, # Changed from list to single string/None from state
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
    session_history: list,
    caption_instruction: str | None = None,
    caption_prefix: str = "",
    caption_suffix: str = "",
    ) -> tuple[str, dict, str, list]:
    """
    Generates caption for the currently selected image using an agent/team and model.

    caption_instruction overrides the default "describe this image" prompt;
    caption_prefix / caption_suffix (e.g. a LoRA trigger word) are added verbatim.
    """
    print("\n--- Running: Generate Caption for Selected Image ---")
    start_time = time.time()

    # --- Updated input checks ---
    if not selected_filename:
        return "No image selected in the gallery.", current_captions, "", session_history
    if not agent_or_team_display_name:
         return "Please select an Agent, a Team or '(Direct Agent Call)' for captioning.", current_captions, "", session_history
    if not selected_model_display_name:
         return "Please select a Vision Model for captioning.", current_captions, "", session_history

    image_path = image_paths.get(selected_filename)
    if not image_path or not os.path.isfile(image_path):
        msg = f"Error: Image path not found or invalid for selected file '{selected_filename}'."
        print(msg)
        return msg, current_captions, "", session_history

    is_likely_vision_agent = (
        any(tag in agent_or_team_display_name.lower() for tag in ['llava', 'vision', 'photographer', 'captioner']) or
        agent_or_team_display_name.startswith("[Team]")
    )
    if not is_likely_vision_agent:
        print(f"Warning: Agent/Team '{agent_or_team_display_name}' selected, ensure model '{selected_model_display_name}' is appropriate.")

    status_messages = []
    updated_captions = current_captions.copy()
    processed_count = 0
    error_count = 0
    skipped_count = 0
    last_caption_generated = "" # Store the caption for display

    current_session_history = list(session_history)
    fixed_prompt = (caption_instruction or "").strip() or DEFAULT_CAPTION_INSTRUCTION

    # --- File handling check ---
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
        # Same "<summary> Status: X." first line as the other outcomes, so batch runs count it
        final_status = f"Caption generation for '{selected_filename}' finished. Status: Skipped.\n" + "\n".join(status_messages)
        # Ensure correct return tuple structure (status, captions_dict, last_caption, session_history)
        return final_status, updated_captions, "", current_session_history


    # --- Image loading and Agent call ---
    img = None
    try:
        img = open_image(image_path) # Restricted to PNG/JPEG/BMP/GIF/WEBP decoders
        print(f"    Calling agent/team '{agent_or_team_display_name}' with model '{selected_model_display_name}'...")

        # Call the router, passing PIL image via 'single_image_input' keyword
        response_text, _, _, updated_session_history_list = execute_chat_or_team(
            folder_path=None,
            user_input=fixed_prompt,
            model_with_vision=selected_model_display_name,
            max_tokens_ui=150,
            file_handling_option="Skip",
            limiter_handling_option="Off",
            single_image_input=img, # Correct keyword for the router
            use_ollama_api_options=True,
            release_model_on_change=False,
            selected_role_or_team=agent_or_team_display_name,
            clean_artifacts_flag=True, # <--- THE FIX IS HERE
            # States
            current_settings=settings,
            models_data_state=models_data,
            limiters_data_state=limiters_data,
            teams_data_state=teams_data,
            selected_model_tracker_value=None,
            file_agents_dict=file_agents,
            history_list_state=history_list,
            session_history_list_state=current_session_history
        )
        current_session_history = updated_session_history_list # Capture updated history

        if response_text is None or isinstance(response_text, str) and (response_text.startswith("Error:") or response_text.startswith("⚠️ Error:")):
             raise ValueError(f"Agent/Team returned an error: {response_text}")

        generated_caption = response_text.strip() if isinstance(response_text, str) else "Error: Invalid response type"
        if not generated_caption or generated_caption.startswith("Error:"):
             raise ValueError(f"Agent/Team returned empty or error response: '{generated_caption}'")

        print(f"    Generated Caption (Stripped): {generated_caption[:100]}...")

        action_taken = ""
        generated_caption = f"{caption_prefix or ''}{generated_caption}{caption_suffix or ''}"
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

        try:
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(final_caption_to_write)
            updated_captions[selected_filename] = final_caption_to_write
            # Show exactly what is in the .txt (incl. prefix/suffix and appended text),
            # so "Save THIS Caption" writes back the same content.
            last_caption_generated = final_caption_to_write
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

    end_time = time.time()
    duration = end_time - start_time
    status_prefix = f"Caption generation for '{selected_filename}' finished in {duration:.2f}s. "
    if processed_count == 1: status_prefix += "Status: Success."
    elif error_count >= 1: status_prefix += "Status: Error."
    elif skipped_count == 1: status_prefix += "Status: Skipped."
    else: status_prefix += "Status: Unknown."
    final_status = status_prefix + "\n" + "\n".join(status_messages)
    print(final_status)

    return final_status, updated_captions, last_caption_generated, current_session_history

# MODIFIED: Loops and calls the MODIFIED generate_captions_for_selected
def iter_generate_captions_for_all(
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
    session_history: list,
    caption_instruction: str | None = None,
    caption_prefix: str = "",
    caption_suffix: str = "",
    progress=None,
    ):
    """
    Generates captions for ALL loaded images, one at a time.

    Yields (status, captions_dict, last_caption, session_history) after every
    image, so the UI shows progress and the Stop button can end the batch
    between images (files already written are kept).
    """
    print("\n--- Running: Generate Captions for ALL ---")
    start_time = time.time()

    if not image_paths:
        yield "No images loaded.", current_captions, "", session_history; return
    if not selected_model_display_name:
        yield "Please select a Vision Model.", current_captions, "", session_history; return
    if not agent_or_team_display_name:
        yield "Please select an Agent, a Team or '(Direct Agent Call)'.", current_captions, "", session_history; return

    all_filenames = sorted(image_paths.keys())
    total = len(all_filenames)
    batch_status_messages = []
    batch_updated_captions = dict(current_captions or {})
    batch_session_history = list(session_history or [])
    batch_history_list = list(history_list or [])
    overall_processed = overall_errors = overall_skipped = 0
    last_caption = ""

    def summary(done: int, finished: bool) -> str:
        head = (f"Batch Caption generation finished in {time.time() - start_time:.2f}s.\n" if finished
                else f"⏳ Captioning {done}/{total} ({time.time() - start_time:.0f}s elapsed)...\n")
        return (head + f"Overall: Processed={overall_processed}, Errors={overall_errors}, Skipped={overall_skipped}.\n\n"
                + "--- Details ---\n" + "\n".join(batch_status_messages))

    for index, filename in enumerate(all_filenames, start=1):
        if progress is not None:
            progress((index - 1) / total, desc=f"Captioning {filename} ({index}/{total})")
        single_status, batch_updated_captions, single_last_caption, batch_session_history = generate_captions_for_selected(
            selected_filename=filename,
            agent_or_team_display_name=agent_or_team_display_name,
            selected_model_display_name=selected_model_display_name,
            generate_mode=generate_mode,
            image_paths=image_paths,
            current_captions=batch_updated_captions,
            settings=settings,
            models_data=models_data,
            limiters_data=limiters_data,
            teams_data=teams_data,
            file_agents=file_agents,
            history_list=batch_history_list,
            session_history=batch_session_history,
            caption_instruction=caption_instruction,
            caption_prefix=caption_prefix,
            caption_suffix=caption_suffix,
        )
        if single_last_caption: last_caption = single_last_caption
        if "Status: Success." in single_status: overall_processed += 1
        elif "Status: Error." in single_status: overall_errors += 1
        elif "Status: Skipped." in single_status: overall_skipped += 1
        detail_lines = single_status.split('\n', 1)[1] if '\n' in single_status else "(No details)"
        batch_status_messages.append(f"--- {filename} ---\n{detail_lines}")
        yield summary(index, finished=index == total), batch_updated_captions, last_caption, batch_session_history

    print(summary(total, finished=True))


def generate_captions_for_all(*args, **kwargs) -> tuple[str, dict, str, list]:
    """Blocking variant of iter_generate_captions_for_all; returns its final update."""
    last = None
    for last in iter_generate_captions_for_all(*args, **kwargs):
        pass
    return last
