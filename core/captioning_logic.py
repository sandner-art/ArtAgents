# ArtAgent/core/captioning_logic.py
import os
import re
import time

import gradio as gr # Keep for gr.SelectData type hint if desired
from PIL import Image

# Import utilities and core components
from .app_logic import execute_chat_or_team
from .image_utils import IMAGE_EXTENSIONS, make_thumbnail, open_image
from .ollama_client import OllamaClient, OllamaError
from .dataset import caption_io, guardrails, profiles, provenance, snapshots, structured, tagger, vision
from .dataset.tokens import budget_markdown
from .version import APP_VERSION

DEFAULT_CAPTION_INSTRUCTION = "Describe this image concisely. Focus on the main subject, action, and setting. Do not add commentary."

# Caption sources (config["source"])
SOURCE_VLM = "vlm"                    # Vision model only (agent / team / direct)
SOURCE_TAGGER = "tagger"              # Booru tagger only, no vision model
SOURCE_TAGGER_HINTS = "tagger_hints"  # Tagger tags are given to the vision model as hints
SOURCE_TAGGER_PLUS = "tagger_plus"    # Tagger tags, then the vision model's caption
SOURCE_STRUCTURED = "structured"      # Vision model fills JSON fields; caption assembled from a template
TAGGER_SOURCES = (SOURCE_TAGGER, SOURCE_TAGGER_HINTS, SOURCE_TAGGER_PLUS)

_THUMBNAILS: dict[str, tuple[float, object]] = {}
_THUMBNAIL_CACHE_LIMIT = 3000


def get_thumbnail(image_path: str):
    """Cached gallery thumbnail (re-made when the file changes)."""
    mtime = os.path.getmtime(image_path)
    cached = _THUMBNAILS.get(image_path)
    if cached and cached[0] == mtime:
        return cached[1]
    thumb = make_thumbnail(image_path)
    if len(_THUMBNAILS) >= _THUMBNAIL_CACHE_LIMIT:
        _THUMBNAILS.pop(next(iter(_THUMBNAILS)))
    _THUMBNAILS[image_path] = (mtime, thumb)
    return thumb


def gallery_items(filenames: list[str], image_paths: dict) -> list[tuple]:
    """(thumbnail, filename) pairs for gr.Gallery, in the given order."""
    items = []
    for name in filenames:
        path = (image_paths or {}).get(name)
        if not path:
            continue
        try:
            items.append((get_thumbnail(path), name))
        except Exception as e:
            print(f"Warning: thumbnail failed for {name}: {e}")
    return items

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
                        thumbnail = get_thumbnail(image_abs_path)
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
    return caption_io.caption_path(image_path)


def _dataset_folder(image_paths_dict: dict) -> str | None:
    first = next(iter((image_paths_dict or {}).values()), None)
    return os.path.dirname(first) if first else None


def _snapshot(image_paths_dict: dict, operation: str) -> str:
    """Snapshot before a batch write; returns a status note (never blocks the operation)."""
    folder = _dataset_folder(image_paths_dict)
    if not folder:
        return ""
    try:
        snap_id = snapshots.create_snapshot(folder, image_paths_dict, operation)
        return f"Snapshot saved: {snap_id}."
    except OSError as e:
        print(f"Warning: could not create snapshot: {e}")
        return f"Warning: snapshot failed ({e})."


def batch_edit_captions(
    scope: str,
    selected_filename: str | None,
    text_to_add: str,
    mode: str, # "Append", "Prepend" or "Add as caption variant"
    image_paths_dict: dict,
    caption_data_dict: dict,
    joiner: str = caption_io.DEFAULT_JOINER,
    ) -> tuple[str, dict]:
    """
    Adds text (e.g. a LoRA trigger word or style tag) to captions and writes the .txt files.

    Append/Prepend join on the same line with `joiner` (never a new line: OneTrainer and
    SimpleTuner would read a new line as a separate caption). "Add as caption variant"
    adds the text as an extra caption line on purpose. A snapshot is taken first.
    """
    captions = dict(caption_data_dict or {})
    if not text_to_add or not text_to_add.strip():
        return "Enter the text to add first.", captions
    if not image_paths_dict:
        return "No images loaded.", captions
    if scope == BATCH_SCOPE_SELECTED:
        if not selected_filename or selected_filename not in image_paths_dict:
            return "Select an image in the gallery first (or choose 'All loaded images').", captions
        targets = [selected_filename]
    else:
        targets = list(image_paths_dict.keys())

    snap_note = _snapshot({t: image_paths_dict[t] for t in targets}, f"before batch {mode.lower()}")
    changed, errors = 0, []
    for filename in targets:
        current = captions.get(filename, "")
        new_caption = caption_io.combine(current, text_to_add, mode, joiner)
        if new_caption == current:
            continue
        try:
            caption_io.write_caption(image_paths_dict[filename], new_caption)
            captions[filename] = new_caption
            changed += 1
        except OSError as e:
            errors.append(f"{filename}: {e}")
    verb = {"Append": "Appended", "Prepend": "Prepended"}.get(mode, "Added variant")
    status = f"{verb} text to {changed} caption file(s). {snap_note}".strip()
    if errors:
        status += " Errors:\n" + "\n".join(errors)
    print(status)
    return status, captions


def export_caption_dataset(image_paths_dict: dict, caption_data_dict: dict, export_format: str) -> str:
    """
    Writes metadata.jsonl ({"file_name", "text"} per line; Hugging Face imagefolder /
    diffusers training format) or metadata.csv (file_name,text) next to the images.
    These formats hold one caption per image, so files with variants export their first line.
    """
    import csv
    import json as _json
    if not image_paths_dict:
        return "No images loaded."
    folder = _dataset_folder(image_paths_dict)
    rows, multi = [], 0
    for name in sorted(image_paths_dict):
        variants = caption_io.split_variants((caption_data_dict or {}).get(name, ""))
        multi += len(variants) > 1
        rows.append((name, variants[0] if variants else ""))
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
    if multi:
        status += f" {multi} image(s) have several caption variants; the first line was exported."
    return status


# --- Functions for Agent Caption Generation ---

def default_caption_config() -> dict:
    """Settings used when the caller passes no caption_config (e.g. older code and tests)."""
    return {
        "style": "prose", "joiner": " ", "omit_terms": [], "strip_omitted": False,
        "temperature": None, "seed": None, "max_image_side": None, "retries": 0,
        "max_output_tokens": 150, "budgets": [], "profile": None, "preset": None,
        "source": SOURCE_VLM, "backend": vision.BACKEND_OLLAMA, "openai": {}, "tagger": {}, "structured": {},
    }


_digest_cache: dict[str, tuple[float, dict]] = {}


def _model_digest(settings: dict, model_name: str) -> str:
    """Ollama digest of the model (for provenance), cached for a minute; '' if unknown."""
    base = OllamaClient.from_settings(settings).base_url
    cached = _digest_cache.get(base)
    if not cached or time.time() - cached[0] > 60:
        try:
            digests = {m["name"]: m.get("digest", "") for m in OllamaClient.from_settings(settings, timeout=3).list_models()}
        except OllamaError:
            digests = {}
        _digest_cache[base] = (time.time(), digests)
        cached = _digest_cache[base]
    return (cached[1].get(model_name) or "")[:12]


def _prepare_image(image_path: str, max_side: int | None):
    """Opens the image (whitelisted decoders) and downscales it for the vision model."""
    img = open_image(image_path)
    if not max_side:
        return img
    img.load()
    if max(img.size) <= max_side:
        return img
    resized = img.copy()
    resized.thumbnail((int(max_side), int(max_side)), Image.LANCZOS)
    img.close()
    return resized


def generate_captions_for_selected(
    selected_filename: str | None,
    agent_or_team_display_name: str,
    selected_model_display_name: str,
    generate_mode: str, # Overwrite, Skip, Append, Prepend, Add as caption variant
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
    caption_config: dict | None = None,
    ) -> tuple[str, dict, str, list]:
    """
    Generates a caption for the selected image with an agent/team (or a direct model call).

    The model output is cleaned to one line, checked for glitches (empty, refusal,
    "This image shows…", repetition loops, truncation) and retried with a new seed if
    needed; omitted terms are removed from tag captions (flagged in prose); then the
    trigger prefix/suffix is added and the file is written per `generate_mode`.
    A provenance record (model, digest, instruction hash, options) is saved alongside.
    """
    config = default_caption_config()
    config.update(caption_config or {})
    style = config.get("style") or "prose"
    print("\n--- Running: Generate Caption for Selected Image ---")
    start_time = time.time()

    if not selected_filename:
        return "No image selected in the gallery.", current_captions, "", session_history
    source = config.get("source") or SOURCE_VLM
    backend = config.get("backend") or vision.BACKEND_OLLAMA
    uses_model = source != SOURCE_TAGGER
    openai_cfg = config.get("openai") or {}
    if backend == vision.BACKEND_OPENAI:
        selected_model_display_name = openai_cfg.get("model") or ""
        if uses_model and not (openai_cfg.get("base_url") and selected_model_display_name):
            return "Enter the server URL and model name for the OpenAI-compatible backend.", current_captions, "", session_history
    elif uses_model and not agent_or_team_display_name and source != SOURCE_STRUCTURED:
        return "Please select an Agent, a Team or '(Direct Agent Call)' for captioning.", current_captions, "", session_history
    if uses_model and not selected_model_display_name:
        return "Please select a Vision Model for captioning.", current_captions, "", session_history

    image_path = image_paths.get(selected_filename)
    if not image_path or not os.path.isfile(image_path):
        msg = f"Error: Image path not found or invalid for selected file '{selected_filename}'."
        print(msg)
        return msg, current_captions, "", session_history

    updated_captions = dict(current_captions or {})
    current_session_history = list(session_history or [])
    instruction = (caption_instruction or "").strip() or DEFAULT_CAPTION_INSTRUCTION
    text_path = caption_io.caption_path(image_path)
    text_filename = os.path.basename(text_path)
    caption_exists = os.path.exists(text_path)
    original_caption = updated_captions.get(selected_filename, "")

    def finish(outcome: str, details: list[str], caption_shown: str = ""):
        duration = time.time() - start_time
        status = f"Caption generation for '{selected_filename}' finished in {duration:.2f}s. Status: {outcome}.\n" + "\n".join(details)
        print(status)
        return status, updated_captions, caption_shown, current_session_history

    if caption_exists and generate_mode == caption_io.MODE_SKIP:
        return finish("Skipped", [f"- Skipped {selected_filename}: Caption file '{text_filename}' already exists and mode is Skip."])

    max_output_tokens = int(config.get("max_output_tokens") or 150)
    base_options = {}
    if config.get("temperature") is not None:
        base_options["temperature"] = float(config["temperature"])
    if config.get("seed") is not None and int(config["seed"]) >= 0:
        base_options["seed"] = int(config["seed"])
    omit_terms = list(config.get("omit_terms") or [])
    retries = max(0, int(config.get("retries") or 0))

    img = None
    caption, issues, attempts = "", [], 0
    retry_reasons: list[str] = []
    options = dict(base_options)
    profile = profiles.get_profile(config.get("profile")) if config.get("profile") else {}
    tagger_opts = dict(config.get("tagger") or {})
    tagger_record, fields_record = None, None
    direct = backend == vision.BACKEND_OPENAI or source == SOURCE_STRUCTURED
    target = vision.VisionTarget(backend, selected_model_display_name or "", openai_cfg.get("base_url", ""),
                                 openai_cfg.get("api_key", ""))
    model_label = target.label if uses_model else ""
    used_instruction = instruction
    try:
        img = _prepare_image(image_path, config.get("max_image_side"))

        tag_list: list[str] = []
        if source in TAGGER_SOURCES:
            tcfg = tagger.get_tagger(tagger_opts.get("label"))
            probs, cached = tagger.tag_probabilities(tcfg, image_path, img, reuse=tagger_opts.get("reuse", True))
            selection = tagger.select(probs, tagger_opts)
            tag_list = tagger.format_tags(selection, profile, tagger_opts)
            if omit_terms:
                tag_list = [t for t in tag_list if not guardrails.find_terms(t, omit_terms, "tags")]
            tagger_record = {"model": tcfg.get("id"), "thresholds": selection.get("thresholds"), "cached": cached,
                             "tags": len(tag_list)}
            if source == SOURCE_TAGGER_HINTS:
                if omit_terms:  # Never hint at words the model is told not to mention
                    selection = dict(selection, **{key: [(n, p) for n, p in selection.get(key, [])
                                                         if not guardrails.find_terms(tagger.display_tag(n), omit_terms, "tags")]
                                                   for key in ("general", "characters")})
                hint = tagger.hint_text(selection, float(tagger_opts.get("hint_threshold", 0.5)), profile)
                if hint:
                    used_instruction = f"{instruction}\n\n{hint}"

        def call_model(prompt: str, opts: dict, json_schema: dict | None = None, max_tokens: int = max_output_tokens) -> str:
            nonlocal current_session_history
            if direct:  # OpenAI-compatible servers and structured output: plain model call, no agent persona
                return vision.ask(target, settings, prompt, [img], temperature=opts.get("temperature"),
                                  seed=opts.get("seed"), max_tokens=max_tokens, json_schema=json_schema)
            print(f"    Calling '{agent_or_team_display_name}' with '{selected_model_display_name}' (attempt {attempts})...")
            response_text, _, _, current_session_history = execute_chat_or_team(
                folder_path=None,
                user_input=prompt,
                model_with_vision=selected_model_display_name,
                max_tokens_ui=max_tokens,
                file_handling_option="Skip",
                limiter_handling_option="Off",
                single_image_input=img,
                use_ollama_api_options=None, # Follow App Settings; temperature/seed/num_predict are set explicitly
                release_model_on_change=False,
                selected_role_or_team=agent_or_team_display_name,
                clean_artifacts_flag=True,
                current_settings=settings,
                models_data_state=models_data,
                limiters_data_state=limiters_data,
                teams_data_state=teams_data,
                selected_model_tracker_value=None,
                file_agents_dict=file_agents,
                history_list_state=history_list,
                session_history_list_state=current_session_history,
                ollama_option_overrides=opts or None,
            )
            if response_text is None or (isinstance(response_text, str) and (response_text.startswith("Error:") or response_text.startswith("⚠️ Error:"))):
                raise ValueError(f"Agent/Team returned an error: {response_text}")
            return response_text

        if source == SOURCE_TAGGER:
            caption, attempts = ", ".join(tag_list), 1
            style = "tags"
            if not caption:
                raise ValueError("The tagger found no tags above the thresholds.")
        else:
            sconf = config.get("structured") or {}
            field_ids = list(sconf.get("fields") or structured.DEFAULT_FIELDS)
            vlm_style = "prose" if source == SOURCE_TAGGER_PLUS and style != "tags" else style
            for attempt in range(retries + 1):
                attempts = attempt + 1
                if source == SOURCE_STRUCTURED:
                    prompt = structured.instruction(field_ids, style, sconf.get("extra", ""))
                    raw = call_model(prompt, options, structured.json_schema(field_ids), max(400, 3 * max_output_tokens))
                    try:
                        fields = structured.normalize_fields(vision.parse_json_object(raw), field_ids)
                        caption = structured.render(fields, sconf.get("template") or structured.default_template(field_ids, style),
                                                    style, sconf.get("include"))
                        caption = guardrails.clean_caption(caption, style)
                        fields_record = fields
                        issues = guardrails.check_caption(caption, style, 10 ** 6, omit_terms)
                    except ValueError as parse_error:
                        caption, issues = "", ["invalid JSON"]
                        print(f"    Structured output could not be read: {parse_error}")
                    retryable = [i for i in issues if i in guardrails.RETRYABLE or i == "invalid JSON"]
                else:
                    response_text = call_model(used_instruction, options)
                    caption = guardrails.clean_caption(response_text, vlm_style)
                    issues = guardrails.check_caption(caption, vlm_style, max_output_tokens, omit_terms)
                    retryable = [i for i in issues if i in guardrails.RETRYABLE]
                if not retryable or attempt == retries:
                    break
                retry_reasons.extend(retryable)
                print(f"    Caption issues {retryable}; retrying with a new seed.")
                options = dict(base_options)
                options["seed"] = int(base_options.get("seed", 42)) + attempt + 1
                options["temperature"] = max(float(base_options.get("temperature") or 0.0), 0.3)
            if source == SOURCE_TAGGER_PLUS and caption:
                if style == "tags":
                    caption = ", ".join(dict.fromkeys(tag_list + [t.strip() for t in caption.split(",") if t.strip()]))
                elif tag_list:
                    caption = f"{', '.join(tag_list)}. {caption}"
            if fields_record is not None:
                try:
                    structured.save_fields(image_path, fields_record, model=model_label)
                except OSError as e:
                    print(f"    Warning: could not save structured fields: {e}")

        if not caption:
            raise ValueError("The model returned an empty caption.")
        if omit_terms and style == "tags":
            caption = guardrails.remove_tags(caption, omit_terms)
        elif omit_terms and config.get("strip_omitted"):
            for term in guardrails.find_terms(caption, omit_terms, style):
                caption = re.sub(r"\b" + re.escape(term) + r"\b", "", caption, flags=re.IGNORECASE)
            caption = re.sub(r"\s{2,}", " ", caption).strip(" ,")
        model_caption = caption  # Checked below without the user's own prefix/suffix (trigger, class noun)
        caption = guardrails.apply_prefix_suffix(caption, caption_prefix, caption_suffix)

        joiner = config.get("joiner") or (", " if style == "tags" else " ")
        final_text = caption_io.combine(original_caption if caption_exists else "", caption, generate_mode, joiner)
        caption_io.write_caption(image_path, final_text)
        updated_captions[selected_filename] = final_text

        remaining = list(guardrails.check_caption(model_caption, style, max_output_tokens, omit_terms))
        try:
            model_name = vision.clean_model_name(selected_model_display_name or "")
            provenance.write_provenance(image_path, {
                "model": model_name if uses_model else f"tagger:{(tagger_record or {}).get('model')}",
                "model_digest": _model_digest(settings, model_name) if uses_model and backend == vision.BACKEND_OLLAMA else "",
                "backend": backend if uses_model else "tagger",
                "source": source,
                "tagger": tagger_record,
                "fields": fields_record,
                "agent": "(Direct)" if (direct or not uses_model) else agent_or_team_display_name,
                "profile": config.get("profile"),
                "preset": config.get("preset"),
                "instruction": used_instruction if uses_model else "",
                "instruction_hash": provenance.prompt_hash(used_instruction) if uses_model else "",
                "options": options,
                "max_image_side": config.get("max_image_side"),
                "mode": generate_mode,
                "prefix": caption_prefix, "suffix": caption_suffix,
                "retries": attempts - 1,
                "issues": remaining,
                "caption_hash": provenance.text_hash(final_text),
                "artagents_version": APP_VERSION,
            })
        except OSError as e:
            print(f"    Warning: could not write provenance: {e}")

        action = {caption_io.MODE_APPEND: "Appended", caption_io.MODE_PREPEND: "Prepended",
                  caption_io.MODE_VARIANT: "Added as variant"}.get(generate_mode, "Overwritten" if caption_exists else "Written")
        details = [f"- Success {selected_filename}: Caption generated and file {action}."]
        if attempts > 1:
            details.append(f"  Retried {attempts - 1}x because of: {', '.join(dict.fromkeys(retry_reasons)) or 'glitches'}.")
        if remaining:
            details.append(f"  ⚠️ Check this caption: {', '.join(remaining)}.")
        budget_line = budget_markdown(caption, config.get("budgets") or [])
        if budget_line and ("⚠️" in budget_line or "⛔" in budget_line):
            details.append(f"  Tokens: {budget_line}")
        return finish("Success", details, final_text)

    except Exception as e_gen:
        return finish("Error", [f"- Error processing {selected_filename}: {e_gen}"])
    finally:
        if img is not None:
            try:
                img.close()
            except Exception:
                pass

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
    caption_config: dict | None = None,
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
    cfg = caption_config or {}
    source = cfg.get("source") or SOURCE_VLM
    if cfg.get("backend") == vision.BACKEND_OPENAI:
        if source != SOURCE_TAGGER and not ((cfg.get("openai") or {}).get("base_url") and (cfg.get("openai") or {}).get("model")):
            yield "Enter the server URL and model name for the OpenAI-compatible backend.", current_captions, "", session_history; return
    elif source != SOURCE_TAGGER:
        if not selected_model_display_name:
            yield "Please select a Vision Model.", current_captions, "", session_history; return
        if not agent_or_team_display_name and source != SOURCE_STRUCTURED:
            yield "Please select an Agent, a Team or '(Direct Agent Call)'.", current_captions, "", session_history; return

    all_filenames = sorted(image_paths.keys())
    snap_note = _snapshot(image_paths, "before AI captioning (all)")
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
        return (head + f"Overall: Processed={overall_processed}, Errors={overall_errors}, Skipped={overall_skipped}. {snap_note}\n\n"
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
            caption_config=caption_config,
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
