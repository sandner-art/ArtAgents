# ArtAgent/ui/captions_tab.py
"""Image Captions Editor: layout (create_captions_tab) and event wiring (wire_captions_tab)."""
import gradio as gr

from core import dataset_tools_ui as dtu
from core import dataset_ui as dui
from core.captioning_logic import BATCH_SCOPE_ALL, BATCH_SCOPE_SELECTED, save_caption, export_caption_dataset
from core.dataset import caption_io, profiles, structured, tagger, terms, trainer_export


def create_captions_tab(initial_agent_team_choices, initial_vision_models):
    """Creates the Gradio components for the Captions Editor Tab."""
    profile_choices = dui.profile_labels()
    preset_choices = dui.preset_labels()
    first_profile = profiles.get_profile(profile_choices[0] if profile_choices else None)
    first_preset = profiles.get_preset(preset_choices[0] if preset_choices else None)
    first_style = first_profile.get("style", profiles.STYLE_PROSE)
    agent_default = "(Direct Agent Call)" if "(Direct Agent Call)" in (initial_agent_team_choices or []) else \
        (initial_agent_team_choices[0] if initial_agent_team_choices else None)

    with gr.Tab("Image Captions Editor"):
        view_state = gr.State([])        # Filenames shown in the gallery (in order)
        filter_state = gr.State("")      # Active term filter

        gr.Markdown("## Training Captions")
        with gr.Row():
            captions_folder_path = gr.Textbox(
                label="Image Folder Path", scale=4,
                placeholder="Full path to the folder with images and .txt captions",
            )
            captions_load_button = gr.Button("Load Folder", variant="secondary", scale=1)

        with gr.Row():
            profile_dropdown = gr.Dropdown(profile_choices, value=profile_choices[0] if profile_choices else None,
                                           label="Target model", scale=3, elem_id="caption_profile")
            preset_dropdown = gr.Dropdown(preset_choices, value=preset_choices[0] if preset_choices else None,
                                          label="Dataset type", scale=3, elem_id="caption_preset")
            trigger_textbox = gr.Textbox(label="Trigger word", placeholder="e.g. txcl", scale=2)
            class_textbox = gr.Textbox(label="Class noun", placeholder="e.g. painting / woman / sneaker", scale=2)
        with gr.Accordion("Guidance for this model and dataset type", open=False):
            guidance_md = gr.Markdown(profiles.guidance_markdown(first_profile, first_preset))

        with gr.Row():
            with gr.Column(scale=3):
                filter_md = gr.Markdown("")
                captions_image_gallery = gr.Gallery(
                    label="Images", show_label=False, object_fit="contain", height=560, columns=5,
                    preview=True, elem_id="caption_gallery",
                )
            with gr.Column(scale=2):
                caption_selected_filename_display = gr.Textbox(label="Selected Image Filename", interactive=False)
                captions_caption_display = gr.Textbox(
                    label="Caption / Text Content", lines=7, interactive=True,
                    placeholder="Click an image in the gallery to view/edit its caption...",
                )
                token_md = gr.Markdown("", elem_id="caption_tokens")
                info_md = gr.Markdown("", elem_id="caption_info")
                captions_save_button = gr.Button("Save THIS Caption", variant="primary")
                captions_status_display = gr.Textbox(label="Status", interactive=False, lines=4)

        with gr.Tabs():
            # ---- AI captioning ----
            with gr.Tab("AI Captioning"):
                with gr.Row():
                    source_radio = gr.Radio(list(dtu.SOURCE_CHOICES), value=dtu.SOURCE_DEFAULT, label="Caption source", scale=3)
                    backend_radio = gr.Radio(list(dtu.BACKEND_CHOICES), value=dtu.BACKEND_DEFAULT, label="Vision backend", scale=2)
                source_note_md = gr.Markdown("")
                with gr.Group(visible=False) as openai_group:
                    with gr.Row():
                        oa_url = gr.Textbox(label="Server URL", value="http://localhost:1234/v1", scale=3,
                                            info="LM Studio :1234, vLLM :8000, llama.cpp server :8080 (…/v1)")
                        oa_model = gr.Dropdown([], label="Model", allow_custom_value=True, scale=3)
                        oa_key = gr.Textbox(label="API key (optional)", type="password", scale=2,
                                            info="Not saved. Or set ARTAGENTS_VISION_API_KEY.")
                        oa_list_button = gr.Button("🔄 List models", scale=1)
                    oa_status_md = gr.Markdown("")
                with gr.Row() as ollama_row:
                    caption_model_selector = gr.Dropdown(
                        choices=initial_vision_models, label="Vision Model",
                        value=initial_vision_models[0] if initial_vision_models else None, scale=2,
                    )
                    caption_agent_selector = gr.Dropdown(
                        choices=initial_agent_team_choices, label="Agent / Team", value=agent_default, scale=2,
                        info="(Direct Agent Call) sends the instruction straight to the model; a team can tag, caption and clean up in steps.",
                    )
                with gr.Accordion("Booru tagger", open=True, visible=False) as tagger_group:
                    tagger_labels = tagger.tagger_labels()
                    with gr.Row():
                        tagger_dropdown = gr.Dropdown(tagger_labels, value=tagger_labels[0] if tagger_labels else None,
                                                      label="Tagger model", scale=3)
                        tagger_reuse = gr.Checkbox(value=True, label="Reuse stored results (change thresholds without re-running)", scale=2)
                    tagger_info_md = gr.Markdown(tagger.tagger_info_markdown(tagger_labels[0]) if tagger_labels else "")
                    with gr.Row():
                        general_thr = gr.Slider(0.05, 0.95, value=0.35, step=0.01, label="General threshold")
                        general_mcut = gr.Checkbox(value=False, label="MCut (adaptive)")
                        character_thr = gr.Slider(0.05, 0.99, value=0.85, step=0.01, label="Character threshold")
                        character_mcut = gr.Checkbox(value=False, label="MCut for characters")
                    with gr.Row():
                        hint_thr = gr.Slider(0.1, 0.99, value=0.5, step=0.01, label="Hint threshold (tags given to the vision model)")
                        add_rating = gr.Checkbox(value=False, label="Add rating tag (named for the target model)")
                        split_series = gr.Checkbox(value=False, label="Split 'character (series)' into two tags")
                    tagger_blacklist = gr.Textbox(label="Never output these tags (comma-separated)",
                                                  placeholder="e.g. simple background, white background")
                with gr.Accordion("Structured fields", open=True, visible=False) as structured_group:
                    gr.Markdown("The model fills these fields as JSON (stored per image); the caption is assembled from the "
                                "template. Edit fields later in the **Attributes** tab and re-render without calling the model.")
                    field_checkboxes = gr.CheckboxGroup(dtu.FIELD_LABELS, value=dtu.DEFAULT_FIELD_LABELS, label="Fields to ask for")
                    field_template = gr.Textbox(label="Caption template ({field} placeholders; empty fields are dropped)",
                                                value=structured.default_template(structured.DEFAULT_FIELDS, first_style))
                with gr.Row():
                    style_radio = gr.Radio(list(profiles.STYLE_LABELS.values()),
                                           value=profiles.STYLE_LABELS[first_style], label="Caption format", scale=2)
                    word_target = gr.Slider(0, 200, value=first_profile.get("word_target") or 0, step=5,
                                            label="Length target (words or tags; 0 = no target)", scale=3)
                mismatch_md = gr.Markdown("")
                caption_options_box = gr.CheckboxGroup(
                    [label for label, _ in profiles.CAPTION_OPTIONS.values()],
                    value=profiles.default_options(first_profile, first_preset), label="Caption rules",
                )
                with gr.Row():
                    omit_textbox = gr.Textbox(
                        label="Never mention (comma-separated)", value=", ".join(first_preset.get("omit_terms") or []),
                        info="Told to the model; matching tags are removed, prose is flagged.", scale=3,
                    )
                    extra_textbox = gr.Textbox(label="Extra instruction", placeholder="Anything else the captioner should follow", scale=2)
                caption_instruction = gr.Textbox(
                    label="Caption Instruction (built from the settings above; edit freely)",
                    value=profiles.build_instruction(first_profile, first_preset, first_style,
                                                     profiles.default_options(first_profile, first_preset),
                                                     first_profile.get("word_target"), first_preset.get("omit_terms") or []),
                    lines=4,
                )
                with gr.Row():
                    caption_prefix = gr.Textbox(label="Prefix (trigger)", placeholder="filled from trigger + class noun", scale=3)
                    quality_checkbox = gr.Checkbox(
                        label="Add quality tags (none for this model)", value=False,
                        interactive=bool(first_profile.get("quality_prefix")), scale=2,
                    )
                    caption_suffix = gr.Textbox(label="Suffix", placeholder="e.g. ', film grain'", scale=2)
                with gr.Row():
                    caption_generate_mode = gr.Radio(caption_io.FILE_MODES, value=caption_io.MODE_OVERWRITE,
                                                     label="If a caption already exists", scale=4)
                    joiner_dropdown = gr.Dropdown(list(caption_io.JOINER_CHOICES), value=dui.joiner_label_for(first_style),
                                                  label="Join appended text with", scale=1)
                with gr.Accordion("Reproducibility & retries", open=False):
                    with gr.Row():
                        temperature_slider = gr.Slider(0, 1.5, value=0, step=0.05, label="Temperature (0 = most consistent)")
                        seed_number = gr.Number(value=42, precision=0, label="Seed (-1 = random)")
                        max_side_slider = gr.Slider(0, 2048, value=1024, step=64, label="Downscale images to (px, 0 = off)")
                        retries_slider = gr.Slider(0, 3, value=2, step=1, label="Retries on glitches")
                with gr.Row():
                    caption_generate_selected_button = gr.Button("Generate Caption for SELECTED Image", variant="secondary")
                    caption_generate_all_button = gr.Button("Generate Captions for ALL Loaded Images", variant="secondary")
                    caption_stop_button = gr.Button("⏹ Stop", variant="stop", elem_id="caption_stop")

            # ---- Tags & terms ----
            with gr.Tab("Tags & Terms"):
                gr.Markdown("How often each tag/word appears across the dataset. **Click a row** to show only those images. "
                            "Terms in most images can bind to your trigger; singletons are often noise.")
                with gr.Row():
                    term_mode_radio = gr.Radio(terms.TERM_MODES, value=dui.term_mode_for(first_style), label="Count", scale=3)
                    refresh_terms_button = gr.Button("🔄 Refresh", scale=1)
                    clear_filter_button = gr.Button("Show all images", scale=1)
                common_md = gr.Markdown("")
                term_table = gr.Dataframe(headers=dui.TERM_HEADERS, datatype=["str", "number", "number", "str"],
                                          interactive=False, wrap=True, max_height=360, elem_id="term_table")
                with gr.Row():
                    bulk_op_radio = gr.Radio(dui.BULK_OPS, value=dui.BULK_RENAME, label="Bulk edit", scale=3)
                    bulk_scope_radio = gr.Radio([dui.FILTER_SCOPE, dui.ALL_SCOPE], value=dui.ALL_SCOPE, label="Apply to", scale=2)
                with gr.Row():
                    bulk_find = gr.Textbox(label="Tag / text to find", scale=2)
                    bulk_replace = gr.Textbox(label="Replace with (new tag name)", scale=2)
                    bulk_regex = gr.Checkbox(label="Regular expression", value=False, scale=1)
                    bulk_apply_button = gr.Button("Apply (snapshot first)", variant="secondary", scale=1)

            # ---- Batch edit, normalize & export ----
            with gr.Tab("Batch Edit & Export"):
                gr.Markdown("Add text such as a trigger word or style tag to existing captions. Append/Prepend stay on the "
                            "same line using the joiner from the AI Captioning tab; *Add as caption variant* adds a new "
                            "line, which OneTrainer/SimpleTuner read as an alternative caption.")
                with gr.Row():
                    captions_batch_scope = gr.Radio([BATCH_SCOPE_ALL, BATCH_SCOPE_SELECTED], value=BATCH_SCOPE_ALL, label="Apply to")
                    captions_batch_text = gr.Textbox(label="Text to Add")
                with gr.Row():
                    captions_prepend_button = gr.Button("Prepend Text")
                    captions_append_button = gr.Button("Append Text")
                    captions_variant_button = gr.Button("Add as Caption Variant")
                gr.Markdown("**Normalize caption files**: fixes files with several lines or the old '---' separators "
                            "written by earlier ArtAgents versions.")
                with gr.Row():
                    normalize_how = gr.Radio([dui.NORMALIZE_MERGE, dui.NORMALIZE_VARIANTS], value=dui.NORMALIZE_MERGE, label="How", scale=3)
                    normalize_button = gr.Button("Normalize (snapshot first)", scale=1)
                with gr.Row():
                    captions_export_format = gr.Dropdown(["metadata.jsonl", "metadata.csv"], value="metadata.jsonl",
                                                         label="Export captions for training (written into the image folder)", scale=3)
                    captions_export_button = gr.Button("Export", scale=1)
                with gr.Accordion("Trainer config", open=False):
                    gr.Markdown("Writes a dataset config for your trainer into the folder's `.artagents/export`. "
                                "Your folders are never renamed; repeats go into the config.")
                    with gr.Row():
                        trainer_dropdown = gr.Dropdown(list(trainer_export.TRAINER_LABELS), value=list(trainer_export.TRAINER_LABELS)[0],
                                                       label="Trainer", scale=3)
                        tr_resolution = gr.Number(value=1024, precision=0, label="Resolution", scale=1)
                        tr_repeats = gr.Number(value=1, precision=0, label="Repeats", scale=1)
                        tr_batch = gr.Number(value=1, precision=0, label="Batch size", scale=1)
                    with gr.Row():
                        tr_keep = gr.Number(value=1, precision=0, label="Keep tokens (from the prefix)", scale=1)
                        tr_shuffle = gr.Checkbox(value=first_style == "tags", label="Shuffle tags", scale=1)
                        tr_dropout = gr.Slider(0, 0.3, value=0.05, step=0.01, label="Caption dropout", scale=2)
                        tr_min_bucket = gr.Number(value=256, precision=0, label="Min bucket", scale=1)
                        tr_max_bucket = gr.Number(value=2048, precision=0, label="Max bucket", scale=1)
                    trainer_write_button = gr.Button("Write config")
                    trainer_code = gr.Code(value="", label="Config", interactive=False)
                with gr.Accordion("Repeats calculator", open=False):
                    with gr.Row():
                        rep_target = gr.Number(value=100, precision=0, label="Target images per epoch per concept", scale=2)
                        rep_epochs = gr.Number(value=10, precision=0, label="Epochs", scale=1)
                        rep_batch = gr.Number(value=1, precision=0, label="Batch size", scale=1)
                    rep_other = gr.Textbox(label="Other concept folders (one 'name: image count' per line)", lines=2,
                                           placeholder="style_b: 35")
                    rep_button = gr.Button("Calculate")
                    rep_md = gr.Markdown("")

            # ---- Attributes (structured fields) ----
            with gr.Tab("Attributes"):
                gr.Markdown("Structured fields per image (from the *Structured fields* caption source). Values that are "
                            "the same in most images are flagged: caption them to keep them changeable, or leave them "
                            "out so the trigger learns them.")
                with gr.Row():
                    sheet_load_button = gr.Button("🔄 Load sheet", scale=1)
                    sheet_save_button = gr.Button("Save sheet", scale=1)
                    sheet_export_button = gr.Button("Export CSV", scale=1)
                sheet_columns_state = gr.State([])
                sheet_table = gr.Dataframe(value=[[""]], headers=["image"], interactive=True, wrap=True, max_height=380,
                                           elem_id="attribute_sheet")
                sheet_summary = gr.Dataframe(headers=dtu.SHEET_SUMMARY_HEADERS, interactive=False, wrap=True,
                                             elem_id="attribute_summary")
                sheet_file = gr.File(label="attributes.csv", visible=True, interactive=False)
                gr.Markdown("**Render captions from fields** (snapshot first). Unticked fields are left out of the captions.")
                with gr.Row():
                    render_fields = gr.CheckboxGroup(dtu.FIELD_LABELS, value=dtu.DEFAULT_FIELD_LABELS, label="Fields in captions", scale=4)
                    render_scope = gr.Radio([dui.FILTER_SCOPE, dui.ALL_SCOPE], value=dui.ALL_SCOPE, label="Apply to", scale=1)
                render_template = gr.Textbox(label="Template (empty = fields in order)", value="")
                render_button = gr.Button("Render captions (uses the trigger prefix/suffix from AI Captioning)")

            # ---- Dataset health ----
            with gr.Tab("Health & Audit"):
                gr.Markdown("Checks run on your computer (no AI). Flags are relative to this dataset and are suggestions: "
                            "nothing is deleted. Quarantine moves an image to `.artagents/quarantine`.")
                with gr.Row():
                    health_resolution = gr.Number(value=1024, precision=0, label="Training resolution", scale=1)
                    health_button = gr.Button("Scan dataset", variant="primary", scale=1)
                    health_show_button = gr.Button("Show flagged images", scale=1)
                    quarantine_button = gr.Button("Quarantine selected image", scale=1)
                    unquarantine_button = gr.Button("Restore quarantined", scale=1)
                health_flagged_state = gr.State([])
                health_md = gr.Markdown("")
                health_table = gr.Dataframe(headers=dtu.ISSUE_HEADERS, interactive=False, wrap=True, max_height=320,
                                            elem_id="health_table")
                with gr.Accordion("Bucket simulator (kohya-style aspect-ratio buckets)", open=False):
                    with gr.Row():
                        bk_resolution = gr.Number(value=1024, precision=0, label="Resolution")
                        bk_min = gr.Number(value=256, precision=0, label="Min bucket")
                        bk_max = gr.Number(value=2048, precision=0, label="Max bucket")
                        bk_batch = gr.Number(value=1, precision=0, label="Batch size")
                        bk_repeats = gr.Number(value=1, precision=0, label="Repeats")
                        bk_no_upscale = gr.Checkbox(value=False, label="bucket_no_upscale")
                    bk_button = gr.Button("Simulate buckets")
                    bk_md = gr.Markdown("")
                    bk_buckets = gr.Dataframe(headers=dtu.BUCKET_HEADERS, interactive=False, wrap=True, elem_id="bucket_table")
                    bk_images = gr.Dataframe(headers=dtu.BUCKET_IMAGE_HEADERS, interactive=False, wrap=True, max_height=320)
                with gr.Accordion("Caption consistency audit", open=False):
                    with gr.Row():
                        audit_triggers = gr.Textbox(label="Other triggers (multi-concept, comma-separated)", scale=3)
                        audit_use_tagger = gr.Checkbox(value=True, label="Compare with tagger results", scale=1)
                        audit_tagger_button = gr.Button("Run tagger for comparison", scale=1)
                    audit_button = gr.Button("Run audit")
                    audit_md = gr.Markdown("")
                    audit_table = gr.Dataframe(headers=dtu.AUDIT_HEADERS, interactive=False, wrap=True, max_height=320,
                                               elem_id="audit_table")
                with gr.Accordion("Cross-model verifier (experimental)", open=False):
                    gr.Markdown("A second vision model answers yes / no for each statement in a caption, plus two "
                                "statements about objects that are not mentioned as controls. Use a model from a "
                                "**different family** than the captioner. Results are hints; captions are not changed.")
                    with gr.Row():
                        verifier_model = gr.Dropdown(list(initial_vision_models or []) + [dtu.VERIFIER_OPENAI],
                                                     label="Verifier model", scale=3)
                        verify_scope = gr.Radio(["Selected image", dui.FILTER_SCOPE, dui.ALL_SCOPE], value="Selected image",
                                                label="Verify", scale=3)
                        verify_button = gr.Button("Verify", scale=1)
                        verify_stop = gr.Button("⏹ Stop", variant="stop", scale=1)
                    verify_status = gr.Markdown("")
                    verify_table = gr.Dataframe(headers=dtu.VERIFY_HEADERS, interactive=False, wrap=True, max_height=320,
                                                elem_id="verify_table")

            # ---- Edit pairs ----
            with gr.Tab("Edit Pairs"):
                gr.Markdown("For edit models (FLUX.1 Kontext, FLUX.2 edit, Qwen-Image-Edit): the **result** folder holds "
                            "the edited images, the **input** folder(s) the originals with the same file names. Each "
                            "result gets an instruction caption; inputs get none. Uses the vision model and backend "
                            "from AI Captioning.")
                with gr.Row():
                    pair_target = gr.Textbox(label="Result folder (targets)", scale=3)
                    pair_controls = gr.Textbox(label="Input folder(s), one per line (up to 3)", lines=2, scale=3)
                    pair_load_button = gr.Button("Load pairs", scale=1)
                pairs_state = gr.State([])
                pair_index_state = gr.State(None)
                pair_status = gr.Markdown("")
                pair_gallery = gr.Gallery(label="Pairs (input → result)", columns=6, height=360, allow_preview=False,
                                          elem_id="pair_gallery")
                pair_table = gr.Dataframe(headers=dtu.PAIR_HEADERS, interactive=False, wrap=True, max_height=200)
                with gr.Row():
                    pair_instruction = gr.Textbox(label="Instruction for the selected pair (one phrasing per line)", lines=3, scale=4)
                    pair_save_button = gr.Button("Save instruction", scale=1)
                with gr.Row():
                    pair_paraphrases = gr.Slider(0, 10, value=0, step=1, label="Extra phrasings (BFL suggests 5–10 in total)", scale=2)
                    pair_trigger = gr.Textbox(label="Trigger prefix (optional)", scale=2)
                    pair_mode = gr.Radio([caption_io.MODE_OVERWRITE, caption_io.MODE_SKIP], value=caption_io.MODE_SKIP,
                                         label="If an instruction exists", scale=2)
                pair_extra = gr.Textbox(label="Extra instruction for the model", placeholder="e.g. Mention the new colour exactly.")
                with gr.Row():
                    pair_scope = gr.Radio(["Selected pair", "All pairs"], value="All pairs", label="Write instructions for", scale=2)
                    pair_generate_button = gr.Button("Write instructions", variant="primary", scale=1)
                    pair_stop = gr.Button("⏹ Stop", variant="stop", scale=1)
                with gr.Row():
                    pair_export_format = gr.Dropdown(dtu.EDIT_EXPORTS, value=dtu.EDIT_EXPORTS[0], label="Export for", scale=3)
                    pair_export_button = gr.Button("Export", scale=1)
                pair_export_code = gr.Code(value="", label="Export", interactive=False)
                pair_export_file = gr.File(label="Exported file", interactive=False)

            # ---- Snapshots ----
            with gr.Tab("Snapshots"):
                gr.Markdown("A snapshot of all captions is saved automatically before every batch change "
                            "(in the folder's hidden `.artagents/snapshots`). Restoring also snapshots the current state first.")
                with gr.Row():
                    snapshot_dropdown = gr.Dropdown([], label="Snapshot", scale=4, elem_id="snapshot_dropdown")
                    snapshot_refresh_button = gr.Button("🔄", scale=0, min_width=50)
                    snapshot_diff_button = gr.Button("Compare with current", scale=1)
                    snapshot_restore_button = gr.Button("Restore", variant="stop", scale=1)
                snapshot_diff_md = gr.Markdown("")

    return {
        "view_state": view_state, "filter_state": filter_state,
        "captions_folder_path": captions_folder_path, "captions_load_button": captions_load_button,
        "profile_dropdown": profile_dropdown, "preset_dropdown": preset_dropdown,
        "trigger_textbox": trigger_textbox, "class_textbox": class_textbox, "guidance_md": guidance_md,
        "filter_md": filter_md, "captions_image_gallery": captions_image_gallery,
        "caption_selected_filename_display": caption_selected_filename_display,
        "captions_caption_display": captions_caption_display, "token_md": token_md, "info_md": info_md,
        "captions_save_button": captions_save_button, "captions_status_display": captions_status_display,
        "caption_model_selector": caption_model_selector, "caption_agent_selector": caption_agent_selector,
        "style_radio": style_radio, "word_target": word_target, "mismatch_md": mismatch_md,
        "caption_options_box": caption_options_box, "omit_textbox": omit_textbox, "extra_textbox": extra_textbox,
        "caption_instruction": caption_instruction, "caption_prefix": caption_prefix,
        "quality_checkbox": quality_checkbox, "caption_suffix": caption_suffix,
        "caption_generate_mode": caption_generate_mode, "joiner_dropdown": joiner_dropdown,
        "temperature_slider": temperature_slider, "seed_number": seed_number,
        "max_side_slider": max_side_slider, "retries_slider": retries_slider,
        "caption_generate_selected_button": caption_generate_selected_button,
        "caption_generate_all_button": caption_generate_all_button, "caption_stop_button": caption_stop_button,
        "term_mode_radio": term_mode_radio, "refresh_terms_button": refresh_terms_button,
        "clear_filter_button": clear_filter_button, "common_md": common_md, "term_table": term_table,
        "bulk_op_radio": bulk_op_radio, "bulk_scope_radio": bulk_scope_radio, "bulk_find": bulk_find,
        "bulk_replace": bulk_replace, "bulk_regex": bulk_regex, "bulk_apply_button": bulk_apply_button,
        "captions_batch_scope": captions_batch_scope, "captions_batch_text": captions_batch_text,
        "captions_prepend_button": captions_prepend_button, "captions_append_button": captions_append_button,
        "captions_variant_button": captions_variant_button, "normalize_how": normalize_how,
        "normalize_button": normalize_button, "captions_export_format": captions_export_format,
        "captions_export_button": captions_export_button, "snapshot_dropdown": snapshot_dropdown,
        "snapshot_refresh_button": snapshot_refresh_button, "snapshot_diff_button": snapshot_diff_button,
        "snapshot_restore_button": snapshot_restore_button, "snapshot_diff_md": snapshot_diff_md,
        # Tier 2
        "source_radio": source_radio, "backend_radio": backend_radio, "source_note_md": source_note_md,
        "openai_group": openai_group, "oa_url": oa_url, "oa_model": oa_model, "oa_key": oa_key,
        "oa_list_button": oa_list_button, "oa_status_md": oa_status_md, "ollama_row": ollama_row,
        "tagger_group": tagger_group, "tagger_dropdown": tagger_dropdown, "tagger_reuse": tagger_reuse,
        "tagger_info_md": tagger_info_md, "general_thr": general_thr, "general_mcut": general_mcut,
        "character_thr": character_thr, "character_mcut": character_mcut, "hint_thr": hint_thr,
        "add_rating": add_rating, "split_series": split_series, "tagger_blacklist": tagger_blacklist,
        "structured_group": structured_group, "field_checkboxes": field_checkboxes, "field_template": field_template,
        "trainer_dropdown": trainer_dropdown, "tr_resolution": tr_resolution, "tr_repeats": tr_repeats,
        "tr_batch": tr_batch, "tr_keep": tr_keep, "tr_shuffle": tr_shuffle, "tr_dropout": tr_dropout,
        "tr_min_bucket": tr_min_bucket, "tr_max_bucket": tr_max_bucket, "trainer_write_button": trainer_write_button,
        "trainer_code": trainer_code, "rep_target": rep_target, "rep_epochs": rep_epochs, "rep_batch": rep_batch,
        "rep_other": rep_other, "rep_button": rep_button, "rep_md": rep_md,
        "sheet_load_button": sheet_load_button, "sheet_save_button": sheet_save_button,
        "sheet_export_button": sheet_export_button, "sheet_columns_state": sheet_columns_state,
        "sheet_table": sheet_table, "sheet_summary": sheet_summary, "sheet_file": sheet_file,
        "render_fields": render_fields, "render_scope": render_scope, "render_template": render_template,
        "render_button": render_button,
        "health_resolution": health_resolution, "health_button": health_button, "health_show_button": health_show_button,
        "quarantine_button": quarantine_button, "unquarantine_button": unquarantine_button,
        "health_flagged_state": health_flagged_state, "health_md": health_md, "health_table": health_table,
        "bk_resolution": bk_resolution, "bk_min": bk_min, "bk_max": bk_max, "bk_batch": bk_batch,
        "bk_repeats": bk_repeats, "bk_no_upscale": bk_no_upscale, "bk_button": bk_button, "bk_md": bk_md,
        "bk_buckets": bk_buckets, "bk_images": bk_images,
        "audit_triggers": audit_triggers, "audit_use_tagger": audit_use_tagger, "audit_tagger_button": audit_tagger_button,
        "audit_button": audit_button, "audit_md": audit_md, "audit_table": audit_table,
        "verifier_model": verifier_model, "verify_scope": verify_scope, "verify_button": verify_button,
        "verify_stop": verify_stop, "verify_status": verify_status, "verify_table": verify_table,
        "pair_target": pair_target, "pair_controls": pair_controls, "pair_load_button": pair_load_button,
        "pairs_state": pairs_state, "pair_index_state": pair_index_state, "pair_status": pair_status,
        "pair_gallery": pair_gallery, "pair_table": pair_table, "pair_instruction": pair_instruction,
        "pair_save_button": pair_save_button, "pair_paraphrases": pair_paraphrases, "pair_trigger": pair_trigger,
        "pair_mode": pair_mode, "pair_extra": pair_extra, "pair_scope": pair_scope,
        "pair_generate_button": pair_generate_button, "pair_stop": pair_stop, "pair_export_format": pair_export_format,
        "pair_export_button": pair_export_button, "pair_export_code": pair_export_code, "pair_export_file": pair_export_file,
    }


def wire_captions_tab(c, *, paths_state, captions_state, selected_state, settings_state, models_data_state,
                      limiters_data_state, teams_data_state, file_agents_state, history_list_state, session_history_state):
    """Registers all Captions-tab events. `c` is the dict returned by create_captions_tab."""
    status = c["captions_status_display"]
    editor_outputs = [c["captions_caption_display"], c["token_md"], c["info_md"]]
    sync_inputs = [selected_state, captions_state, paths_state, c["profile_dropdown"]]
    term_inputs = [captions_state, c["term_mode_radio"], c["trigger_textbox"], c["class_textbox"]]

    # --- profile / preset / instruction composition ---
    c["profile_dropdown"].change(
        fn=dui.on_profile_or_preset_change,
        inputs=[c["profile_dropdown"], c["preset_dropdown"], c["trigger_textbox"], c["class_textbox"],
                c["captions_caption_display"], c["extra_textbox"]],
        outputs=[c["style_radio"], c["caption_options_box"], c["word_target"], c["omit_textbox"], c["caption_prefix"],
                 c["quality_checkbox"], c["caption_instruction"], c["mismatch_md"], c["guidance_md"],
                 c["joiner_dropdown"], c["term_mode_radio"], c["token_md"], c["class_textbox"]],
    )
    c["preset_dropdown"].change(
        fn=dui.on_profile_or_preset_change,
        inputs=[c["profile_dropdown"], c["preset_dropdown"], c["trigger_textbox"], c["class_textbox"],
                c["captions_caption_display"], c["extra_textbox"]],
        outputs=[c["style_radio"], c["caption_options_box"], c["word_target"], c["omit_textbox"], c["caption_prefix"],
                 c["quality_checkbox"], c["caption_instruction"], c["mismatch_md"], c["guidance_md"],
                 c["joiner_dropdown"], c["term_mode_radio"], c["token_md"], c["class_textbox"]],
    )
    instruction_inputs = [c["profile_dropdown"], c["preset_dropdown"], c["style_radio"], c["caption_options_box"],
                          c["word_target"], c["omit_textbox"], c["class_textbox"], c["extra_textbox"]]
    for component in (c["style_radio"], c["caption_options_box"], c["word_target"], c["omit_textbox"],
                      c["class_textbox"], c["extra_textbox"]):
        component.change(fn=dui.compose_instruction, inputs=instruction_inputs,
                         outputs=[c["caption_instruction"], c["mismatch_md"]])
    prefix_inputs = [c["preset_dropdown"], c["profile_dropdown"], c["style_radio"], c["trigger_textbox"],
                     c["class_textbox"], c["quality_checkbox"]]
    for component in (c["trigger_textbox"], c["class_textbox"], c["quality_checkbox"]):
        component.change(fn=dui.compose_prefix, inputs=prefix_inputs, outputs=[c["caption_prefix"]])

    # --- loading, selecting, editing ---
    c["captions_load_button"].click(
        fn=dui.load_folder,
        inputs=[c["captions_folder_path"], c["profile_dropdown"], c["term_mode_radio"], c["trigger_textbox"], c["class_textbox"]],
        outputs=[c["captions_image_gallery"], paths_state, captions_state, status, selected_state,
                 c["captions_caption_display"], c["caption_selected_filename_display"], c["view_state"], c["filter_state"],
                 c["filter_md"], c["term_table"], c["snapshot_dropdown"], c["token_md"], c["info_md"]],
    )
    c["captions_image_gallery"].select(
        fn=dui.select_image,
        inputs=[c["view_state"], captions_state, paths_state, c["profile_dropdown"]],
        outputs=[c["captions_caption_display"], selected_state, c["caption_selected_filename_display"], c["token_md"], c["info_md"]],
    )
    c["captions_caption_display"].change(fn=dui.token_report, inputs=[c["captions_caption_display"], c["profile_dropdown"]],
                                         outputs=[c["token_md"]])
    c["captions_save_button"].click(
        fn=save_caption, inputs=[selected_state, c["captions_caption_display"], paths_state, captions_state],
        outputs=[status, captions_state],
    ).then(fn=dui.term_rows, inputs=term_inputs, outputs=[c["term_table"]]
    ).then(fn=dui.sync_editor, inputs=sync_inputs, outputs=editor_outputs)

    # --- AI captioning ---
    gen_extra = [c["caption_instruction"], c["caption_prefix"], c["caption_suffix"], c["profile_dropdown"],
                 c["preset_dropdown"], c["style_radio"], c["joiner_dropdown"], c["omit_textbox"],
                 c["temperature_slider"], c["seed_number"], c["max_side_slider"], c["retries_slider"]]
    shared = [settings_state, models_data_state, limiters_data_state, teams_data_state, file_agents_state,
              history_list_state, session_history_state]
    source_inputs = [c["source_radio"], c["backend_radio"], c["oa_url"], c["oa_model"], c["oa_key"], c["tagger_dropdown"],
                     c["general_thr"], c["character_thr"], c["general_mcut"], c["character_mcut"], c["hint_thr"],
                     c["add_rating"], c["split_series"], c["tagger_blacklist"], c["tagger_reuse"],
                     c["field_checkboxes"], c["field_template"]]
    gen_extra = gen_extra + source_inputs
    c["caption_generate_selected_button"].click(
        fn=dtu.generate_selected,
        inputs=[selected_state, c["caption_agent_selector"], c["caption_model_selector"], c["caption_generate_mode"],
                paths_state, captions_state] + shared + gen_extra,
        outputs=[status, captions_state, c["captions_caption_display"], session_history_state],
    ).then(fn=dui.term_rows, inputs=term_inputs, outputs=[c["term_table"]]
    ).then(fn=dui.sync_editor, inputs=sync_inputs, outputs=editor_outputs)
    all_event = c["caption_generate_all_button"].click(
        fn=dtu.generate_all,
        inputs=[paths_state, captions_state, c["caption_agent_selector"], c["caption_model_selector"],
                c["caption_generate_mode"]] + shared + gen_extra,
        outputs=[status, captions_state, c["captions_caption_display"], session_history_state],
    )
    # The batch shows the LAST image's caption; switch back to the selected image afterwards (also after Stop)
    all_event.then(fn=dui.sync_editor, inputs=sync_inputs, outputs=editor_outputs
    ).then(fn=dui.term_rows, inputs=term_inputs, outputs=[c["term_table"]]
    ).then(fn=dui.snapshot_dropdown, inputs=[paths_state], outputs=[c["snapshot_dropdown"]])
    c["caption_stop_button"].click(fn=None, cancels=[all_event]).then(
        fn=dui.sync_editor, inputs=sync_inputs, outputs=editor_outputs)

    # --- caption source, backend, tagger, structured ---
    source_outputs = [c["tagger_group"], c["structured_group"], c["openai_group"], c["ollama_row"], c["source_note_md"],
                      c["field_template"]]
    for component in (c["source_radio"], c["backend_radio"]):
        component.change(fn=dtu.on_source_change,
                         inputs=[c["source_radio"], c["backend_radio"], c["field_checkboxes"], c["style_radio"]],
                         outputs=source_outputs)
    for component in (c["field_checkboxes"], c["style_radio"]):
        component.change(fn=dtu.on_fields_change, inputs=[c["field_checkboxes"], c["style_radio"]],
                         outputs=[c["field_template"]])
    c["oa_list_button"].click(fn=dtu.list_openai_models, inputs=[c["oa_url"], c["oa_key"]],
                              outputs=[c["oa_model"], c["oa_status_md"]])
    c["tagger_dropdown"].change(fn=dtu.tagger_info, inputs=[c["tagger_dropdown"]], outputs=[c["tagger_info_md"]])

    # --- tags & terms ---
    c["refresh_terms_button"].click(fn=dui.term_rows, inputs=term_inputs, outputs=[c["term_table"]])
    for component in (c["trigger_textbox"], c["class_textbox"]):  # Trigger/class terms are never flagged
        component.change(fn=dui.term_rows, inputs=term_inputs, outputs=[c["term_table"]])
    c["term_mode_radio"].change(fn=dui.term_rows, inputs=term_inputs, outputs=[c["term_table"]])
    c["term_table"].select(
        fn=dui.filter_by_term, inputs=[captions_state, paths_state, c["term_mode_radio"]],
        outputs=[c["captions_image_gallery"], c["view_state"], c["filter_state"], c["filter_md"], c["common_md"]],
    )
    c["clear_filter_button"].click(
        fn=dui.clear_filter, inputs=[paths_state],
        outputs=[c["captions_image_gallery"], c["view_state"], c["filter_state"], c["filter_md"], c["common_md"]],
    )
    c["bulk_apply_button"].click(
        fn=dui.bulk_edit,
        inputs=[c["bulk_op_radio"], c["bulk_find"], c["bulk_replace"], c["bulk_regex"], c["bulk_scope_radio"],
                c["view_state"], captions_state, paths_state, c["term_mode_radio"], c["trigger_textbox"], c["class_textbox"]],
        outputs=[status, captions_state, c["term_table"]],
    ).then(fn=dui.sync_editor, inputs=sync_inputs, outputs=editor_outputs
    ).then(fn=dui.snapshot_dropdown, inputs=[paths_state], outputs=[c["snapshot_dropdown"]])

    # --- batch add, normalize, export ---
    batch_inputs = [c["captions_batch_scope"], selected_state, c["captions_batch_text"]]
    batch_tail = [paths_state, captions_state, c["joiner_dropdown"], c["style_radio"], c["profile_dropdown"]]
    for button, mode in ((c["captions_append_button"], caption_io.MODE_APPEND),
                         (c["captions_prepend_button"], caption_io.MODE_PREPEND),
                         (c["captions_variant_button"], caption_io.MODE_VARIANT)):
        button.click(
            fn=lambda scope, sel, text, paths, caps, joiner, style, profile, _mode=mode:
                dui.batch_add(scope, sel, text, _mode, paths, caps, joiner, style, profile),
            inputs=batch_inputs + batch_tail, outputs=[status, captions_state],
        ).then(fn=dui.sync_editor, inputs=sync_inputs, outputs=editor_outputs
        ).then(fn=dui.term_rows, inputs=term_inputs, outputs=[c["term_table"]]
        ).then(fn=dui.snapshot_dropdown, inputs=[paths_state], outputs=[c["snapshot_dropdown"]])
    c["normalize_button"].click(
        fn=dui.normalize_files,
        inputs=[c["normalize_how"], c["joiner_dropdown"], c["style_radio"], c["profile_dropdown"], paths_state, captions_state],
        outputs=[status, captions_state],
    ).then(fn=dui.sync_editor, inputs=sync_inputs, outputs=editor_outputs
    ).then(fn=dui.term_rows, inputs=term_inputs, outputs=[c["term_table"]]
    ).then(fn=dui.snapshot_dropdown, inputs=[paths_state], outputs=[c["snapshot_dropdown"]])
    c["captions_export_button"].click(fn=export_caption_dataset,
                                      inputs=[paths_state, captions_state, c["captions_export_format"]], outputs=[status])

    # --- snapshots ---
    c["snapshot_refresh_button"].click(fn=dui.snapshot_dropdown, inputs=[paths_state], outputs=[c["snapshot_dropdown"]])
    c["snapshot_diff_button"].click(fn=dui.diff_with_snapshot, inputs=[c["snapshot_dropdown"], paths_state, captions_state],
                                    outputs=[c["snapshot_diff_md"]])
    c["snapshot_restore_button"].click(
        fn=dui.restore_from_snapshot,
        inputs=[c["snapshot_dropdown"], paths_state, captions_state, c["term_mode_radio"], c["trigger_textbox"], c["class_textbox"]],
        outputs=[status, captions_state, c["term_table"], c["snapshot_dropdown"]],
    ).then(fn=dui.sync_editor, inputs=sync_inputs, outputs=editor_outputs)

    # --- trainer export & repeats ---
    c["caption_prefix"].change(fn=dtu.keep_tokens_from_prefix, inputs=[c["caption_prefix"]], outputs=[c["tr_keep"]])
    c["trainer_write_button"].click(
        fn=dtu.write_trainer_config,
        inputs=[c["trainer_dropdown"], paths_state, captions_state, c["tr_resolution"], c["tr_repeats"], c["tr_batch"],
                c["tr_keep"], c["tr_shuffle"], c["tr_dropout"], c["tr_min_bucket"], c["tr_max_bucket"], c["trigger_textbox"]],
        outputs=[status, c["trainer_code"]],
    )
    c["rep_button"].click(fn=dtu.repeats_table,
                          inputs=[paths_state, c["rep_target"], c["rep_epochs"], c["rep_batch"], c["rep_other"]],
                          outputs=[c["rep_md"]])

    # --- attribute sheet ---
    sheet_outputs = [c["sheet_table"], c["sheet_columns_state"], c["sheet_summary"], status]
    c["sheet_load_button"].click(fn=dtu.load_sheet, inputs=[paths_state], outputs=sheet_outputs)
    c["sheet_save_button"].click(fn=dtu.save_sheet, inputs=[c["sheet_table"], c["sheet_columns_state"], paths_state],
                                 outputs=[status, c["sheet_summary"]])
    c["sheet_export_button"].click(fn=dtu.export_sheet, inputs=[c["sheet_table"], c["sheet_columns_state"], paths_state],
                                   outputs=[status, c["sheet_file"]])
    c["render_button"].click(
        fn=dtu.render_from_fields,
        inputs=[c["render_fields"], c["render_template"], c["style_radio"], c["profile_dropdown"], c["caption_prefix"],
                c["caption_suffix"], c["render_scope"], c["view_state"], paths_state, captions_state, c["joiner_dropdown"]],
        outputs=[status, captions_state],
    ).then(fn=dui.sync_editor, inputs=sync_inputs, outputs=editor_outputs
    ).then(fn=dui.term_rows, inputs=term_inputs, outputs=[c["term_table"]]
    ).then(fn=dui.snapshot_dropdown, inputs=[paths_state], outputs=[c["snapshot_dropdown"]])

    # --- health, buckets, audit, verifier ---
    c["health_button"].click(
        fn=dtu.run_health,
        inputs=[paths_state, captions_state, c["profile_dropdown"], c["trigger_textbox"], c["health_resolution"]],
        outputs=[c["health_md"], c["health_table"], c["health_flagged_state"]],
    )
    gallery_view_outputs = [c["captions_image_gallery"], c["view_state"], c["filter_state"], c["filter_md"], c["common_md"]]
    c["health_show_button"].click(fn=dtu.show_images, inputs=[c["health_flagged_state"], paths_state],
                                  outputs=gallery_view_outputs)
    c["quarantine_button"].click(
        fn=dtu.quarantine_selected, inputs=[selected_state, paths_state, captions_state],
        outputs=[status, paths_state, captions_state, c["captions_image_gallery"], c["view_state"], selected_state],
    ).then(fn=dui.sync_editor, inputs=sync_inputs, outputs=editor_outputs
    ).then(fn=dui.term_rows, inputs=term_inputs, outputs=[c["term_table"]])
    c["unquarantine_button"].click(fn=dtu.restore_quarantined, inputs=[paths_state, c["captions_folder_path"]],
                                   outputs=[status])
    c["bk_button"].click(
        fn=dtu.simulate_buckets,
        inputs=[paths_state, c["bk_resolution"], c["bk_min"], c["bk_max"], c["bk_batch"], c["bk_repeats"], c["bk_no_upscale"]],
        outputs=[c["bk_md"], c["bk_images"], c["bk_buckets"]],
    )
    c["audit_button"].click(
        fn=dtu.run_audit,
        inputs=[captions_state, paths_state, c["profile_dropdown"], c["preset_dropdown"], c["style_radio"],
                c["trigger_textbox"], c["class_textbox"], c["audit_triggers"], c["omit_textbox"], c["tagger_dropdown"],
                c["audit_use_tagger"]],
        outputs=[c["audit_md"], c["audit_table"]],
    )
    c["audit_tagger_button"].click(fn=dtu.run_tagger_compare, inputs=[paths_state, c["tagger_dropdown"]],
                                   outputs=[c["audit_md"]])
    verify_event = c["verify_button"].click(
        fn=dtu.iter_verify,
        inputs=[c["verify_scope"], selected_state, c["view_state"], paths_state, captions_state, settings_state,
                c["verifier_model"], c["caption_model_selector"], c["backend_radio"], c["oa_url"], c["oa_model"],
                c["oa_key"], c["profile_dropdown"], c["style_radio"], c["trigger_textbox"], c["class_textbox"]],
        outputs=[c["verify_status"], c["verify_table"]],
    )
    c["verify_stop"].click(fn=None, cancels=[verify_event])

    # --- edit pairs ---
    c["pair_load_button"].click(
        fn=dtu.load_pairs, inputs=[c["pair_target"], c["pair_controls"]],
        outputs=[c["pairs_state"], c["pair_gallery"], c["pair_table"], c["pair_status"], c["pair_index_state"],
                 c["pair_instruction"]],
    )
    c["pair_gallery"].select(fn=dtu.select_pair, inputs=[c["pairs_state"]],
                             outputs=[c["pair_index_state"], c["pair_instruction"]])
    c["pair_save_button"].click(fn=dtu.save_pair_instruction,
                                inputs=[c["pair_index_state"], c["pair_instruction"], c["pairs_state"]],
                                outputs=[c["pair_status"], c["pair_gallery"], c["pair_table"]])
    pair_event = c["pair_generate_button"].click(
        fn=dtu.iter_pair_instructions,
        inputs=[c["pair_scope"], c["pair_index_state"], c["pairs_state"], settings_state, c["caption_model_selector"],
                c["backend_radio"], c["oa_url"], c["oa_model"], c["oa_key"], c["pair_paraphrases"], c["pair_trigger"],
                c["pair_extra"], c["pair_mode"], c["temperature_slider"], c["seed_number"], c["max_side_slider"]],
        outputs=[c["pair_status"], c["pair_gallery"], c["pair_table"]],
    )
    c["pair_stop"].click(fn=None, cancels=[pair_event])
    c["pair_export_button"].click(
        fn=dtu.export_pairs,
        inputs=[c["pair_export_format"], c["pairs_state"], c["pair_target"], c["pair_controls"], c["trigger_textbox"]],
        outputs=[c["pair_status"], c["pair_export_code"], c["pair_export_file"]],
    )
