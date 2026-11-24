# ArtAgent/ui/captions_tab.py
"""Image Captions Editor: layout (create_captions_tab) and event wiring (wire_captions_tab)."""
import gradio as gr

from core import dataset_ui as dui
from core.captioning_logic import BATCH_SCOPE_ALL, BATCH_SCOPE_SELECTED, save_caption, export_caption_dataset
from core.dataset import caption_io, profiles, terms


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
                    caption_model_selector = gr.Dropdown(
                        choices=initial_vision_models, label="Vision Model",
                        value=initial_vision_models[0] if initial_vision_models else None, scale=2,
                    )
                    caption_agent_selector = gr.Dropdown(
                        choices=initial_agent_team_choices, label="Agent / Team", value=agent_default, scale=2,
                        info="(Direct Agent Call) sends the instruction straight to the model; a team can tag, caption and clean up in steps.",
                    )
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
    c["caption_generate_selected_button"].click(
        fn=dui.generate_selected,
        inputs=[selected_state, c["caption_agent_selector"], c["caption_model_selector"], c["caption_generate_mode"],
                paths_state, captions_state] + shared + gen_extra,
        outputs=[status, captions_state, c["captions_caption_display"], session_history_state],
    ).then(fn=dui.term_rows, inputs=term_inputs, outputs=[c["term_table"]]
    ).then(fn=dui.sync_editor, inputs=sync_inputs, outputs=editor_outputs)
    all_event = c["caption_generate_all_button"].click(
        fn=dui.generate_all,
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
