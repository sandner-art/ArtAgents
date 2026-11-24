# ArtAgent/ui/lora_lab_tab.py
"""LoRA Lab: validation prompts, LoRA test sweeps, metrics, blind votes and experiment records."""
import gradio as gr

from core import lab_ui
from core.image_gen_ui import default_backend
from core.lab import experiments, metrics, validation, voting


def create_lora_lab_tab(initial_settings, vision_models, text_models):
    backends = lab_ui.lab_backend_labels()
    start_backend = default_backend(initial_settings) if default_backend(initial_settings) in backends else backends[0]
    embedders = metrics.available_embedders()
    with gr.Tab("LoRA Lab", id="lora_lab"):
        gr.Markdown("Test what your captions and training settings did: render the same prompts and seeds with each "
                    "LoRA, strength and a no-LoRA baseline, compare grids, score them and vote blind. "
                    "Records keep the evidence together.")
        with gr.Tabs():
            # ---- validation prompts ----
            with gr.Tab("Validation Prompts"):
                gr.Markdown("Four buckets: **training-like** (fidelity), **held-out** (new contexts), **trigger-less** "
                            "(leakage) and **unrelated** (does the base model still work?). Use `[trigger]` in prompts. "
                            "Flags use the kohya / FluxGym syntax (`--w 832 --h 1216 --l 4.5 --s 28 --d 7`).")
                with gr.Row():
                    v_name = gr.Dropdown(validation.list_sets(), label="Set", allow_custom_value=True, scale=3)
                    v_version = gr.Dropdown([], label="Version", scale=1)
                    v_load = gr.Button("Load", scale=1)
                    v_refresh = gr.Button("🔄", scale=0, min_width=50)
                with gr.Row():
                    v_trigger = gr.Textbox(label="Trigger", scale=2)
                    v_class = gr.Textbox(label="Class noun", scale=2)
                    v_style = gr.Checkbox(label="Style LoRA", value=False, scale=1)
                    v_seeds = gr.Textbox(label="Seeds", value="42, 43, 44, 45", scale=2)
                with gr.Row():
                    v_width = gr.Number(value=1024, precision=0, label="Width")
                    v_height = gr.Number(value=1024, precision=0, label="Height")
                    v_steps = gr.Number(value=0, precision=0, label="Steps (0 = default)")
                    v_cfg = gr.Number(value=0, label="CFG (0 = default)")
                    v_negative = gr.Textbox(label="Negative prompt", scale=2)
                with gr.Row():
                    v_folder = gr.Textbox(label="Dataset folder (for training-like prompts)", scale=3)
                    v_defaults = gr.Button("Fill starter prompts", scale=1)
                    v_draft_model = gr.Dropdown(text_models, label="Draft with model", scale=2)
                    v_draft_count = gr.Number(value=8, precision=0, label="How many", scale=1)
                    v_draft = gr.Button("Draft held-out prompts", scale=1)
                v_table = gr.Dataframe(headers=validation.HEADERS, value=[["held-out", "", "", ""]], interactive=True,
                                       wrap=True, max_height=420, column_count=(4, "fixed"), elem_id="validation_table")
                with gr.Row():
                    v_save = gr.Button("Save as a new version", variant="primary")
                    v_export = gr.Button("Export kohya / FluxGym sample prompts")
                v_status = gr.Markdown("")
                v_code = gr.Code(value="", label="sample_prompts.txt", interactive=False)
                v_file = gr.File(label="Download", interactive=False)

            # ---- test sweep ----
            with gr.Tab("Test Sweep"):
                with gr.Row():
                    s_backend = gr.Radio(backends, value=start_backend, label="Image backend", scale=3)
                    s_token = gr.Textbox(label="GRADIENT API token (optional, not saved)", type="password", scale=2,
                                         visible=start_backend == "GRADIENT")
                    s_refresh = gr.Button("🔄 Connect", scale=1)
                s_status = gr.Markdown("")
                with gr.Accordion("GRADIENT options", open=True, visible=start_backend == "GRADIENT") as s_gradient_group:
                    gr.Markdown("Applied to every image of the sweep. The baseline is a real no-LoRA render; LoRA files can "
                                "be absolute paths from your trainer's output folder.")
                    with gr.Row():
                        s_scheduler = gr.Dropdown([lab_ui.DEFAULT_OPTION], value=lab_ui.DEFAULT_OPTION, label="Scheduler")
                        s_precision = gr.Dropdown([lab_ui.DEFAULT_OPTION], value=lab_ui.DEFAULT_OPTION, label="Precision")
                        s_vae = gr.Dropdown([lab_ui.DEFAULT_OPTION], value=lab_ui.DEFAULT_OPTION, label="VAE")
                        s_guidance = gr.Textbox(label="CFG / schedule override", placeholder="e.g. 4.5, linear(9,3), 9@0,4@20")
                with gr.Row():
                    s_model = gr.Dropdown([], label="Checkpoint", scale=3, allow_custom_value=True)
                    s_workflow = gr.Dropdown([], label="ComfyUI workflow (needs %lora_name%)", scale=2,
                                             visible=start_backend == "ComfyUI")
                with gr.Row():
                    s_loras = gr.Dropdown([], multiselect=True, label="LoRA files (e.g. one per epoch)", scale=3)
                    s_extra_loras = gr.Textbox(label="Other LoRA names (one per line)", lines=2, scale=2)
                with gr.Row():
                    s_strengths = gr.Textbox(label="Strengths", value="0.6, 0.8, 1.0", scale=2)
                    s_baseline = gr.Checkbox(label="No-LoRA baseline", value=True, scale=1)
                    s_buckets = gr.CheckboxGroup(validation.BUCKETS, value=validation.BUCKETS, label="Buckets", scale=4)
                with gr.Row():
                    s_vname = gr.Dropdown(validation.list_sets(), label="Validation set", scale=2)
                    s_vversion = gr.Dropdown([], label="Version", scale=1)
                    s_max_prompts = gr.Number(value=0, precision=0, label="Max prompts (0 = all)", scale=1)
                    s_run_name = gr.Textbox(label="Run name", scale=2)
                s_count = gr.Markdown("")
                with gr.Row():
                    s_run = gr.Button("Run test sweep", variant="primary")
                    s_stop = gr.Button("⏹ Stop", variant="stop")
                s_progress = gr.Markdown("")
                s_preview = gr.Gallery(label="Latest image / grids", columns=2, height=520, elem_id="lab_grids")
                with gr.Accordion("Earlier runs", open=False):
                    with gr.Row():
                        s_runs = gr.Dropdown([], label="Run", scale=4)
                        s_runs_refresh = gr.Button("🔄", scale=0, min_width=50)
                    s_run_info = gr.Markdown("")
                with gr.Accordion("GRADIENT on another machine: sweep file out, results folder back", open=False):
                    gr.Markdown("**Export** writes a sweep file for this sweep (all settings above; for a GRADIENT you "
                                "can't connect to, type its model name as the Checkpoint). In GRADIENT: Queue → "
                                "**Import test sweep…**. GRADIENT writes a results folder when the sweep is done; "
                                "**Import** turns it into a normal run here.")
                    with gr.Row():
                        x_export = gr.Button("📤 Export sweep for GRADIENT", scale=1)
                        x_file = gr.File(label="Sweep file", interactive=False, scale=2)
                    with gr.Row():
                        x_folder = gr.Textbox(label="Results folder from GRADIENT", scale=3,
                                              placeholder="…/GRADIENT/outputs/sweeps/<run>-<id>")
                        x_import = gr.Button("📥 Import GRADIENT results", scale=1)
                    x_status = gr.Markdown("")
                with gr.Accordion("Copy a LoRA into ComfyUI", open=False):
                    with gr.Row():
                        c_src = gr.Textbox(label="LoRA file (.safetensors)", scale=3)
                        c_dst = gr.Textbox(label="ComfyUI folder", placeholder="C:/ComfyUI", scale=3)
                        c_button = gr.Button("Copy", scale=1)
                    c_status = gr.Markdown("")

            # ---- score & vote ----
            with gr.Tab("Score & Vote"):
                with gr.Row():
                    m_run = gr.Dropdown([], label="Test run", scale=4)
                    m_refresh = gr.Button("🔄", scale=0, min_width=50)
                gr.Markdown("**Blind pairwise votes.** Same prompt and seed, two conditions, labels hidden, sides random. "
                            "Votes are fitted with Bradley-Terry; win rates show 95% Wilson intervals.")
                with gr.Row():
                    vote_baseline = gr.Checkbox(label="Include the baseline", value=True)
                    vote_start = gr.Button("Start / continue voting", variant="primary")
                vote_pairs = gr.State([])
                vote_index = gr.State(0)
                vote_prompt = gr.Markdown("")
                with gr.Row():
                    vote_left = gr.Image(label="A", interactive=False, height=420, elem_id="vote_left")
                    vote_right = gr.Image(label="B", interactive=False, height=420, elem_id="vote_right")
                with gr.Row():
                    vote_a = gr.Button("A is better")
                    vote_tie = gr.Button("Same / can't tell")
                    vote_b = gr.Button("B is better")
                vote_status = gr.Markdown("")
                vote_table = gr.Dataframe(headers=voting.RESULT_HEADERS, interactive=False, elem_id="vote_table")
                with gr.Accordion("Vision-model judge (experimental)", open=False):
                    gr.Markdown("The judge sees each pair in both orders; if its answer flips, it is position-biased "
                                "on that pair and the vote is not counted. Trust it only if it agrees with your votes.")
                    with gr.Row():
                        judge_model = gr.Dropdown(vision_models, label="Judge model", scale=2)
                        judge_criterion = gr.Textbox(label="Judge on", placeholder="e.g. how close the style is to the "
                                                     "training images, and image quality", scale=3)
                        judge_max = gr.Number(value=20, precision=0, label="Max pairs", scale=1)
                        judge_button = gr.Button("Run judge", scale=1)
                    judge_status = gr.Markdown("")
                    judge_table = gr.Dataframe(headers=voting.RESULT_HEADERS, interactive=False)
                with gr.Accordion("Metric scoring (experimental)", open=False):
                    gr.Markdown("Fidelity to the training images, CLIP-T prompt adherence, CMMD drift and leakage vs "
                                "the baseline, with bootstrap intervals. Metrics are signals, not verdicts: DINO agrees "
                                "with human preference only weakly (r≈0.32), and aesthetic scores favour photorealism.")
                    embed_note = "" if embedders else "⚠️ " + metrics.install_hint()
                    with gr.Row():
                        m_embedder = gr.Dropdown(embedders, value=embedders[0] if embedders else None,
                                                 label="Embedding model", scale=2)
                        m_train = gr.Textbox(label="Training image folder", scale=3)
                        m_score = gr.Button("Score run", scale=1)
                    m_note = gr.Markdown(embed_note)
                    m_table = gr.Dataframe(headers=metrics.SCORE_HEADERS, interactive=False, wrap=True)

            # ---- experiments ----
            with gr.Tab("Experiments"):
                gr.Markdown("One record per training experiment: dataset, caption set, trainer settings, LoRA files, "
                            "validation set, test run, metrics, votes and notes (stored in "
                            "`experiments/artagents_experiments.sqlite`).")
                e_id = gr.State(None)
                e_table = gr.Dataframe(headers=experiments.LIST_HEADERS, value=experiments.list_rows(), interactive=False,
                                       wrap=True, elem_id="experiments_table")
                with gr.Row():
                    e_name = gr.Textbox(label="Name", scale=3)
                    e_new = gr.Button("New", scale=1)
                with gr.Row():
                    e_dataset = gr.Textbox(label="Dataset folder", scale=3)
                    e_caption_set = gr.Textbox(label="Caption set", scale=2)
                    e_snapshot = gr.Textbox(label="Caption snapshot", scale=2)
                    e_from_dataset = gr.Button("Fill from dataset", scale=1)
                e_caption_settings = gr.Textbox(label="How the caption set was made", lines=3)
                with gr.Row():
                    e_trainer = gr.Textbox(label="Trainer", placeholder="e.g. kohya sd-scripts 0.9", scale=2)
                    e_trainer_settings = gr.Textbox(label="Trainer settings (rank, LR, steps, …)", lines=3, scale=4)
                with gr.Row():
                    e_lora_paths = gr.Textbox(label="LoRA files (one per line)", lines=2, scale=3)
                    e_vset = gr.Textbox(label="Validation set", scale=1)
                    e_vversion = gr.Textbox(label="Version", scale=1)
                with gr.Row():
                    e_run = gr.Dropdown([], label="Test run", scale=3, allow_custom_value=True)
                    e_from_run = gr.Button("Fill from run", scale=1)
                e_run_dir = gr.Textbox(label="Run folder", interactive=True)
                e_notes = gr.Textbox(label="Notes / conclusions", lines=3)
                with gr.Row():
                    e_save = gr.Button("Save experiment", variant="primary")
                    e_export = gr.Button("Export report (Markdown + HTML)")
                e_status = gr.Markdown("")
                e_files = gr.File(label="Report", file_count="multiple", interactive=False)
                e_preview = gr.Markdown("")

    return {k: v for k, v in locals().items() if isinstance(v, (gr.components.Component, gr.blocks.BlockContext))
            or k in ("vote_pairs", "vote_index", "e_id")}


def wire_lora_lab_tab(c, *, settings_state):
    exp_inputs = [c["e_name"], c["e_dataset"], c["e_snapshot"], c["e_caption_set"], c["e_caption_settings"],
                  c["e_trainer"], c["e_trainer_settings"], c["e_lora_paths"], c["e_vset"], c["e_vversion"],
                  c["e_run_dir"], c["e_notes"]]

    # validation
    c["v_refresh"].click(fn=lab_ui.validation_names, outputs=[c["v_name"]])
    c["v_name"].change(fn=lab_ui.validation_versions, inputs=[c["v_name"]], outputs=[c["v_version"]])
    c["v_load"].click(fn=lab_ui.load_validation, inputs=[c["v_name"], c["v_version"]],
                      outputs=[c["v_table"], c["v_seeds"], c["v_trigger"], c["v_width"], c["v_height"], c["v_steps"],
                               c["v_cfg"], c["v_status"]])
    c["v_defaults"].click(fn=lab_ui.fill_default_prompts, inputs=[c["v_folder"], c["v_trigger"], c["v_class"], c["v_style"]],
                          outputs=[c["v_table"], c["v_status"]])
    c["v_draft"].click(fn=lab_ui.draft_prompts,
                       inputs=[c["v_table"], c["v_folder"], c["v_class"], c["v_style"], c["v_draft_model"],
                               c["v_draft_count"], settings_state],
                       outputs=[c["v_table"], c["v_status"]])
    c["v_save"].click(fn=lab_ui.save_validation,
                      inputs=[c["v_name"], c["v_table"], c["v_seeds"], c["v_trigger"], c["v_width"], c["v_height"],
                              c["v_steps"], c["v_cfg"], c["v_negative"]],
                      outputs=[c["v_status"], c["v_name"], c["v_version"]]
                      ).then(fn=lab_ui.validation_names, outputs=[c["s_vname"]])
    c["v_export"].click(fn=lab_ui.export_kohya_prompts, inputs=[c["v_name"], c["v_version"]],
                        outputs=[c["v_status"], c["v_code"], c["v_file"]])

    # sweep
    backend_outputs = [c["s_model"], c["s_loras"], c["s_workflow"], c["s_status"], c["s_gradient_group"],
                       c["s_scheduler"], c["s_precision"], c["s_vae"]]
    backend_inputs = [c["s_backend"], settings_state, c["s_token"]]
    c["s_refresh"].click(fn=lab_ui.refresh_lab_backend, inputs=backend_inputs, outputs=backend_outputs)
    c["s_model"].change(fn=lab_ui.gradient_precisions, inputs=[c["s_backend"], c["s_model"], settings_state, c["s_token"]],
                        outputs=[c["s_precision"]])
    c["s_backend"].change(fn=lambda b: gr.update(visible=b == "GRADIENT"), inputs=[c["s_backend"]], outputs=[c["s_token"]]
                          ).then(fn=lab_ui.refresh_lab_backend, inputs=backend_inputs, outputs=backend_outputs)
    c["s_vname"].change(fn=lab_ui.validation_versions, inputs=[c["s_vname"]], outputs=[c["s_vversion"]])

    def _count(vname, vversion, loras, extra, strengths, baseline, buckets, max_prompts):
        try:
            data = lab_ui.validation.load(vname, int(vversion) if vversion else None) if vname else {"prompts": [], "seeds": []}
        except Exception:
            return ""
        rows = lab_ui.validation.to_rows(data)
        all_loras = list(loras or []) + [l for l in (extra or "").splitlines() if l.strip()]
        return lab_ui.count_images(rows, ", ".join(str(s) for s in data.get("seeds", [])), all_loras, strengths, baseline,
                                   buckets, max_prompts)

    count_inputs = [c["s_vname"], c["s_vversion"], c["s_loras"], c["s_extra_loras"], c["s_strengths"], c["s_baseline"],
                    c["s_buckets"], c["s_max_prompts"]]
    for comp in count_inputs:
        comp.change(fn=_count, inputs=count_inputs, outputs=[c["s_count"]])
    sweep_event = c["s_run"].click(
        fn=lab_ui.iter_lora_sweep,
        inputs=[settings_state, c["s_backend"], c["s_model"], c["s_workflow"], c["s_loras"], c["s_extra_loras"],
                c["s_strengths"], c["s_baseline"], c["s_vname"], c["s_vversion"], c["s_buckets"], c["s_max_prompts"],
                c["s_run_name"], c["s_scheduler"], c["s_precision"], c["s_vae"], c["s_guidance"], c["s_token"]],
        outputs=[c["s_progress"], c["s_preview"], c["s_runs"], c["m_run"]],
    )
    c["s_stop"].click(fn=lab_ui.stop_lab_backend, inputs=[c["s_backend"], settings_state, c["s_token"]],
                      outputs=[c["s_progress"]], cancels=[sweep_event])
    c["x_export"].click(
        fn=lab_ui.export_gradient_sweep,
        inputs=[settings_state, c["s_model"], c["s_loras"], c["s_extra_loras"], c["s_strengths"], c["s_baseline"],
                c["s_vname"], c["s_vversion"], c["s_buckets"], c["s_max_prompts"], c["s_run_name"], c["s_scheduler"],
                c["s_precision"], c["s_vae"], c["s_guidance"], c["s_token"]],
        outputs=[c["x_status"], c["x_file"], c["s_runs"], c["m_run"]])
    c["x_import"].click(fn=lab_ui.import_gradient_results, inputs=[settings_state, c["x_folder"]],
                        outputs=[c["x_status"], c["s_preview"], c["s_runs"], c["m_run"]])
    c["s_runs_refresh"].click(fn=lab_ui.run_choices, inputs=[settings_state], outputs=[c["s_runs"], c["m_run"]])
    c["s_runs"].change(fn=lab_ui.show_run, inputs=[settings_state, c["s_runs"]], outputs=[c["s_preview"], c["s_run_info"]])
    c["c_button"].click(fn=lab_ui.copy_lora, inputs=[c["c_src"], c["c_dst"]], outputs=[c["c_status"]])

    # score & vote
    c["m_refresh"].click(fn=lab_ui.run_choices, inputs=[settings_state], outputs=[c["s_runs"], c["m_run"]])
    c["m_run"].change(fn=lab_ui.vote_results, inputs=[settings_state, c["m_run"]], outputs=[c["vote_table"], c["judge_status"]])
    vote_outputs = [c["vote_index"], c["vote_left"], c["vote_right"], c["vote_prompt"], c["vote_status"], c["vote_table"]]
    c["vote_start"].click(fn=lab_ui.start_voting, inputs=[settings_state, c["m_run"], c["vote_baseline"]],
                          outputs=[c["vote_pairs"], c["vote_index"], c["vote_left"], c["vote_right"], c["vote_prompt"],
                                   c["vote_status"]])
    for button, choice in ((c["vote_a"], "left"), (c["vote_b"], "right"), (c["vote_tie"], voting.TIE)):
        button.click(fn=lambda pairs, idx, settings, run, _choice=choice: lab_ui.cast_vote(_choice, pairs, idx, settings, run),
                     inputs=[c["vote_pairs"], c["vote_index"], settings_state, c["m_run"]], outputs=vote_outputs)
    c["judge_button"].click(fn=lab_ui.iter_judge,
                            inputs=[settings_state, c["m_run"], c["judge_model"], c["judge_criterion"], c["judge_max"]],
                            outputs=[c["judge_status"], c["judge_table"]])
    c["m_score"].click(fn=lab_ui.score_run, inputs=[settings_state, c["m_run"], c["m_train"], c["m_embedder"]],
                       outputs=[c["m_table"], c["m_note"]])

    # experiments
    c["e_table"].select(fn=lab_ui.select_experiment, outputs=[c["e_id"]] + exp_inputs)
    c["e_new"].click(fn=lab_ui.new_experiment, outputs=[c["e_id"]] + exp_inputs)
    c["e_from_run"].click(fn=lab_ui.prefill_from_run, inputs=[settings_state, c["e_run"]],
                          outputs=[c["e_run_dir"], c["e_lora_paths"], c["e_vset"], c["e_vversion"]])
    c["e_from_dataset"].click(fn=lab_ui.prefill_from_dataset, inputs=[c["e_dataset"], c["e_caption_set"]],
                              outputs=[c["e_snapshot"], c["e_caption_settings"]])
    c["e_save"].click(fn=lab_ui.save_experiment, inputs=[c["e_id"], settings_state] + exp_inputs,
                      outputs=[c["e_status"], c["e_id"], c["e_table"]])
    c["e_export"].click(fn=lab_ui.export_experiment, inputs=[c["e_id"]], outputs=[c["e_status"], c["e_files"], c["e_preview"]])
    c["s_runs"].change(fn=lambda run: gr.update(value=run), inputs=[c["s_runs"]], outputs=[c["e_run"]])
    c["m_refresh"].click(fn=lambda settings: lab_ui.run_choices(settings)[0], inputs=[settings_state], outputs=[c["e_run"]])
