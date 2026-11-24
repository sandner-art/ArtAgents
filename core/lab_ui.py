# ArtAgent/core/lab_ui.py
"""Gradio callbacks for Tier 3: caption sets & sweeps (Captions tab) and the LoRA Lab tab."""
from __future__ import annotations

import json
import os
import random

import gradio as gr

from . import captioning_logic as cl
from . import dataset_tools_ui as dtu
from . import dataset_ui as dui
from .dataset import caption_io, profiles, vision
from .image_gen import BACKENDS, ImageGenError, get_backend, list_workflows
from .image_gen_settings import backend_url, image_gen_settings
from .lab import caption_sets as cs
from .lab import experiments, lora_sweep, metrics, validation, voting
from .utils import get_absolute_path


def _pil(path):
    """In-memory copy for Gradio image components (works for output folders outside the project)."""
    if not path or not os.path.exists(path):
        return None
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB").copy()


def _pils(paths):
    return [(img, os.path.basename(p)) for p in paths for img in [_pil(p)] if img is not None]


# =====================================================================
# Caption sets (Captions tab → Caption Sets)
# =====================================================================
def _folder(image_paths) -> str | None:
    return cl._dataset_folder(image_paths) if image_paths else None


def set_choices(image_paths, value=None):
    names = [cs.ACTIVE] + cs.discover_sets(image_paths or {})
    return gr.update(choices=names, value=value if value in names else names[0])


def compare_sets(image_paths, profile_label, trigger):
    if not image_paths:
        return [], "No images loaded."
    budgets = profiles.get_profile(profile_label).get("budgets") or []
    names = [cs.ACTIVE] + cs.discover_sets(image_paths)
    rows = [cs.set_stats(image_paths, n, budgets, trigger or "") for n in names]
    folder = _folder(image_paths)
    manifest = cs.load_manifest(folder).get("sets", {})
    lines = [f"**{len(names) - 1} caption set(s).** Train on a set with kohya-ss / musubi-tuner: "
             "`caption_extension = \".<set>.txt\"`, or make it active."]
    for n in names[1:]:
        st = manifest.get(n, {}).get("settings", {})
        if st:
            lines.append(f"- **{n}**: {st.get('strategy', '')}" + (f" · {st.get('model')}" if st.get("model") else "")
                         + (f" · {st.get('profile')}" if st.get("profile") else ""))
    return rows, "\n".join(lines)


def set_histogram(image_paths, name):
    if not image_paths or not name:
        return []
    return cs.length_histogram(image_paths, name)


def save_active_as_set(image_paths, captions, name, description):
    if not image_paths:
        return "No images loaded.", gr.update()
    try:
        n = cs.save_active_as_set(image_paths, captions, name, description)
    except cs.CaptionSetError as e:
        return f"⚠️ {e}", gr.update()
    return f"Saved {n} caption(s) as set '{cs.validate_name(name)}'.", set_choices(image_paths, cs.validate_name(name))


def activate_set(image_paths, captions, name):
    captions = dict(captions or {})
    if not image_paths or not name or name == cs.ACTIVE:
        return "Pick a caption set first.", captions
    try:
        content, snap, missing = cs.activate_set(image_paths, name)
    except (cs.CaptionSetError, OSError) as e:
        return f"⚠️ {e}", captions
    captions.update(content)
    note = f" {missing} image(s) have no caption in this set; their active caption was kept." if missing else ""
    return f"Set '{name}' is now the active captions ({len(content)} files). Snapshot saved: {snap}.{note}", captions


def delete_set(image_paths, name):
    if not image_paths or not name or name == cs.ACTIVE:
        return "Pick a caption set first.", gr.update()
    try:
        target = cs.delete_set(image_paths, name)
    except (cs.CaptionSetError, OSError) as e:
        return f"⚠️ {e}", gr.update()
    return f"Moved set '{name}' to {target} (nothing deleted).", set_choices(image_paths)


def materialize_set(image_paths, name, mode_label):
    mode = next((k for k, v in cs.VARIANT_MODES.items() if v == mode_label), "lines")
    if not image_paths:
        return "No images loaded."
    try:
        out, n = cs.materialize(image_paths, name or cs.ACTIVE, mode)
    except (cs.CaptionSetError, OSError) as e:
        return f"⚠️ {e}"
    extra = " In kohya set enable_multiline_caption = true." if mode == "kohya_multiline" else ""
    return f"Wrote {n} caption file(s) with images (hard-linked where possible) to {out}.{extra}"


def iter_caption_sweep(strategy_labels, set_prefix, image_paths, captions, agent, model, settings, models_data,
                       limiters, teams, file_agents, history_list, session, instruction, prefix, suffix,
                       profile_label, preset_label, style_label, joiner_label, omit_text, temperature, seed, max_side,
                       retries, option_labels, class_noun, extra, *source_args):
    """Generates one caption set per strategy without touching the active captions."""
    if not image_paths:
        yield "No images loaded.", gr.update(), gr.update()
        return
    strategies = [cs.STRATEGY_LABELS[l] for l in strategy_labels or [] if l in cs.STRATEGY_LABELS]
    if not strategies:
        yield "Pick at least one strategy.", gr.update(), gr.update()
        return
    prefix_name = (set_prefix or "sweep").strip().lower()
    try:
        cs.validate_name(f"{prefix_name}-x")
    except cs.CaptionSetError as e:
        yield f"⚠️ {e}", gr.update(), gr.update()
        return
    for strat in strategies + ["short_prose", "long_prose"]:
        clash = cs.collisions(image_paths, f"{prefix_name}-{strat.replace('_', '-')}")
        if clash:  # e.g. image 'a.png' + set 'sweep-x' would write 'a.sweep-x.txt', another image's caption
            yield (f"⚠️ The prefix '{prefix_name}' clashes with image file names ({', '.join(clash[:3])}). "
                   "Choose another prefix.", gr.update(), gr.update())
            return
    folder = _folder(image_paths)
    base = dui.build_caption_config(profile_label, preset_label, style_label, joiner_label, omit_text, temperature,
                                    seed, max_side, retries)
    base.update(dtu.source_config(*source_args))
    profile = profiles.get_profile(profile_label)
    tags_label, prose_label = profiles.STYLE_LABELS["tags"], profiles.STYLE_LABELS["prose"]
    names = sorted(image_paths)
    model_strats = [s for s in strategies if cs.STRATEGIES[s][1] == "model"]
    # Derived strategies need their inputs: generate those sets too
    needs = {"random_length": ["short_prose", "long_prose"], "long_plus_trigger": ["long_prose"]}
    for s in strategies:
        for dep in needs.get(s, []):
            if dep not in model_strats:
                model_strats.append(dep)
    if "blend_95_5" in strategies and not any(s in model_strats for s in ("long_prose", "short_prose", "tags")):
        model_strats.append("short_prose")
    generated: dict[str, dict[str, str]] = {}
    log, total = [], len(model_strats) * len(names)
    done = 0
    for strat in model_strats:
        set_name = f"{prefix_name}-{strat.replace('_', '-')}"
        cfg = dict(base, extension=cs.extension(set_name))
        if strat == "tags":
            cfg["style"], cfg["joiner"] = "tags", ", "
            if cfg.get("source") not in cl.TAGGER_SOURCES:
                cfg["source"] = cl.SOURCE_VLM
            instr = dui.compose_instruction(profile_label, preset_label, tags_label, option_labels,
                                            profile.get("word_target") or 25, omit_text, class_noun, extra)[0]
        elif strat in cs.WORD_TARGETS:
            cfg["style"], cfg["joiner"], cfg["source"] = "prose", " ", cl.SOURCE_VLM
            instr = dui.compose_instruction(profile_label, preset_label, prose_label, option_labels,
                                            cs.WORD_TARGETS[strat], omit_text, class_noun, extra)[0]
            cfg["max_output_tokens"] = max(int(cfg.get("max_output_tokens") or 150), 4 * cs.WORD_TARGETS[strat])
        else:  # structured
            cfg["source"] = cl.SOURCE_STRUCTURED
            instr = instruction
        errors = 0
        for name in names:
            status, _, _, session = cl.generate_captions_for_selected(
                name, agent, model, caption_io.MODE_OVERWRITE, image_paths, captions, settings, models_data, limiters,
                teams, file_agents, history_list, session, caption_instruction=instr, caption_prefix=prefix,
                caption_suffix=suffix, caption_config=cfg)
            errors += int("Status: Error" in status)
            done += 1
            yield f"Caption sweep: {cs.STRATEGIES[strat][0]} — {done}/{total} image(s).", gr.update(), gr.update()
        generated[strat] = cs.read_set(image_paths, set_name)
        cs.record_set(folder, set_name, {"strategy": cs.STRATEGIES[strat][0], "model": vision.clean_model_name(model or ""),
                                         "agent": agent, "profile": profile.get("id"), "preset": profiles.get_preset(preset_label).get("id"),
                                         "source": cfg.get("source"), "backend": cfg.get("backend"), "instruction": instr,
                                         "temperature": cfg.get("temperature"), "seed": cfg.get("seed")})
        log.append(f"- {set_name}: {len(generated[strat])} caption(s)" + (f", {errors} error(s)" if errors else ""))
    derived = [s for s in strategies if cs.STRATEGIES[s][1] == "derived"]
    for strat in derived:
        set_name = f"{prefix_name}-{strat.replace('_', '-')}"
        try:
            content = cs.derive(strat, image_paths, captions or {}, generated, prefix or "",
                                profile.get("quality_prefix") or "", int(seed) if seed is not None and int(seed) >= 0 else 42)
            n = cs.write_set(image_paths, set_name, content, {"strategy": cs.STRATEGIES[strat][0], "derived_from":
                                                              sorted(generated) or ["active captions"]})
            log.append(f"- {set_name}: {n} caption(s)")
        except cs.CaptionSetError as e:
            log.append(f"- {set_name}: ⚠️ {e}")
    rows, _ = compare_sets(image_paths, profile_label, (prefix or "").strip().rstrip(","))
    yield "Caption sweep finished. Active captions were not changed.\n" + "\n".join(log), rows, set_choices(image_paths)


# =====================================================================
# LoRA Lab: validation prompt sets
# =====================================================================
def validation_names():
    return gr.update(choices=validation.list_sets())


def validation_versions(name):
    versions = validation.list_versions(name) if name else []
    return gr.update(choices=[str(v) for v in versions], value=str(versions[-1]) if versions else None)


def load_validation(name, version):
    if not name:
        return gr.update(), "", "", gr.update(), gr.update(), gr.update(), gr.update(), "Pick a set."
    try:
        data = validation.load(name, int(version) if version else None)
    except (validation.ValidationError, OSError, ValueError) as e:
        return gr.update(), "", "", gr.update(), gr.update(), gr.update(), gr.update(), f"⚠️ {e}"
    d = data.get("defaults", {})
    return (validation.to_rows(data), ", ".join(str(s) for s in data.get("seeds", [])), data.get("trigger", ""),
            d.get("width", 1024), d.get("height", 1024), d.get("steps") or 0, d.get("cfg") or 0,
            f"Loaded {name} v{data['version']} ({len(data['prompts'])} prompts, created {data['created']}).")


def fill_default_prompts(folder, trigger, class_noun, style_lora):
    captions = {}
    if folder and os.path.isdir(folder):
        from .dataset import caption_io as cio
        from .image_utils import IMAGE_EXTENSIONS
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(tuple(IMAGE_EXTENSIONS)):
                text = cio.read_caption(os.path.join(folder, f))
                if text.strip():
                    captions[f] = text
    rows = validation.default_rows(trigger, class_noun, captions, bool(style_lora))
    note = f"{len(captions)} caption(s) read for training-like prompts." if captions else "No dataset captions: training-like prompts are empty."
    return rows, f"Starter prompts for all four buckets. {note} Edit freely, then save as a version."


def draft_prompts(frame, folder, class_noun, style_lora, model, count, settings):
    rows = dtu._rows_from_frame(frame)
    captions = {}
    if folder and os.path.isdir(folder):
        for f in sorted(os.listdir(folder)):
            if f.endswith(".txt") and f.count(".") == 1:
                captions[f] = open(os.path.join(folder, f), encoding="utf-8", errors="replace").read()
    if not model:
        return gr.update(), "Pick a model to draft prompts with."
    try:
        raw = vision.ask(vision.VisionTarget(vision.BACKEND_OLLAMA, model), settings or {},
                         validation.draft_prompt(captions, class_noun or "", int(count or 8), bool(style_lora)),
                         system=validation.DRAFT_SYSTEM, temperature=0.8, max_tokens=60 * int(count or 8))
    except vision.VisionError as e:
        return gr.update(), f"⚠️ {e}"
    prompts = validation.parse_draft(raw, int(count or 8))
    if not prompts:
        return gr.update(), "The model returned no usable prompts (each must contain [trigger])."
    rows = [r for r in rows if any(str(c).strip() for c in r)] + [["held-out", p, "", ""] for p in prompts]
    return rows, f"Added {len(prompts)} drafted held-out prompt(s). Review them before saving."


def save_validation(name, frame, seeds, trigger, width, height, steps, cfg, negative):
    defaults = {"width": int(width or 1024), "height": int(height or 1024), "steps": int(steps) if steps else None,
                "cfg": float(cfg) if cfg else None, "negative": negative or ""}
    try:
        version = validation.save_new_version(name, dtu._rows_from_frame(frame), seeds, trigger, defaults)
    except (validation.ValidationError, OSError) as e:
        return f"⚠️ {e}", gr.update(), gr.update()
    return (f"Saved {name.strip()} v{version}.", gr.update(choices=validation.list_sets(), value=name.strip()),
            gr.update(choices=[str(v) for v in validation.list_versions(name.strip())], value=str(version)))


def export_kohya_prompts(name, version):
    try:
        data = validation.load(name, int(version) if version else None)
    except (validation.ValidationError, OSError, ValueError) as e:
        return f"⚠️ {e}", gr.update(value=""), None
    text = validation.kohya_sample_prompts(data)
    path = os.path.join(get_absolute_path(validation.SETS_DIR), f"{name}.v{data['version']}.sample_prompts.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Wrote {path}", gr.update(value=text), dtu._downloadable(path)


def count_images(frame, seeds, loras, strengths, baseline, buckets, max_prompts=0):
    rows = validation.clean_rows(dtu._rows_from_frame(frame)) if frame is not None else []
    n_prompts = len([r for r in rows if not buckets or r["bucket"] in buckets])
    if max_prompts and int(max_prompts) > 0:
        n_prompts = min(n_prompts, int(max_prompts))
    try:
        n_seeds = len(validation.parse_seeds(seeds))
        n_strengths = len(lora_sweep.parse_strengths(strengths))
    except (validation.ValidationError, ValueError):
        return "Check the seeds and strengths."
    n = validation.image_count(n_prompts, n_seeds, len(loras or []), n_strengths, bool(baseline))
    return f"**{n} image(s)** = {n_prompts} prompt(s) × {n_seeds} seed(s) × ({len(loras or [])} LoRA(s) × {n_strengths} strength(s){' + baseline' if baseline else ''})."


# =====================================================================
# LoRA Lab: test sweep
# =====================================================================
LAB_BACKENDS = ["ComfyUI", "Forge / A1111 / SD.Next"]


def runs_root(settings) -> str:
    out = image_gen_settings(settings)["output_dir"]
    base = out if os.path.isabs(out) else get_absolute_path(out)
    return os.path.join(os.path.dirname(os.path.normpath(base)), "lora_tests")


def refresh_lab_backend(backend_label, settings):
    url = backend_url(settings, backend_label)
    try:
        backend = get_backend(backend_label, url)
        ok, message = backend.check()
        models = backend.list_models() if ok else []
        loras = backend.list_loras() if ok and hasattr(backend, "list_loras") else []
    except ImageGenError as e:
        ok, message, models, loras = False, str(e), [], []
    is_comfy = backend_label == "ComfyUI"
    workflows = list_workflows() if is_comfy else []
    default_wf = "lora_txt2img.json" if "lora_txt2img.json" in workflows else (workflows[0] if workflows else None)
    status = f"{'✅' if ok else '⚠️'} {backend_label} at {url}: {message}" + (f" · {len(loras)} LoRA(s) found." if ok else "")
    return (gr.update(choices=models, value=models[0] if models else None),
            gr.update(choices=loras, value=[]), gr.update(visible=is_comfy, choices=workflows, value=default_wf), status)


def iter_lora_sweep(settings, backend_label, model, workflow, loras, extra_loras, strengths, baseline, vname, vversion,
                    buckets, max_prompts, run_name):
    lora_list = list(loras or []) + [l.strip() for l in (extra_loras or "").splitlines() if l.strip()]
    if not vname:
        yield "Pick a validation prompt set.", gr.update(), gr.update(), gr.update()
        return
    try:
        vset = validation.load(vname, int(vversion) if vversion else None)
        strength_list = lora_sweep.parse_strengths(strengths)
    except (validation.ValidationError, ValueError, OSError) as e:
        yield f"⚠️ {e}", gr.update(), gr.update(), gr.update()
        return
    if not lora_list and not baseline:
        yield "Pick at least one LoRA (or include the baseline).", gr.update(), gr.update(), gr.update()
        return
    if backend_label == "ComfyUI" and not lora_list:
        yield "ComfyUI needs at least one LoRA for the %lora_name% slot (the baseline uses it at strength 0).", gr.update(), gr.update(), gr.update()
        return
    jobs = lora_sweep.build_jobs(vset, lora_list, strength_list if lora_list else [], bool(baseline), buckets or None,
                                 int(max_prompts) if max_prompts else None)
    if not jobs:
        yield "No images to make: check the buckets and prompts.", gr.update(), gr.update(), gr.update()
        return
    root = runs_root(settings)
    run_dir = lora_sweep.new_run_dir(root, run_name or vname)
    backend = get_backend(backend_label, backend_url(settings, backend_label))
    meta = {"name": run_name or vname, "validation_set": vname, "validation_version": vset["version"],
            "trigger": vset.get("trigger", ""), "loras": lora_list, "strengths": strength_list, "baseline": bool(baseline)}
    last = None
    errors = 0
    try:
        for done, total, message, path in lora_sweep.run_iter(backend, backend_label, jobs, run_dir, meta, model,
                                                              workflow if backend_label == "ComfyUI" else None):
            errors += int("⚠" in message)
            last = path or last
            yield (f"{message}\n{done}/{total} image(s) in {run_dir}" + (f" · {errors} error(s)" if errors else ""),
                   gr.update(value=_pils([last]) if last else None), gr.update(), gr.update())
    finally:
        grids = lora_sweep.make_grids(run_dir) if os.path.exists(os.path.join(run_dir, "run.json")) else []
    yield (f"Finished: {len(jobs)} job(s), {errors} error(s). Grids and run.json are in {run_dir}.",
           gr.update(value=_pils(grids)), gr.update(choices=lora_sweep.list_runs(root), value=os.path.basename(run_dir)),
           gr.update(choices=lora_sweep.list_runs(root), value=os.path.basename(run_dir)))


def stop_lab_backend(backend_label, settings):
    try:
        get_backend(backend_label, backend_url(settings, backend_label)).interrupt()
    except Exception:
        pass


def copy_lora(src, comfy_dir):
    try:
        dst = lora_sweep.copy_lora((src or "").strip().strip('"'), (comfy_dir or "").strip().strip('"'))
    except (OSError, ValueError) as e:
        return f"⚠️ {e}"
    return f"Copied to {dst}. Press 🔄 to refresh the LoRA list."


def run_choices(settings):
    runs = lora_sweep.list_runs(runs_root(settings))
    return gr.update(choices=runs, value=runs[0] if runs else None), gr.update(choices=runs, value=runs[0] if runs else None)


def show_run(settings, run_name):
    if not run_name:
        return gr.update(value=None), ""
    run_dir = os.path.join(runs_root(settings), run_name)
    grids = sorted(os.path.join(run_dir, f) for f in os.listdir(run_dir) if f.startswith("grid_"))
    if not grids and os.path.exists(os.path.join(run_dir, "run.json")):
        grids = lora_sweep.make_grids(run_dir)
    data = lora_sweep.load_run(run_dir)
    return gr.update(value=_pils(grids)), (f"{data.get('name')} · {data.get('backend')} · {len(data.get('results', []))}/"
                                    f"{len(data.get('jobs', []))} images · validation {data.get('validation_set')} v{data.get('validation_version')}")


# =====================================================================
# LoRA Lab: metrics & voting
# =====================================================================
def embedder_choices():
    available = metrics.available_embedders()
    note = "" if available else "⚠️ " + metrics.install_hint()
    return gr.update(choices=available, value=available[0] if available else None), note


def score_run(settings, run_name, train_folder, embedder_label, progress=gr.Progress()):
    if not run_name:
        return [], "Pick a test run."
    if not embedder_label:
        return [], "⚠️ " + metrics.install_hint()
    run_dir = os.path.join(runs_root(settings), run_name)
    from .image_utils import IMAGE_EXTENSIONS
    train = [os.path.join(train_folder, f) for f in sorted(os.listdir(train_folder))
             if f.lower().endswith(tuple(IMAGE_EXTENSIONS))] if train_folder and os.path.isdir(train_folder) else []
    try:
        progress(0.1, desc="Loading the embedding model")
        result = metrics.score_run(lora_sweep.load_run(run_dir), run_dir, train, metrics.Embedder(embedder_label))
    except (metrics.MetricsError, OSError) as e:
        return [], f"⚠️ {e}"
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"embedder": embedder_label, "headers": metrics.SCORE_HEADERS, "rows": result["rows"]}, f, indent=2)
    return result["rows"], result["summary"]


def start_voting(settings, run_name, include_baseline):
    if not run_name:
        return [], 0, None, None, "", "Pick a test run."
    run_dir = os.path.join(runs_root(settings), run_name)
    pairs = voting.make_pairs(lora_sweep.load_run(run_dir), run_dir, seed=random.randint(0, 10 ** 6),
                              include_baseline=bool(include_baseline))
    voted = {v["key"] for v in voting.load_votes(run_dir, "user")}
    pairs = [p for p in pairs if voting.pair_key(p) not in voted]
    if not pairs:
        return [], 0, None, None, "", "No pairs left to vote on (or the run has fewer than two conditions per prompt and seed)."
    return (pairs, 0, _pil(os.path.join(run_dir, pairs[0]["left_path"])), _pil(os.path.join(run_dir, pairs[0]["right_path"])),
            f"**Prompt:** {pairs[0]['prompt']}", f"Pair 1 of {len(pairs)}. Labels are hidden; sides are random.")


def cast_vote(choice, pairs, index, settings, run_name):
    if not pairs or index is None or index >= len(pairs):
        return index, gr.update(), gr.update(), "", "Start voting first.", gr.update()
    run_dir = os.path.join(runs_root(settings), run_name)
    voting.record_vote(run_dir, pairs[index], choice, "user")
    index += 1
    table = voting.results_table(voting.load_votes(run_dir, "user"))
    if index >= len(pairs):
        return index, None, None, "", f"All {len(pairs)} pairs voted. Results below.", table
    p = pairs[index]
    return (index, _pil(os.path.join(run_dir, p["left_path"])), _pil(os.path.join(run_dir, p["right_path"])), f"**Prompt:** {p['prompt']}",
            f"Pair {index + 1} of {len(pairs)}.", table)


def vote_results(settings, run_name):
    if not run_name:
        return [], ""
    run_dir = os.path.join(runs_root(settings), run_name)
    return voting.results_table(voting.load_votes(run_dir, "user")), voting.judge_agreement(run_dir)


def iter_judge(settings, run_name, model, criterion, max_pairs):
    if not run_name or not model:
        yield "Pick a test run and a judge model.", gr.update()
        return
    run_dir = os.path.join(runs_root(settings), run_name)
    pairs = voting.make_pairs(lora_sweep.load_run(run_dir), run_dir, seed=1)
    user_keys = {v["key"] for v in voting.load_votes(run_dir, "user")}
    # Pairs you already voted on first, so the judge's agreement with you can be measured
    pairs = sorted(pairs, key=lambda p: voting.pair_key(p) not in user_keys)[:int(max_pairs or 20)]
    target = vision.VisionTarget(vision.BACKEND_OLLAMA, model)
    counted = biased = 0
    for i, pair in enumerate(pairs, start=1):
        try:
            result = voting.judge_pair(pair, run_dir, target, settings or {}, criterion or "")
        except (vision.VisionError, OSError) as e:
            yield f"⚠️ {e}", gr.update()
            return
        if result["consistent"]:
            voting.record_vote(run_dir, pair, result["choice"], "judge")
            counted += 1
        else:
            biased += 1
        yield f"Judge: {i}/{len(pairs)} pairs · {counted} counted · {biased} position-biased (answer changed with the order).", gr.update()
    yield (f"Judge finished: {counted} vote(s) counted, {biased} pair(s) skipped as position-biased.\n"
           + voting.judge_agreement(run_dir), voting.results_table(voting.load_votes(run_dir, "judge")))


# =====================================================================
# LoRA Lab: experiments
# =====================================================================
EXP_INPUT_KEYS = ["name", "dataset", "snapshot", "caption_set", "caption_set_settings", "trainer", "trainer_settings",
                  "lora_paths", "validation_set", "validation_version", "run_dir", "notes"]


def experiments_table():
    return experiments.list_rows()


def prefill_from_run(settings, run_name):
    """Fills run folder, LoRAs and validation set from a test run (other fields unchanged)."""
    if not run_name:
        return gr.update(), gr.update(), gr.update(), gr.update()
    run_dir = os.path.join(runs_root(settings), run_name)
    data = lora_sweep.load_run(run_dir)
    return run_dir, "\n".join(data.get("loras", [])), data.get("validation_set", ""), str(data.get("validation_version", ""))


def prefill_from_dataset(folder, caption_set):
    """Caption-set manifest and the newest snapshot id for the dataset folder."""
    if not folder or not os.path.isdir(folder):
        return gr.update(), gr.update()
    from .dataset import snapshots
    snaps = snapshots.list_snapshots(folder)
    settings_text = ""
    if caption_set and caption_set != cs.ACTIVE:
        entry = cs.load_manifest(folder).get("sets", {}).get(caption_set, {})
        settings_text = json.dumps(entry.get("settings", {}), indent=2, ensure_ascii=False) if entry else ""
    return (snaps[0]["id"] if snaps else ""), settings_text


def save_experiment(exp_id, settings, *values):
    record = dict(zip(EXP_INPUT_KEYS, values))
    run_dir = record.get("run_dir") or ""
    if run_dir and os.path.isdir(run_dir):
        mpath = os.path.join(run_dir, "metrics.json")
        if os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as f:
                record["metrics"] = json.load(f).get("rows", [])
        record["votes"] = voting.results_table(voting.load_votes(run_dir, "user"))
    try:
        new_id = experiments.save(record, int(exp_id) if exp_id else None)
    except (ValueError, OSError) as e:
        return f"⚠️ {e}", exp_id, gr.update()
    return f"Saved experiment #{new_id}.", new_id, experiments.list_rows()


def select_experiment(evt: gr.SelectData):
    row = getattr(evt, "row_value", None)
    if not row:
        return [None] + [gr.update()] * len(EXP_INPUT_KEYS)
    exp = experiments.get(int(row[0]))
    return [exp["id"]] + [exp.get(k) or "" for k in EXP_INPUT_KEYS]


def new_experiment():
    return [None] + [""] * len(EXP_INPUT_KEYS)


def export_experiment(exp_id):
    if not exp_id:
        return "Select or save an experiment first.", None, ""
    try:
        md_path, html_path = experiments.export_report(int(exp_id))
    except (ValueError, OSError) as e:
        return f"⚠️ {e}", None, ""
    with open(md_path, encoding="utf-8") as f:
        md = f.read()
    preview = "\n".join(l for l in md.splitlines() if not l.startswith("![")).replace("## Grids", "## Grids\n\n_(in the exported files)_")
    return f"Wrote {md_path} and {html_path}.", [dtu._downloadable(md_path), dtu._downloadable(html_path)], preview


def lab_backend_labels():
    return [b for b in LAB_BACKENDS if b in BACKENDS]
