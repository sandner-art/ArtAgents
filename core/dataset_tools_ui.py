# ArtAgent/core/dataset_tools_ui.py
"""
Gradio callbacks for the Tier 2 dataset tools in the Captions tab:
caption sources (tagger / hybrid / structured) and the OpenAI-compatible backend,
the attribute sheet, Dataset Health (scan, buckets, audit, verifier), trainer
export and edit-pair mode. Logic lives in core/dataset/*; this module adapts it
to Gradio inputs and outputs.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import gradio as gr

from . import captioning_logic as cl
from . import dataset_ui as dui
from .dataset import (audit, buckets, caption_io, edit_pairs, guardrails, health, profiles, structured, tagger,
                      trainer_export, vision)
from .image_utils import open_image

def _downloadable(path: str) -> str:
    """
    Copy of an exported file in the temp dir for gr.File: Gradio only serves files from the working
    directory or temp dir (ArtAgents does not open your folders to the browser with allowed_paths).
    """
    folder = tempfile.mkdtemp(prefix="artagents_export_")
    copy = os.path.join(folder, os.path.basename(path))
    shutil.copy2(path, copy)
    return copy


# ---------- caption source & backend ----------
SOURCE_CHOICES = {
    "Vision model": cl.SOURCE_VLM,
    "Booru tagger only (no vision model)": cl.SOURCE_TAGGER,
    "Tagger tags as hints for the vision model": cl.SOURCE_TAGGER_HINTS,
    "Tagger tags + vision-model caption": cl.SOURCE_TAGGER_PLUS,
    "Structured fields (JSON) → caption": cl.SOURCE_STRUCTURED,
}
SOURCE_DEFAULT = "Vision model"
BACKEND_CHOICES = {"Ollama": vision.BACKEND_OLLAMA,
                   "OpenAI-compatible server (LM Studio, vLLM, llama.cpp…)": vision.BACKEND_OPENAI}
BACKEND_DEFAULT = "Ollama"
VERIFIER_OPENAI = "(OpenAI-compatible server from AI Captioning)"
FIELD_LABELS = [structured.field_label(f) for f in structured.FIELDS]
DEFAULT_FIELD_LABELS = [structured.field_label(f) for f in structured.DEFAULT_FIELDS]


def _field_ids(labels) -> list[str]:
    return [structured.label_to_id(lab) for lab in labels or []]


def source_config(source_label, backend_label, oa_url, oa_model, oa_key, tagger_label, general_thr, character_thr,
                  general_mcut, character_mcut, hint_thr, add_rating, split_series, blacklist, reuse, field_labels,
                  template) -> dict:
    """Tier 2 part of the caption config (merged into dataset_ui.build_caption_config)."""
    fields = _field_ids(field_labels) or list(structured.DEFAULT_FIELDS)
    return {
        "source": SOURCE_CHOICES.get(source_label, cl.SOURCE_VLM),
        "backend": BACKEND_CHOICES.get(backend_label, vision.BACKEND_OLLAMA),
        "openai": {"base_url": (oa_url or "").strip(), "model": (oa_model or "").strip(), "api_key": (oa_key or "").strip()},
        "tagger": {"label": tagger_label, "general_threshold": float(general_thr or 0.35),
                   "character_threshold": float(character_thr or 0.85), "general_mcut": bool(general_mcut),
                   "character_mcut": bool(character_mcut), "hint_threshold": float(hint_thr or 0.5),
                   "add_rating": bool(add_rating), "split_series": bool(split_series),
                   "blacklist": profiles.parse_terms(blacklist), "reuse": bool(reuse)},
        "structured": {"fields": fields, "template": (template or "").strip()},
    }


def on_source_change(source_label, backend_label, field_labels, style_label):
    """Shows the panels that matter for the chosen source and backend."""
    source = SOURCE_CHOICES.get(source_label, cl.SOURCE_VLM)
    backend = BACKEND_CHOICES.get(backend_label, vision.BACKEND_OLLAMA)
    uses_tagger = source in cl.TAGGER_SOURCES
    uses_model = source != cl.SOURCE_TAGGER
    note = ""
    if uses_model and (backend == vision.BACKEND_OPENAI or source == cl.SOURCE_STRUCTURED):
        note = "ℹ️ This sends the instruction straight to the model; the Agent / Team selection is not used."
    if source == cl.SOURCE_TAGGER:
        note = "ℹ️ Captions are the tagger's tags only: the vision model, instruction and agent are not used."
    style = "tags" if style_label == profiles.STYLE_LABELS.get("tags") else "prose"
    template = structured.default_template(_field_ids(field_labels) or structured.DEFAULT_FIELDS, style)
    return (gr.update(visible=uses_tagger), gr.update(visible=source == cl.SOURCE_STRUCTURED),
            gr.update(visible=uses_model and backend == vision.BACKEND_OPENAI),
            gr.update(visible=uses_model and backend == vision.BACKEND_OLLAMA), note, template)


def on_fields_change(field_labels, style_label):
    style = "tags" if style_label == profiles.STYLE_LABELS.get("tags") else "prose"
    return structured.default_template(_field_ids(field_labels) or structured.DEFAULT_FIELDS, style)


def list_openai_models(url, key):
    from .openai_compat import OpenAICompatClient, OpenAICompatError
    try:
        models = OpenAICompatClient(url, key).list_models()
    except OpenAICompatError as e:
        return gr.update(), f"⚠️ {e}"
    if not models:
        return gr.update(), "The server answered but lists no models."
    return gr.update(choices=models, value=models[0]), f"✅ {len(models)} model(s) on {url}."


def tagger_info(label):
    return tagger.tagger_info_markdown(label)


def generate_selected(selected, agent, model, mode, image_paths, captions, settings, models_data, limiters, teams,
                      file_agents, history_list, session, instruction, prefix, suffix, profile_label, preset_label,
                      style_label, joiner_label, omit_text, temperature, seed, max_side, retries, *source_args):
    config = dui.build_caption_config(profile_label, preset_label, style_label, joiner_label, omit_text,
                                      temperature, seed, max_side, retries)
    config.update(source_config(*source_args))
    return cl.generate_captions_for_selected(selected, agent, model, mode, image_paths, captions, settings, models_data,
                                             limiters, teams, file_agents, history_list, session,
                                             caption_instruction=instruction, caption_prefix=prefix,
                                             caption_suffix=suffix, caption_config=config)


def generate_all(image_paths, captions, agent, model, mode, settings, models_data, limiters, teams, file_agents,
                 history_list, session, instruction, prefix, suffix, profile_label, preset_label, style_label,
                 joiner_label, omit_text, temperature, seed, max_side, retries, *source_args):
    config = dui.build_caption_config(profile_label, preset_label, style_label, joiner_label, omit_text,
                                      temperature, seed, max_side, retries)
    config.update(source_config(*source_args))
    yield from cl.iter_generate_captions_for_all(image_paths, captions, agent, model, mode, settings, models_data,
                                                 limiters, teams, file_agents, history_list, session,
                                                 caption_instruction=instruction, caption_prefix=prefix,
                                                 caption_suffix=suffix, caption_config=config)


# ---------- attribute sheet ----------
SHEET_SUMMARY_HEADERS = ["Field", "Images", "Distinct", "Most common", "%", "Note"]


def load_sheet(image_paths):
    if not image_paths:
        return gr.update(value=[[""]], headers=["image"]), [], [], "No images loaded."
    columns = structured.sheet_columns(image_paths)
    rows = structured.sheet_rows(image_paths, columns)
    if not rows:
        return (gr.update(value=[[""]], headers=["image"]), [], [],
                "No structured fields yet. Generate captions with the source 'Structured fields (JSON) → caption'.")
    headers = ["image"] + [structured.field_label(c) for c in columns]
    return (gr.update(value=rows, headers=headers), columns, structured.field_summary(columns, rows),
            f"{len(rows)} image(s) with fields. Edit cells, then Save sheet.")


def _rows_from_frame(frame) -> list[list]:
    if frame is None:
        return []
    if isinstance(frame, dict):  # Check before .values: dicts have a values() method
        return [list(r) for r in frame.get("data", [])]
    if hasattr(frame, "values") and hasattr(frame.values, "tolist"):  # pandas DataFrame
        return [list(r) for r in frame.values.tolist()]
    return [list(r) for r in frame]


def save_sheet(frame, columns, image_paths):
    if not columns:
        return "Load the sheet first.", []
    rows = _rows_from_frame(frame)
    changed = structured.save_sheet(image_paths or {}, columns, rows)
    return f"Saved fields for {changed} image(s).", structured.field_summary(columns, rows)


def render_from_fields(include_labels, template, style_label, profile_label, prefix, suffix, scope, view,
                       image_paths, captions, joiner_label):
    captions = dict(captions or {})
    if not image_paths:
        return "No images loaded.", captions
    style = profiles.style_of(profiles.get_profile(profile_label), style_label)
    include = _field_ids(include_labels)
    template = (template or "").strip() or structured.default_template(include or structured.DEFAULT_FIELDS, style)
    targets = [n for n in dui._targets(scope, view, image_paths) if structured.load_fields(image_paths[n])]
    if not targets:
        return "No images with structured fields in this selection.", captions
    snap_note = cl._snapshot({t: image_paths[t] for t in targets}, "before render from fields")
    written, errors = 0, []
    for name in targets:
        text = structured.render(structured.load_fields(image_paths[name]), template, style, include or None)
        text = guardrails.apply_prefix_suffix(guardrails.clean_caption(text, style), prefix or "", suffix or "")
        try:
            caption_io.write_caption(image_paths[name], text)
        except OSError as e:
            errors.append(f"{name}: {e}")
            continue
        captions[name] = text
        written += 1
    return f"Rendered {written} caption(s) from fields. {snap_note}".strip() + dui._error_note(errors), captions


def export_sheet(frame, columns, image_paths):
    if not columns or not image_paths:
        return "Load the sheet first.", None
    folder = cl._dataset_folder(image_paths)
    out_dir = os.path.join(folder, ".artagents", "export")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "attributes.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(structured.sheet_csv(columns, _rows_from_frame(frame)))
    return f"Wrote {path}", _downloadable(path)


# ---------- dataset health ----------
ISSUE_HEADERS = ["Image", "Issue", "Detail"]
BUCKET_IMAGE_HEADERS = ["Image", "Size", "Bucket", "Scale", "Cropped %", "Cut from", "Note"]
BUCKET_HEADERS = ["Bucket", "Aspect", "Images", "Per epoch", "Note"]
AUDIT_HEADERS = ["Image", "Check", "Detail"]
VERIFY_HEADERS = ["Image", "Disputed claims", "Unsure", "Controls failed", "Verifier", "Note"]


def run_health(image_paths, captions, profile_label, trigger, resolution, progress=gr.Progress()):
    if not image_paths:
        return "No images loaded.", [], []
    profile = profiles.get_profile(profile_label)
    result = health.scan(image_paths, captions or {}, int(resolution or 1024), profile.get("budgets") or [],
                         (trigger or "").strip(), progress=progress)
    return result["summary"], result["issues"], result["flagged"]


def show_images(names, image_paths, label="flagged by the health scan"):
    names = [n for n in names or [] if n in (image_paths or {})]
    if not names:
        return gr.update(), gr.update(), gr.update(), "Nothing to show: run the scan first.", ""
    md = f"🔎 Showing {len(names)}/{len(image_paths)} images {label}."
    return cl.gallery_items(names, image_paths), names, label, md, ""


def quarantine_selected(selected, image_paths, captions):
    image_paths, captions = dict(image_paths or {}), dict(captions or {})
    if not selected or selected not in image_paths:
        return "Select an image in the gallery first.", image_paths, captions, gr.update(), gr.update(), None
    try:
        moved = health.quarantine(image_paths[selected])
    except OSError as e:
        return f"Could not move {selected}: {e}", image_paths, captions, gr.update(), gr.update(), selected
    image_paths.pop(selected, None)
    captions.pop(selected, None)
    view = sorted(image_paths)
    return (f"Moved {selected} (and its caption) to {os.path.dirname(moved)}. Nothing was deleted.",
            image_paths, captions, cl.gallery_items(view, image_paths), view, None)


def restore_quarantined(image_paths, folder_text):
    folder = cl._dataset_folder(image_paths) or (folder_text or "").strip()
    if not folder:
        return "Load a folder first."
    restored = health.restore_quarantine(folder)
    if not restored:
        return "Nothing in quarantine (or the files already exist in the dataset)."
    return f"Moved {len(restored)} file(s) back: {', '.join(restored[:10])}. Press Load Folder to see them."


def _sizes(image_paths) -> dict[str, tuple[int, int]]:
    sizes = {}
    for name, path in (image_paths or {}).items():
        try:
            with open_image(path) as img:
                sizes[name] = img.size
        except Exception:
            continue
    return sizes


def simulate_buckets(image_paths, resolution, min_bucket, max_bucket, batch_size, repeats, no_upscale):
    if not image_paths:
        return "No images loaded.", [], []
    result = buckets.simulate(_sizes(image_paths), int(resolution or 1024), int(min_bucket or 256),
                              int(max_bucket or 2048), 64, int(batch_size or 1), int(repeats or 1), bool(no_upscale))
    note = "\n\n_Simulates kohya-ss bucketing (other trainers bucket similarly but not identically)._"
    return result["summary"] + note, result["rows"], result["buckets"]


def run_audit(captions, image_paths, profile_label, preset_label, style_label, trigger, class_noun, extra_triggers,
              omit_text, tagger_label, use_tagger):
    if not image_paths:
        return "No images loaded.", []
    style = profiles.style_of(profiles.get_profile(profile_label), style_label)
    trig = (trigger or "").strip()
    triggers = ([f"{trig} {class_noun.strip()}".strip()] if trig and (class_noun or "").strip() and style == "tags"
                else ([trig] if trig else []))
    triggers += [t for t in profiles.parse_terms(extra_triggers) if t not in triggers]
    result = audit.audit(captions or {}, image_paths, style, triggers, profiles.parse_terms(omit_text),
                         tagger_label if use_tagger else None)
    return result["summary"], result["rows"]


def run_tagger_compare(image_paths, tagger_label, progress=gr.Progress()):
    """Stores tagger probabilities for every image (no captions are written)."""
    if not image_paths:
        return "No images loaded."
    cfg = tagger.get_tagger(tagger_label)
    done, cached, errors = 0, 0, []
    names = sorted(image_paths)
    for i, name in enumerate(names):
        progress((i, len(names)), desc=f"Tagging {name}")
        try:
            _, was_cached = tagger.tag_probabilities(cfg, image_paths[name])
            done += 1
            cached += int(was_cached)
        except tagger.TaggerError as e:
            return f"⚠️ {e}"
        except Exception as e:
            errors.append(f"{name}: {e}")
    msg = f"Tagger results stored for {done} image(s) ({cached} reused). Run the audit to compare."
    return msg + ("\n" + "\n".join(errors[:5]) if errors else "")


def _verifier_target(verifier_model, backend_label, oa_url, oa_model, oa_key) -> vision.VisionTarget:
    if verifier_model == VERIFIER_OPENAI:
        return vision.VisionTarget(vision.BACKEND_OPENAI, oa_model or "", oa_url or "", oa_key or "")
    return vision.VisionTarget(vision.BACKEND_OLLAMA, verifier_model or "")


def iter_verify(scope, selected, view, image_paths, captions, settings, verifier_model, captioner_model, backend_label,
                oa_url, oa_model, oa_key, profile_label, style_label, trigger, class_noun):
    """Generator: verifies captions one image at a time (Stop ends it)."""
    if not image_paths:
        yield "No images loaded.", []
        return
    if not verifier_model:
        yield "Choose a verifier model.", []
        return
    target = _verifier_target(verifier_model, backend_label, oa_url, oa_model, oa_key)
    style = profiles.style_of(profiles.get_profile(profile_label), style_label)
    skip = [t for t in ((trigger or "").strip(), (class_noun or "").strip(),
                        f"{(trigger or '').strip()} {(class_noun or '').strip()}".strip()) if t]
    if scope == dui.FILTER_SCOPE and view:
        names = [n for n in view if n in image_paths]
    elif scope == "Selected image":
        names = [selected] if selected in (image_paths or {}) else []
    else:
        names = sorted(image_paths)
    names = [n for n in names if (captions or {}).get(n, "").strip()]
    if not names:
        yield "No captioned images to verify in this selection.", audit.verification_rows(image_paths, captions or {})
        return
    warn = ""
    if vision.clean_model_name(verifier_model) == vision.clean_model_name(captioner_model or ""):
        warn = " ⚠ Same model as the captioner: a model from a different family catches more mistakes."
    disputed, errors = 0, []
    for i, name in enumerate(names, start=1):
        try:
            img = cl._prepare_image(image_paths[name], 1024)
            try:
                result = audit.verify(image_paths[name], captions[name], img, target, settings or {}, style, skip)
            finally:
                img.close()
            disputed += int(bool(result["disputed"]))
        except (vision.VisionError, OSError, ValueError) as e:
            errors.append(f"{name}: {e}")
        status = f"Verified {i}/{len(names)} caption(s); {disputed} with disputed claims.{warn}"
        if errors:
            status += f"\n{len(errors)} error(s): " + "; ".join(errors[-3:])
        yield status, audit.verification_rows(image_paths, captions or {})


def verification_table(image_paths, captions):
    return audit.verification_rows(image_paths or {}, captions or {})


# ---------- trainer export ----------
def _counts(image_paths, captions):
    variants = sum(1 for n in image_paths if caption_io.variant_count((captions or {}).get(n, "")) > 1)
    webp = sum(1 for p in image_paths.values() if p.lower().endswith(".webp"))
    return variants, webp


def write_trainer_config(trainer_label, image_paths, captions, resolution, repeats, batch_size, keep_tokens, shuffle,
                         dropout, min_bucket, max_bucket, trigger):
    if not image_paths:
        return "No images loaded.", gr.update(value="")
    trainer = trainer_export.TRAINER_LABELS.get(trainer_label, "kohya")
    folder = cl._dataset_folder(image_paths)
    s = trainer_export.default_settings(folder, len(image_paths))
    variants, webp = _counts(image_paths, captions)
    s.update({"resolution": int(resolution or 1024), "repeats": int(repeats or 1), "batch_size": int(batch_size or 1),
              "keep_tokens": int(keep_tokens or 0), "shuffle": bool(shuffle), "caption_dropout": float(dropout or 0),
              "min_bucket": int(min_bucket or 256), "max_bucket": int(max_bucket or 2048), "variant_files": variants,
              "webp_files": webp, "trigger": (trigger or "").strip()})
    try:
        path, text = trainer_export.write_config(trainer, s)
    except OSError as e:
        return f"Could not write the config: {e}", gr.update(value="")
    return f"Wrote {path}", gr.update(value=text, language=trainer_export.code_language(trainer))


def keep_tokens_from_prefix(prefix):
    return trainer_export.keep_tokens_for(prefix)


def repeats_table(image_paths, target, epochs, batch_size, other_concepts):
    concepts = {}
    if image_paths:
        folder = cl._dataset_folder(image_paths)
        concepts[os.path.basename(os.path.normpath(folder)) or "dataset"] = len(image_paths)
    for line in (other_concepts or "").splitlines():
        if ":" in line:
            name, _, count = line.rpartition(":")
            try:
                concepts[name.strip()] = int(count.strip())
            except ValueError:
                continue
    return trainer_export.repeats_markdown(concepts, int(target or 100), int(epochs or 10), int(batch_size or 1))


# ---------- edit pairs ----------
PAIR_HEADERS = ["Pair", "Issue", "Detail"]


def _pair_gallery(pairs: list[dict]) -> list[tuple]:
    items = []
    for pair in pairs:
        for i, control in enumerate(pair["controls"]):
            if control:
                items.append((cl.get_thumbnail(control), f"{pair['stem']} · input{'' if len(pair['controls']) == 1 else ' ' + str(i + 1)}"))
        lines = caption_io.split_variants(caption_io.read_caption(pair["target"]))
        label = f"{pair['stem']} · result" + (f": {lines[0][:60]}" if lines else " (no instruction)")
        items.append((cl.get_thumbnail(pair["target"]), label))
    return items


def load_pairs(target_dir, controls_text):
    found = edit_pairs.find_pairs((target_dir or "").strip(), edit_pairs.parse_control_dirs(controls_text))
    pairs = found["pairs"]
    rows = edit_pairs.validate(pairs)
    msg = [f"{len(pairs)} pair(s) found."] + found["errors"]
    if found["orphans"]:
        msg.append(f"{len(found['orphans'])} control image(s) without a result: {', '.join(found['orphans'][:8])}")
    complete = sum(1 for p in pairs if all(p["controls"]))
    msg.append(f"{complete} complete; {len(rows)} thing(s) to check.")
    return pairs, _pair_gallery(pairs), rows, " ".join(msg), None, ""


def select_pair(evt: gr.SelectData, pairs):
    if not pairs or evt is None:
        return None, ""
    per = len(pairs[0]["controls"]) + 1
    flat = []
    for idx, pair in enumerate(pairs):
        flat += [idx] * (sum(1 for c in pair["controls"] if c) + 1)
    index = evt.index if isinstance(evt.index, int) else (evt.index[0] if evt.index else 0)
    pair_index = flat[index] if 0 <= index < len(flat) else index // per
    pair = pairs[pair_index]
    return pair_index, caption_io.read_caption(pair["target"])


def save_pair_instruction(pair_index, text, pairs):
    if pair_index is None or not pairs or pair_index >= len(pairs):
        return "Select a pair first.", gr.update(), gr.update()
    caption_io.write_caption(pairs[pair_index]["target"], (text or "").strip())
    return f"Saved the instruction for {pairs[pair_index]['stem']}.", _pair_gallery(pairs), edit_pairs.validate(pairs)


def iter_pair_instructions(scope, pair_index, pairs, settings, model, backend_label, oa_url, oa_model, oa_key,
                           paraphrases, trigger_prefix, extra, mode, temperature, seed, max_side):
    if not pairs:
        yield "Load pairs first.", gr.update(), gr.update()
        return
    backend = BACKEND_CHOICES.get(backend_label, vision.BACKEND_OLLAMA)
    target = (vision.VisionTarget(vision.BACKEND_OPENAI, oa_model or "", oa_url or "", oa_key or "")
              if backend == vision.BACKEND_OPENAI else vision.VisionTarget(vision.BACKEND_OLLAMA, model or ""))
    if scope == "Selected pair" and (pair_index is None or pair_index >= len(pairs)):
        yield "Select a pair in the gallery first (or choose 'All pairs').", gr.update(), gr.update()
        return
    indices = [pair_index] if scope == "Selected pair" else list(range(len(pairs)))
    done, skipped, errors = 0, 0, []
    for n, i in enumerate(indices, start=1):
        pair = pairs[i]
        if mode == caption_io.MODE_SKIP and caption_io.read_caption(pair["target"]).strip():
            skipped += 1
            continue
        try:
            lines = edit_pairs.generate_instruction(pair, target, settings or {}, int(paraphrases or 0), trigger_prefix or "",
                                                    extra or "", float(temperature) if temperature is not None else None,
                                                    int(seed) if seed is not None and int(seed) >= 0 else None,
                                                    int(max_side or 0))
            caption_io.write_caption(pair["target"], "\n".join(lines))
            done += 1
        except (vision.VisionError, ValueError, OSError) as e:
            errors.append(f"{pair['stem']}: {e}")
        status = f"Wrote {done} instruction(s), skipped {skipped}, {n}/{len(indices)} processed."
        if errors:
            status += f"\n{len(errors)} error(s): " + "; ".join(errors[-3:])
        yield status, _pair_gallery(pairs), gr.update()
    final = f"Wrote {done} instruction(s), skipped {skipped} (already had one), {len(indices)} pair(s) processed."
    if errors:
        final += f"\n{len(errors)} error(s): " + "; ".join(errors[-3:])
    yield final, _pair_gallery(pairs), edit_pairs.validate(pairs)


EDIT_EXPORTS = ["ai-toolkit (datasets YAML)", "musubi-tuner (dataset TOML)", "musubi-tuner (metadata JSONL)", "fal (zip of _start/_end pairs)"]


def export_pairs(fmt, pairs, target_dir, controls_text, trigger):
    if not pairs:
        return "Load pairs first.", gr.update(value=""), None
    target_dir = (target_dir or "").strip()
    controls = edit_pairs.parse_control_dirs(controls_text)
    out_dir = os.path.join(target_dir, ".artagents", "export")
    os.makedirs(out_dir, exist_ok=True)
    s = trainer_export.default_settings(target_dir, len(pairs))
    s.update({"control_dirs": controls, "trigger": (trigger or "").strip()})
    if fmt.startswith("ai-toolkit"):
        path, text = trainer_export.write_config("ai_toolkit", s)
        return f"Wrote {path}", gr.update(value=text, language="yaml"), _downloadable(path)
    if fmt == "musubi-tuner (dataset TOML)":
        path, text = trainer_export.write_config("musubi", s)
        return f"Wrote {path}", gr.update(value=text, language=None), _downloadable(path)
    if fmt.startswith("musubi-tuner (metadata"):
        text = edit_pairs.metadata_jsonl(pairs)
        path = os.path.join(out_dir, "edit_pairs_musubi.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return (f"Wrote {path} ({text.count(chr(10))} pairs with an instruction)", gr.update(value=text, language="json"),
                _downloadable(path))
    path = os.path.join(out_dir, "edit_pairs_fal.zip")
    n = edit_pairs.fal_zip(pairs, path)
    return f"Wrote {path} ({n} pairs)", gr.update(value=""), _downloadable(path)
