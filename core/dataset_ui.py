# ArtAgent/core/dataset_ui.py
"""
Gradio callbacks for the Image Captions Editor. All dataset logic lives in
core.dataset (UI-independent, tested on its own); this module only adapts it.
"""
from __future__ import annotations

import re

import gradio as gr

from . import captioning_logic as cl
from .dataset import caption_io, profiles, provenance, snapshots, terms
from .dataset.tokens import budget_markdown

FILTER_SCOPE = "Filtered images"
ALL_SCOPE = "All loaded images"
BULK_RENAME = "Rename tag"
BULK_DELETE = "Delete tag"
BULK_REPLACE = "Find & replace"
BULK_OPS = [BULK_RENAME, BULK_DELETE, BULK_REPLACE]
NORMALIZE_MERGE = "Merge lines into one caption"
NORMALIZE_VARIANTS = "Keep lines as clean caption variants"
TERM_HEADERS = ["Term", "Images", "%", "Note"]


# ---------- choices ----------
def profile_labels() -> list[str]:
    return [p["label"] for p in profiles.load_profiles()]


def preset_labels() -> list[str]:
    return [p["label"] for p in profiles.load_presets()]


def joiner_value(joiner_label: str | None, style: str) -> str:
    if joiner_label in caption_io.JOINER_CHOICES:
        return caption_io.JOINER_CHOICES[joiner_label]
    return ", " if style == profiles.STYLE_TAGS else " "


def joiner_label_for(style: str) -> str:
    wanted = ", " if style == profiles.STYLE_TAGS else " "
    return next(label for label, value in caption_io.JOINER_CHOICES.items() if value == wanted)


def term_mode_for(style: str) -> str:
    return terms.MODE_TAGS if style == profiles.STYLE_TAGS else terms.MODE_WORDS


# ---------- profile / preset / instruction ----------
def compose_prefix(preset_label, profile_label, style_label, trigger, class_noun, use_quality):
    profile, preset = profiles.get_profile(profile_label), profiles.get_preset(preset_label)
    style = profiles.style_of(profile, style_label)
    prefix = profiles.compose_prefix(preset, trigger, class_noun, style)
    quality = (profile.get("quality_prefix") or "").strip()
    if use_quality and quality:
        prefix = f"{prefix}{quality}, " if prefix else f"{quality}, "
    return prefix


def compose_instruction(profile_label, preset_label, style_label, option_labels, word_target, omit_text, class_noun, extra):
    profile, preset = profiles.get_profile(profile_label), profiles.get_preset(preset_label)
    style = profiles.style_of(profile, style_label)
    instruction = profiles.build_instruction(profile, preset, style, option_labels, word_target,
                                             profiles.parse_terms(omit_text), class_noun, extra)
    return instruction, profiles.style_mismatch_warning(profile, style)


def on_profile_or_preset_change(profile_label, preset_label, trigger, class_noun, current_caption, extra):
    """Applies profile + preset defaults to every captioning control."""
    profile, preset = profiles.get_profile(profile_label), profiles.get_preset(preset_label)
    style = profile.get("style", profiles.STYLE_PROSE)
    option_labels = profiles.default_options(profile, preset)
    omit_text = ", ".join(preset.get("omit_terms") or [])
    word_target = profile.get("word_target") or 0
    instruction = profiles.build_instruction(profile, preset, style, option_labels, word_target,
                                             preset.get("omit_terms") or [], class_noun, extra)
    quality = (profile.get("quality_prefix") or "").strip()
    return (
        gr.update(value=profiles.STYLE_LABELS[style]),                     # style radio
        gr.update(value=option_labels),                                    # options
        gr.update(value=word_target),                                      # word target
        omit_text,                                                         # omit terms
        compose_prefix(preset_label, profile_label, profiles.STYLE_LABELS[style], trigger, class_noun, False),
        gr.update(label=f"Add quality tags: {quality}" if quality else "Add quality tags (none for this model)",
                  value=False, interactive=bool(quality)),                 # quality checkbox
        instruction,
        "",                                                                # mismatch warning
        profiles.guidance_markdown(profile, preset),
        gr.update(value=joiner_label_for(style)),                          # joiner
        gr.update(value=term_mode_for(style)),                             # term mode
        token_report(current_caption, profile_label),
        gr.update(info=f"e.g. {preset['class_examples']}" if preset.get("class_examples") else None),
    )


def token_report(text, profile_label) -> str:
    if not text:
        return ""
    return budget_markdown(text, profiles.get_profile(profile_label).get("budgets") or [])


def caption_info(filename, captions, image_paths) -> str:
    """Variant count and provenance for the selected image."""
    if not filename:
        return ""
    bits = []
    count = caption_io.variant_count((captions or {}).get(filename, ""))
    if count > 1:
        bits.append(f"📑 {count} caption variants (one per line)")
    path = (image_paths or {}).get(filename)
    if path:
        summary = provenance.summarize(path, (captions or {}).get(filename, ""))
        if summary:
            bits.append(f"🧾 {summary}")
    return " · ".join(bits)


def build_caption_config(profile_label, preset_label, style_label, joiner_label, omit_text,
                         temperature, seed, max_side, retries) -> dict:
    profile = profiles.get_profile(profile_label)
    style = profiles.style_of(profile, style_label)
    return {
        "style": style,
        "joiner": joiner_value(joiner_label, style),
        "omit_terms": profiles.parse_terms(omit_text),
        "temperature": float(temperature) if temperature is not None else None,
        "seed": int(seed) if seed is not None and int(seed) >= 0 else None,
        "max_image_side": int(max_side) if max_side else None,
        "retries": int(retries or 0),
        "max_output_tokens": int(profile.get("max_output_tokens") or 150),
        "budgets": profile.get("budgets") or [],
        "profile": profile.get("id"),
        "preset": profiles.get_preset(preset_label).get("id"),
    }


# ---------- loading / selection / filtering ----------
def _snapshot_choices(image_paths) -> list[tuple[str, str]]:
    folder = cl._dataset_folder(image_paths)
    if not folder:
        return []
    return [(f"{s['created']} · {s['operation']} ({s['count']} files)", s["id"]) for s in snapshots.list_snapshots(folder)]


def snapshot_dropdown(image_paths):
    choices = _snapshot_choices(image_paths)
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def term_rows(captions, term_mode, trigger, class_noun):
    trigger, class_noun = (trigger or "").strip(), (class_noun or "").strip()
    protected = [trigger, class_noun, f"{trigger} {class_noun}".strip()]
    return terms.term_table(captions or {}, term_mode, protected)


def load_folder(folder, profile_label, term_mode, trigger, class_noun):
    gallery, paths, captions, status, first, caption, filename = cl.load_images_and_captions(folder)
    view = sorted(paths)
    multi = sum(1 for text in captions.values() if caption_io.needs_normalizing(text))
    if multi:
        status += (f" {multi} caption file(s) contain several lines or old '---' separators; "
                   "see Batch Edit → Normalize caption files.")
    return (gallery, paths, captions, status, first, caption, filename, view, "",
            _filter_markdown("", view, paths), term_rows(captions, term_mode, trigger, class_noun),
            snapshot_dropdown(paths), token_report(caption, profile_label), caption_info(first, captions, paths))


def select_image(evt: gr.SelectData, view, captions, image_paths, profile_label):
    """Gallery click → caption editor. Uses the index into the current (possibly filtered) view."""
    index = evt.index if evt is not None else None
    if isinstance(index, (list, tuple)):
        index = index[0]
    view = view or sorted(image_paths or {})
    if index is None or not (0 <= index < len(view)):
        return "", None, None, "", ""
    name = view[index]
    caption = (captions or {}).get(name, "")
    return caption, name, name, token_report(caption, profile_label), caption_info(name, captions, image_paths)


def _filter_markdown(term, view, image_paths) -> str:
    total = len(image_paths or {})
    if not term:
        return f"Showing all {total} images." if total else ""
    return f"🔎 Showing {len(view)}/{total} images with **{term}**. Bulk edits can target just these."


def filter_by_term(evt: gr.SelectData, captions, image_paths, term_mode):
    """Click a row in the term table → show only images whose caption has that term."""
    row = getattr(evt, "row_value", None)
    term = row[0] if row else (evt.value if isinstance(evt.value, str) else None)
    if not term:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    view = terms.files_with_term(captions or {}, str(term), term_mode)
    common = terms.common_terms(captions or {}, view, term_mode)
    common_md = ("Shared by all filtered images: " + ", ".join(common[:40])) if len(view) > 1 and common else ""
    return cl.gallery_items(view, image_paths), view, str(term), _filter_markdown(str(term), view, image_paths), common_md


def clear_filter(image_paths):
    view = sorted(image_paths or {})
    return cl.gallery_items(view, image_paths), view, "", _filter_markdown("", view, image_paths), ""


# ---------- bulk edits ----------
def _targets(scope, view, image_paths):
    if scope == FILTER_SCOPE and view:
        return [n for n in view if n in (image_paths or {})]
    return sorted(image_paths or {})


def bulk_edit(op, find_text, replace_text, use_regex, scope, view, captions, image_paths, term_mode, trigger, class_noun):
    """Rename/delete a tag or find & replace across captions (snapshot first)."""
    captions = dict(captions or {})
    if not image_paths:
        return "No images loaded.", captions, gr.update()
    if not (find_text or "").strip():
        return "Enter the tag or text to find.", captions, gr.update()
    if op == BULK_RENAME and not (replace_text or "").strip():
        return "Enter the new tag name.", captions, gr.update()
    if op == BULK_REPLACE and use_regex:
        try:
            re.compile(find_text)
        except re.error as e:
            return f"Invalid regular expression: {e}", captions, gr.update()

    targets = _targets(scope, view, image_paths)
    snap_note = cl._snapshot({t: image_paths[t] for t in targets}, f"before {op.lower()}")
    changed, errors = 0, []
    for name in targets:
        old = captions.get(name, "")
        if op == BULK_RENAME:
            new = terms.rename_tag(old, find_text, replace_text)
        elif op == BULK_DELETE:
            new = terms.delete_tag(old, find_text)
        else:
            new = terms.replace_text(old, find_text, replace_text or "", bool(use_regex))
        if new != old:
            try:
                caption_io.write_caption(image_paths[name], new)
            except OSError as e:
                errors.append(f"{name}: {e}")
                continue
            captions[name] = new
            changed += 1
    status = f"{op}: changed {changed} of {len(targets)} caption file(s). {snap_note}".strip() + _error_note(errors)
    return status, captions, term_rows(captions, term_mode, trigger, class_noun)


def normalize_files(how, joiner_label, style_label, profile_label, image_paths, captions):
    """Fixes files written by the old Append/Prepend ('\\n\\n---\\n\\n') or with stray blank lines."""
    captions = dict(captions or {})
    if not image_paths:
        return "No images loaded.", captions
    style = profiles.style_of(profiles.get_profile(profile_label), style_label)
    joiner = joiner_value(joiner_label, style)
    targets = [n for n in sorted(image_paths) if caption_io.needs_normalizing(captions.get(n, "")) or
               (how == NORMALIZE_MERGE and caption_io.variant_count(captions.get(n, "")) > 1)]
    if not targets:
        return "Nothing to normalize: every caption file is clean.", captions
    snap_note = cl._snapshot({t: image_paths[t] for t in targets}, "before normalize")
    mode = "merge" if how == NORMALIZE_MERGE else "variants"
    done, errors = 0, []
    for name in targets:
        new = caption_io.normalize(captions.get(name, ""), mode, joiner)
        try:
            caption_io.write_caption(image_paths[name], new)
        except OSError as e:
            errors.append(f"{name}: {e}")
            continue
        captions[name] = new
        done += 1
    return f"Normalized {done} caption file(s) ({how.lower()}). {snap_note}".strip() + _error_note(errors), captions


def _error_note(errors: list[str]) -> str:
    if not errors:
        return ""
    shown = "\n".join(errors[:10]) + (f"\n… and {len(errors) - 10} more" if len(errors) > 10 else "")
    return f"\n⚠ Could not write {len(errors)} file(s); they were left unchanged:\n{shown}"


# ---------- snapshots ----------
def diff_with_snapshot(snap_id, image_paths, captions):
    folder = cl._dataset_folder(image_paths)
    if not folder or not snap_id:
        return "Load a folder and pick a snapshot."
    try:
        return snapshots.diff_snapshot(folder, snap_id, captions or {})[1]
    except (OSError, ValueError) as e:
        return f"Could not read snapshot: {e}"


def restore_from_snapshot(snap_id, image_paths, captions, term_mode, trigger, class_noun):
    folder = cl._dataset_folder(image_paths)
    if not folder or not snap_id:
        return "Load a folder and pick a snapshot.", captions, gr.update(), gr.update()
    try:
        safety_id, changed = snapshots.restore_snapshot(folder, snap_id, image_paths)
    except (OSError, ValueError) as e:
        return f"Restore failed: {e}", captions, gr.update(), gr.update()
    reloaded = {name: caption_io.read_caption(path) for name, path in image_paths.items()}
    status = f"Restored {changed} caption file(s) from {snap_id}. The previous state was saved as {safety_id}."
    return status, reloaded, term_rows(reloaded, term_mode, trigger, class_noun), snapshot_dropdown(image_paths)


# ---------- generation wrappers ----------
def generate_selected(selected, agent, model, mode, image_paths, captions, settings, models_data, limiters, teams,
                      file_agents, history_list, session, instruction, prefix, suffix,
                      profile_label, preset_label, style_label, joiner_label, omit_text, temperature, seed, max_side, retries):
    config = build_caption_config(profile_label, preset_label, style_label, joiner_label, omit_text,
                                  temperature, seed, max_side, retries)
    return cl.generate_captions_for_selected(selected, agent, model, mode, image_paths, captions, settings, models_data,
                                             limiters, teams, file_agents, history_list, session,
                                             caption_instruction=instruction, caption_prefix=prefix,
                                             caption_suffix=suffix, caption_config=config)


def generate_all(image_paths, captions, agent, model, mode, settings, models_data, limiters, teams, file_agents,
                 history_list, session, instruction, prefix, suffix,
                 profile_label, preset_label, style_label, joiner_label, omit_text, temperature, seed, max_side, retries):
    config = build_caption_config(profile_label, preset_label, style_label, joiner_label, omit_text,
                                  temperature, seed, max_side, retries)
    yield from cl.iter_generate_captions_for_all(image_paths, captions, agent, model, mode, settings, models_data,
                                                 limiters, teams, file_agents, history_list, session,
                                                 caption_instruction=instruction, caption_prefix=prefix,
                                                 caption_suffix=suffix, caption_config=config)


def batch_add(scope, selected, text, mode, image_paths, captions, joiner_label, style_label, profile_label):
    style = profiles.style_of(profiles.get_profile(profile_label), style_label)
    return cl.batch_edit_captions(scope, selected, text, mode, image_paths, captions, joiner=joiner_value(joiner_label, style))


def sync_editor(selected, captions, image_paths, profile_label):
    """After batch operations: show the selected image's current caption, tokens and info."""
    caption = (captions or {}).get(selected, "") if selected else ""
    return caption, token_report(caption, profile_label), caption_info(selected, captions, image_paths)
