# ArtAgent/tests/test_dataset_ui.py
"""Tests for core/dataset_ui.py (Gradio callbacks of the Image Captions editor), with real files in tmp_path."""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import dataset_ui as ui
from core.dataset import profiles, provenance, snapshots, terms

GEN_SELECTED_PATH = "core.dataset_ui.cl.generate_captions_for_selected"
GEN_ALL_PATH = "core.dataset_ui.cl.iter_generate_captions_for_all"
BATCH_EDIT_PATH = "core.dataset_ui.cl.batch_edit_captions"

PONY = profiles.get_profile("pony_v6")["label"]
SDXL = profiles.get_profile("sd_sdxl")["label"]
STYLE = profiles.get_preset("style")["label"]
GENERAL = profiles.get_preset("general")["label"]
COMMA, SPACE = "comma  ( , )", "space  (   )"


def _update(value):
    return isinstance(value, dict) and value.get("__type__") == "update"


@pytest.fixture
def dataset(tmp_path):
    """a/b/c images; a and b have tag captions, c has none."""
    for name in ("a", "b", "c"):
        Image.new("RGB", (16, 16), "white").save(tmp_path / f"{name}.png")
    (tmp_path / "a.txt").write_text("txcl, cat, sofa", encoding="utf-8")
    (tmp_path / "b.txt").write_text("txcl, cat, window", encoding="utf-8")
    paths = {f"{n}.png": str(tmp_path / f"{n}.png") for n in ("a", "b", "c")}
    captions = {"a.png": "txcl, cat, sofa", "b.png": "txcl, cat, window", "c.png": ""}
    return tmp_path, paths, captions


# --- choices / small helpers ---

def test_labels_come_from_yaml():
    assert PONY in ui.profile_labels()
    assert STYLE in ui.preset_labels()


@pytest.mark.parametrize("label, style, expected", [
    (COMMA, profiles.STYLE_PROSE, ", "),
    (SPACE, profiles.STYLE_TAGS, " "),
    (None, profiles.STYLE_TAGS, ", "),
    ("unknown", profiles.STYLE_PROSE, " "),
])
def test_joiner_value(label, style, expected):
    assert ui.joiner_value(label, style) == expected


def test_joiner_label_and_term_mode_for_style():
    assert ui.joiner_label_for(profiles.STYLE_TAGS) == COMMA
    assert ui.joiner_label_for(profiles.STYLE_PROSE) == SPACE
    assert ui.term_mode_for(profiles.STYLE_TAGS) == terms.MODE_TAGS
    assert ui.term_mode_for(profiles.STYLE_PROSE) == terms.MODE_WORDS


# --- prefix / instruction / profile change ---

def test_compose_prefix_with_and_without_quality():
    assert ui.compose_prefix(STYLE, PONY, "Tags", "txcl", "painting", False) == "txcl painting, "
    quality = profiles.get_profile("pony_v6")["quality_prefix"]
    assert ui.compose_prefix(STYLE, PONY, "Tags", "txcl", "painting", True) == f"txcl painting, {quality}, "
    assert ui.compose_prefix(STYLE, PONY, "Tags", "", "", True) == f"{quality}, "
    assert ui.compose_prefix(STYLE, SDXL, "Natural language", "txcl", "", True) == "txcl, " # No quality tags for SDXL


def test_compose_instruction_returns_mismatch_warning():
    instruction, warning = ui.compose_instruction(PONY, STYLE, "Natural language", [], 30, "watercolor", "painting", "")
    assert instruction.startswith("Write one caption")
    assert "Do not use these words: watercolor." in instruction
    assert warning.startswith("⚠️")
    _, no_warning = ui.compose_instruction(PONY, STYLE, "Tags", [], 30, "", "painting", "")
    assert no_warning == ""


def test_on_profile_or_preset_change_tags_profile():
    out = ui.on_profile_or_preset_change(PONY, STYLE, "txcl", "painting", "txcl, cat", "")
    assert len(out) == 13
    (style, options, word_target, omit_text, prefix, quality, instruction, mismatch, guidance,
     joiner, term_mode, tokens, class_info) = out
    assert style["value"] == "Tags"
    assert options["value"] == profiles.default_options(profiles.get_profile("pony_v6"), profiles.get_preset("style"))
    assert word_target["value"] == 25
    assert omit_text.startswith("style, art style, watercolor")
    assert prefix == "txcl painting, "
    assert quality["interactive"] is True and quality["value"] is False and "score_9" in quality["label"]
    assert instruction.startswith("Write tags for this image")
    assert mismatch == ""
    assert "**Target model: Pony Diffusion V6" in guidance
    assert joiner["value"] == COMMA
    assert term_mode["value"] == terms.MODE_TAGS
    assert tokens.startswith("CLIP: ")
    assert class_info["info"] == "e.g. painting, illustration, photo, artwork"


def test_on_profile_or_preset_change_prose_profile_without_quality():
    out = ui.on_profile_or_preset_change(SDXL, GENERAL, "", "", "", "")
    assert out[0]["value"] == "Natural language"
    assert out[3] == "" # No omit terms for the general preset
    assert out[4] == "" # No trigger, no prefix
    assert out[5]["interactive"] is False and out[5]["label"] == "Add quality tags (none for this model)"
    assert out[9]["value"] == SPACE
    assert out[10]["value"] == terms.MODE_WORDS
    assert out[11] == "" # No caption, no token report
    assert out[12]["info"] is None


def test_token_report():
    assert ui.token_report("", PONY) == ""
    assert ui.token_report("a cat", PONY) == "CLIP: 2/75 (max 225) ✅"


def test_build_caption_config():
    config = ui.build_caption_config(PONY, STYLE, None, None, "watercolor, oil painting", 0.2, -1, 768, 2)
    assert config == {
        "style": "tags", "joiner": ", ", "omit_terms": ["watercolor", "oil painting"],
        "temperature": 0.2, "seed": None, "max_image_side": 768, "retries": 2,
        "max_output_tokens": 160, "budgets": profiles.get_profile("pony_v6")["budgets"],
        "profile": "pony_v6", "preset": "style",
    }
    other = ui.build_caption_config(SDXL, GENERAL, "Tags", SPACE, "", None, 42.0, 0, None)
    assert other["style"] == "tags" and other["joiner"] == " "
    assert other["temperature"] is None and other["seed"] == 42
    assert other["max_image_side"] is None and other["retries"] == 0


# --- caption info / loading / selection ---

def test_caption_info_variants_and_provenance(dataset):
    folder, paths, _ = dataset
    assert ui.caption_info(None, {}, paths) == ""
    assert ui.caption_info("a.png", {"a.png": "one"}, paths) == ""
    provenance.write_provenance(paths["a.png"], {"model": "gemma", "agent": "Captioner", "timestamp": "t"})
    info = ui.caption_info("a.png", {"a.png": "one\ntwo"}, paths)
    assert info == "📑 2 caption variants (one per line) · 🧾 gemma · Captioner · t"


def test_load_folder(dataset):
    folder, paths, captions = dataset
    (folder / "c.txt").write_text("old a\n\n---\n\nold b", encoding="utf-8")
    out = ui.load_folder(str(folder), PONY, terms.MODE_TAGS, "txcl", "")
    assert len(out) == 14
    gallery, loaded_paths, loaded_captions, status, first, caption, filename, view, term, filter_md, rows, snaps, tokens, info = out
    assert loaded_paths == paths
    assert view == ["a.png", "b.png", "c.png"] and term == ""
    assert "1 caption file(s) contain several lines or old '---' separators" in status
    assert first == filename == "a.png" and caption == "txcl, cat, sofa"
    assert filter_md == "Showing all 3 images."
    assert {r[0]: r[3] for r in rows}["txcl"] == "trigger/class"
    assert _update(snaps) and snaps["choices"] == [] and snaps["value"] is None
    assert tokens.startswith("CLIP: ")


def test_select_image_uses_current_view(dataset):
    _, paths, captions = dataset
    view = ["b.png", "c.png"] # Filtered gallery
    out = ui.select_image(SimpleNamespace(index=0), view, captions, paths, PONY)
    assert out[:3] == ("txcl, cat, window", "b.png", "b.png")
    assert out[3].startswith("CLIP: ")
    assert ui.select_image(SimpleNamespace(index=[1, 0]), view, captions, paths, PONY)[1] == "c.png" # Index as list
    assert ui.select_image(SimpleNamespace(index=0), None, captions, paths, PONY)[1] == "a.png" # No view: all sorted


@pytest.mark.parametrize("evt", [None, SimpleNamespace(index=None), SimpleNamespace(index=5), SimpleNamespace(index=-1)])
def test_select_image_invalid_events(dataset, evt):
    _, paths, captions = dataset
    assert ui.select_image(evt, ["a.png"], captions, paths, PONY) == ("", None, None, "", "")


def test_filter_by_term_and_clear(dataset):
    _, paths, captions = dataset
    gallery, view, term, md, common = ui.filter_by_term(
        SimpleNamespace(row_value=["sofa", 1, 33.3, ""], value="sofa", index=[0, 0]), captions, paths, terms.MODE_TAGS)
    assert view == ["a.png"] and term == "sofa"
    assert [label for _, label in gallery] == ["a.png"]
    assert md == "🔎 Showing 1/3 images with **sofa**. Bulk edits can target just these."
    assert common == "" # Only one image
    _, view2, _, _, common2 = ui.filter_by_term(SimpleNamespace(row_value=["cat", 2, 66.7, ""]), captions, paths, terms.MODE_TAGS)
    assert view2 == ["a.png", "b.png"]
    assert common2 == "Shared by all filtered images: cat, txcl"

    gallery, view, term, md, common = ui.clear_filter(paths)
    assert view == ["a.png", "b.png", "c.png"] and term == "" and md == "Showing all 3 images." and common == ""


def test_filter_by_term_without_term_changes_nothing(dataset):
    _, paths, captions = dataset
    out = ui.filter_by_term(SimpleNamespace(row_value=None, value=None), captions, paths, terms.MODE_TAGS)
    assert len(out) == 5 and all(_update(o) for o in out)


# --- bulk edits ---

def _bulk(op, find, replace, dataset, scope=ui.ALL_SCOPE, view=None, regex=False):
    _, paths, captions = dataset
    return ui.bulk_edit(op, find, replace, regex, scope, view, captions, paths, terms.MODE_TAGS, "txcl", "")


def test_bulk_rename_writes_files_and_snapshots(dataset):
    folder, paths, _ = dataset
    status, captions, rows = _bulk(ui.BULK_RENAME, "cat", "kitty", dataset)
    assert status.startswith("Rename tag: changed 2 of 3 caption file(s). Snapshot saved:")
    assert captions["a.png"] == "txcl, kitty, sofa"
    assert (folder / "b.txt").read_text(encoding="utf-8") == "txcl, kitty, window"
    assert not (folder / "c.txt").exists() # Unchanged empty caption not written
    assert "kitty" in [r[0] for r in rows]
    snap = snapshots.list_snapshots(str(folder))[0]
    assert snap["operation"] == "before rename tag"
    assert snapshots.snapshot_captions(str(folder), snap["id"])["a.png"] == "txcl, cat, sofa"


def test_bulk_delete_with_filter_scope(dataset):
    folder, _, _ = dataset
    status, captions, _ = _bulk(ui.BULK_DELETE, "cat", "", dataset, scope=ui.FILTER_SCOPE, view=["b.png"])
    assert status.startswith("Delete tag: changed 1 of 1 caption file(s).")
    assert captions["b.png"] == "txcl, window"
    assert (folder / "a.txt").read_text(encoding="utf-8") == "txcl, cat, sofa" # Outside the filter


def test_bulk_filter_scope_without_view_targets_all(dataset):
    status, *_ = _bulk(ui.BULK_DELETE, "cat", "", dataset, scope=ui.FILTER_SCOPE, view=[])
    assert "changed 2 of 3" in status


def test_bulk_replace_regex(dataset):
    status, captions, _ = _bulk(ui.BULK_REPLACE, r"s\w+a", "couch", dataset, regex=True)
    assert captions["a.png"] == "txcl, cat, couch"
    assert "changed 1 of 3" in status


@pytest.mark.parametrize("op, find, replace, regex, expected", [
    (ui.BULK_RENAME, "", "x", False, "Enter the tag or text to find."),
    (ui.BULK_RENAME, "cat", " ", False, "Enter the new tag name."),
    (ui.BULK_REPLACE, "([", "x", True, "Invalid regular expression:"),
])
def test_bulk_edit_validation(dataset, op, find, replace, regex, expected):
    folder, _, captions = dataset
    status, returned, rows = _bulk(op, find, replace, dataset, regex=regex)
    assert status.startswith(expected)
    assert returned == captions and _update(rows)
    assert snapshots.list_snapshots(str(folder)) == [] # Nothing snapshotted or written


def test_bulk_edit_without_images():
    status, captions, _ = ui.bulk_edit(ui.BULK_DELETE, "x", "", False, ui.ALL_SCOPE, None, {}, {}, terms.MODE_TAGS, "", "")
    assert status == "No images loaded."


def test_bulk_edit_reports_write_errors_and_keeps_state_in_sync(dataset):
    folder, paths, captions = dataset
    paths = dict(paths, **{"z.png": str(folder / "missing_dir" / "z.png")})
    captions = dict(captions, **{"z.png": "txcl, cat"})
    status, returned, _ = ui.bulk_edit(ui.BULK_RENAME, "cat", "kitty", False, ui.ALL_SCOPE, None, captions, paths,
                                       terms.MODE_TAGS, "txcl", "")
    assert "z.png" in status
    assert returned["a.png"] == "txcl, kitty, sofa" # Written file is reflected in the state
    assert returned["z.png"] == "txcl, cat"


# --- normalize ---

def test_normalize_files_merge(dataset):
    folder, paths, captions = dataset
    captions = dict(captions, **{"a.png": "x\n\n---\n\ny", "b.png": "v1\nv2"})
    status, result = ui.normalize_files(ui.NORMALIZE_MERGE, COMMA, None, PONY, paths, captions)
    assert status.startswith("Normalized 2 caption file(s) (merge lines into one caption).")
    assert result["a.png"] == "x, y" and result["b.png"] == "v1, v2"
    assert (folder / "a.txt").read_text(encoding="utf-8") == "x, y"


def test_normalize_files_variants_keeps_clean_multiline(dataset):
    folder, paths, captions = dataset
    captions = dict(captions, **{"a.png": "x\n\n---\n\ny", "b.png": "v1\nv2"})
    status, result = ui.normalize_files(ui.NORMALIZE_VARIANTS, None, None, SDXL, paths, captions)
    assert status.startswith("Normalized 1 caption file(s)")
    assert result["a.png"] == "x\ny"
    assert result["b.png"] == "v1\nv2" # Already clean variants: untouched
    assert (folder / "a.txt").read_text(encoding="utf-8") == "x\ny"


def test_normalize_files_nothing_to_do(dataset):
    _, paths, captions = dataset
    assert ui.normalize_files(ui.NORMALIZE_MERGE, None, None, PONY, paths, captions)[0] == \
        "Nothing to normalize: every caption file is clean."
    assert ui.normalize_files(ui.NORMALIZE_MERGE, None, None, PONY, {}, {})[0] == "No images loaded."


# --- snapshots in the UI ---

def test_diff_and_restore_from_snapshot(dataset):
    folder, paths, captions = dataset
    _, edited, _ = _bulk(ui.BULK_RENAME, "cat", "kitty", dataset)
    snap_id = snapshots.list_snapshots(str(folder))[0]["id"]

    assert ui.diff_with_snapshot(None, paths, edited) == "Load a folder and pick a snapshot."
    assert ui.diff_with_snapshot("../bad", paths, edited).startswith("Could not read snapshot:")
    diff = ui.diff_with_snapshot(snap_id, paths, edited)
    assert diff.startswith("2 caption file(s) differ.") and "+txcl, kitty, sofa" in diff

    status, reloaded, rows, dropdown = ui.restore_from_snapshot(snap_id, paths, edited, terms.MODE_TAGS, "txcl", "")
    assert status.startswith(f"Restored 2 caption file(s) from {snap_id}. The previous state was saved as ")
    assert reloaded == captions
    assert (folder / "a.txt").read_text(encoding="utf-8") == "txcl, cat, sofa"
    assert "cat" in [r[0] for r in rows]
    assert _update(dropdown) and len(dropdown["choices"]) == 2


def test_restore_from_snapshot_errors(dataset):
    _, paths, captions = dataset
    assert ui.restore_from_snapshot("", paths, captions, terms.MODE_TAGS, "", "")[0] == "Load a folder and pick a snapshot."
    assert ui.restore_from_snapshot("x", {}, captions, terms.MODE_TAGS, "", "")[0] == "Load a folder and pick a snapshot."
    assert ui.restore_from_snapshot(".bad", paths, captions, terms.MODE_TAGS, "", "")[0].startswith("Restore failed:")


def test_snapshot_dropdown_lists_newest_first(dataset):
    folder, paths, _ = dataset
    assert ui.snapshot_dropdown({})["choices"] == []
    _bulk(ui.BULK_DELETE, "sofa", "", dataset)
    dropdown = ui.snapshot_dropdown(paths)
    (label, value), = dropdown["choices"]
    assert dropdown["value"] == value
    assert "before delete tag (2 files)" in label


# --- generation / batch wrappers ---

GEN_STATE = ({}, [], {}, {}, {}, [], [])


def test_generate_selected_builds_caption_config(dataset):
    _, paths, captions = dataset
    with patch(GEN_SELECTED_PATH, return_value=("ok", {}, "", [])) as mock_gen:
        result = ui.generate_selected("a.png", "Captioner", "m (VISION)", "Overwrite", paths, captions, *GEN_STATE,
                                      "instr", "txcl, ", "", PONY, STYLE, None, None, "watercolor", 0.0, 7, 768, 1)
    assert result == ("ok", {}, "", [])
    kwargs = mock_gen.call_args.kwargs
    assert kwargs["caption_instruction"] == "instr" and kwargs["caption_prefix"] == "txcl, "
    config = kwargs["caption_config"]
    assert config["style"] == "tags" and config["seed"] == 7 and config["retries"] == 1
    assert config["omit_terms"] == ["watercolor"] and config["preset"] == "style"


def test_generate_all_streams_with_config(dataset):
    _, paths, captions = dataset
    with patch(GEN_ALL_PATH, return_value=iter([("1/2", {}, "", []), ("done", {}, "", [])])) as mock_gen:
        updates = list(ui.generate_all(paths, captions, "Captioner", "m", "Overwrite", *GEN_STATE,
                                       "instr", "", "", SDXL, GENERAL, None, SPACE, "", None, -1, None, 0))
    assert [u[0] for u in updates] == ["1/2", "done"]
    assert mock_gen.call_args.kwargs["caption_config"]["joiner"] == " "


def test_batch_add_uses_selected_joiner(dataset):
    folder, paths, captions = dataset
    status, updated = ui.batch_add(ui.ALL_SCOPE, None, "ohwx", "Append", paths, captions, SPACE, None, PONY)
    assert updated["a.png"] == "txcl, cat, sofa ohwx"
    with patch(BATCH_EDIT_PATH, return_value=("s", {})) as mock_batch:
        ui.batch_add(ui.ALL_SCOPE, None, "ohwx", "Append", paths, captions, None, None, PONY)
    assert mock_batch.call_args.kwargs["joiner"] == ", " # Tag profile default


def test_sync_editor(dataset):
    _, paths, captions = dataset
    assert ui.sync_editor(None, captions, paths, PONY) == ("", "", "")
    caption, tokens, info = ui.sync_editor("a.png", captions, paths, PONY)
    assert caption == "txcl, cat, sofa" and tokens.startswith("CLIP: ") and info == ""
