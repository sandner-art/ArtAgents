# ArtAgent/tests/test_dataset_tools_ui.py
"""Tests for core/dataset_tools_ui.py (Gradio glue for the Tier 2 dataset tools)."""
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import captioning_logic as cl
from core import dataset_tools_ui as dtu
from core import dataset_ui as dui
from core.dataset import audit, caption_io, profiles, structured, trainer_export

PONY = "Pony Diffusion V6 (SDXL, booru tags)"
FLUX = "FLUX.1 [dev] / Chroma (natural language)"
TAGS = profiles.STYLE_LABELS["tags"]
PROSE = profiles.STYLE_LABELS["prose"]
NO_PROGRESS = lambda *a, **k: None  # noqa: E731


@pytest.fixture(autouse=True)
def _private_tempdir(tmp_path, monkeypatch):
    """Download copies (artagents_export_*) go to this test's tmp dir instead of the system temp dir."""
    scratch = tmp_path / "tempdir"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))


@pytest.fixture
def dataset(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    paths = {}
    for i, name in enumerate(["a.png", "b.png", "c.png"]):
        Image.new("RGB", (600, 400), (40 * i, 80, 120)).save(d / name)
        paths[name] = str(d / name)
    captions = {"a.png": "txcl, cat", "b.png": "txcl, dog", "c.png": ""}
    for name, text in captions.items():
        if text:
            caption_io.write_caption(paths[name], text)
    return d, paths, captions


# --- source config & visibility ---

SOURCE_ARGS = ["Tagger tags as hints for the vision model", list(dtu.BACKEND_CHOICES)[1], " http://h:1/v1 ", " jc ", " k ",
               "WD SwinV2 v3 (general use)", 0.4, 0.9, True, False, 0.6, True, False, "simple background, , watermark",
               False, ["Subject", "Hair"], " {subject} "]


def test_source_config():
    cfg = dtu.source_config(*SOURCE_ARGS)
    assert cfg["source"] == cl.SOURCE_TAGGER_HINTS and cfg["backend"] == "openai"
    assert cfg["openai"] == {"base_url": "http://h:1/v1", "model": "jc", "api_key": "k"}
    t = cfg["tagger"]
    assert t["label"] == "WD SwinV2 v3 (general use)" and t["general_threshold"] == 0.4 and t["character_threshold"] == 0.9
    assert t["general_mcut"] is True and t["character_mcut"] is False and t["hint_threshold"] == 0.6
    assert t["add_rating"] is True and t["blacklist"] == ["simple background", "watermark"] and t["reuse"] is False
    assert cfg["structured"] == {"fields": ["subject", "hair"], "template": "{subject}"}


def test_source_config_defaults_for_unknown_and_empty():
    cfg = dtu.source_config("??", "??", None, None, None, None, None, None, None, None, None, None, None, None, None, [], None)
    assert cfg["source"] == cl.SOURCE_VLM and cfg["backend"] == "ollama"
    assert cfg["tagger"]["general_threshold"] == 0.35 and cfg["tagger"]["character_threshold"] == 0.85
    assert cfg["structured"]["fields"] == structured.DEFAULT_FIELDS


@pytest.mark.parametrize("source, backend, tagger_v, struct_v, openai_v, ollama_v, note", [
    ("Vision model", "Ollama", False, False, False, True, ""),
    ("Vision model", list(dtu.BACKEND_CHOICES)[1], False, False, True, False, "Agent / Team selection is not used"),
    ("Booru tagger only (no vision model)", list(dtu.BACKEND_CHOICES)[1], True, False, False, False, "tagger's tags only"),
    ("Tagger tags + vision-model caption", "Ollama", True, False, False, True, ""),
    ("Structured fields (JSON) → caption", "Ollama", False, True, False, True, "Agent / Team selection is not used"),
])
def test_on_source_change(source, backend, tagger_v, struct_v, openai_v, ollama_v, note):
    tagger_u, struct_u, openai_u, ollama_u, md, template = dtu.on_source_change(source, backend, ["Subject", "Action / pose"], PROSE)
    assert (tagger_u["visible"], struct_u["visible"], openai_u["visible"], ollama_u["visible"]) == \
        (tagger_v, struct_v, openai_v, ollama_v)
    assert (note in md) if note else md == ""
    assert template == "{subject} {action}"


def test_on_fields_change():
    assert dtu.on_fields_change(["Subject", "Setting"], TAGS) == "{subject}, {setting}"
    assert dtu.on_fields_change([], PROSE) == structured.default_template(structured.DEFAULT_FIELDS, "prose")


def test_list_openai_models():
    with patch("core.openai_compat.OpenAICompatClient") as cls:
        cls.return_value.list_models.return_value = ["a", "b"]
        update, md = dtu.list_openai_models("http://h:1", "k")
    assert update["choices"] == ["a", "b"] and update["value"] == "a" and md.startswith("✅ 2 model(s)")
    with patch("core.openai_compat.OpenAICompatClient") as cls:
        cls.return_value.list_models.return_value = []
        _, md = dtu.list_openai_models("http://h:1", "")
    assert "lists no models" in md
    _, md = dtu.list_openai_models("", "")
    assert md.startswith("⚠️ Enter the server URL")


def test_generate_selected_merges_source_config():
    with patch("core.captioning_logic.generate_captions_for_selected", return_value=("ok", {}, "", [])) as gen:
        dtu.generate_selected("a.png", "Agent", "m", "Overwrite", {}, {}, {}, [], {}, {}, {}, [], [], "instr", "p, ", "",
                              PONY, "Style (painting, illustration, photo look)", TAGS, "comma  ( , )", "watercolor",
                              0, 42, 1024, 2, *SOURCE_ARGS)
    cfg = gen.call_args.kwargs["caption_config"]
    assert cfg["profile"] == "pony_v6" and cfg["style"] == "tags" and cfg["seed"] == 42
    assert cfg["source"] == cl.SOURCE_TAGGER_HINTS and cfg["openai"]["model"] == "jc"
    assert gen.call_args.kwargs["caption_prefix"] == "p, "


def test_generate_all_merges_source_config():
    with patch("core.captioning_logic.iter_generate_captions_for_all", return_value=iter([("done", {}, "", [])])) as gen:
        out = list(dtu.generate_all({}, {}, "Agent", "m", "Overwrite", {}, [], {}, {}, {}, [], [], "i", "", "",
                                    FLUX, "General / no preset", PROSE, "space  (   )", "", 0, -1, 0, 0, *SOURCE_ARGS))
    assert out == [("done", {}, "", [])]
    cfg = gen.call_args.kwargs["caption_config"]
    assert cfg["seed"] is None and cfg["max_image_side"] is None and cfg["backend"] == "openai"


# --- attribute sheet ---

@pytest.fixture
def with_fields(dataset):
    d, paths, captions = dataset
    structured.save_fields(paths["a.png"], {"subject": "a cat", "setting": "a sofa"}, model="m")
    structured.save_fields(paths["b.png"], {"subject": "a dog", "setting": "a park"}, model="m")
    return d, paths, captions


def test_load_sheet(with_fields):
    _, paths, _ = with_fields
    update, columns, summary, status = dtu.load_sheet(paths)
    assert columns == ["subject", "setting"]
    assert update["headers"] == ["image", "Subject", "Setting"]
    assert update["value"] == [["a.png", "a cat", "a sofa"], ["b.png", "a dog", "a park"]]
    assert summary[0][0] == "Subject" and status.startswith("2 image(s) with fields")


def test_load_sheet_empty(dataset):
    _, paths, _ = dataset
    assert dtu.load_sheet(paths)[3].startswith("No structured fields yet")
    assert dtu.load_sheet({})[3] == "No images loaded."


def test_save_sheet_from_dataframe(with_fields):
    _, paths, _ = with_fields
    frame = pd.DataFrame([["a.png", "a kitten", "a sofa"], ["b.png", "a dog", "a park"]],
                         columns=["image", "Subject", "Setting"])
    status, summary = dtu.save_sheet(frame, ["subject", "setting"], paths)
    assert status == "Saved fields for 1 image(s)."
    assert structured.load_fields(paths["a.png"])["subject"] == "a kitten"
    assert dtu.save_sheet(frame, [], paths)[0] == "Load the sheet first."


def test_rows_from_frame_variants():
    assert dtu._rows_from_frame(None) == []
    assert dtu._rows_from_frame([("a", "b")]) == [["a", "b"]]
    assert dtu._rows_from_frame(pd.DataFrame([["a", "b"]], columns=["x", "y"])) == [["a", "b"]]


def test_rows_from_frame_dict_payload():
    assert dtu._rows_from_frame({"headers": ["x", "y"], "data": [["a", "b"]]}) == [["a", "b"]]


def test_render_from_fields_snapshot_first(with_fields):
    d, paths, captions = with_fields
    with patch("core.captioning_logic._snapshot", wraps=cl._snapshot) as snap:
        status, new = dtu.render_from_fields(["Subject", "Setting"], "", TAGS, PONY, "txcl, ", "", dui.ALL_SCOPE, None,
                                             paths, captions, "comma  ( , )")
    assert snap.call_args[0][1] == "before render from fields"
    assert set(snap.call_args[0][0]) == {"a.png", "b.png"}          # c.png has no fields
    assert new["a.png"] == "txcl, a cat, a sofa" and new["c.png"] == ""
    assert (d / "b.txt").read_text(encoding="utf-8") == "txcl, a dog, a park"
    assert status.startswith("Rendered 2 caption(s) from fields.")


def test_render_from_fields_include_template_and_scope(with_fields):
    _, paths, captions = with_fields
    status, new = dtu.render_from_fields(["Setting"], "{setting} with {subject}", PROSE, FLUX, "", "", dui.FILTER_SCOPE,
                                         ["b.png"], paths, captions, "space  (   )")
    assert new["b.png"] == "A park with." or new["b.png"].startswith("A park")
    assert new["a.png"] == "txcl, cat"                                # Not in the filtered view
    assert status.startswith("Rendered 1 caption(s)")


def test_render_from_fields_nothing(dataset):
    _, paths, captions = dataset
    assert dtu.render_from_fields([], "", TAGS, PONY, "", "", dui.ALL_SCOPE, None, paths, captions, "")[0] == \
        "No images with structured fields in this selection."
    assert dtu.render_from_fields([], "", TAGS, PONY, "", "", dui.ALL_SCOPE, None, {}, {}, "")[0] == "No images loaded."


def test_export_sheet(with_fields):
    d, paths, _ = with_fields
    frame = pd.DataFrame([["a.png", "a cat", "a sofa"]], columns=["image", "Subject", "Setting"])
    status, download = dtu.export_sheet(frame, ["subject", "setting"], paths)
    written = d / ".artagents" / "export" / "attributes.csv"
    assert status == f"Wrote {written}"
    assert written.read_text(encoding="utf-8").splitlines() == ["image,Subject,Setting", "a.png,a cat,a sofa"]
    # The download is a copy in the temp dir (Gradio does not serve the dataset folder)
    assert os.path.basename(download) == "attributes.csv" and not download.startswith(str(d))
    assert open(download, encoding="utf-8").read() == written.read_text(encoding="utf-8")
    assert dtu.export_sheet(frame, [], paths) == ("Load the sheet first.", None)


# --- health ---

def test_run_health(dataset):
    _, paths, captions = dataset
    summary, rows, flagged = dtu.run_health(paths, captions, PONY, "txcl", 512, progress=NO_PROGRESS)
    assert summary.startswith("**3 image(s) scanned")
    assert ["c.png", "no caption file", ""] in rows
    assert "c.png" in flagged
    assert dtu.run_health({}, {}, PONY, "", 1024, progress=NO_PROGRESS) == ("No images loaded.", [], [])


def test_show_images(dataset):
    _, paths, _ = dataset
    gallery, view, label, md, common = dtu.show_images(["c.png", "zzz.png"], paths)
    assert view == ["c.png"] and len(gallery) == 1 and md.startswith("🔎 Showing 1/3 images flagged")
    out = dtu.show_images([], paths)
    assert out[3] == "Nothing to show: run the scan first."


def test_quarantine_selected(dataset):
    d, paths, captions = dataset
    status, new_paths, new_caps, gallery, view, selected = dtu.quarantine_selected("a.png", paths, captions)
    assert "Moved a.png" in status and "Nothing was deleted" in status
    assert "a.png" not in new_paths and "a.png" not in new_caps and view == ["b.png", "c.png"] and selected is None
    assert (d / ".artagents" / "quarantine" / "a.png").exists() and (d / ".artagents" / "quarantine" / "a.txt").exists()
    assert "a.png" in paths   # The input dict is not mutated
    assert dtu.quarantine_selected(None, paths, captions)[0] == "Select an image in the gallery first."
    assert dtu.restore_quarantined(new_paths, "").startswith("Moved 2 file(s) back")
    assert dtu.restore_quarantined(new_paths, "") == "Nothing in quarantine (or the files already exist in the dataset)."
    assert dtu.restore_quarantined({}, "") == "Load a folder first."


def test_simulate_buckets(dataset):
    _, paths, _ = dataset
    summary, rows, bucket_rows = dtu.simulate_buckets(paths, 512, 256, 1024, 2, 1, False)
    assert summary.startswith("3 image(s) in 1 bucket(s) at 512px") and "kohya-ss" in summary
    assert rows[0][2] == "640×384" and bucket_rows[0][2] == 3
    assert dtu.simulate_buckets({}, 512, 256, 1024, 1, 1, False) == ("No images loaded.", [], [])


@pytest.mark.parametrize("style, trigger, cls, extra, expected", [
    (TAGS, "txcl", "painting", "zkx, txcl painting", ["txcl painting", "zkx"]),
    (TAGS, "txcl", "", "", ["txcl"]),
    (PROSE, "txcl", "painting", "", ["txcl"]),
    (TAGS, "", "painting", "zkx", ["zkx"]),
])
def test_run_audit_trigger_list(dataset, style, trigger, cls, extra, expected):
    _, paths, captions = dataset
    with patch("core.dataset.audit.audit", return_value={"summary": "s", "rows": []}) as run:
        dtu.run_audit(captions, paths, PONY, "General / no preset", style, trigger, cls, extra, "", "T", False)
    assert run.call_args[0][3] == expected
    assert run.call_args[0][5] is None   # use_tagger off


def test_run_audit_real(dataset):
    _, paths, captions = dataset
    summary, rows = dtu.run_audit(captions, paths, PONY, "General / no preset", TAGS, "txcl", "", "", "dog", None, False)
    assert ["b.png", audit.CHECK_OMITTED, "dog"] in rows and "Trigger 'txcl': in 2/2" in summary
    assert dtu.run_audit({}, {}, PONY, "", TAGS, "", "", "", "", None, False) == ("No images loaded.", [])


def test_run_tagger_compare(dataset):
    _, paths, _ = dataset
    with patch("core.dataset.tagger.tag_probabilities", side_effect=[({}, True), ({}, False), RuntimeError("bad")]):
        msg = dtu.run_tagger_compare(paths, "WD SwinV2 v3 (general use)", progress=NO_PROGRESS)
    assert msg.startswith("Tagger results stored for 2 image(s) (1 reused).") and "c.png: bad" in msg
    from core.dataset.tagger import TaggerError
    with patch("core.dataset.tagger.tag_probabilities", side_effect=TaggerError("needs onnxruntime")):
        assert dtu.run_tagger_compare(paths, "x", progress=NO_PROGRESS) == "⚠️ needs onnxruntime"
    assert dtu.run_tagger_compare({}, "x", progress=NO_PROGRESS) == "No images loaded."


# --- verifier ---

def _verify_args(scope, selected=None, view=None, paths=None, captions=None, verifier="other-vlm (VISION)",
                 captioner="vision-model (VISION)"):
    return [scope, selected, view, paths, captions, {}, verifier, captioner, "Ollama", "http://h:1", "oa-model", "k",
            PONY, TAGS, "txcl", ""]


def test_iter_verify_scopes(dataset):
    _, paths, captions = dataset
    done = []

    def fake_verify(path, caption, img, target, settings, style, skip):
        done.append((os.path.basename(path), target.model, style, skip))
        return {"disputed": ["cat"] if "cat" in caption else []}

    with patch("core.dataset.audit.verify", side_effect=fake_verify):
        out = list(dtu.iter_verify(*_verify_args(dui.ALL_SCOPE, paths=paths, captions=captions)))
        assert [d[0] for d in done] == ["a.png", "b.png"]       # c.png has no caption
        assert done[0][1:3] == ("other-vlm (VISION)", "tags") and set(done[0][3]) == {"txcl"}
        assert out[-1][0].startswith("Verified 2/2 caption(s); 1 with disputed claims.")
        done.clear()
        list(dtu.iter_verify(*_verify_args("Selected image", selected="b.png", paths=paths, captions=captions)))
        assert [d[0] for d in done] == ["b.png"]
        done.clear()
        list(dtu.iter_verify(*_verify_args(dui.FILTER_SCOPE, view=["b.png", "zzz.png"], paths=paths, captions=captions)))
        assert [d[0] for d in done] == ["b.png"]


def test_iter_verify_same_model_warning_and_errors(dataset):
    _, paths, captions = dataset
    from core.dataset.vision import VisionError
    with patch("core.dataset.audit.verify", side_effect=[VisionError("timeout"), {"disputed": []}]):
        out = list(dtu.iter_verify(*_verify_args(dui.ALL_SCOPE, paths=paths, captions=captions,
                                                 verifier="vision-model (VISION)")))
    assert "Same model as the captioner" in out[-1][0]
    assert "1 error(s): a.png: timeout" in out[-1][0]


def test_iter_verify_openai_target(dataset):
    _, paths, captions = dataset
    with patch("core.dataset.audit.verify", return_value={"disputed": []}) as verify:
        list(dtu.iter_verify(*_verify_args("Selected image", selected="a.png", paths=paths, captions=captions,
                                           verifier=dtu.VERIFIER_OPENAI)))
    target = verify.call_args[0][3]
    assert (target.backend, target.model, target.base_url, target.api_key) == ("openai", "oa-model", "http://h:1", "k")


def test_iter_verify_guards(dataset):
    _, paths, captions = dataset
    assert list(dtu.iter_verify(*_verify_args(dui.ALL_SCOPE, paths={}, captions={})))[0][0] == "No images loaded."
    assert list(dtu.iter_verify(*_verify_args(dui.ALL_SCOPE, paths=paths, captions=captions, verifier=None)))[0][0] == \
        "Choose a verifier model."
    out = list(dtu.iter_verify(*_verify_args("Selected image", selected=None, paths=paths, captions=captions)))
    assert out[0][0] == "No captioned images to verify in this selection."


def test_verification_table(dataset):
    _, paths, captions = dataset
    assert dtu.verification_table(paths, captions) == []
    assert dtu.verification_table(None, None) == []


# --- trainer export ---

def test_write_trainer_config(dataset):
    d, paths, captions = dataset
    captions = dict(captions, **{"a.png": "txcl, cat\nalt"})
    label = trainer_export.TRAINERS["kohya"][0]
    status, update = dtu.write_trainer_config(label, paths, captions, 768, 4, 2, 1, True, 0.1, 256, 1536, "txcl")
    path = d / ".artagents" / "export" / "dataset_kohya.toml"
    assert status == f"Wrote {path}"
    assert update["value"] == path.read_text(encoding="utf-8") and update["language"] is None
    assert "1 caption file(s) have several lines" in update["value"] and "enable_wildcard = true" in update["value"]
    assert "num_repeats = 4" in update["value"] and "max_bucket_reso = 1536" in update["value"]
    status, update = dtu.write_trainer_config(trainer_export.TRAINERS["onetrainer"][0], paths, captions, 1024, 1, 1, 1,
                                              False, 0, 256, 2048, "")
    assert update["language"] == "json"
    assert dtu.write_trainer_config(label, {}, {}, 1, 1, 1, 1, False, 0, 1, 1, "")[0] == "No images loaded."


def test_write_trainer_config_counts_webp(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "a.webp")
    _, update = dtu.write_trainer_config(trainer_export.TRAINERS["ai_toolkit"][0], {"a.webp": str(tmp_path / "a.webp")},
                                         {}, 1024, 1, 1, 1, False, 0.05, 256, 2048, "txcl")
    assert "1 WEBP image(s)" in update["value"]


def test_keep_tokens_and_repeats_table(dataset):
    _, paths, _ = dataset
    assert dtu.keep_tokens_from_prefix("a, b, ") == 2
    md = dtu.repeats_table(paths, 30, 5, 1, "style_b: 15\nbad line\nx: notanumber")
    assert "| ds | 3 | 10 | 30 | 30 |" in md and "| style_b | 15 | 2 | 30 | 30 |" in md
    assert "x" not in [row.split("|")[1].strip() for row in md.splitlines() if row.startswith("| x")]
    assert dtu.repeats_table({}, 100, 1, 1, "") == "Load a folder first."


# --- edit pairs ---

@pytest.fixture
def pair_dirs(tmp_path):
    t, c = tmp_path / "target", tmp_path / "control"
    t.mkdir(), c.mkdir()
    for name in ("p1", "p2"):
        Image.new("RGB", (1000, 1000), "red").save(t / f"{name}.png")
        Image.new("RGB", (1000, 1000), "blue").save(c / f"{name}.jpg")
    Image.new("RGB", (1000, 1000), "red").save(t / "p3.png")     # No control
    return t, c


def test_load_pairs(pair_dirs):
    t, c = pair_dirs
    pairs, gallery, rows, status, index, text = dtu.load_pairs(str(t), str(c))
    assert [p["stem"] for p in pairs] == ["p1", "p2", "p3"]
    assert len(gallery) == 5                                   # 2 × (input + result) + 1 result without input
    assert gallery[0][1] == "p1 · input" and gallery[1][1] == "p1 · result (no instruction)"
    assert ["p3", "no matching control image", "control"] in rows
    assert status.startswith("3 pair(s) found.") and "2 complete" in status
    assert index is None and text == ""


def test_select_pair_maps_gallery_index(pair_dirs):
    t, c = pair_dirs
    pairs = dtu.load_pairs(str(t), str(c))[0]
    caption_io.write_caption(pairs[1]["target"], "Make it blue.")
    for gallery_index, expected in ((0, 0), (1, 0), (2, 1), (3, 1), (4, 2)):
        evt = MagicMock(index=gallery_index)
        idx, text = dtu.select_pair(evt, pairs)
        assert idx == expected
    assert dtu.select_pair(MagicMock(index=2), pairs)[1] == "Make it blue."
    assert dtu.select_pair(None, pairs) == (None, "")


def test_save_pair_instruction(pair_dirs):
    t, c = pair_dirs
    pairs = dtu.load_pairs(str(t), str(c))[0]
    status, gallery, rows = dtu.save_pair_instruction(0, " Make it blue. \n", pairs)
    assert status == "Saved the instruction for p1."
    assert all(not (stem == "p1" and issue == "no instruction yet") for stem, issue, _ in rows)
    assert (t / "p1.txt").read_text(encoding="utf-8") == "Make it blue."
    assert gallery[1][1] == "p1 · result: Make it blue."
    assert dtu.save_pair_instruction(None, "x", pairs)[0] == "Select a pair first."


def _pair_args(scope, index, pairs, mode=caption_io.MODE_OVERWRITE, backend="Ollama"):
    return [scope, index, pairs, {}, "vision-model (VISION)", backend, "http://h:1", "oa", "", 2, "txcl", "", mode,
            0.0, 42, 1024]


def test_iter_pair_instructions(pair_dirs):
    t, c = pair_dirs
    pairs = dtu.load_pairs(str(t), str(c))[0]
    caption_io.write_caption(pairs[1]["target"], "existing")
    calls = []

    def fake(pair, target, settings, paraphrases, prefix, extra, temperature, seed, max_side):
        calls.append((pair["stem"], target.backend, target.model, paraphrases, prefix, seed))
        if pair["stem"] == "p3":
            raise ValueError("This pair has no control image.")
        return ["txcl, make it red", "txcl, turn it red"]

    with patch("core.dataset.edit_pairs.generate_instruction", side_effect=fake):
        out = list(dtu.iter_pair_instructions(*_pair_args("All pairs", None, pairs, caption_io.MODE_SKIP)))
    assert [c[0] for c in calls] == ["p1", "p3"]                 # p2 skipped: it has an instruction
    assert calls[0][1:] == ("ollama", "vision-model (VISION)", 2, "txcl", 42)
    assert (t / "p1.txt").read_text(encoding="utf-8") == "txcl, make it red\ntxcl, turn it red"
    assert out[-1][0].startswith("Wrote 1 instruction(s), skipped 1")
    assert "p3: This pair has no control image." in out[-1][0]


def test_iter_pair_instructions_selected_and_openai(pair_dirs):
    t, c = pair_dirs
    pairs = dtu.load_pairs(str(t), str(c))[0]
    with patch("core.dataset.edit_pairs.generate_instruction", return_value=["x"]) as gen:
        list(dtu.iter_pair_instructions(*_pair_args("Selected pair", 1, pairs, backend=list(dtu.BACKEND_CHOICES)[1])))
    assert gen.call_count == 1 and gen.call_args[0][0]["stem"] == "p2"
    target = gen.call_args[0][1]
    assert (target.backend, target.model, target.base_url) == ("openai", "oa", "http://h:1")
    assert list(dtu.iter_pair_instructions(*_pair_args("All pairs", None, [])))[0][0] == "Load pairs first."


def test_iter_pair_instructions_selected_without_selection(pair_dirs):
    t, c = pair_dirs
    pairs = dtu.load_pairs(str(t), str(c))[0]
    with patch("core.dataset.edit_pairs.generate_instruction", return_value=["x"]) as gen:
        out = list(dtu.iter_pair_instructions(*_pair_args("Selected pair", None, pairs)))
    assert gen.call_count == 0
    assert "Select a pair" in out[-1][0]


def test_export_pairs(pair_dirs):
    t, c = pair_dirs
    pairs = dtu.load_pairs(str(t), str(c))[0]
    caption_io.write_caption(pairs[0]["target"], "Make it blue.")
    export_dir = t / ".artagents" / "export"
    status, update, path = dtu.export_pairs(dtu.EDIT_EXPORTS[0], pairs, str(t), str(c), "txcl")
    assert status == f"Wrote {export_dir / 'ai_toolkit_datasets.yaml'}"
    assert os.path.basename(path) == "ai_toolkit_datasets.yaml" and not path.startswith(str(t))
    assert "control_path:" in update["value"] and update["language"] == "yaml"
    status, update, path = dtu.export_pairs(dtu.EDIT_EXPORTS[1], pairs, str(t), str(c), "")
    assert path.endswith("dataset_musubi.toml") and "control_directory" in update["value"]
    status, update, path = dtu.export_pairs(dtu.EDIT_EXPORTS[2], pairs, str(t), str(c), "")
    assert (export_dir / "edit_pairs_musubi.jsonl").is_file()
    rows = [json.loads(l) for l in open(path, encoding="utf-8").read().splitlines()]
    assert rows == [{"image_path": str(t / "p1.png"), "control_path": str(c / "p1.jpg"), "caption": "Make it blue."}]
    assert "(1 pairs with an instruction)" in status
    status, update, path = dtu.export_pairs(dtu.EDIT_EXPORTS[3], pairs, str(t), str(c), "")
    assert path.endswith("edit_pairs_fal.zip") and "(2 pairs)" in status
    assert dtu.export_pairs(dtu.EDIT_EXPORTS[0], [], str(t), "", "")[0] == "Load pairs first."
