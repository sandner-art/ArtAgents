# ArtAgent/tests/test_lab_experiments_ui.py
"""Tests for core/lab/experiments.py and the Tier 3 callbacks in core/lab_ui.py."""
import json
import os
import shutil
import sqlite3
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import lab_ui
from core.dataset import caption_io, profiles, snapshots
from core.image_gen.base import GeneratedImage, GenProgress, GenResult, ImageGenError
from core.lab import caption_sets as cs
from core.lab import experiments as ex
from core.lab import lora_sweep as ls
from core.lab import validation, voting


@pytest.fixture
def app_root(tmp_path):
    """Redirects every get_absolute_path used by the lab (validation sets, experiment DB, report folder)."""
    root = tmp_path / "app"
    root.mkdir()

    def gap(rel):
        return str(root / rel)
    with patch("core.lab.validation.get_absolute_path", side_effect=gap), \
            patch("core.lab.experiments.get_absolute_path", side_effect=gap), \
            patch("core.lab_ui.get_absolute_path", side_effect=gap):
        yield root


# =====================================================================
# experiments.py
# =====================================================================

@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "exp.sqlite")


def test_save_get_update_delete(db):
    i = ex.save({"name": "first", "caption_set": "short", "validation_version": 2, "lora_paths": "/a/x.safetensors\n/a/y.safetensors",
                 "metrics": [["c", "b"]], "votes": [["A", 1.2, 3, 1, 0, "75%"]]}, path=db)
    rec = ex.get(i, db)
    assert rec["name"] == "first" and rec["validation_version"] == "2" and rec["created"] == rec["updated"]
    assert json.loads(rec["metrics"]) == [["c", "b"]] and rec["app_version"]
    assert ex.save({"name": "renamed", "notes": "n"}, exp_id=i, path=db) == i
    rec = ex.get(i, db)
    assert rec["name"] == "renamed" and rec["notes"] == "n" and rec["caption_set"] == "short"  # Fields not given are kept
    j = ex.save({"name": "second", "validation_set": "vs", "validation_version": 1}, path=db)
    rows = ex.list_rows(db)
    assert [r[0] for r in rows] == [j, i]
    assert rows[0][6] == "vs v1" and rows[1][7] == "A (1.2)"  # Votes kept by the partial update
    ex.delete(i, db)
    assert [r[0] for r in ex.list_rows(db)] == [j]
    with pytest.raises(ValueError):
        ex.get(i, db)
    with pytest.raises(ValueError, match="No experiment"):
        ex.save({"name": "x"}, exp_id=999, path=db)
    with pytest.raises(ValueError, match="name"):
        ex.save({"name": "  "}, path=db)


def test_list_rows_lora_names_and_top_vote(db):
    ex.save({"name": "e", "lora_paths": "/a/x.safetensors\n\n/b/y.safetensors", "votes": [["A @ 1", 0.7], ["B", -0.7]]}, path=db)
    ex.save({"name": "bad votes", "votes": "not json"}, path=db)
    rows = ex.list_rows(db)
    assert rows[1][5] == "x.safetensors, y.safetensors" and rows[1][7] == "A @ 1 (0.7)"
    assert rows[0][7] == ""


def test_db_path_and_schema(app_root):
    path = ex.db_path()
    assert path == str(app_root / "experiments" / "artagents_experiments.sqlite") and os.path.isdir(os.path.dirname(path))
    i = ex.save({"name": "x"})
    cols = [r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(experiments)")]
    assert set(ex.FIELDS) <= set(cols) and ex.get(i)["name"] == "x"


def test_report_markdown_sections(db):
    i = ex.save({"name": "Style test", "dataset": "/ds", "snapshot": "snap1", "caption_set": "short",
                 "caption_set_settings": '{"strategy": "Short prose"}', "trainer": "kohya", "trainer_settings": "rank 16",
                 "lora_paths": "/l/a.safetensors", "validation_set": "vs", "validation_version": "3", "run_dir": "/runs/r",
                 "metrics": [["c", "held-out", 2, "0.9", "", "—", "0.1"]], "votes": [["A", 1.0, 2, 0, 0, "100%"]],
                 "notes": "Looks good"}, path=db)
    md = ex.report_markdown(ex.get(i, db), ["/runs/r/grid_p00.png"], "/runs/r")
    for part in ("# Style test", "## Setup", "(caption snapshot `snap1`)", "### How the caption set was made",
                 "### Trainer settings", "rank 16", "## Metrics (experimental)", "| c | held-out |", "## Blind votes",
                 "| A | 1.0 |", "## Notes", "Looks good", "## Grids", "![grid_p00.png](grid_p00.png)", "vs v3", "a.safetensors"):
        assert part in md, part
    empty = ex.report_markdown(ex.get(ex.save({"name": "E"}, path=db), db))
    assert "_none_" in empty and "## Notes" not in empty and "## Grids" not in empty


def test_markdown_to_html():
    md = ("# Title <b>\n\nText with **bold**, `code` and _emphasis_ and grid_p00.png and a_b_c.\n\n- one\n- two\n\n"
          "| H1 | H2 |\n|---|---|\n| a \\| b | <i>x</i> |\n\n```\n<raw> & code\n```\n![g](grid_p01.png)\n")
    html_text = ex.markdown_to_html(md, "T & <x>")
    assert "<title>T &amp; &lt;x&gt;</title>" in html_text
    assert "<h1>Title &lt;b&gt;</h1>" in html_text
    assert "<strong>bold</strong>" in html_text and "<code>code</code>" in html_text and "<em>emphasis</em>" in html_text
    assert "grid_p00.png" in html_text and "a_b_c" in html_text and "<em>p00" not in html_text and "<em>b" not in html_text
    assert "<ul>" in html_text and "<li>one</li>" in html_text and "</ul>" in html_text
    assert "<th>H1</th>" in html_text and "<td>a | b</td>" in html_text and "<td>&lt;i&gt;x&lt;/i&gt;</td>" in html_text
    assert "<pre>" in html_text and "&lt;raw&gt; &amp; code" in html_text
    assert '<img alt="g" src="grid_p01.png"' in html_text
    assert "---" not in html_text.split("<table>")[1].split("</table>")[0]


def test_export_report_into_run_dir(tmp_path, db):
    run = tmp_path / "run"
    run.mkdir()
    Image.new("RGB", (4, 4)).save(run / "grid_p00.png")
    i = ex.save({"name": "R", "run_dir": str(run)}, path=db)
    md_path, html_path = ex.export_report(i, db)
    assert os.path.dirname(md_path) == str(run) and os.path.exists(html_path)
    assert "![grid_p00.png](grid_p00.png)" in open(md_path, encoding="utf-8").read()


def test_export_report_without_run_dir(app_root, db):
    i = ex.save({"name": "R", "run_dir": "/does/not/exist"}, path=db)
    md_path, _ = ex.export_report(i, db)
    assert os.path.dirname(md_path) == str(app_root / "experiments" / "reports")
    assert os.path.basename(md_path) == f"experiment_{i}_report.md"


# =====================================================================
# lab_ui: caption sets
# =====================================================================

@pytest.fixture
def ds(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    for n in ("a.png", "b.png", "c.png"):
        Image.new("RGB", (16, 16), "white").save(d / n)
    paths = {n: str(d / n) for n in ("a.png", "b.png", "c.png")}
    for n, t in (("a", "txcl, a cat"), ("b", "txcl, a dog")):
        (d / f"{n}.txt").write_text(t, encoding="utf-8")
    return d, paths, {"a.png": "txcl, a cat", "b.png": "txcl, a dog"}


def test_set_choices(ds):
    d, paths, caps = ds
    upd = lab_ui.set_choices(paths)
    assert upd["choices"] == [cs.ACTIVE] and upd["value"] == cs.ACTIVE
    cs.write_set(paths, "s", {"a.png": "x"}, {})
    assert lab_ui.set_choices(paths, "s")["value"] == "s"
    assert lab_ui.set_choices(paths, "unknown")["value"] == cs.ACTIVE
    assert lab_ui.set_choices({})["choices"] == [cs.ACTIVE]


def test_save_activate_delete_callbacks(ds):
    d, paths, caps = ds
    status, upd = lab_ui.save_active_as_set(paths, caps, "Backup", "desc")
    assert "Saved 2 caption(s) as set 'backup'" in status and upd["value"] == "backup"
    status, upd = lab_ui.save_active_as_set(paths, caps, "bad name", "")
    assert status.startswith("⚠️")
    assert lab_ui.save_active_as_set({}, caps, "x", "")[0] == "No images loaded."
    cs.write_set(paths, "alt", {"a.png": "alt a"}, {})
    status, new_caps = lab_ui.activate_set(paths, caps, "alt")
    assert "is now the active captions (1 files)" in status and "2 image(s) have no caption" in status
    assert new_caps["a.png"] == "alt a" and new_caps["b.png"] == "txcl, a dog"
    assert snapshots.list_snapshots(str(d))[0]["operation"] == "before activating set alt"
    assert lab_ui.activate_set(paths, caps, cs.ACTIVE)[0] == "Pick a caption set first."
    assert lab_ui.activate_set(paths, caps, "empty")[0].startswith("⚠️")
    status, upd = lab_ui.delete_set(paths, "alt")
    assert "nothing deleted" in status and "alt" not in upd["choices"]
    assert lab_ui.delete_set(paths, cs.ACTIVE)[0] == "Pick a caption set first."


def test_compare_histogram_materialize(ds):
    d, paths, caps = ds
    cs.write_set(paths, "s", {"a.png": "txcl, one\ntwo"}, {"strategy": "Short prose (~15 words)", "model": "m", "profile": "flux1"})
    rows, md = lab_ui.compare_sets(paths, "flux1", "txcl")
    assert [r[0] for r in rows] == [cs.ACTIVE, "s"] and rows[0][1] == 2 and rows[1][1] == 1
    assert "**1 caption set(s).**" in md and "- **s**: Short prose (~15 words) · m · flux1" in md
    assert lab_ui.compare_sets({}, "flux1", "")[1] == "No images loaded."
    assert sum(r[1] for r in lab_ui.set_histogram(paths, "s")) == 1 and lab_ui.set_histogram({}, "s") == []
    status = lab_ui.materialize_set(paths, "s", cs.VARIANT_MODES["kohya_multiline"])
    assert "Wrote 1 caption file(s)" in status and "enable_multiline_caption" in status
    assert "---===---" in open(d / ".artagents" / "export" / "s-kohya_multiline" / "a.txt").read()
    assert lab_ui.materialize_set(paths, "missing", cs.VARIANT_MODES["lines"]).startswith("⚠️")
    assert lab_ui.materialize_set({}, "s", "x") == "No images loaded."


# ---------- caption sweep ----------

SOURCE_ARGS = ("Vision model", "Ollama", "", "", "", None, 0.35, 0.85, False, False, 0.5, False, False, "", True, [], "")


def _sweep(paths, caps, labels, prefix="sweep", seed=42):
    calls = []

    def fake_generate(name, agent, model, mode, image_paths, captions, settings, *rest, caption_instruction=None,
                      caption_prefix="", caption_suffix="", caption_config=None):
        calls.append({"name": name, "mode": mode, "cfg": dict(caption_config), "instr": caption_instruction,
                      "captions_id": id(captions)})
        ext = caption_config["extension"]
        text = f"{caption_prefix}{ext.split('.')[1]} caption for {name}"
        caption_io.write_caption(image_paths[name], text, ext)
        return "Status: Success.", captions, text, rest[-1]

    with patch("core.captioning_logic.generate_captions_for_selected", side_effect=fake_generate):
        events = list(lab_ui.iter_caption_sweep(
            labels, prefix, paths, caps, "(Direct Agent Call)", "vm (VISION)", {}, [], {}, {}, {}, [], [],
            "Describe.", "txcl, ", "", "flux1", "general", profiles.STYLE_LABELS["prose"], "space  (   )", "", 0.0, seed,
            1024, 1, [], "", "", *SOURCE_ARGS))
    return events, calls


def test_caption_sweep_model_and_derived_sets(ds):
    d, paths, caps = ds
    before = {f: open(d / f, encoding="utf-8").read() for f in ("a.txt", "b.txt")}
    labels = [cs.STRATEGIES["short_prose"][0], cs.STRATEGIES["random_length"][0], cs.STRATEGIES["trigger_only"][0]]
    events, calls = _sweep(paths, caps, labels)
    final = events[-1][0]
    assert "Caption sweep finished. Active captions were not changed." in final
    # random_length needs long prose too: it was added
    exts = sorted({c["cfg"]["extension"] for c in calls})
    assert exts == [".sweep-long-prose.txt", ".sweep-short-prose.txt"] and len(calls) == 6
    assert all(c["mode"] == "Overwrite" for c in calls)
    short = [c for c in calls if "short" in c["cfg"]["extension"]][0]
    long_ = [c for c in calls if "long" in c["cfg"]["extension"]][0]
    assert short["cfg"]["style"] == "prose" and short["cfg"]["source"] == "vlm"
    assert long_["cfg"]["max_output_tokens"] >= 240 and short["instr"] != long_["instr"]
    assert set(cs.discover_sets(paths)) == {"sweep-short-prose", "sweep-long-prose", "sweep-random-length", "sweep-trigger-only"}
    assert cs.read_set(paths, "sweep-trigger-only") == {n: "txcl" for n in paths}
    assert set(cs.read_set(paths, "sweep-random-length")) == set(paths)
    assert {f: open(d / f, encoding="utf-8").read() for f in ("a.txt", "b.txt")} == before
    manifest = cs.load_manifest(str(d))["sets"]
    assert manifest["sweep-short-prose"]["settings"]["strategy"] == cs.STRATEGIES["short_prose"][0]
    assert manifest["sweep-short-prose"]["settings"]["model"] == "vm" and manifest["sweep-short-prose"]["settings"]["profile"] == "flux1"
    assert manifest["sweep-random-length"]["settings"]["derived_from"] == ["long_prose", "short_prose"]
    assert [r[0] for r in events[-1][1]][0] == cs.ACTIVE
    assert len(events) == 1 + 6                                       # One progress event per image + final


def test_caption_sweep_tags_and_structured(ds):
    d, paths, caps = ds
    events, calls = _sweep(paths, caps, [cs.STRATEGIES["tags"][0], cs.STRATEGIES["structured"][0]], prefix="exp")
    tags = [c for c in calls if c["cfg"]["extension"] == ".exp-tags.txt"]
    structured_calls = [c for c in calls if c["cfg"]["extension"] == ".exp-structured.txt"]
    assert len(tags) == 3 and tags[0]["cfg"]["style"] == "tags" and tags[0]["cfg"]["joiner"] == ", "
    assert tags[0]["cfg"]["source"] == "vlm"
    assert structured_calls[0]["cfg"]["source"] == "structured" and structured_calls[0]["instr"] == "Describe."


def test_caption_sweep_blend_adds_a_model_set(ds):
    d, paths, caps = ds
    events, calls = _sweep(paths, caps, [cs.STRATEGIES["blend_95_5"][0]])
    assert {c["cfg"]["extension"] for c in calls} == {".sweep-short-prose.txt"}
    assert "sweep-blend-95-5" in cs.discover_sets(paths)


def test_caption_sweep_derived_error_reported(ds):
    d, paths, caps = ds
    events, _ = _sweep(paths, caps, [cs.STRATEGIES["quality_on"][0]])  # FLUX has no quality tags
    assert "sweep-quality-on: ⚠️" in events[-1][0]


@pytest.mark.parametrize("labels, prefix, msg", [([], "sweep", "Pick at least one strategy"),
                                                 (["unknown"], "sweep", "Pick at least one strategy"),
                                                 ([cs.STRATEGIES["short_prose"][0]], "bad prefix", "⚠️")])
def test_caption_sweep_input_errors(ds, labels, prefix, msg):
    d, paths, caps = ds
    events, calls = _sweep(paths, caps, labels, prefix=prefix)
    assert msg in events[-1][0] and not calls
    assert "No images loaded." in list(lab_ui.iter_caption_sweep([], "", {}, {}, *([None] * 24), *SOURCE_ARGS))[0][0]


# =====================================================================
# lab_ui: validation prompts
# =====================================================================

def test_fill_default_prompts(ds):
    d, paths, caps = ds
    rows, note = lab_ui.fill_default_prompts(str(d), "txcl", "cat", False)
    assert "2 caption(s) read" in note
    assert [r for r in rows if r[0] == "training-like"][0][1] == "[trigger], a cat"
    rows, note = lab_ui.fill_default_prompts("/nope", "", "", True)
    assert "No dataset captions" in note and not [r for r in rows if r[0] == "training-like"]


def test_save_load_export_validation(app_root):
    rows = [["held-out", "[trigger] cat", "", ""], ["unrelated", "a car", "", "--s 10"]]
    status, name_upd, ver_upd = lab_ui.save_validation(" my set ", rows, "1, 2", "txcl", 832, 1216, 28, 4.5, "blurry")
    assert status == "Saved my set v1." and name_upd["value"] == "my set" and ver_upd["value"] == "1"
    data = validation.load("my set")
    assert data["defaults"] == {"width": 832, "height": 1216, "steps": 28, "cfg": 4.5, "negative": "blurry"}
    status, *_ = lab_ui.save_validation("x", rows, "", "", 1024, 1024, 0, 0, "")
    assert status.startswith("⚠️")
    loaded = lab_ui.load_validation("my set", "1")
    assert loaded[0] == validation.to_rows(data) and loaded[1] == "1, 2" and loaded[2] == "txcl"
    assert loaded[3:7] == (832, 1216, 28, 4.5) and "Loaded my set v1" in loaded[7]
    assert lab_ui.load_validation("", None)[-1] == "Pick a set."
    assert lab_ui.load_validation("missing", None)[-1].startswith("⚠️")
    assert lab_ui.validation_versions("my set")["choices"] == ["1"]
    assert lab_ui.validation_names()["choices"] == ["my set"]
    status, code, dl = lab_ui.export_kohya_prompts("my set", "1")
    assert status.startswith("Wrote ") and str(app_root) in status
    assert "txcl cat --n blurry --w 832 --h 1216 --l 4.5 --s 28 --d 1" in code["value"]  # Set-wide negative included
    assert os.path.exists(dl) and open(dl, encoding="utf-8").read() == code["value"]
    assert lab_ui.export_kohya_prompts("missing", None)[0].startswith("⚠️")


def test_kohya_export_includes_default_negative(app_root):
    lab_ui.save_validation("neg", [["held-out", "[trigger] cat", "", ""]], "1", "txcl", 512, 512, 0, 0, "blurry")
    _, code, _ = lab_ui.export_kohya_prompts("neg", "1")
    assert "--n blurry" in code["value"]


def test_count_images():
    rows = [["held-out", "a", "", ""], ["unrelated", "b", "", ""], ["held-out", "", "", ""]]
    text = lab_ui.count_images(rows, "1, 2", ["x", "y"], "0.5, 1", True, ["held-out"])
    assert text.startswith("**10 image(s)** = 1 prompt(s) × 2 seed(s)")
    assert lab_ui.count_images(rows, "1", [], "1", False, []).startswith("**0 image(s)**")
    assert lab_ui.count_images(rows, "x", [], "1", True, []) == "Check the seeds and strengths."


def test_draft_prompts(tmp_path):
    (tmp_path / "a.txt").write_text("txcl cat", encoding="utf-8")
    with patch("core.dataset.vision.ask", return_value="1. [trigger] cat in space\nno trigger") as ask:
        rows, status = lab_ui.draft_prompts([["held-out", "x", "", ""], ["", "", "", ""]], str(tmp_path), "cat", False, "m", 4, {})
    assert rows == [["held-out", "x", "", ""], ["held-out", "[trigger] cat in space", "", ""]]
    assert "Added 1 drafted" in status and "txcl cat" in ask.call_args[0][2]
    assert lab_ui.draft_prompts([], "", "", False, None, 3, {})[1] == "Pick a model to draft prompts with."
    with patch("core.dataset.vision.ask", return_value="nothing useful"):
        assert "no usable prompts" in lab_ui.draft_prompts([], "", "", False, "m", 3, {})[1]


# =====================================================================
# lab_ui: test sweep, voting, experiments
# =====================================================================

class FakeBackend:
    def __init__(self, ok=True):
        self.ok, self.requests, self.interrupted = ok, [], False

    def check(self):
        return self.ok, "ok" if self.ok else "down"

    def list_models(self):
        return ["base.safetensors"]

    def list_loras(self):
        return ["e04.safetensors", "e08.safetensors"]

    def interrupt(self):
        self.interrupted = True

    def generate_iter(self, req):
        self.requests.append(req)
        yield GenProgress(None, "x")
        yield GenResult([GeneratedImage(Image.new("RGB", (32, 32), (len(self.requests) * 10 % 255, 0, 0)), req.seed)])


@pytest.fixture
def lab(app_root, tmp_path):
    settings = {"image_generation": {"output_dir": str(tmp_path / "outputs" / "images")}}
    validation.save_new_version("vs", [["held-out", "[trigger] cat", "", ""], ["unrelated", "a car", "", ""]],
                                "1, 2", "txcl", {"width": 512, "height": 512})
    return settings


def test_runs_root(tmp_path):
    settings = {"image_generation": {"output_dir": str(tmp_path / "out" / "images")}}
    assert lab_ui.runs_root(settings) == str(tmp_path / "out" / "lora_tests")


def test_refresh_lab_backend():
    backend = FakeBackend()
    with patch("core.lab_ui.get_backend", return_value=backend):
        models, loras, workflow, status = lab_ui.refresh_lab_backend("ComfyUI", {})
    assert models["value"] == "base.safetensors" and loras["choices"] == ["e04.safetensors", "e08.safetensors"]
    assert loras["value"] == [] and workflow["visible"] and workflow["value"] == "lora_txt2img.json"
    assert status.startswith("✅") and "2 LoRA(s) found" in status
    with patch("core.lab_ui.get_backend", return_value=FakeBackend(ok=False)):
        models, loras, workflow, status = lab_ui.refresh_lab_backend("Forge / A1111 / SD.Next", {})
    assert status.startswith("⚠️") and models["choices"] == [] and not workflow["visible"]
    with patch("core.lab_ui.get_backend", side_effect=ImageGenError("unreachable")):
        assert "unreachable" in lab_ui.refresh_lab_backend("ComfyUI", {})[3]


def _sweep_lab(settings, backend_label="ComfyUI", loras=("e04.safetensors",), extra="", baseline=True, vname="vs",
               strengths="0.5, 1", backend=None):
    backend = backend or FakeBackend()
    with patch("core.lab_ui.get_backend", return_value=backend):
        events = list(lab_ui.iter_lora_sweep(settings, backend_label, "base.safetensors", "lora_txt2img.json",
                                             list(loras), extra, strengths, baseline, vname, "1", None, 0, "my run"))
    return events, backend


def test_iter_lora_sweep_comfyui(lab):
    events, backend = _sweep_lab(lab, extra="other.safetensors\n")
    final = events[-1]
    assert final[0].startswith("Finished: 20 job(s), 0 error(s).")      # 2 prompts × 2 seeds × (1 + 2 LoRAs × 2)
    root = lab_ui.runs_root(lab)
    runs = ls.list_runs(root)
    assert len(runs) == 1 and runs[0].endswith("-my-run") and final[2]["value"] == runs[0] and final[3]["value"] == runs[0]
    assert len(final[1]["value"]) == 2 and all(isinstance(g[0], Image.Image) for g in final[1]["value"])
    data = ls.load_run(os.path.join(root, runs[0]))
    assert data["validation_set"] == "vs" and data["loras"] == ["e04.safetensors", "other.safetensors"]
    assert backend.requests[0].loras == [("e04.safetensors", 0.0)] and backend.requests[0].workflow == "lora_txt2img.json"
    assert backend.requests[0].prompt == "txcl cat" and backend.requests[0].width == 512
    assert len(events) == 21


def test_iter_lora_sweep_forge_baseline_only(lab):
    events, backend = _sweep_lab(lab, "Forge / A1111 / SD.Next", loras=(), strengths="")
    assert events[-1][0].startswith("Finished: 4 job(s)")
    assert all(r.loras == [] and r.workflow is None for r in backend.requests)


@pytest.mark.parametrize("kwargs, msg", [
    ({"vname": ""}, "Pick a validation prompt set."),
    ({"vname": "missing"}, "⚠️"),
    ({"strengths": "strong"}, "⚠️"),
    ({"loras": (), "baseline": False}, "Pick at least one LoRA"),
    ({"loras": ()}, "ComfyUI needs at least one LoRA"),
])
def test_iter_lora_sweep_input_errors(lab, kwargs, msg):
    events, backend = _sweep_lab(lab, **kwargs)
    assert msg in events[-1][0] and not backend.requests


def test_run_choices_show_run_and_copy(lab, tmp_path):
    _sweep_lab(lab)
    a, b = lab_ui.run_choices(lab)
    assert a["choices"] == b["choices"] and len(a["choices"]) == 1
    gallery, info = lab_ui.show_run(lab, a["value"])
    assert len(gallery["value"]) == 2 and "12/12 images" in info and "vs v1" in info and "ComfyUI" in info
    assert lab_ui.show_run(lab, None)[1] == ""
    src = tmp_path / "x.safetensors"
    src.write_bytes(b"1")
    (tmp_path / "comfy" / "models" / "loras").mkdir(parents=True)
    assert lab_ui.copy_lora(f'"{src}"', str(tmp_path / "comfy")).startswith("Copied to")
    assert lab_ui.copy_lora(str(src), str(tmp_path / "comfy")).startswith("⚠️")


def test_stop_lab_backend():
    backend = FakeBackend()
    with patch("core.lab_ui.get_backend", return_value=backend):
        lab_ui.stop_lab_backend("ComfyUI", {})
    assert backend.interrupted
    with patch("core.lab_ui.get_backend", side_effect=RuntimeError("x")):
        lab_ui.stop_lab_backend("ComfyUI", {})                   # Never raises


def test_voting_flow(lab):
    _sweep_lab(lab, strengths="1")
    run = ls.list_runs(lab_ui.runs_root(lab))[0]
    pairs, index, left, right, prompt, status = lab_ui.start_voting(lab, run, True)
    assert len(pairs) == 4 and index == 0 and isinstance(left, Image.Image) and "Pair 1 of 4" in status
    assert prompt.startswith("**Prompt:**")
    i = 0
    for choice in ("left", "right", "tie", "left"):
        i, left, right, prompt, status, table = lab_ui.cast_vote(choice, pairs, i, lab, run)
    assert i == 4 and left is None and "All 4 pairs voted" in status
    assert sum(r[2] + r[3] for r in table) == 6 and sum(r[4] for r in table) == 2
    # Everything voted: nothing left
    assert "No pairs left" in lab_ui.start_voting(lab, run, True)[-1]
    assert lab_ui.cast_vote("left", [], 0, lab, run)[4] == "Start voting first."
    table, agreement = lab_ui.vote_results(lab, run)
    assert len(table) == 2 and "No pairs voted by both" in agreement
    assert lab_ui.start_voting(lab, None, True)[-1] == "Pick a test run."


def test_iter_judge(lab):
    _sweep_lab(lab, strengths="1")
    run = ls.list_runs(lab_ui.runs_root(lab))[0]
    with patch("core.dataset.vision.ask", return_value='{"better": "first"}'):
        events = list(lab_ui.iter_judge(lab, run, "m (VISION)", "", 3))
    assert "0 vote(s) counted, 3 pair(s) skipped as position-biased" in events[-1][0]
    assert lab_ui.iter_judge(lab, None, "m", "", 3).__next__()[0] == "Pick a test run and a judge model."


def test_score_run_callback_without_embedder(lab):
    _sweep_lab(lab, strengths="1")
    run = ls.list_runs(lab_ui.runs_root(lab))[0]
    rows, note = lab_ui.score_run(lab, run, "/nowhere", None)
    assert rows == [] and "pip install" in note
    assert lab_ui.score_run(lab, None, "", "x")[1] == "Pick a test run."


def test_experiment_callbacks(lab, ds):
    d, paths, caps = ds
    _sweep_lab(lab, strengths="1")
    run = ls.list_runs(lab_ui.runs_root(lab))[0]
    run_dir = os.path.join(lab_ui.runs_root(lab), run)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({"rows": [["c", "held-out", 2, "0.9", "", "—", "0.1"]]}, f)
    pairs = voting.make_pairs(ls.load_run(run_dir), run_dir)
    voting.record_vote(run_dir, pairs[0], "left")
    run_dir_v, loras, vset, vver = lab_ui.prefill_from_run(lab, run)
    assert run_dir_v == run_dir and loras == "e04.safetensors" and vset == "vs" and vver == "1"
    cs.write_set(paths, "s", {"a.png": "x"}, {"strategy": "Short"})
    snapshots.create_snapshot(str(d), paths, "test")
    snap, settings_text = lab_ui.prefill_from_dataset(str(d), "s")
    assert snap == snapshots.list_snapshots(str(d))[0]["id"] and json.loads(settings_text) == {"strategy": "Short"}
    assert lab_ui.prefill_from_dataset(str(d), cs.ACTIVE)[1] == ""
    values = ["Exp 1", str(d), snap, "s", settings_text, "kohya", "rank 8", loras, vset, vver, run_dir, "notes"]
    status, exp_id, table = lab_ui.save_experiment(None, lab, *values)
    assert status == f"Saved experiment #{exp_id}." and table[0][0] == exp_id
    rec = ex.get(exp_id)
    assert json.loads(rec["metrics"])[0][0] == "c" and json.loads(rec["votes"])[0][2] == 1
    selected = lab_ui.select_experiment(SimpleNamespace(row_value=[exp_id, "Exp 1"]))
    assert selected[0] == exp_id and selected[1:] == values
    assert lab_ui.select_experiment(SimpleNamespace(row_value=None))[0] is None
    assert lab_ui.new_experiment() == [None] + [""] * len(lab_ui.EXP_INPUT_KEYS)
    status, files, md = lab_ui.export_experiment(exp_id)
    assert status.startswith("Wrote ") and len(files) == 2 and all(os.path.exists(f) for f in files)
    assert "# Exp 1" in md and "![" not in md and "_(in the exported files)_" in md
    md_file = next(f for f in files if f.endswith(".md"))
    assert "![grid_p00.png](grid_p00.png)" in open(md_file, encoding="utf-8").read()
    assert lab_ui.export_experiment(None)[0] == "Select or save an experiment first."
    assert lab_ui.save_experiment(None, lab, *([""] * 12))[0].startswith("⚠️")


def test_updating_experiment_keeps_metrics_when_run_dir_is_gone(app_root, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps({"rows": [["c", "held-out"]]}), encoding="utf-8")
    values = ["exp", "", "", "", "", "", "", "", "", "", str(run), "notes"]
    _, exp_id, _ = lab_ui.save_experiment(None, {}, *values)
    shutil.rmtree(run)
    values[-1] = "edited notes"
    lab_ui.save_experiment(exp_id, {}, *values)
    assert json.loads(ex.get(exp_id)["metrics"] or "[]") == [["c", "held-out"]]


def test_experiments_table_and_embedder_choices(app_root):
    assert lab_ui.experiments_table() == []
    with patch("core.lab.metrics.importlib.util.find_spec", return_value=None):
        upd, note = lab_ui.embedder_choices()
    assert upd["choices"] == [] and note.startswith("⚠️")
    assert lab_ui.lab_backend_labels() == ["ComfyUI", "Forge / A1111 / SD.Next"]
