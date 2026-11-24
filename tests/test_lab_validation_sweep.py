# ArtAgent/tests/test_lab_validation_sweep.py
"""Tests for core/lab/validation.py and core/lab/lora_sweep.py."""
import os
import sys
from unittest.mock import patch

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.image_gen.base import GeneratedImage, GenProgress, GenResult, ImageGenError
from core.lab import lora_sweep as ls
from core.lab import validation as v


@pytest.fixture
def sets_dir(tmp_path):
    root = tmp_path / "app"
    root.mkdir()
    with patch("core.lab.validation.get_absolute_path", side_effect=lambda rel: str(root / rel)):
        yield root / v.SETS_DIR


ROWS = [["training-like", "[trigger], a cat", "", ""], ["held-out", "[trigger] on a beach", "blurry", "--w 832"],
        ["", "  ", "", ""], ["weird", "no bucket prompt", None, None]]


# ---------- storage ----------

def test_save_versions_increment_and_never_overwrite(sets_dir):
    assert v.list_sets() == []
    v1 = v.save_new_version("my set", ROWS, "1, 2 3", "txcl", {"width": 1024})
    v2 = v.save_new_version("my set", ROWS[:1], "5", "txcl", {})
    assert (v1, v2) == (1, 2)
    assert v.list_sets() == ["my set"] and v.list_versions("my set") == [1, 2]
    first = v.load("my set", 1)
    assert first["seeds"] == [1, 2, 3] and first["trigger"] == "txcl" and first["defaults"] == {"width": 1024}
    assert [p["bucket"] for p in first["prompts"]] == ["training-like", "held-out", "held-out"]
    assert v.load("my set")["version"] == 2                                   # Latest by default
    assert v.load("my set", "1")["version"] == 1
    # The 'x' mode refuses to overwrite an existing version file
    with pytest.raises(FileExistsError):
        with patch("core.lab.validation.list_versions", return_value=[1]):
            v.save_new_version("my set", ROWS, "1", "", {})


def test_list_ignores_other_files(sets_dir):
    v.save_new_version("a", ROWS, "1", "", {})
    (sets_dir / "a.v1.sample_prompts.txt").write_text("x", encoding="utf-8")
    (sets_dir / "notes.json").write_text("{}", encoding="utf-8")
    assert v.list_sets() == ["a"] and v.list_versions("a") == [1] and v.list_versions("b") == []


@pytest.mark.parametrize("name, rows, seeds, msg", [
    ("", ROWS, "1", "name"), ("bad/name", ROWS, "1", "name"), (".hidden", ROWS, "1", "name"),
    ("ok", [["held-out", "", "", ""]], "1", "no prompts"), ("ok", ROWS, " ", "seed"), ("ok", ROWS, "1, x", "Not a seed"),
])
def test_save_errors(sets_dir, name, rows, seeds, msg):
    with pytest.raises(v.ValidationError, match=msg):
        v.save_new_version(name, rows, seeds, "", {})


def test_load_missing(sets_dir):
    with pytest.raises(v.ValidationError):
        v.load("nope")


def test_clean_rows_and_to_rows_round_trip():
    prompts = v.clean_rows(ROWS + [["unrelated", "a car"]])
    assert prompts[-1] == {"bucket": "unrelated", "prompt": "a car", "negative": "", "flags": ""}
    assert prompts[2]["bucket"] == "held-out"
    assert v.clean_rows(v.to_rows({"prompts": prompts})) == prompts
    assert v.clean_rows(None) == []


def test_parse_seeds():
    assert v.parse_seeds("1, 2,3 4\n5") == [1, 2, 3, 4, 5]
    assert v.parse_seeds("") == [] and v.parse_seeds(None) == []
    assert v.parse_seeds("-1") == [-1]


@pytest.mark.parametrize("prompt, trigger, expected", [
    ("[trigger], a cat", "txcl", "txcl, a cat"),
    ("[trigger], a cat", "", "a cat"),
    ("a cat, [trigger] style", "", "a cat, style"),
    ("[trigger]  painting  of [trigger]", "zz", "zz painting of zz"),
    ("a cat", "txcl", "a cat"),
])
def test_render(prompt, trigger, expected):
    assert v.render(prompt, trigger) == expected


# ---------- default prompts ----------

def test_default_rows_subject():
    caps = {"1.png": "txcl woman, red dress\nvariant", "2.png": "a woman in a park", "3.png": "  "}
    rows = v.default_rows("txcl", "woman", caps, per_bucket=3)
    by = {}
    for r in rows:
        by.setdefault(r[0], []).append(r[1])
    assert by["training-like"] == ["[trigger], a woman in a park", "[trigger] woman, red dress"]
    assert by["held-out"] == [t.format(subject="[trigger] woman") for t in v.HELD_OUT_TEMPLATES[:3]]
    assert by["trigger-less"] == [t.format(subject="woman") for t in v.HELD_OUT_TEMPLATES[:3]]
    assert by["unrelated"] == v.UNRELATED[:3]
    # Each trigger-less prompt is its held-out prompt minus the trigger
    assert [h.replace("[trigger] ", "") for h in by["held-out"]] == by["trigger-less"]


def test_default_rows_without_class_or_captions():
    rows = v.default_rows("", "", None, per_bucket=2)
    assert not [r for r in rows if r[0] == "training-like"]
    assert [r[1] for r in rows if r[0] == "held-out"][0] == "[trigger] in a snowy forest"
    assert [r[1] for r in rows if r[0] == "trigger-less"][0] == "a person in a snowy forest"


def test_default_rows_style_lora():
    rows = v.default_rows("txcl", "painting", {}, style_lora=True, per_bucket=2)
    held = [r[1] for r in rows if r[0] == "held-out"]
    less = [r[1] for r in rows if r[0] == "trigger-less"]
    assert held == ["a lighthouse on a cliff, [trigger] painting style", "a portrait of an old fisherman, [trigger] painting style"]
    assert less == ["a lighthouse on a cliff, painting style", "a portrait of an old fisherman, painting style"]
    no_class = v.default_rows("txcl", "", {}, style_lora=True, per_bucket=1)
    assert [r[1] for r in no_class if r[0] == "trigger-less"] == ["a lighthouse on a cliff"]
    assert [r[1] for r in no_class if r[0] == "held-out"] == ["a lighthouse on a cliff, [trigger] style"]


def test_default_rows_trigger_case_insensitive_once():
    rows = v.default_rows("TXCL", "", {"1": "txcl cat, txcl"}, per_bucket=1)
    assert rows[0][1] == "[trigger] cat, txcl"


def test_default_rows_trigger_whole_word_only():
    rows = v.default_rows("art", "", {"1.png": "a smart cat"}, per_bucket=1)
    assert rows[0][1] == "[trigger], a smart cat"


# ---------- drafting ----------

def test_draft_prompt_and_parse():
    text = v.draft_prompt({"1": "txcl cat on a sofa\nvariant", "2": "  "}, "cat", 5, False)
    assert "subject (class: cat)" in text and "- txcl cat on a sofa" in text and "Write 5 new prompts" in text
    assert "style" in v.draft_prompt({}, "", 3, True)
    raw = "Here are prompts:\n1. [trigger] cat in space\n- \"[trigger] cat underwater\"\nno trigger here\n2) [trigger] cat in space\n* [trigger] cat at night"
    assert v.parse_draft(raw, 10) == ["[trigger] cat in space", "[trigger] cat underwater", "[trigger] cat at night"]
    assert v.parse_draft(raw, 1) == ["[trigger] cat in space"]
    assert v.parse_draft("", 3) == []


# ---------- flags & sample prompts ----------

def test_flags_for_and_parse_flags_round_trip():
    defaults = {"width": 832, "height": 1216, "cfg": 4.5, "steps": 28}
    flags = v.flags_for(defaults, 7, "blurry", "--x 1")
    assert flags == "--n blurry --w 832 --h 1216 --l 4.5 --s 28 --d 7 --x 1"
    assert v.parse_flags(flags) == {"width": 832, "height": 1216, "cfg": 4.5, "steps": 28, "seed": 7}
    assert v.flags_for({}, None) == ""
    assert v.parse_flags("--w abc --h 512 --q 3") == {"height": 512}
    assert v.parse_flags("") == {} and v.parse_flags(None) == {}


def test_kohya_sample_prompts():
    data = {"name": "s", "version": 2, "trigger": "txcl", "seeds": [1, 2], "defaults": {"width": 512, "height": 512},
            "prompts": [{"bucket": "held-out", "prompt": "[trigger] cat", "negative": "ugly", "flags": ""},
                        {"bucket": "unrelated", "prompt": "a car", "negative": "", "flags": "--s 10"}]}
    text = v.kohya_sample_prompts(data)
    lines = text.strip().splitlines()
    assert lines[0].startswith("# s v2") and "txcl" in lines[0]
    assert lines[1:] == ["txcl cat --n ugly --w 512 --h 512 --d 1", "txcl cat --n ugly --w 512 --h 512 --d 2",
                         "a car --w 512 --h 512 --d 1 --s 10", "a car --w 512 --h 512 --d 2 --s 10"]
    only = v.kohya_sample_prompts(data, ["unrelated"]).strip().splitlines()
    assert len(only) == 3 and all("a car" in l for l in only[1:])


def test_kohya_sample_prompts_respects_prompt_seed():
    data = {"name": "s", "version": 1, "trigger": "t", "seeds": [1, 2], "defaults": {},
            "prompts": [{"bucket": "held-out", "prompt": "[trigger] cat", "negative": "", "flags": "--d 7"}]}
    lines = v.kohya_sample_prompts(data).strip().splitlines()[1:]
    assert lines == ["t cat --d 7"]


def test_image_count():
    assert v.image_count(10, 4, 2, 3) == 10 * 4 * 7
    assert v.image_count(10, 4, 2, 3, baseline=False) == 240
    assert v.image_count(5, 2, 0, 3) == 10
    assert v.image_count(0, 4, 2, 3) == 0


# ---------- LoRA sweep: jobs ----------

VSET = {"trigger": "txcl", "seeds": [1, 2], "defaults": {"width": 768, "height": 512, "steps": 20, "cfg": 5.0, "negative": "bad"},
        "prompts": [{"bucket": "held-out", "prompt": "[trigger] cat", "negative": "", "flags": ""},
                    {"bucket": "unrelated", "prompt": "a car", "negative": "rusty", "flags": "--d 9 --w 1024 --s 30 --l 7"},
                    {"bucket": "trigger-less", "prompt": "cat", "negative": "", "flags": ""}]}


def test_parse_strengths():
    assert ls.parse_strengths("0.6, 0.8 1") == [0.6, 0.8, 1.0]
    assert ls.parse_strengths("") == []
    with pytest.raises(ValueError, match="Not a strength"):
        ls.parse_strengths("0.5, strong")


def test_lora_label():
    assert ls.lora_label("/x/style_e04.safetensors") == "style_e04"
    assert ls.lora_label("style.v2.PT") == "style.v2"
    assert ls.lora_label("plain") == "plain"


def test_build_jobs_order_and_values():
    jobs = ls.build_jobs(VSET, ["a.safetensors", "b.safetensors"], [0.5, 1.0])
    assert [j.index for j in jobs] == list(range(len(jobs)))
    # prompt 0: 2 seeds × (baseline + 4) ; prompt 1: its own seed 9 × 5 ; prompt 2: 2 × 5
    assert len(jobs) == 10 + 5 + 10
    first = jobs[:5]
    assert first[0].lora is None and first[0].condition == ls.BASELINE and first[0].seed == 1
    assert [(j.lora, j.strength) for j in first[1:]] == [("a.safetensors", 0.5), ("a.safetensors", 1.0),
                                                         ("b.safetensors", 0.5), ("b.safetensors", 1.0)]
    assert first[0].prompt == "txcl cat" and first[0].negative == "bad"
    assert (first[0].width, first[0].height, first[0].steps, first[0].cfg) == (768, 512, 20, 5.0)
    car = [j for j in jobs if j.prompt_index == 1]
    assert {j.seed for j in car} == {9} and car[0].negative == "rusty"
    assert (car[0].width, car[0].height, car[0].steps, car[0].cfg) == (1024, 512, 30, 7.0)
    assert jobs[1].condition == "a @ 0.5" and jobs[2].condition == "a @ 1"


def test_build_jobs_filters():
    assert {j.bucket for j in ls.build_jobs(VSET, ["a"], [1], buckets=["unrelated"])} == {"unrelated"}
    jobs = ls.build_jobs(VSET, ["a"], [1], include_baseline=False, max_prompts=1)
    assert {j.prompt_index for j in jobs} == {0} and all(j.lora for j in jobs) and len(jobs) == 2
    assert ls.build_jobs(VSET, [], [], include_baseline=False) == []
    assert len(ls.build_jobs(VSET, [], [1.0])) == 5     # Baseline only


def test_request_for_baselines():
    jobs = ls.build_jobs(VSET, ["a.safetensors"], [0.8], max_prompts=1)
    base, lora = jobs[0], jobs[1]
    r = ls.request_for(base, "ComfyUI", "ckpt", "lora_txt2img.json", "euler", "a.safetensors")
    assert r.loras == [("a.safetensors", 0.0)] and r.model == "ckpt" and r.workflow == "lora_txt2img.json"
    assert r.batch_size == 1 and r.seed == base.seed and r.negative_prompt == "bad"
    assert ls.request_for(base, "Forge / A1111 / SD.Next", "", None, "", "a.safetensors").loras == []
    assert ls.request_for(base, "ComfyUI", None, None, None, None).loras == []
    assert ls.request_for(lora, "ComfyUI", None, None, None, "x").loras == [("a.safetensors", 0.8)]
    r = ls.request_for(lora, "Forge / A1111 / SD.Next", "", "", "", None)
    assert r.model is None and r.sampler is None and r.workflow is None


# ---------- LoRA sweep: running ----------

class FakeBackend:
    def __init__(self, fail_on=()):
        self.requests = []
        self.fail_on = set(fail_on)

    def generate_iter(self, req):
        self.requests.append(req)
        yield GenProgress(0.5, "half")
        if len(self.requests) - 1 in self.fail_on:
            raise ImageGenError("boom")
        color = (int(255 * (req.loras[0][1] if req.loras else 0) / 1.5), req.seed % 255, 100)
        yield GenResult([GeneratedImage(Image.new("RGB", (64, 48), color), req.seed)])


def _run(tmp_path, fail_on=(), loras=("a.safetensors",), backend_label="ComfyUI"):
    jobs = ls.build_jobs(VSET, list(loras), [0.5, 1.0], max_prompts=2)
    run_dir = ls.new_run_dir(str(tmp_path / "runs"), "my test!")
    backend = FakeBackend(fail_on)
    events = list(ls.run_iter(backend, backend_label, jobs, run_dir, {"name": "t"}, model="ckpt", workflow="wf.json"))
    return jobs, run_dir, backend, events


def test_new_run_dir_name(tmp_path):
    with patch("core.lab.lora_sweep.time.strftime", return_value="20260101-120000"):
        path = ls.new_run_dir(str(tmp_path), "My Test: v2 !!")
    assert os.path.basename(path) == "20260101-120000-My-Test-v2" and os.path.isdir(path)
    with patch("core.lab.lora_sweep.time.strftime", return_value="20260101-120001"):
        assert os.path.basename(ls.new_run_dir(str(tmp_path), "///")) == "20260101-120001-test"


def test_new_run_dir_same_second_is_unique(tmp_path):
    with patch("core.lab.lora_sweep.time.strftime", return_value="20260101-120000"):
        a = ls.new_run_dir(str(tmp_path), "x")
        b = ls.new_run_dir(str(tmp_path), "x")
    assert a != b and os.path.isdir(b)


def test_run_iter_saves_images_and_run_json(tmp_path):
    jobs, run_dir, backend, events = _run(tmp_path)
    assert [e[0] for e in events] == list(range(1, len(jobs) + 1)) and all(e[1] == len(jobs) for e in events)
    data = ls.load_run(run_dir)
    assert data["name"] == "t" and data["backend"] == "ComfyUI" and data["model"] == "ckpt" and data["workflow"] == "wf.json"
    assert len(data["jobs"]) == len(jobs) and "finished" in data and "started" in data
    assert all(r["path"] and r["error"] is None for r in data["results"])
    first = data["results"][0]["path"]
    assert first.startswith("0000_p00_s1_baseline") and os.path.exists(os.path.join(run_dir, first))
    assert "@" in data["results"][1]["path"]
    # ComfyUI baseline uses the first LoRA at strength 0
    assert backend.requests[0].loras == [("a.safetensors", 0.0)]
    assert backend.requests[1].loras == [("a.safetensors", 0.5)]


def test_run_iter_error_recorded_and_continues(tmp_path):
    jobs, run_dir, backend, events = _run(tmp_path, fail_on={1})
    assert len(events) == len(jobs)
    assert "⚠ boom" in events[1][2] and events[1][3] is None
    data = ls.load_run(run_dir)
    assert data["results"][1] == {"job": 1, "path": None, "error": "boom"}
    assert data["results"][2]["path"]


def test_run_iter_saves_run_json_after_each_job(tmp_path):
    jobs = ls.build_jobs(VSET, ["a"], [1.0], max_prompts=1)
    run_dir = ls.new_run_dir(str(tmp_path), "t")
    gen = ls.run_iter(FakeBackend(), "Forge / A1111 / SD.Next", jobs, run_dir, {})
    next(gen)
    assert len(ls.load_run(run_dir)["results"]) == 1 and "finished" not in ls.load_run(run_dir)
    gen.close()                                   # Stop keeps finished images
    assert len(ls.load_run(run_dir)["results"]) == 1


def test_run_iter_no_image_is_an_error(tmp_path):
    class Empty:
        def generate_iter(self, req):
            yield GenResult([])
    jobs = ls.build_jobs(VSET, [], [], max_prompts=1)
    run_dir = ls.new_run_dir(str(tmp_path), "t")
    list(ls.run_iter(Empty(), "ComfyUI", jobs, run_dir, {}))
    assert all(r["error"] == "The backend returned no image." for r in ls.load_run(run_dir)["results"])


def test_forge_baseline_has_no_lora(tmp_path):
    _, _, backend, _ = _run(tmp_path, backend_label="Forge / A1111 / SD.Next")
    assert backend.requests[0].loras == []


def test_make_grids(tmp_path):
    jobs, run_dir, _, _ = _run(tmp_path, fail_on={2})
    grids = ls.make_grids(run_dir, cell=64)
    assert [os.path.basename(g) for g in grids] == ["grid_p00.png", "grid_p01.png"]
    with Image.open(grids[0]) as g:
        conds, seeds = 3, 2                       # baseline + 2 strengths; seeds 1, 2
        assert g.size == (190 + 64 * seeds, 78 + 64 * conds)
        # Baseline row first: its tile is dark red (strength 0 → red 0), the 0.5 row is brighter
        base_px = g.getpixel((190 + 32, 78 + 32))
        assert base_px[0] == 0
        # The failed job (index 2 = seed 1, strength 1.0, third row) is drawn as an empty tile
        missing = g.getpixel((190 + 32, 78 + 2 * 64 + 32))
        assert missing == (255, 255, 255)
    with Image.open(grids[1]) as g:
        assert g.size == (190 + 64, 78 + 64 * 3)  # Prompt 1 has its own seed only


def test_list_runs(tmp_path):
    root = tmp_path / "runs"
    assert ls.list_runs(str(root)) == []
    for name in ("20260101-a", "20260102-b"):
        (root / name).mkdir(parents=True)
        (root / name / "run.json").write_text("{}", encoding="utf-8")
    (root / "20260103-empty").mkdir()
    assert ls.list_runs(str(root)) == ["20260102-b", "20260101-a"]


def test_copy_lora(tmp_path):
    src = tmp_path / "style.safetensors"
    src.write_bytes(b"lora")
    comfy = tmp_path / "ComfyUI"
    (comfy / "models" / "loras").mkdir(parents=True)
    dst = ls.copy_lora(str(src), str(comfy))
    assert dst == str(comfy / "models" / "loras" / "style.safetensors") and open(dst, "rb").read() == b"lora"
    with pytest.raises(FileExistsError):
        ls.copy_lora(str(src), str(comfy))
    loras_dir = tmp_path / "loras_only"
    loras_dir.mkdir()
    assert ls.copy_lora(str(src), str(loras_dir)) == str(loras_dir / "style.safetensors")
    with pytest.raises(FileNotFoundError):
        ls.copy_lora(str(tmp_path / "missing.safetensors"), str(comfy))
    bad = tmp_path / "notes.txt"
    bad.write_text("x")
    with pytest.raises(ValueError):
        ls.copy_lora(str(bad), str(comfy))
    with pytest.raises(FileNotFoundError):
        ls.copy_lora(str(src), str(tmp_path / "nowhere"))
    (tmp_path / "C2" / "models").mkdir(parents=True)      # models/ but no loras/ folder
    with pytest.raises(FileNotFoundError):
        ls.copy_lora(str(src), str(tmp_path / "C2"))
