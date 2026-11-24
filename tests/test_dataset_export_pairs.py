# ArtAgent/tests/test_dataset_export_pairs.py
"""Tests for core/dataset/trainer_export.py and core/dataset/edit_pairs.py."""
import json
import os
import sys
import tomllib
import zipfile
from unittest.mock import patch

import pytest
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import edit_pairs as ep, trainer_export as te
from core.dataset.vision import VisionTarget
from core.version import APP_VERSION

ASK_PATH = "core.dataset.edit_pairs.ask"


# ======================= trainer export =======================

@pytest.mark.parametrize("images, target, expected", [(20, 100, 5), (30, 100, 3), (300, 100, 1), (0, 100, 1),
                                                       (7, 100, 14), (40, 100, 2)])
def test_repeats_for(images, target, expected):
    assert te.repeats_for(images, target) == expected


def test_steps_per_epoch():
    assert te.steps_per_epoch(10, 3, 4) == 8
    assert te.steps_per_epoch(10, 0, 0) == 10
    assert te.steps_per_epoch(-1, 1, 1) == 0


@pytest.mark.parametrize("prefix, n", [("txcl painting, ", 1), ("txcl, painting, ", 2), ("", 0), (None, 0), (" , ", 0)])
def test_keep_tokens_for(prefix, n):
    assert te.keep_tokens_for(prefix) == n


def test_repeats_markdown():
    md = te.repeats_markdown({"style_a": 20, "style_b": 50}, 100, 10, 2)
    assert "| style_a | 20 | 5 | 100 | 50 |" in md
    assert "| style_b | 50 | 2 | 100 | 50 |" in md
    assert "**About 1000 steps** for 10 epoch(s) at batch size 2" in md
    assert te.repeats_markdown({}, 100, 10, 1) == "Load a folder first."


def _settings(folder="/data/my ds", **kw):
    s = te.default_settings(folder, 12)
    s.update(kw)
    return s


def test_default_settings():
    s = te.default_settings("/data/set_a/", 5)
    assert s["name"] == "set_a" and s["num_images"] == 5 and s["repeats"] == 1 and s["control_dirs"] == []


def test_kohya_toml():
    text = te.render_config("kohya", _settings(repeats=10, keep_tokens=2, shuffle=True, caption_dropout=0.05))
    assert text.startswith(f"# Written by ArtAgents {APP_VERSION} for /data/my ds (12 images).")
    data = tomllib.loads(text)
    assert data["general"] == {"shuffle_caption": True, "caption_extension": ".txt", "keep_tokens": 2}
    ds = data["datasets"][0]
    assert ds["resolution"] == 1024 and ds["enable_bucket"] is True and ds["bucket_reso_steps"] == 64
    assert ds["min_bucket_reso"] == 256 and ds["max_bucket_reso"] == 2048
    sub = ds["subsets"][0]
    assert sub == {"image_dir": "/data/my ds", "num_repeats": 10, "caption_dropout_rate": 0.05}


def test_kohya_variants_enable_wildcard_and_no_dropout():
    data = tomllib.loads(te.render_config("kohya", _settings(variant_files=3, caption_dropout=0)))
    sub = data["datasets"][0]["subsets"][0]
    assert sub["enable_wildcard"] is True and "caption_dropout_rate" not in sub


@pytest.mark.parametrize("folder", [r"C:\Users\me\datasets\style", r"C:\Users\O'Brien\data", "/tmp/it's here",
                                    "/data/ünïcødé"])
def test_toml_paths_parse(folder):
    for trainer in ("kohya", "musubi", "diffusion_pipe"):
        data = tomllib.loads(te.render_config(trainer, _settings(folder)))
        if trainer == "kohya":
            assert data["datasets"][0]["subsets"][0]["image_dir"] == folder
        elif trainer == "musubi":
            assert data["datasets"][0]["image_directory"] == folder
        else:
            assert data["directory"][0]["path"] == folder


def test_musubi_toml():
    data = tomllib.loads(te.render_config("musubi", _settings("/data/my ds", repeats=3, resolution=768, batch_size=2)))
    assert data["general"] == {"resolution": [768, 768], "caption_extension": ".txt", "batch_size": 2,
                               "enable_bucket": True, "bucket_no_upscale": False}
    ds = data["datasets"][0]
    assert ds["image_directory"] == "/data/my ds" and ds["num_repeats"] == 3
    assert ds["cache_directory"] == "/data/my ds_cache"
    assert "control_directory" not in ds


def test_musubi_controls():
    one = tomllib.loads(te.render_config("musubi", _settings(control_dirs=["/data/ctrl"])))
    assert one["datasets"][0]["control_directory"] == "/data/ctrl"
    text = te.render_config("musubi", _settings(control_dirs=["/a", "/b"]))
    assert "control_directory" not in tomllib.loads(text)["datasets"][0]
    assert "control_path_0" in text


def test_musubi_variant_warning():
    assert "not verified" in te.render_config("musubi", _settings(variant_files=2))


def test_ai_toolkit_yaml():
    text = te.render_config("ai_toolkit", _settings(r"C:\data\set", trigger="txcl", shuffle=True, caption_dropout=0.05,
                                                    webp_files=2))
    assert "WEBP image(s)" in text and 'trigger_word: "txcl"' in text
    data = yaml.safe_load(text)
    entry = data["datasets"][0]
    assert entry == {"folder_path": r"C:\data\set", "caption_ext": "txt", "caption_dropout_rate": 0.05,
                     "shuffle_tokens": True, "cache_latents_to_disk": True, "resolution": [512, 768, 1024]}


def test_ai_toolkit_low_resolution_and_controls():
    entry = yaml.safe_load(te.render_config("ai_toolkit", _settings(resolution=512, control_dirs=["/c"])))["datasets"][0]
    assert entry["resolution"] == [512] and entry["control_path"] == "/c"
    entry = yaml.safe_load(te.render_config("ai_toolkit", _settings(control_dirs=["/a", "/b", "/c", "/d"])))["datasets"][0]
    assert entry["control_path"] == ["/a", "/b", "/c"]


def test_webp_warning_only_for_ai_toolkit():
    assert "WEBP" not in te.render_config("kohya", _settings(webp_files=3))


def test_onetrainer_json():
    data = json.loads(te.render_config("onetrainer", _settings(repeats=4, shuffle=True, keep_tokens=2)))
    assert len(data) == 1
    c = data[0]
    assert c["__version"] == 2 and c["name"] == "my ds" and c["path"] == "/data/my ds"
    assert c["type"] == "STANDARD" and c["balancing"] == 4.0 and c["balancing_strategy"] == "REPEATS"
    assert c["text"] == {"prompt_source": "sample", "prompt_path": "", "enable_tag_shuffling": True,
                         "tag_delimiter": ",", "keep_tags_count": 2}
    assert "repeats" not in c


@pytest.mark.parametrize("repeats, expected", [(1, 0), (5, 4), (0, 0)])
def test_simpletuner_json_repeats(repeats, expected):
    data = json.loads(te.render_config("simpletuner", _settings(repeats=repeats)))
    backend, text = data
    assert backend["repeats"] == expected
    assert backend["id"] == "my-ds" and backend["type"] == "local" and backend["caption_strategy"] == "textfile"
    assert backend["instance_data_dir"] == "/data/my ds" and backend["resolution_type"] == "pixel_area"
    assert backend["minimum_image_size"] == 256 and backend["cache_dir_vae"] == "cache/vae/my-ds"
    assert "disable_multiline_split" not in backend
    assert text["dataset_type"] == "text_embeds" and text["default"] is True


def test_simpletuner_variants():
    backend = json.loads(te.render_config("simpletuner", _settings(variant_files=1)))[0]
    assert backend["disable_multiline_split"] is False


def test_diffusion_pipe_toml():
    data = tomllib.loads(te.render_config("diffusion_pipe", _settings(repeats=6, resolution=512)))
    assert data["resolutions"] == [512] and data["enable_ar_bucket"] is True
    assert data["directory"] == [{"path": "/data/my ds", "num_repeats": 6}]


def test_write_config(tmp_path):
    path, text = te.write_config("kohya", _settings(str(tmp_path)))
    assert path == str(tmp_path / ".artagents" / "export" / "dataset_kohya.toml")
    assert open(path, encoding="utf-8").read() == text


def test_trainer_labels_and_language():
    assert set(te.TRAINER_LABELS.values()) == set(te.TRAINERS)
    assert te.code_language("ai_toolkit") == "yaml"
    assert te.code_language("onetrainer") == "json"
    assert te.code_language("kohya") is None


# ======================= edit pairs =======================

def _img(path, size=(100, 100), color=(200, 0, 0)):
    Image.new("RGB", size, color).save(path)
    return str(path)


@pytest.fixture
def pairs_dirs(tmp_path):
    target, c1, c2 = tmp_path / "target", tmp_path / "control1", tmp_path / "control2"
    for d in (target, c1, c2):
        d.mkdir()
    _img(target / "A.png", (1000, 1000))
    _img(target / "b.jpg", (1000, 500))
    _img(target / "c.png", (1000, 1000))
    (target / "notes.txt").write_text("x", encoding="utf-8")
    _img(c1 / "a.jpg", (1000, 1000))
    _img(c1 / "b.png", (1000, 1000))         # Aspect differs from the target
    _img(c1 / "orphan.png")
    _img(c2 / "a.webp", (200, 200))           # Too small
    return target, c1, c2


def test_parse_control_dirs():
    assert ep.parse_control_dirs(' "/a" \n\n/b ; /c') == ["/a", "/b", "/c"]
    assert ep.parse_control_dirs(None) == []


def test_find_pairs(pairs_dirs):
    target, c1, c2 = pairs_dirs
    found = ep.find_pairs(str(target), [str(c1), str(c2)])
    assert found["errors"] == []
    pairs = {p["stem"]: p for p in found["pairs"]}
    assert set(pairs) == {"A", "b", "c"}
    assert pairs["A"]["controls"] == [str(c1 / "a.jpg"), str(c2 / "a.webp")]   # Case-insensitive stem match
    assert pairs["c"]["controls"] == [None, None]
    assert found["orphans"] == ["orphan.png"]


def test_find_pairs_errors(tmp_path):
    found = ep.find_pairs(str(tmp_path / "nope"), [str(tmp_path / "c1")])
    assert found["pairs"] == []
    assert any("Target folder not found" in e for e in found["errors"])
    assert any("Control folder not found" in e for e in found["errors"])
    many = [str(tmp_path)] * 4
    assert any("4 control folders" in e for e in ep.find_pairs(str(tmp_path), many)["errors"])


def test_validate(pairs_dirs):
    target, c1, c2 = pairs_dirs
    (target / "A.txt").write_text("Make it blue.", encoding="utf-8")
    rows = ep.validate(ep.find_pairs(str(target), [str(c1), str(c2)])["pairs"])
    by = {}
    for stem, issue, detail in rows:
        by.setdefault(stem, []).append((issue, detail))
    assert (ep.ISSUE_CONTROL_SIZE, "control 2 is 0.04 MP (about 1 MP is typical)") in by["A"]
    assert all(issue != ep.ISSUE_NO_INSTRUCTION for issue, _ in by["A"])
    assert (ep.ISSUE_ASPECT, "target 1000×500, control 1 1000×1000") in by["b"]
    assert (ep.ISSUE_NO_CONTROL, "control 1") in by["c"] and (ep.ISSUE_NO_CONTROL, "control 2") in by["c"]
    assert (ep.ISSUE_NO_INSTRUCTION, "") in by["c"]


def test_validate_single_control_label_and_unreadable(tmp_path):
    t, c = tmp_path / "t", tmp_path / "c"
    t.mkdir(), c.mkdir()
    _img(t / "x.png", (1000, 1000))
    (c / "x.png").write_bytes(b"broken")
    rows = ep.validate(ep.find_pairs(str(t), [str(c)])["pairs"])
    assert ["x", "cannot be opened", "x.png"] in rows


def test_instruction_prompt():
    one = ep.instruction_prompt(1)
    assert one.startswith("The first image is the original; the last image is the edited result.")
    assert one.rstrip().endswith("Answer with the instruction only.")
    many = ep.instruction_prompt(2, paraphrases=5, trigger="txcl", extra="Name the colour.")
    assert many.startswith("The first 2 images are the inputs")
    assert "Then write 5 more phrasings" in many and "Do not write the word 'txcl'" in many
    assert many.rstrip().endswith("Name the colour.")


def test_clean_instruction_lines():
    raw = '1. Make the car red.\n2) "Turn the car red."\n- make the car red.\n• Paint it red\n\nHere are some phrasings:\n'
    assert ep.clean_instruction_lines(raw, 10) == ["Make the car red.", "Turn the car red.", "Paint it red"]
    assert ep.clean_instruction_lines(raw, 2) == ["Make the car red.", "Turn the car red."]
    assert ep.clean_instruction_lines("", 3) == []
    assert ep.clean_instruction_lines("x", 0) == ["x"]   # At least one line


def test_clean_instruction_lines_strips_label():
    assert ep.clean_instruction_lines("Instruction: Make the car red.", 1) == ["Make the car red."]


def test_generate_instruction_image_order_and_trigger(pairs_dirs):
    target, c1, c2 = pairs_dirs
    pair = {s["stem"]: s for s in ep.find_pairs(str(target), [str(c1), str(c2)])["pairs"]}["A"]
    seen = {}

    def fake_ask(tgt, settings, prompt, images, **kw):
        seen["sizes"] = [im.size for im in images]
        seen["prompt"], seen["kw"] = prompt, kw
        return "1. Make the background blue.\n2. Change the background to blue."

    with patch(ASK_PATH, side_effect=fake_ask):
        lines = ep.generate_instruction(pair, VisionTarget("ollama", "m"), {}, paraphrases=1, trigger_prefix="txcl",
                                        max_side=512)
    # Controls first (1000² → 512², 200²), then the target
    assert seen["sizes"] == [(512, 512), (200, 200), (512, 512)]
    assert "The first 2 images are the inputs" in seen["prompt"]
    assert seen["kw"] == {"temperature": 0.0, "seed": 42, "max_tokens": 100}
    assert lines == ["txcl, make the background blue.", "txcl, change the background to blue."]


def test_generate_instruction_prefix_variants(pairs_dirs):
    target, c1, _ = pairs_dirs
    pair = {s["stem"]: s for s in ep.find_pairs(str(target), [str(c1)])["pairs"]}["A"]
    with patch(ASK_PATH, return_value="txcl, already there\nMake it red"):
        lines = ep.generate_instruction(pair, VisionTarget("ollama", "m"), {}, 1, "txcl,")
    assert lines == ["txcl, already there", "txcl, make it red"]
    with patch(ASK_PATH, return_value="Make it red"):
        assert ep.generate_instruction(pair, VisionTarget("ollama", "m"), {}, 0, "") == ["Make it red"]


def test_generate_instruction_errors(pairs_dirs):
    target, c1, _ = pairs_dirs
    pairs = {s["stem"]: s for s in ep.find_pairs(str(target), [str(c1)])["pairs"]}
    with pytest.raises(ValueError, match="no control image"):
        ep.generate_instruction(pairs["c"], VisionTarget("ollama", "m"), {})
    with patch(ASK_PATH, return_value="\n\n"), pytest.raises(ValueError, match="no instruction"):
        ep.generate_instruction(pairs["A"], VisionTarget("ollama", "m"), {})


def test_metadata_jsonl(pairs_dirs):
    target, c1, c2 = pairs_dirs
    (target / "A.txt").write_text("Make it blue.\nTurn it blue.", encoding="utf-8")
    (target / "b.txt").write_text("Crop it.", encoding="utf-8")
    one = [json.loads(l) for l in ep.metadata_jsonl(ep.find_pairs(str(target), [str(c1)])["pairs"]).splitlines()]
    assert {"image_path": str(target / "A.png"), "control_path": str(c1 / "a.jpg"), "caption": "Make it blue."} in one
    assert all("control_path_0" not in r for r in one)
    two = [json.loads(l) for l in ep.metadata_jsonl(ep.find_pairs(str(target), [str(c1), str(c2)])["pairs"]).splitlines()]
    a = [r for r in two if r["image_path"].endswith("A.png")][0]
    assert a["control_path_0"] == str(c1 / "a.jpg") and a["control_path_1"] == str(c2 / "a.webp")
    assert ep.metadata_jsonl([]) == ""


def test_metadata_jsonl_skips_pairs_without_control(pairs_dirs):
    target, c1, _ = pairs_dirs
    (target / "c.txt").write_text("Make it green.", encoding="utf-8")   # c has no control image
    rows = [json.loads(l) for l in ep.metadata_jsonl(ep.find_pairs(str(target), [str(c1)])["pairs"]).splitlines()]
    assert all(r["image_path"] != str(target / "c.png") for r in rows)


def test_fal_zip(pairs_dirs, tmp_path):
    target, c1, _ = pairs_dirs
    (target / "A.txt").write_text("Make it blue.\nAlt", encoding="utf-8")
    pairs = ep.find_pairs(str(target), [str(c1)])["pairs"]
    out = tmp_path / "pairs.zip"
    assert ep.fal_zip(pairs, str(out)) == 2      # c has no control
    names = sorted(zipfile.ZipFile(out).namelist())
    assert names == ["0001.txt", "0001_end.png", "0001_start.jpg", "0002_end.jpg", "0002_start.png"]
    assert zipfile.ZipFile(out).read("0001.txt").decode() == "Make it blue."
