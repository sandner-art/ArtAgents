# ArtAgent/tests/test_dataset_tagger.py
"""Tests for core/dataset/tagger.py (ONNX booru tagger, thresholds, profile-aware output, sidecars)."""
import csv
import json
import os
import sys
import time
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import profiles, tagger as tg

GEN, CHAR, RATING = tg.CATEGORY_GENERAL, tg.CATEGORY_CHARACTER, tg.CATEGORY_RATING


def _tags(**kw):
    """name=(p, category) shorthand."""
    return dict(kw)


SAMPLE = {
    "general": (0.9, RATING), "sensitive": (0.1, RATING), "questionable": (0.02, RATING), "explicit": (0.01, RATING),
    "1girl": (0.98, GEN), "solo": (0.95, GEN), "long_hair": (0.6, GEN), "smile": (0.36, GEN), "hat": (0.30, GEN),
    "^_^": (0.5, GEN), "hatsune_miku_(vocaloid)": (0.9, CHAR), "other_char": (0.2, CHAR),
}


# --- mcut ---

def test_mcut_threshold_largest_gap():
    # Gaps: 0.9-0.85=0.05, 0.85-0.3=0.55, 0.3-0.25=0.05 -> threshold midway between 0.85 and 0.3
    assert tg.mcut_threshold([0.3, 0.9, 0.85, 0.25]) == pytest.approx(0.575)


@pytest.mark.parametrize("probs", [[], [0.5]])
def test_mcut_threshold_too_few(probs):
    assert tg.mcut_threshold(probs) == 0.0


# --- select ---

def test_select_default_thresholds():
    sel = tg.select(SAMPLE)
    assert sel["rating"] == ("general", 0.9)
    assert [n for n, _ in sel["general"]] == ["1girl", "solo", "long_hair", "^_^", "smile"]  # > 0.35, by probability
    assert [n for n, _ in sel["characters"]] == ["hatsune_miku_(vocaloid)"]                # > 0.85
    assert sel["thresholds"] == (0.35, 0.85)


def test_select_threshold_is_strict():
    sel = tg.select({"a": (0.35, GEN), "b": (0.3501, GEN)})
    assert [n for n, _ in sel["general"]] == ["b"]


def test_select_custom_thresholds():
    sel = tg.select(SAMPLE, {"general_threshold": 0.7, "character_threshold": 0.1})
    assert [n for n, _ in sel["general"]] == ["1girl", "solo"]
    assert [n for n, _ in sel["characters"]] == ["hatsune_miku_(vocaloid)", "other_char"]


def test_select_mcut_general():
    sel = tg.select(SAMPLE, {"general_mcut": True})
    # General probs: .98 .95 .6 .5 .36 .30 -> biggest gap .95 -> .6
    assert sel["thresholds"][0] == pytest.approx(0.775)
    assert [n for n, _ in sel["general"]] == ["1girl", "solo"]


def test_select_mcut_character_floor():
    # Characters 0.12 and 0.05: MCut would be 0.085, but the floor is 0.15
    sel = tg.select({"a": (0.12, CHAR), "b": (0.05, CHAR)}, {"character_mcut": True})
    assert sel["thresholds"][1] == 0.15
    assert sel["characters"] == []


def test_select_no_ratings():
    assert tg.select({"x": (0.9, GEN)})["rating"] is None


# --- display / format ---

@pytest.mark.parametrize("name, keep, expected", [
    ("long_hair", False, "long hair"),
    ("long_hair", True, "long_hair"),
    ("^_^", False, "^_^"),
    ("o_o", False, "o_o"),
    ("hatsune_miku_(vocaloid)", False, "hatsune miku (vocaloid)"),
])
def test_display_tag(name, keep, expected):
    assert tg.display_tag(name, keep) == expected


@pytest.mark.parametrize("name, is_count", [
    ("1girl", True), ("2girls", True), ("6+girls", True), ("multiple_girls", True), ("multiple girls", True),
    ("no_humans", True), ("1other", True), ("1boy", True), ("solo", False), ("girl", False), ("1girl_hat", False),
])
def test_is_count(name, is_count):
    assert tg._is_count(name) is is_count


def _sel(general=(), characters=(), rating=None):
    return {"general": list(general), "characters": list(characters), "rating": rating}


def test_format_tags_default_order_count_character_rating_general():
    sel = _sel([("smile", 0.9), ("1girl", 0.99), ("long_hair", 0.8)], [("hatsune_miku_(vocaloid)", 0.9)], ("general", 0.9))
    out = tg.format_tags(sel, {}, {"add_rating": True})
    assert out == ["1girl", "hatsune miku (vocaloid)", "general", "smile", "long hair"]


def test_format_tags_rating_only_when_asked():
    sel = _sel([("smile", 0.9)], rating=("explicit", 0.9))
    assert tg.format_tags(sel, {}, {}) == ["smile"]


@pytest.mark.parametrize("profile_id, rating, expected", [
    ("pony_v6", "general", "rating_safe"),
    ("pony_v6", "sensitive", "rating_questionable"),
    ("pony_v6", "questionable", "rating_questionable"),
    ("pony_v6", "explicit", "rating_explicit"),
    ("illustrious", "general", "general"),
    ("illustrious", "questionable", "nsfw"),
    ("noobai", "general", "safe"),
    ("noobai", "questionable", "nsfw"),
    ("noobai", "explicit", "explicit"),
    ("sd_sdxl", "questionable", "questionable"),  # No map: tagger name as is
])
def test_format_tags_rating_map_from_models_yaml(profile_id, rating, expected):
    out = tg.format_tags(_sel([("solo", 0.9)], rating=(rating, 0.8)), profiles.get_profile(profile_id), {"add_rating": True})
    assert expected in out


def test_format_tags_pony_puts_rating_first():
    sel = _sel([("solo", 0.9), ("1girl", 0.95)], [("miku", 0.9)], ("general", 0.9))
    out = tg.format_tags(sel, profiles.get_profile("pony_v6"), {"add_rating": True})
    assert out == ["rating_safe", "1girl", "miku", "solo"]


def test_format_tags_illustrious_order():
    sel = _sel([("solo", 0.9), ("1girl", 0.95)], [("miku", 0.9)], ("general", 0.9))
    out = tg.format_tags(sel, profiles.get_profile("illustrious"), {"add_rating": True})
    assert out == ["1girl", "miku", "general", "solo"]


def test_format_tags_blacklist_matches_spaces_and_underscores():
    sel = _sel([("simple_background", 0.9), ("white_background", 0.8), ("solo", 0.7)], [("miku", 0.9)])
    out = tg.format_tags(sel, {}, {"blacklist": ["simple background", "White_Background", " miku "]})
    assert out == ["solo"]


def test_format_tags_blacklist_applies_to_mapped_rating():
    sel = _sel([("solo", 0.9)], rating=("general", 0.9))
    out = tg.format_tags(sel, profiles.get_profile("pony_v6"), {"add_rating": True, "blacklist": ["rating_safe"]})
    assert out == ["solo"]


def test_format_tags_split_series():
    sel = _sel(characters=[("hatsune_miku_(vocaloid)", 0.9), ("plain_name", 0.9)])
    assert tg.format_tags(sel, {}, {"split_series": True}) == ["hatsune miku", "vocaloid", "plain name"]
    assert tg.format_tags(sel, {}, {}) == ["hatsune miku (vocaloid)", "plain name"]


def test_format_tags_keeps_underscores_for_profile():
    sel = _sel([("long_hair", 0.9), ("^_^", 0.8)])
    assert tg.format_tags(sel, {"underscores": True}, {}) == ["long_hair", "^_^"]
    assert tg.format_tags(sel, {}, {}) == ["long hair", "^_^"]


def test_format_tags_dedupes_case_insensitively():
    sel = _sel([("Solo", 0.9), ("solo", 0.8)])
    assert tg.format_tags(sel, {}, {}) == ["Solo"]


def test_format_tags_unknown_groups_appended_after_custom_order():
    # A profile order that omits groups still outputs them (after the listed ones)
    sel = _sel([("solo", 0.9), ("1girl", 0.9)], [("miku", 0.9)])
    out = tg.format_tags(sel, {"tagger": {"order": ["general"]}}, {})
    assert out[0] == "solo" and set(out[1:]) == {"1girl", "miku"}


# --- hints ---

def test_hint_text_threshold_and_characters():
    sel = _sel([("1girl", 0.9), ("long_hair", 0.6), ("smile", 0.4)], [("hatsune_miku_(vocaloid)", 0.9)])
    text = tg.hint_text(sel, 0.5)
    assert text.splitlines()[0].startswith("Tags from an automatic image tagger")
    assert "Characters: hatsune miku (vocaloid)" in text
    assert "Tags: 1girl, long hair" in text and "smile" not in text


def test_hint_text_empty():
    assert tg.hint_text(_sel([("smile", 0.4)]), 0.5) == ""


def test_summarize_selection():
    sel = tg.select(SAMPLE)
    assert tg.summarize_selection(sel) == "5 general (> 0.35), 1 character (> 0.85), rating general 0.90"


# --- labels / registry ---

def _write_labels(path, rows, header=("tag_id", "name", "category", "count")):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_load_labels(tmp_path):
    p = tmp_path / "tags.csv"
    _write_labels(p, [[0, "general", 9, 1], [1, "1girl", 0, 1], [2, "miku", 4, 1], [3, "odd", "x", 1], [4, "blank", "", 1]])
    names, cats = tg.load_labels(str(p))
    assert names == ["general", "1girl", "miku", "odd", "blank"]
    assert cats.tolist() == [9, 0, 4, 0, 0]


def test_load_labels_needs_name_column(tmp_path):
    p = tmp_path / "tags.csv"
    _write_labels(p, [[0, "x"]], header=("id", "tag"))
    with pytest.raises(tg.TaggerError, match="no 'name' column"):
        tg.load_labels(str(p))


def test_registry_has_wd_models_with_licenses():
    taggers = tg.load_taggers()
    ids = [t["id"] for t in taggers]
    assert ids[0] == "wd-swinv2-v3" and "wd-eva02-large-v3" in ids
    assert all(t.get("license") for t in taggers)
    assert tg.get_tagger("WD EVA02-Large v3 (most accurate, slower)")["repo"] == "SmilingWolf/wd-eva02-large-tagger-v3"
    assert tg.get_tagger("unknown")["id"] == "wd-swinv2-v3"
    assert tg.tagger_labels()[0] == taggers[0]["label"]


def test_registry_falls_back_to_builtin():
    with patch("core.dataset.tagger._load_yaml_list", return_value=[]):
        assert [t["id"] for t in tg.load_taggers()] == ["wd-swinv2-v3"]


def test_tagger_info_markdown_mentions_onnxruntime_when_missing():
    with patch("core.dataset.tagger.onnxruntime_available", return_value=False):
        md = tg.tagger_info_markdown("WD SwinV2 v3 (general use)")
    assert "huggingface.co/SmilingWolf/wd-swinv2-tagger-v3" in md and "Apache-2.0" in md
    assert "pip install onnxruntime" in md
    with patch("core.dataset.tagger.onnxruntime_available", return_value=True):
        assert "pip install" not in tg.tagger_info_markdown("WD SwinV2 v3 (general use)")


# --- files ---

def test_resolve_files_local_folder(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    (tmp_path / "selected_tags.csv").write_text("name,category\n", encoding="utf-8")
    model, labels = tg.resolve_files({"folder": str(tmp_path)})
    assert model == str(tmp_path / "model.onnx") and labels == str(tmp_path / "selected_tags.csv")


def test_resolve_files_local_folder_missing_file(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"x")
    with pytest.raises(tg.TaggerError, match="Tagger file not found: .*selected_tags.csv"):
        tg.resolve_files({"folder": str(tmp_path)})


def test_resolve_files_needs_repo_or_folder():
    with pytest.raises(tg.TaggerError, match="neither 'repo' nor 'folder'"):
        tg.resolve_files({"label": "Broken"})


def test_resolve_files_download_error_is_helpful():
    with patch("huggingface_hub.hf_hub_download", side_effect=OSError("offline")), \
            pytest.raises(tg.TaggerError, match="Could not download org/repo from Hugging Face: OSError: offline"):
        tg.resolve_files({"repo": "org/repo"})


def test_load_model_without_onnxruntime(tmp_path, monkeypatch):
    (tmp_path / "model.onnx").write_bytes(b"x")
    _write_labels(tmp_path / "selected_tags.csv", [[0, "a", 0, 1]])
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import onnxruntime -> ImportError
    monkeypatch.setattr(tg, "_SESSIONS", {})
    with pytest.raises(tg.TaggerError, match="needs onnxruntime: pip install onnxruntime"):
        tg.load_model({"folder": str(tmp_path)})


def test_onnxruntime_available_uses_find_spec():
    with patch("importlib.util.find_spec", return_value=None):
        assert tg.onnxruntime_available() is False


# --- preprocess ---

def test_preprocess_default_wd_layout():
    img = Image.new("RGB", (40, 20), (200, 100, 50))
    arr = tg.preprocess(img, 40)
    assert arr.shape == (1, 40, 40, 3) and arr.dtype == np.float32
    # Pad rows (top) are white; image rows are BGR 0..255
    assert arr[0, 0, 0].tolist() == [255, 255, 255]
    assert arr[0, 20, 20].tolist() == [50, 100, 200]


def test_preprocess_resizes_to_model_size():
    assert tg.preprocess(Image.new("RGB", (300, 100)), 64).shape == (1, 64, 64, 3)


def test_preprocess_alpha_composited_on_pad_colour():
    img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    arr = tg.preprocess(img, 10, {"pad": [127, 127, 127], "channels": "RGB"})
    assert arr[0, 5, 5].tolist() == [127, 127, 127]


def test_preprocess_rgb_unit_nchw():
    img = Image.new("RGB", (8, 8), (255, 0, 51))
    arr = tg.preprocess(img, 8, {"channels": "RGB", "normalize": "unit", "layout": "NCHW"})
    assert arr.shape == (1, 3, 8, 8)
    assert arr[0, :, 4, 4].tolist() == pytest.approx([1.0, 0.0, 0.2])


def test_preprocess_imagenet_rgb():
    img = Image.new("RGB", (4, 4), (255, 255, 255))
    arr = tg.preprocess(img, 4, {"channels": "RGB", "normalize": "imagenet"})
    expected = (1 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    assert arr[0, 1, 1].tolist() == pytest.approx(expected.tolist(), rel=1e-5)


def test_preprocess_imagenet_bgr_uses_matching_channel_statistics():
    img = Image.new("RGB", (4, 4), (255, 0, 0))  # Pure red
    arr = tg.preprocess(img, 4, {"channels": "BGR", "normalize": "imagenet"})
    # Channel order B, G, R: red should be normalized with R's mean/std
    red = (1 - 0.485) / 0.229
    blue = (0 - 0.406) / 0.225
    assert arr[0, 1, 1].tolist() == pytest.approx([blue, (0 - 0.456) / 0.224, red], rel=1e-5)


# --- sidecar ---

def test_sidecar_roundtrip(tmp_path):
    image = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(image)
    tags = {"solo": (0.9, GEN), "general": (0.8, RATING), "low": (0.03, GEN)}
    tg.save_sidecar(str(image), {"id": "m1"}, tags)
    path = tmp_path / ".artagents" / "tagger" / "a.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["model"] == "m1" and data["floor"] == tg.SIDECAR_FLOOR
    assert list(data["tags"]) == ["solo", "general", "low"]  # Sorted by probability
    assert tg.sidecar_tags(tg.load_sidecar(str(image))) == tags


def test_load_sidecar_missing_or_broken(tmp_path):
    image = tmp_path / "a.png"
    assert tg.load_sidecar(str(image)) == {}
    path = tmp_path / ".artagents" / "tagger"
    path.mkdir(parents=True)
    (path / "a.json").write_text("{broken", encoding="utf-8")
    assert tg.load_sidecar(str(image)) == {}
    (path / "a.json").write_text("[1]", encoding="utf-8")
    assert tg.load_sidecar(str(image)) == {}


def test_sidecar_tags_skips_malformed():
    assert tg.sidecar_tags({"tags": {"a": [0.5, 0], "b": [0.5], "c": "x"}}) == {"a": (0.5, 0)}


def test_tag_probabilities_reuses_until_image_or_model_changes(tmp_path):
    image = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(image)
    cfg = {"id": "m1"}
    with patch("core.dataset.tagger.predict", return_value={"solo": (0.9, GEN)}) as predict:
        tags, cached = tg.tag_probabilities(cfg, str(image))
        assert (tags, cached) == ({"solo": (0.9, GEN)}, False)
        tags, cached = tg.tag_probabilities(cfg, str(image))
        assert cached is True and predict.call_count == 1
        # Different model -> runs again
        tg.tag_probabilities({"id": "m2"}, str(image))
        assert predict.call_count == 2
        # reuse=False -> runs again
        tg.tag_probabilities({"id": "m2"}, str(image), reuse=False)
        assert predict.call_count == 3
        # Image changed -> runs again
        later = time.time() + 10
        os.utime(image, (later, later))
        _, cached = tg.tag_probabilities({"id": "m2"}, str(image))
        assert cached is False and predict.call_count == 4


def test_tag_probabilities_uses_given_image(tmp_path):
    image = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(image)
    given = Image.new("RGB", (2, 2))
    with patch("core.dataset.tagger.predict", return_value={}) as predict:
        tg.tag_probabilities({"id": "m"}, str(image), given)
    assert predict.call_args[0][1] is given


def test_tag_probabilities_survives_unwritable_sidecar(tmp_path):
    image = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(image)
    with patch("core.dataset.tagger.predict", return_value={"x": (0.5, GEN)}), \
            patch("core.dataset.tagger.save_sidecar", side_effect=OSError("read-only")):
        assert tg.tag_probabilities({"id": "m"}, str(image)) == ({"x": (0.5, GEN)}, False)


# --- vocabulary ---

def test_vocabulary_from_folder(tmp_path):
    _write_labels(tmp_path / "selected_tags.csv", [[0, "long_hair", 0, 1], [1, "^_^", 0, 1], [2, "Miku", 4, 1]])
    tg._VOCAB.pop("vocab-test", None)
    vocab = tg.vocabulary({"id": "vocab-test", "folder": str(tmp_path)})
    assert vocab == {"long hair": 0, "^_^": 0, "miku": 4}
    tg._VOCAB.pop("vocab-test", None)


def test_vocabulary_missing_folder_file(tmp_path):
    tg._VOCAB.pop("vocab-missing", None)
    assert tg.vocabulary({"id": "vocab-missing", "folder": str(tmp_path)}) is None


def test_vocabulary_repo_never_downloads():
    tg._VOCAB.pop("vocab-repo", None)
    with patch("huggingface_hub.hf_hub_download", side_effect=OSError("not cached")) as dl:
        assert tg.vocabulary({"id": "vocab-repo", "repo": "org/repo"}) is None
    assert dl.call_args[1]["local_files_only"] is True


# --- real inference with a tiny ONNX model ---

FAKE_TAGS = [
    ("general", 9, (0, 0, 0), 2.0), ("sensitive", 9, (0, 0, 0), -1.0), ("questionable", 9, (0, 0, 0), -3.0),
    ("explicit", 9, (0, 0, 0), -4.0), ("1girl", 0, (0, 0, 0), 2.5), ("solo", 0, (0, 0, 0), 1.5),
    ("red_theme", 0, (-6, -6, 8), -1.0), ("blue_theme", 0, (8, -6, -6), -1.0), ("rare", 0, (0, 0, 0), -9.0),
    ("hatsune_miku_(vocaloid)", 4, (0, 0, 0), 3.0),
]


@pytest.fixture
def fake_tagger(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper
    W = np.array([t[2] for t in FAKE_TAGS], dtype=np.float32).T
    B = np.array([t[3] for t in FAKE_TAGS], dtype=np.float32)
    nodes = [
        helper.make_node("ReduceMean", ["input", "axes"], ["mean"], keepdims=0),
        helper.make_node("Div", ["mean", "scale"], ["unit"]),
        helper.make_node("MatMul", ["unit", "W"], ["logits0"]),
        helper.make_node("Add", ["logits0", "B"], ["logits"]),
        helper.make_node("Sigmoid", ["logits"], ["output"]),
    ]
    inits = [numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "axes"),
             numpy_helper.from_array(np.array(255.0, dtype=np.float32), "scale"),
             numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "B")]
    graph = helper.make_graph(nodes, "fake", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 32, 32, 3])],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, len(FAKE_TAGS)])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    folder = tmp_path / "fake_tagger"
    folder.mkdir()
    onnx.save(model, str(folder / "model.onnx"))
    _write_labels(folder / "selected_tags.csv", [[i, n, c, 1] for i, (n, c, _, _) in enumerate(FAKE_TAGS)])
    tg._SESSIONS.clear()
    yield {"id": "fake", "label": "Fake", "folder": str(folder)}
    tg._SESSIONS.clear()


def test_real_inference_red_vs_blue(fake_tagger):
    red = tg.predict(fake_tagger, Image.new("RGB", (32, 32), (230, 20, 20)))
    blue = tg.predict(fake_tagger, Image.new("RGB", (32, 32), (20, 20, 230)))
    p = lambda d, k: d.get(k, (0.0, 0))[0]  # Absent = below the sidecar floor
    assert p(red, "red_theme") > 0.5 > p(red, "blue_theme")
    assert p(blue, "blue_theme") > 0.5 > p(blue, "red_theme")  # Confirms the BGR channel order
    assert red["1girl"] == (pytest.approx(0.924, abs=1e-3), 0)
    assert "rare" not in red                    # Below the sidecar floor
    assert red["explicit"][1] == RATING         # Ratings are always kept
    assert red["hatsune_miku_(vocaloid)"][1] == CHAR


def test_real_inference_input_size_read_from_model(fake_tagger):
    _, _, size, names, _ = tg.load_model(fake_tagger)
    assert size == 32 and names[0] == "general"


def test_real_inference_end_to_end_format(fake_tagger, tmp_path):
    image = tmp_path / "red.png"
    Image.new("RGB", (64, 48), (230, 20, 20)).save(image)
    tags, cached = tg.tag_probabilities(fake_tagger, str(image))
    assert not cached and (tmp_path / ".artagents" / "tagger" / "red.json").is_file()
    out = tg.format_tags(tg.select(tags), profiles.get_profile("pony_v6"), {"add_rating": True})
    assert out[:3] == ["rating_safe", "1girl", "hatsune miku (vocaloid)"]
    assert "red theme" in out


def test_real_inference_label_count_mismatch(fake_tagger):
    path = os.path.join(fake_tagger["folder"], "selected_tags.csv")
    _write_labels(path, [[0, "only_one", 0, 1]])
    tg._SESSIONS.clear()
    with pytest.raises(tg.TaggerError, match=f"returned {len(FAKE_TAGS)} scores but the labels file has 1 tags"):
        tg.predict(fake_tagger, Image.new("RGB", (8, 8)))


def test_real_inference_sigmoid_option(fake_tagger):
    plain = tg.predict(fake_tagger, Image.new("RGB", (8, 8)))
    double = tg.predict(dict(fake_tagger, sigmoid=True), Image.new("RGB", (8, 8)))
    # Applying sigmoid to probabilities squeezes them towards 0.5..0.73
    assert double["1girl"][0] == pytest.approx(1 / (1 + np.exp(-plain["1girl"][0])), abs=1e-3)
