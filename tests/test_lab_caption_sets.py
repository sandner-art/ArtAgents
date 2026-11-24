# ArtAgent/tests/test_lab_caption_sets.py
"""Tests for core/lab/caption_sets.py."""
import hashlib
import json
import os
import sys
from unittest.mock import patch

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import audit, caption_io, snapshots
from core.dataset.tokens import count_tokens
from core.lab import caption_sets as cs

CLIP = [{"tokenizer": "clip", "budget": 10, "max": 20, "label": "CLIP"}]


@pytest.fixture
def ds(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        Image.new("RGB", (16, 16), "white").save(d / name)
    paths = {n: str(d / n) for n in ("a.png", "b.png", "c.png")}
    return d, paths


def _caps(d, ext=".txt", **texts):
    for stem, text in texts.items():
        (d / f"{stem}{ext}").write_text(text, encoding="utf-8")


# ---------- names & manifest ----------

@pytest.mark.parametrize("name, ok", [
    ("short-prose", "short-prose"), ("Short_Prose", "short_prose"), ("  set1 ", "set1"), ("a" * 40, "a" * 40),
])
def test_validate_name_ok(name, ok):
    assert cs.validate_name(name) == ok


@pytest.mark.parametrize("name", ["", "-x", "_x", "a.b", "a b", "a/b", "a" * 41, "txt", "caption", "captions", "é"])
def test_validate_name_rejects(name):
    with pytest.raises(cs.CaptionSetError):
        cs.validate_name(name)


def test_extension():
    assert cs.extension("short") == ".short.txt"


def test_manifest_record_and_update(ds, tmp_path):
    d, _ = ds
    assert cs.load_manifest(str(d)) == {"sets": {}}
    with patch("core.lab.caption_sets.time.strftime", return_value="T1"):
        cs.record_set(str(d), "s", {"strategy": "x"}, "first")
    with patch("core.lab.caption_sets.time.strftime", return_value="T2"):
        cs.record_set(str(d), "s", {"strategy": "y"})
    entry = cs.load_manifest(str(d))["sets"]["s"]
    assert entry == {"created": "T1", "updated": "T2", "settings": {"strategy": "y"}, "description": "first"}


def test_load_manifest_corrupt(ds):
    d, _ = ds
    (d / ".artagents").mkdir()
    (d / ".artagents" / "caption_sets.json").write_text("{bad", encoding="utf-8")
    assert cs.load_manifest(str(d)) == {"sets": {}}
    (d / ".artagents" / "caption_sets.json").write_text("[1]", encoding="utf-8")
    assert cs.load_manifest(str(d)) == {"sets": {}}


# ---------- discover / read / write ----------

def test_discover_sets_disk_and_manifest(ds):
    d, paths = ds
    _caps(d, a="active")
    _caps(d, ".long.txt", a="x")
    _caps(d, ".tags.txt", b="y")
    (d / "zzz.other.txt").write_text("no image zzz", encoding="utf-8")
    (d / "a.Bad Name.txt").write_text("invalid set name", encoding="utf-8")
    cs.record_set(str(d), "planned", {})
    assert cs.discover_sets(paths) == ["long", "planned", "tags"]
    assert cs.discover_sets({}) == []


def test_discover_ignores_active_caption_of_dotted_stem_without_sibling(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    Image.new("RGB", (8, 8)).save(d / "photo.v2.png")
    (d / "photo.v2.txt").write_text("active", encoding="utf-8")
    paths = {"photo.v2.png": str(d / "photo.v2.png")}
    assert cs.discover_sets(paths) == []


def test_discover_does_not_mistake_dotted_image_caption_for_a_set(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    for n in ("a.png", "a.b.png"):
        Image.new("RGB", (8, 8)).save(d / n)
    (d / "a.b.txt").write_text("caption of a.b.png", encoding="utf-8")
    paths = {n: str(d / n) for n in ("a.png", "a.b.png")}
    assert cs.discover_sets(paths) == []


def test_read_set_active_and_named(ds):
    d, paths = ds
    _caps(d, a="act a", b="")
    _caps(d, ".s.txt", c="set c")
    assert cs.read_set(paths, cs.ACTIVE) == {"a.png": "act a", "b.png": ""}
    assert cs.read_set(paths, None) == cs.read_set(paths, "") == cs.read_set(paths, cs.ACTIVE)
    assert cs.read_set(paths, "s") == {"c.png": "set c"}
    assert cs.read_set(paths, "missing") == {}


def test_write_set_validates_and_records(ds):
    d, paths = ds
    n = cs.write_set(paths, "My_Set", {"a.png": "one", "b.png": None, "zzz.png": "unknown"}, {"strategy": "t"}, "desc")
    assert n == 1
    assert (d / "a.my_set.txt").read_text(encoding="utf-8") == "one"
    assert not (d / "b.my_set.txt").exists()
    entry = cs.load_manifest(str(d))["sets"]["my_set"]
    assert entry["settings"] == {"strategy": "t"} and entry["description"] == "desc"
    with pytest.raises(cs.CaptionSetError):
        cs.write_set(paths, "bad name", {"a.png": "x"}, {})


def test_save_active_as_set(ds):
    d, paths = ds
    n = cs.save_active_as_set(paths, {"a.png": "cap a", "b.png": "  ", "c.png": "cap c\nvariant"}, "copy")
    assert n == 2
    assert cs.read_set(paths, "copy") == {"a.png": "cap a", "c.png": "cap c\nvariant"}
    assert cs.load_manifest(str(d))["sets"]["copy"]["settings"]["strategy"] == "copy of the active captions"
    with pytest.raises(cs.CaptionSetError, match="no captions"):
        cs.save_active_as_set(paths, {"a.png": " "}, "empty")


# ---------- activate / delete ----------

def test_activate_set_snapshot_first_and_keeps_others(ds):
    d, paths = ds
    _caps(d, a="old a", c="old c")
    _caps(d, ".s.txt", a="new a", b="new b")
    content, snap, missing = cs.activate_set(paths, "s")
    assert content == {"a.png": "new a", "b.png": "new b"} and missing == 1
    assert caption_io.read_caption(paths["a.png"]) == "new a"
    assert caption_io.read_caption(paths["b.png"]) == "new b"
    assert caption_io.read_caption(paths["c.png"]) == "old c"           # Not in the set: kept
    snap_caps = snapshots.snapshot_captions(str(d), snap)
    assert snap_caps == {"a.png": "old a", "b.png": None, "c.png": "old c"}
    assert cs.load_manifest(str(d))["active_from"]["set"] == "s"
    # Restoring the snapshot undoes it exactly
    snapshots.restore_snapshot(str(d), snap, paths)
    assert caption_io.read_caption(paths["a.png"]) == "old a" and not (d / "b.txt").exists()


def test_activate_set_errors(ds):
    d, paths = ds
    with pytest.raises(cs.CaptionSetError, match="no caption files"):
        cs.activate_set(paths, "nothing")
    with pytest.raises(cs.CaptionSetError):
        cs.activate_set(paths, cs.ACTIVE)


def test_delete_set_moves_files_and_manifest_entry(ds):
    d, paths = ds
    cs.write_set(paths, "s", {"a.png": "x", "b.png": "y"}, {"strategy": "t"})
    target = cs.delete_set(paths, "s")
    assert os.path.isdir(target) and target.startswith(str(d / ".artagents" / "deleted_sets"))
    assert sorted(os.listdir(target)) == ["a.s.txt", "b.s.txt", "manifest_entry.json"]
    assert json.load(open(os.path.join(target, "manifest_entry.json")))["settings"] == {"strategy": "t"}
    assert not (d / "a.s.txt").exists()
    assert "s" not in cs.load_manifest(str(d))["sets"] and cs.discover_sets(paths) == []


def test_delete_set_twice_in_same_second_keeps_both(ds):
    d, paths = ds
    with patch("core.lab.caption_sets.time.strftime", return_value="20260101-000000"):
        cs.write_set(paths, "s", {"a.png": "first"}, {})
        t1 = cs.delete_set(paths, "s")
        cs.write_set(paths, "s", {"a.png": "second"}, {})
        t2 = cs.delete_set(paths, "s")
    kept = set()
    for t in {t1, t2}:
        for f in os.listdir(t):
            if f.endswith(".s.txt"):
                kept.add(open(os.path.join(t, f), encoding="utf-8").read())
    assert kept == {"first", "second"}


# ---------- statistics ----------

def test_set_stats_numbers(ds):
    d, paths = ds
    long_text = "txcl " + " ".join(["word"] * 15)
    cs.write_set(paths, "s", {"a.png": "txcl, a cat", "b.png": long_text, "c.png": "a dog\nsecond line"}, {})
    row = cs.set_stats(paths, "s", CLIP, "txcl")
    tokens = [count_tokens(t, "clip")[0] for t in ("txcl, a cat", long_text, "a dog")]
    assert row[0] == "s" and row[1] == 3
    assert row[2] == round(sum(tokens) / 3, 1) and row[3] == max(tokens)
    assert row[4] == round(100 / 3, 1)                     # Only the long caption is over 10 CLIP tokens
    assert row[6] == round(200 / 3, 1)                     # Trigger in 2 of 3
    assert row[8] == 1                                     # One multi-line file
    assert row[9] == "—"                                   # Nothing verified


def test_set_stats_empty_and_no_budget(ds):
    _, paths = ds
    assert cs.set_stats(paths, "none") == ["none", 0, "", "", "", "", "", "", "", ""]
    assert cs.set_stats(paths, None)[0] == cs.ACTIVE
    cs.write_set(paths, "s", {"a.png": "a cat"}, {})
    row = cs.set_stats(paths, "s")
    assert row[4] == "" and row[6] == ""                   # No budgets / no trigger: blank columns


def test_set_stats_verifier_only_counts_matching_hash(ds):
    d, paths = ds
    cs.write_set(paths, "s", {"a.png": "a cat", "b.png": "a dog"}, {})

    def h(t):
        return hashlib.sha256(t.strip().encode("utf-8")).hexdigest()[:16]
    audit.save_verification(paths["a.png"], {"caption_hash": h("a cat"), "disputed": ["a cat"]})
    audit.save_verification(paths["b.png"], {"caption_hash": h("other text"), "disputed": ["x"]})
    assert cs.set_stats(paths, "s")[9] == "1/1"
    audit.save_verification(paths["b.png"], {"caption_hash": h("a dog"), "disputed": []})
    assert cs.set_stats(paths, "s")[9] == "1/2"


def test_length_histogram_bins(ds):
    d, paths = ds
    texts = {"a.png": "cat", "b.png": " ".join(["word"] * 30), "c.png": " ".join(["word"] * 300)}
    cs.write_set(paths, "s", texts, {})
    rows = cs.length_histogram(paths, "s")
    labels = [r[0] for r in rows]
    assert labels == ["≤10", "11–20", "21–40", "41–75", "76–150", "151–225", ">225"]
    counts = dict(rows)
    assert counts["≤10"] == 1 and counts["21–40"] == 1 and counts[">225"] == 1 and sum(counts.values()) == 3
    assert sum(c for _, c in cs.length_histogram(paths, "missing")) == 0


# ---------- derived strategies ----------

NAMES = [f"{i:02d}.png" for i in range(20)]
PATHS = {n: f"/x/{n}" for n in NAMES}


def test_derive_trigger_only():
    out = cs.derive("trigger_only", PATHS, {}, {}, "txcl painting, ")
    assert set(out) == set(NAMES) and set(out.values()) == {"txcl painting"}
    with pytest.raises(cs.CaptionSetError, match="trigger"):
        cs.derive("trigger_only", PATHS, {}, {}, " , ")


def test_derive_random_length_deterministic():
    short = {n: f"short {n}" for n in NAMES}
    long_ = {n: f"long {n}" for n in NAMES}
    a = cs.derive("random_length", PATHS, {}, {"short_prose": short, "long_prose": long_}, "txcl, ", seed=1)
    b = cs.derive("random_length", PATHS, {}, {"short_prose": short, "long_prose": long_}, "txcl, ", seed=1)
    c = cs.derive("random_length", PATHS, {}, {"short_prose": short, "long_prose": long_}, "txcl, ", seed=2)
    assert a == b and a != c and set(a) == set(NAMES)
    assert all(v in (short[k], long_[k], "txcl") for k, v in a.items())
    assert len(set(v.split()[0] for v in a.values())) == 3          # All three lengths occur
    with pytest.raises(cs.CaptionSetError, match="short and long"):
        cs.derive("random_length", PATHS, {}, {"short_prose": short}, "txcl")


def test_derive_random_length_without_trigger_uses_two_choices():
    out = cs.derive("random_length", PATHS, {}, {"short_prose": {n: "s" for n in NAMES}, "long_prose": {n: "l" for n in NAMES}}, "")
    assert set(out.values()) <= {"s", "l"}


def test_derive_long_plus_trigger():
    long_ = {"00.png": "A long caption.\nold variant", "01.png": "Another."}
    out = cs.derive("long_plus_trigger", PATHS, {}, {"long_prose": long_}, "txcl, ")
    assert out == {"00.png": "A long caption.\ntxcl", "01.png": "Another.\ntxcl"}
    with pytest.raises(cs.CaptionSetError):
        cs.derive("long_plus_trigger", PATHS, {}, {"long_prose": long_}, "")
    with pytest.raises(cs.CaptionSetError):
        cs.derive("long_plus_trigger", PATHS, {}, {}, "txcl")


def test_derive_blend_keeps_about_five_percent_of_active():
    active = {n: f"human {n}" for n in NAMES}
    model = {n: f"model {n}" for n in NAMES}
    out = cs.derive("blend_95_5", PATHS, active, {"long_prose": model}, "", seed=3)
    human = [k for k, v in out.items() if v.startswith("human")]
    assert len(out) == 20 and len(human) == 1
    assert out == cs.derive("blend_95_5", PATHS, active, {"long_prose": model}, "", seed=3)
    # Falls back to short prose or tags
    assert len(cs.derive("blend_95_5", PATHS, active, {"tags": model}, "")) == 20
    with pytest.raises(cs.CaptionSetError, match="model-written"):
        cs.derive("blend_95_5", PATHS, active, {}, "")


def test_derive_blend_large_dataset_rounds():
    names = [f"{i:03d}.png" for i in range(100)]
    paths = {n: n for n in names}
    out = cs.derive("blend_95_5", paths, {n: "h" for n in names}, {"short_prose": {n: "m" for n in names}}, "")
    assert list(out.values()).count("h") == 5


def test_derive_quality_on():
    active = {"00.png": "txcl, 1girl, smile\nvariant", "01.png": "1girl", "02.png": "txcl",
              "03.png": "txcl, score_9, score_8_up, x", "04.png": "  "}
    q = "score_9, score_8_up"
    out = cs.derive("quality_on", PATHS, active, {}, "txcl, ", q)
    assert out["00.png"] == "txcl, score_9, score_8_up, 1girl, smile\nvariant"
    assert out["01.png"] == "score_9, score_8_up, 1girl"
    assert out["02.png"] == "txcl, score_9, score_8_up"
    assert out["03.png"] == active["03.png"]                 # Already has them
    assert "04.png" not in out
    with pytest.raises(cs.CaptionSetError, match="quality tags"):
        cs.derive("quality_on", PATHS, active, {}, "txcl", "")


def test_derive_unknown_strategy():
    with pytest.raises(cs.CaptionSetError):
        cs.derive("short_prose", PATHS, {}, {}, "")


def test_strategy_tables_consistent():
    assert set(cs.STRATEGY_LABELS.values()) == set(cs.STRATEGIES)
    assert set(cs.WORD_TARGETS) <= {k for k, v in cs.STRATEGIES.items() if v[1] == "model"}


# ---------- materialize ----------

def test_materialize_lines(ds):
    d, paths = ds
    cs.write_set(paths, "s", {"a.png": "one\n\n---\n\ntwo", "b.png": "single"}, {})
    out, n = cs.materialize(paths, "s", "lines")
    assert n == 2 and out == str(d / ".artagents" / "export" / "s-lines")
    assert sorted(os.listdir(out)) == ["a.png", "a.txt", "b.png", "b.txt"]
    assert open(os.path.join(out, "a.txt"), encoding="utf-8").read() == "one\ntwo"


def test_materialize_kohya_multiline(ds):
    d, paths = ds
    cs.write_set(paths, "s", {"a.png": "one\ntwo\nthree"}, {})
    out, _ = cs.materialize(paths, "s", "kohya_multiline")
    text = open(os.path.join(out, "a.txt"), encoding="utf-8").read()
    assert text == "one\n---===---\ntwo\n---===---\nthree"
    assert all(line.startswith("---===---") for line in text.splitlines()[1::2])


def test_materialize_duplicate(ds):
    d, paths = ds
    cs.write_set(paths, "s", {"a.png": "one\ntwo\nthree", "b.png": "solo"}, {})
    out, n = cs.materialize(paths, "s", "duplicate")
    assert n == 4
    assert sorted(os.listdir(out)) == ["a.png", "a.txt", "a__v2.png", "a__v2.txt", "a__v3.png", "a__v3.txt", "b.png", "b.txt"]
    assert open(os.path.join(out, "a__v3.txt"), encoding="utf-8").read() == "three"


def test_materialize_hardlinks_or_copies(ds):
    d, paths = ds
    cs.write_set(paths, "s", {"a.png": "x"}, {})
    out, _ = cs.materialize(paths, "s", "lines")
    assert os.path.samefile(os.path.join(out, "a.png"), paths["a.png"])            # Hard link here
    with patch("core.lab.caption_sets.os.link", side_effect=OSError("cross-device")):
        out, _ = cs.materialize(paths, "s", "lines")
    assert not os.path.samefile(os.path.join(out, "a.png"), paths["a.png"])
    assert open(os.path.join(out, "a.png"), "rb").read() == open(paths["a.png"], "rb").read()


def test_materialize_rerun_replaces_folder_and_active(ds):
    d, paths = ds
    cs.write_set(paths, "s", {"a.png": "x", "b.png": "y"}, {})
    out, _ = cs.materialize(paths, "s", "lines")
    (d / "b.s.txt").unlink()
    out2, n = cs.materialize(paths, "s", "lines")
    assert out2 == out and n == 1 and sorted(os.listdir(out)) == ["a.png", "a.txt"]
    _caps(d, a="active a")
    out3, _ = cs.materialize(paths, cs.ACTIVE, "lines")
    assert out3.endswith(os.path.join("export", "active-lines"))
    with pytest.raises(cs.CaptionSetError):
        cs.materialize(paths, "missing", "lines")
