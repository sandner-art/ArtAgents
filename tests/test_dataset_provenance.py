# ArtAgent/tests/test_dataset_provenance.py
"""Tests for core/dataset/provenance.py (which model/prompt/settings produced each caption)."""
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import provenance as pv

STRFTIME_PATH = "core.dataset.provenance.time.strftime"


def test_provenance_path_is_hidden_next_to_dataset(tmp_path):
    image = tmp_path / "photo.final.png"
    assert pv.provenance_path(str(image)) == str(tmp_path / ".artagents" / "provenance" / "photo.final.json")


def test_prompt_hash_is_stable_and_short():
    h = pv.prompt_hash("Describe the image.")
    assert h == pv.prompt_hash("Describe the image.")
    assert len(h) == 16 and all(c in "0123456789abcdef" for c in h)
    assert h != pv.prompt_hash("Describe the image!")
    assert pv.prompt_hash("a", "b") != pv.prompt_hash("ab") # Parts are separated
    assert pv.prompt_hash(None) == pv.prompt_hash("")


def test_write_and_read_provenance(tmp_path):
    image = str(tmp_path / "a.png")
    with patch(STRFTIME_PATH, return_value="2026-11-24 12:00:00"):
        path = pv.write_provenance(image, {"model": "gemma", "options": {"seed": 1}})
    assert path == pv.provenance_path(image)
    data = pv.read_provenance(image)
    assert data == {"image": "a.png", "latest": {"model": "gemma", "options": {"seed": 1}, "timestamp": "2026-11-24 12:00:00"},
                    "history": []}
    assert not [n for n in os.listdir(os.path.dirname(path)) if n.endswith(".tmp")]


def test_write_provenance_keeps_explicit_timestamp_and_does_not_mutate_input(tmp_path):
    record = {"model": "m", "timestamp": "fixed"}
    pv.write_provenance(str(tmp_path / "a.png"), record)
    assert pv.read_provenance(str(tmp_path / "a.png"))["latest"]["timestamp"] == "fixed"
    record2 = {"model": "m"}
    pv.write_provenance(str(tmp_path / "b.png"), record2)
    assert record2 == {"model": "m"}


def test_history_keeps_previous_records_newest_first_and_is_capped(tmp_path):
    image = str(tmp_path / "a.png")
    for i in range(pv.HISTORY_LENGTH + 3):
        pv.write_provenance(image, {"model": f"m{i}", "timestamp": str(i)})
    data = pv.read_provenance(image)
    last = pv.HISTORY_LENGTH + 2
    assert data["latest"]["model"] == f"m{last}"
    assert len(data["history"]) == pv.HISTORY_LENGTH
    assert [h["model"] for h in data["history"][:3]] == [f"m{last - 1}", f"m{last - 2}", f"m{last - 3}"]


def test_read_provenance_missing_or_corrupt(tmp_path):
    image = str(tmp_path / "a.png")
    assert pv.read_provenance(image) == {}
    path = pv.provenance_path(image)
    os.makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("{broken")
    assert pv.read_provenance(image) == {}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(["not", "a", "dict"], f)
    assert pv.read_provenance(image) == {}


def test_corrupt_file_is_replaced_on_next_write(tmp_path):
    image = str(tmp_path / "a.png")
    path = pv.provenance_path(image)
    os.makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("{broken")
    pv.write_provenance(image, {"model": "m"})
    assert pv.read_provenance(image)["latest"]["model"] == "m"


def test_summarize(tmp_path):
    image = str(tmp_path / "a.png")
    assert pv.summarize(image) == ""
    pv.write_provenance(image, {"model": "gemma4:e4b", "agent": "(Direct Agent Call)", "timestamp": "2026-11-24 12:00",
                                "retries": 0, "issues": []})
    assert pv.summarize(image) == "gemma4:e4b · (Direct Agent Call) · 2026-11-24 12:00"
    pv.write_provenance(image, {"model": "m", "agent": "", "timestamp": "t", "retries": 1, "issues": ["meta opener"]})
    assert pv.summarize(image) == "m · t · 1 retry · issues: meta opener"
    pv.write_provenance(image, {"timestamp": "t", "retries": 2})
    assert pv.summarize(image) == "? · t · 2 retries"


def test_summarize_flags_edited_and_removed_captions(tmp_path):
    image = str(tmp_path / "x.png")
    pv.write_provenance(image, {"model": "m", "agent": "a", "caption_hash": pv.text_hash("a, b\n")})
    assert "edited since" not in pv.summarize(image, "a, b")  # Trailing whitespace is ignored
    assert pv.summarize(image, "a, b, c").endswith("edited since")
    assert pv.summarize(image, "") == ""  # Caption gone (e.g. restored to before the AI run)
    assert pv.summarize(image).startswith("m · a")  # No current text: plain summary
