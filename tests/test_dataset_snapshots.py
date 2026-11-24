# ArtAgent/tests/test_dataset_snapshots.py
"""Tests for core/dataset/snapshots.py (automatic caption snapshots = undo for batch operations)."""
import json
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import snapshots as s

STRFTIME_PATH = "core.dataset.snapshots.time.strftime"


def _fixed_time(stamp="20261124-120000", created="2026-11-24 12:00:00"):
    return lambda fmt, *a: stamp if fmt == "%Y%m%d-%H%M%S" else created


@pytest.fixture
def dataset(tmp_path):
    """Three images: a and b captioned, c without a caption."""
    for name in ("a", "b", "c"):
        (tmp_path / f"{name}.png").write_bytes(b"png")
    (tmp_path / "a.txt").write_text("caption a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("caption b\nvariant b", encoding="utf-8")
    paths = {f"{n}.png": str(tmp_path / f"{n}.png") for n in ("a", "b", "c")}
    return tmp_path, paths


def test_create_snapshot_copies_captions_and_records_missing(dataset):
    folder, paths = dataset
    with patch(STRFTIME_PATH, side_effect=_fixed_time()):
        snap_id = s.create_snapshot(str(folder), paths, "Before Batch Append!")
    assert snap_id == "20261124-120000-before-batch-append"
    snap_dir = folder / ".artagents" / "snapshots" / snap_id
    assert (snap_dir / "a.txt").read_text(encoding="utf-8") == "caption a"
    assert (snap_dir / "b.txt").read_text(encoding="utf-8") == "caption b\nvariant b"
    assert not (snap_dir / "c.txt").exists()
    manifest = json.loads((snap_dir / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(manifest.pop("created_ns"), int)  # Orders snapshots made in the same second
    assert manifest == {"id": snap_id, "operation": "Before Batch Append!", "created": "2026-11-24 12:00:00",
                        "files": {"a.png": "a.txt", "b.png": "b.txt", "c.png": None}}


def test_snapshot_ids_are_unique_within_one_second(dataset):
    folder, paths = dataset
    with patch(STRFTIME_PATH, side_effect=_fixed_time()):
        ids = [s.create_snapshot(str(folder), paths, "op") for _ in range(3)]
    assert ids == ["20261124-120000-op", "20261124-120000-op-2", "20261124-120000-op-3"]


def test_snapshot_slug_fallback():
    assert s._slug("") == "snapshot"
    assert s._slug("!!!") == "snapshot"
    assert s._slug("Before restore of 20261124-x") == "before-restore-of-20261124-x"
    assert len(s._slug("x" * 100)) == 40


def test_list_snapshots_newest_first_with_counts(dataset):
    folder, paths = dataset
    with patch(STRFTIME_PATH, side_effect=_fixed_time("20261124-120000", "2026-11-24 12:00:00")):
        first = s.create_snapshot(str(folder), paths, "first")
    with patch(STRFTIME_PATH, side_effect=_fixed_time("20261124-120005", "2026-11-24 12:00:05")):
        second = s.create_snapshot(str(folder), {"a.png": paths["a.png"]}, "second")
    listed = s.list_snapshots(str(folder))
    assert [x["id"] for x in listed] == [second, first]
    assert listed[0] == {"id": second, "operation": "second", "created": "2026-11-24 12:00:05", "count": 1}
    assert listed[1]["count"] == 2 # c.png had no caption


def test_list_snapshots_same_second_keeps_creation_order(dataset):
    folder, paths = dataset
    with patch(STRFTIME_PATH, side_effect=_fixed_time()):
        s.create_snapshot(str(folder), paths, "zzz first")
        newest = s.create_snapshot(str(folder), paths, "aaa second")
    assert s.list_snapshots(str(folder))[0]["id"] == newest


def test_list_snapshots_skips_broken_entries(dataset):
    folder, paths = dataset
    good = s.create_snapshot(str(folder), paths, "good")
    root = folder / ".artagents" / "snapshots"
    (root / "no-manifest").mkdir()
    (root / "bad-json").mkdir()
    (root / "bad-json" / "manifest.json").write_text("{not json", encoding="utf-8")
    assert [x["id"] for x in s.list_snapshots(str(folder))] == [good]


def test_list_snapshots_without_folder(tmp_path):
    assert s.list_snapshots(str(tmp_path)) == []


def test_prune_keeps_newest(dataset):
    folder, paths = dataset
    for i in range(5):
        with patch(STRFTIME_PATH, side_effect=_fixed_time(f"20261124-12000{i}")):
            s.create_snapshot(str(folder), paths, f"op{i}")
    s.prune_snapshots(str(folder), keep=2)
    assert [x["operation"] for x in s.list_snapshots(str(folder))] == ["op4", "op3"]


def test_create_snapshot_prunes(dataset):
    folder, paths = dataset
    with patch.object(s, "prune_snapshots") as mock_prune:
        s.create_snapshot(str(folder), paths, "op")
    mock_prune.assert_called_once_with(str(folder))
    assert s.MAX_SNAPSHOTS == 50


def test_snapshot_captions(dataset):
    folder, paths = dataset
    snap_id = s.create_snapshot(str(folder), paths, "op")
    assert s.snapshot_captions(str(folder), snap_id) == {"a.png": "caption a", "b.png": "caption b\nvariant b", "c.png": None}


@pytest.mark.parametrize("bad_id", ["", "../outside", "a/b", os.path.join("a", "b"), ".hidden", ".."])
def test_invalid_snapshot_ids_are_rejected(dataset, bad_id):
    folder, _ = dataset
    with pytest.raises(ValueError, match="Invalid snapshot id"):
        s.snapshot_captions(str(folder), bad_id)


def test_unknown_snapshot_id_raises_oserror(dataset):
    folder, _ = dataset
    with pytest.raises(OSError):
        s.snapshot_captions(str(folder), "20990101-000000-nothing")


def test_restore_snapshot_is_exact(dataset):
    folder, paths = dataset
    snap_id = s.create_snapshot(str(folder), paths, "before edit")
    (folder / "a.txt").write_text("edited a", encoding="utf-8")
    (folder / "c.txt").write_text("new caption c", encoding="utf-8") # Did not exist at snapshot time
    # b unchanged

    safety_id, changed = s.restore_snapshot(str(folder), snap_id, paths)

    assert changed == 2
    assert (folder / "a.txt").read_text(encoding="utf-8") == "caption a"
    assert (folder / "b.txt").read_text(encoding="utf-8") == "caption b\nvariant b"
    assert not (folder / "c.txt").exists() # Removed again
    # The state before the restore was saved, so the restore itself can be undone
    assert s.snapshot_captions(str(folder), safety_id) == {"a.png": "edited a", "b.png": "caption b\nvariant b",
                                                           "c.png": "new caption c"}
    assert s.list_snapshots(str(folder))[0]["operation"] == f"before restore of {snap_id}"
    assert not [n for n in os.listdir(folder) if n.startswith(".tmp-")]


def test_restore_recreates_deleted_caption(dataset):
    folder, paths = dataset
    snap_id = s.create_snapshot(str(folder), paths, "op")
    os.remove(folder / "a.txt")
    _, changed = s.restore_snapshot(str(folder), snap_id, paths)
    assert changed == 1
    assert (folder / "a.txt").read_text(encoding="utf-8") == "caption a"


def test_restore_only_touches_loaded_images(dataset):
    folder, paths = dataset
    snap_id = s.create_snapshot(str(folder), paths, "op")
    (folder / "a.txt").write_text("edited a", encoding="utf-8")
    (folder / "b.txt").write_text("edited b", encoding="utf-8")
    _, changed = s.restore_snapshot(str(folder), snap_id, {"a.png": paths["a.png"], "z.png": str(folder / "z.png")})
    assert changed == 1
    assert (folder / "a.txt").read_text(encoding="utf-8") == "caption a"
    assert (folder / "b.txt").read_text(encoding="utf-8") == "edited b" # Not loaded: untouched


def test_diff_snapshot(dataset):
    folder, paths = dataset
    snap_id = s.create_snapshot(str(folder), paths, "op")
    same = {"a.png": "caption a", "b.png": "caption b\nvariant b", "c.png": ""}
    assert s.diff_snapshot(str(folder), snap_id, same) == (0, "No differences: the current captions match this snapshot.")

    changed, md = s.diff_snapshot(str(folder), snap_id, dict(same, **{"a.png": "caption a, ohwx", "c.png": "new"}))
    assert changed == 2
    assert md.startswith("2 caption file(s) differ.")
    assert "**a.png**" in md and "-caption a" in md and "+caption a, ohwx" in md
    assert "**c.png**" in md and "+new" in md
    assert "**b.png**" not in md


def test_diff_snapshot_limits_listed_files(dataset):
    folder, paths = dataset
    snap_id = s.create_snapshot(str(folder), paths, "op")
    changed, md = s.diff_snapshot(str(folder), snap_id, {"a.png": "x", "b.png": "y", "c.png": "z"}, max_files=1)
    assert changed == 3
    assert md.count("```diff") == 1
    assert md.endswith("…and 2 more file(s).")
