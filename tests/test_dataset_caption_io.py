# ArtAgent/tests/test_dataset_caption_io.py
"""Tests for core/dataset/caption_io.py (caption files, one caption per line, Append/Prepend/variants)."""
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import caption_io as cio
from core.dataset.caption_io import (
    MODE_APPEND, MODE_OVERWRITE, MODE_PREPEND, MODE_SKIP, MODE_VARIANT,
    caption_path, combine, join_on_line, needs_normalizing, normalize, read_caption,
    single_line, split_variants, variant_count, write_caption, write_text_atomic,
)


def _tmp_leftovers(folder):
    return [n for n in os.listdir(folder) if n.startswith(".tmp-")]


# --- paths, read / write ---

def test_caption_path_replaces_extension():
    assert caption_path(os.path.join("d", "img.final.png")) == os.path.join("d", "img.final.txt")
    assert caption_path(os.path.join("d", "img.jpg"), ".caption") == os.path.join("d", "img.caption")


def test_write_and_read_caption_roundtrip(tmp_path):
    image = tmp_path / "a.png"
    path = write_caption(str(image), "ünïcode caption, ok")
    assert path == str(tmp_path / "a.txt")
    assert read_caption(str(image)) == "ünïcode caption, ok"
    assert _tmp_leftovers(tmp_path) == []


def test_read_caption_missing_returns_empty(tmp_path):
    assert read_caption(str(tmp_path / "none.png")) == ""


def test_write_text_atomic_uses_lf_newlines(tmp_path):
    path = tmp_path / "c.txt"
    write_text_atomic(str(path), "line one\nline two")
    assert path.read_bytes() == b"line one\nline two"


def test_write_text_atomic_replaces_existing_file(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("old", encoding="utf-8")
    write_text_atomic(str(path), "new")
    assert path.read_text(encoding="utf-8") == "new"


def test_write_text_atomic_failure_keeps_old_file_and_cleans_temp(tmp_path):
    """A failed write never leaves half a caption or a stray temp file behind."""
    path = tmp_path / "c.txt"
    path.write_text("original", encoding="utf-8")
    with patch("core.dataset.caption_io.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            write_text_atomic(str(path), "new text")
    assert path.read_text(encoding="utf-8") == "original"
    assert _tmp_leftovers(tmp_path) == []


def test_write_caption_into_missing_folder_raises(tmp_path):
    with pytest.raises(OSError):
        write_caption(str(tmp_path / "missing" / "a.png"), "x")


# --- single_line ---

@pytest.mark.parametrize("raw, joiner, expected", [
    ("", " ", ""),
    ("A cat.\nOn a sofa.", " ", "A cat. On a sofa."),
    ("1girl\nsolo\nsmile", ", ", "1girl, solo, smile"),
    ("1girl,\nsolo", ", ", "1girl, solo"), # Line already ends with the separator: not doubled
    ("# Caption\n- **A cat**\n* on a sofa\n1. sleeping", " ", "Caption A cat on a sofa sleeping"),
    ("first\n\n---\n\nsecond", " ", "first second"), # Old separator lines dropped
    ("  lots   of\t spaces  ", " ", "lots of spaces"),
    ("a\r\nb", " ", "a b"),
    ("a\nb", "", "a b"), # Empty joiner falls back to a space
])
def test_single_line(raw, joiner, expected):
    assert single_line(raw, joiner) == expected


# --- split_variants / variant_count ---

@pytest.mark.parametrize("text, expected", [
    ("", []),
    ("one caption", ["one caption"]),
    ("a\nb\n", ["a", "b"]),
    ("a\n\n---\n\nb", ["a", "b"]), # Old Append separator
    ("a\n===\nb\n***\nc\n___\nd", ["a", "b", "c", "d"]),
    ("a\n---===---\nb", ["a", "b"]), # kohya multiline separator
    ("  padded  \r\n\r\nnext", ["padded", "next"]),
])
def test_split_variants(text, expected):
    assert split_variants(text) == expected
    assert variant_count(text) == len(expected)


# --- join_on_line ---

@pytest.mark.parametrize("first, second, joiner, expected", [
    ("a", "b", ", ", "a, b"),
    ("a", "b", " ", "a b"),
    ("a.", "b", ". ", "a. b"), # Separator already there: not doubled
    ("a,", "b", ", ", "a, b"),
    ("", "b", ", ", "b"),
    ("a", "", ", ", "a"),
    ("  a  ", "  b ", " ", "a b"),
])
def test_join_on_line(first, second, joiner, expected):
    assert join_on_line(first, second, joiner) == expected


def test_join_on_line_second_starts_with_separator():
    assert join_on_line("a", ", b", ", ") == "a, b"


# --- combine ---

@pytest.mark.parametrize("existing, new, mode, joiner, expected", [
    ("old caption", "new part", MODE_APPEND, ", ", "old caption, new part"),
    ("old caption", "new part", MODE_PREPEND, ", ", "new part, old caption"),
    ("old caption", "new part", MODE_APPEND, " ", "old caption new part"),
    ("old caption", "new part", MODE_OVERWRITE, ", ", "new part"),
    ("old caption", "new part", MODE_SKIP, ", ", "old caption"),
    ("old caption", "new part", MODE_VARIANT, ", ", "old caption\nnew part"),
    ("", "new part", MODE_APPEND, ", ", "new part"), # Nothing to append to
    ("", "new part", MODE_PREPEND, ", ", "new part"),
    ("", "new part", MODE_VARIANT, ", ", "new part"),
    ("", "new part", MODE_SKIP, ", ", "new part"), # No existing caption: written even in Skip
    ("v1\nv2", "x", MODE_APPEND, ", ", "v1, x\nv2, x"), # Every variant line gets the text
    ("v1\nv2", "x", MODE_PREPEND, " ", "x v1\nx v2"),
    ("a\n\n---\n\nb", "c", MODE_VARIANT, ", ", "a\nb\nc"), # Old separators cleaned
    ("a\nb", "b", MODE_VARIANT, ", ", "a\nb"), # Duplicate variant not added twice
    ("old", "multi\nline\noutput", MODE_APPEND, ", ", "old, multi, line, output"), # New text kept on one line
])
def test_combine(existing, new, mode, joiner, expected):
    assert combine(existing, new, mode, joiner) == expected


def test_combine_never_writes_old_separator():
    for mode in (MODE_APPEND, MODE_PREPEND, MODE_VARIANT):
        result = combine("first caption", "second caption", mode, " ")
        assert "---" not in result
        assert "\n\n" not in result


def test_combine_default_joiner_is_comma():
    assert cio.DEFAULT_JOINER == ", "
    assert combine("a", "b", MODE_APPEND) == "a, b"


def test_combine_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown caption file mode"):
        combine("a", "b", "Sideways")


def test_file_modes_and_joiner_choices():
    assert cio.FILE_MODES == [MODE_OVERWRITE, MODE_SKIP, MODE_APPEND, MODE_PREPEND, MODE_VARIANT]
    assert set(cio.JOINER_CHOICES.values()) == {", ", " ", ". "}


# --- needs_normalizing / normalize ---

@pytest.mark.parametrize("text, expected", [
    ("", False),
    ("one line", False),
    ("one line\n", False), # Trailing newline is fine
    ("v1\nv2", False), # Clean variants
    ("a\n\n---\n\nb", True), # Old separator
    ("a\n---===---\nb", True),
    ("a\n\nb", True), # Blank line between captions
])
def test_needs_normalizing(text, expected):
    assert needs_normalizing(text) is expected


def test_normalize_merge_and_variants():
    assert normalize("a\n\n---\n\nb", "merge", ", ") == "a, b"
    assert normalize("a\n\n---\n\nb", "merge", " ") == "a b"
    assert normalize("a\n\n---\n\nb\n\na", "variants") == "a\nb" # Deduplicated, order kept
    assert normalize("", "merge") == ""


def test_normalize_unknown_mode_raises():
    with pytest.raises(ValueError):
        normalize("a", "explode")


def test_normalize_merge_drops_duplicate_tags():
    # 'Add as caption variant' + merge must not repeat the trigger tag
    assert normalize("zkx painting, 1girl, solo\nzkx painting", "merge", ", ") == "zkx painting, 1girl, solo"
    assert normalize("Red Hair, smile\nred hair, hat", "merge", ", ") == "Red Hair, smile, hat"
    assert normalize("one. \ntwo.", "merge", ". ") == "one. two."  # Prose is never de-duplicated
