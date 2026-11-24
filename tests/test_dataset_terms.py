# ArtAgent/tests/test_dataset_terms.py
"""Tests for core/dataset/terms.py (term statistics, binding-risk notes, bulk tag edits)."""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import terms as t
from core.dataset.terms import MODE_PHRASES, MODE_TAGS, MODE_WORDS


# --- caption_terms ---

def test_caption_terms_tags_normalized_across_variants():
    caption = "1girl, Long_Hair, smile\nlong hair, outdoors"
    assert t.caption_terms(caption, MODE_TAGS) == {"1girl", "long hair", "smile", "outdoors"}


def test_caption_terms_words_skip_stopwords_short_words_and_numbers():
    caption = "This image shows a red umbrella in the rain; 3 cats and an ox."
    assert t.caption_terms(caption, MODE_WORDS) == {"red", "umbrella", "rain", "cats"}


def test_caption_terms_phrases():
    caption = "A red umbrella in heavy rain near wooden benches"
    # Pairs with a stopword ('in', 'near') or a short word are skipped
    assert t.caption_terms(caption, MODE_PHRASES) == {"red umbrella", "heavy rain", "wooden benches"}


def test_caption_terms_phrases_do_not_cross_punctuation():
    caption = "Heavy rain. Red umbrella, wooden benches"
    assert t.caption_terms(caption, MODE_PHRASES) == {"heavy rain", "red umbrella", "wooden benches"}


def test_caption_terms_empty():
    assert t.caption_terms("", MODE_TAGS) == set()
    assert t.caption_terms("", MODE_WORDS) == set()


# --- term_table ---

def _captions(n, **specials):
    caps = {f"img{i}.png": f"base{i}" for i in range(n)}
    caps.update(specials)
    return caps


def test_term_table_counts_per_image_and_sorts():
    captions = {"a.png": "cat, sofa, cat", "b.png": "cat, window", "c.png": "dog"}
    rows = t.term_table(captions, MODE_TAGS)
    assert rows[0] == ["cat", 2, 66.7, ""] # Counted once per image; < 4 images: no binding note
    assert [r[0] for r in rows] == ["cat", "dog", "sofa", "window"] # Ties sorted alphabetically


def test_term_table_flags_over_represented_terms():
    captions = {f"{i}.png": "txcl, red scarf, park" if i < 3 else f"txcl, beach{i}" for i in range(5)}
    rows = {r[0]: r for r in t.term_table(captions, MODE_TAGS, ["txcl", "painting"])}
    assert rows["txcl"][3] == "trigger/class" # Protected, never a binding risk
    assert rows["red scarf"][1:] == [3, 60.0, "⚠ over-represented: may bind to the trigger"]
    assert rows["beach3"][3] == "" # Singleton note only from 10 images up


def test_term_table_singletons_need_ten_images():
    captions = {f"{i}.png": f"common, rare{i}" if i == 0 else "common" for i in range(10)}
    rows = {r[0]: r for r in t.term_table(captions, MODE_TAGS)}
    assert rows["rare0"][3] == "singleton"
    assert rows["common"][3].startswith("⚠")


def test_term_table_shows_original_tag_spelling():
    captions = {"a.png": "red_hair, smile", "b.png": "red_hair", "c.png": "red hair"}
    rows = t.term_table(captions, MODE_TAGS)
    assert rows[0][:2] == ["red_hair", 3] # Most common spelling shown, counted together


def test_term_table_protected_matches_normalized():
    rows = t.term_table({f"{i}": "Ohwx_Style" for i in range(4)}, MODE_TAGS, ["ohwx style", ""])
    assert rows[0][3] == "trigger/class"


def test_term_table_empty_and_limit():
    assert t.term_table({}, MODE_TAGS) == []
    captions = {"a.png": ", ".join(f"tag{i}" for i in range(20))}
    assert len(t.term_table(captions, MODE_TAGS, limit=5)) == 5


def test_term_table_handles_none_captions():
    assert t.term_table({"a.png": None, "b.png": "cat"}, MODE_TAGS) == [["cat", 1, 50.0, ""]]


# --- files_with_term / common_terms ---

def test_files_with_term_and_common_terms():
    captions = {"b.png": "cat, sofa", "a.png": "Cat, window", "c.png": "dog, sofa"}
    assert t.files_with_term(captions, "cat", MODE_TAGS) == ["a.png", "b.png"]
    assert t.files_with_term(captions, "CAT", MODE_TAGS) == ["a.png", "b.png"]
    assert t.files_with_term({"a.png": "A red umbrella."}, "Umbrella ", MODE_WORDS) == ["a.png"]
    assert t.common_terms(captions, ["b.png", "c.png"], MODE_TAGS) == ["sofa"]
    assert t.common_terms(captions, [], MODE_TAGS) == []


# --- rename / delete / replace ---

@pytest.mark.parametrize("caption, old, new, expected", [
    ("1girl, long_hair, smile", "long hair", "very long hair", "1girl, very long hair, smile"),
    ("1girl, long_hair, very long hair", "long_hair", "very long hair", "1girl, very long hair"), # Duplicate removed
    ("a, b\nb, c", "b", "x", "a, x\nx, c"), # Every variant line
    ("a, b", "z", "x", "a, b"),
])
def test_rename_tag(caption, old, new, expected):
    assert t.rename_tag(caption, old, new) == expected


@pytest.mark.parametrize("caption, tag, expected", [
    ("1girl, watercolor, smile", "Watercolor", "1girl, smile"),
    ("1girl, ,smile", "x", "1girl, smile"), # Empty tags tidied
    ("watercolor", "watercolor", ""),
    ("a, w\nw, b", "w", "a\nb"),
])
def test_delete_tag(caption, tag, expected):
    assert t.delete_tag(caption, tag) == expected


def test_delete_tag_drops_lines_that_become_empty():
    assert t.delete_tag("ohwx\n1girl, ohwx", "ohwx") == "1girl"


@pytest.mark.parametrize("caption, find, replace, regex, expected", [
    ("A cat on a cat-tree.", "cat", "dog", False, "A dog on a dog-tree."), # Whole word ('-' is not a word char)
    ("A category of cats.", "cat", "dog", False, "A category of cats."), # Not inside words
    ("Watercolor, lake, watercolor style", "watercolor", "", False, "lake, style"), # Tidies empty tags/commas
    ("in the style of x", "style of x", "", False, "in the"),
    ("score_9, score_8_up", r"score_\d(_up)?", "q", True, "q, q"),
    ("A red\nB red", "red", "blue", False, "A blue\nB blue"),
    ("unchanged", "", "x", False, "unchanged"),
])
def test_replace_text(caption, find, replace, regex, expected):
    assert t.replace_text(caption, find, replace, regex) == expected


def test_replace_text_case_sensitive():
    assert t.replace_text("Cat cat", "cat", "dog", False, case_sensitive=True) == "Cat dog"


def test_replace_text_drops_lines_emptied_by_replace():
    assert t.replace_text("ohwx\na cat", "ohwx", "", False) == "a cat"


def test_replace_text_invalid_regex_raises():
    with pytest.raises(re.error):
        t.replace_text("x", "([", "y", True)
