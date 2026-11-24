# ArtAgent/tests/test_dataset_guardrails.py
"""Tests for core/dataset/guardrails.py (cleaning and checking AI-written captions)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import guardrails as g


# --- clean_caption ---

@pytest.mark.parametrize("raw, style, expected", [
    ("A cat on a sofa.", "prose", "A cat on a sofa."),
    ("A cat.\nOn a sofa.", "prose", "A cat. On a sofa."),
    ("Caption: A cat on a sofa.", "prose", "A cat on a sofa."),
    ("description:  \"A cat on a sofa.\"", "prose", "A cat on a sofa."),
    ("'quoted'", "prose", "quoted"),
    ("- **A cat**\n- on a sofa", "prose", "A cat on a sofa"),
    ("1girl, solo, smile", "tags", "1girl, solo, smile"),
    ("1girl\nsolo\nsmile.", "tags", "1girl, solo, smile"),
    ("Tags: 1girl, Long_Hair, long hair, solo, 1girl", "tags", "1girl, Long_Hair, solo"), # Duplicates (incl. _ vs space) removed
    ("1girl, , solo,", "tags", "1girl, solo"),
    ("", "prose", ""),
])
def test_clean_caption(raw, style, expected):
    assert g.clean_caption(raw, style) == expected


def test_normalize_tag():
    assert g.normalize_tag("  Long_Hair ") == "long hair"
    assert g.normalize_tag("blue   eyes") == "blue eyes"
    assert g.normalize_tag(None) == ""


# --- check_caption ---

@pytest.mark.parametrize("text, style, max_tokens, omit, expected", [
    ("A cat sleeps on a red sofa.", "prose", 150, None, []),
    ("", "prose", 150, None, [g.ISSUE_EMPTY]),
    ("   ", "tags", 150, None, [g.ISSUE_EMPTY]),
    ("I'm sorry, but I can't describe this image.", "prose", 150, None, [g.ISSUE_REFUSAL]),
    ("As an AI I cannot see images.", "prose", 150, None, [g.ISSUE_REFUSAL]),
    ("This image shows a cat on a sofa.", "prose", 150, None, [g.ISSUE_META_OPENER]),
    ("The photograph depicts a cat.", "prose", 150, None, [g.ISSUE_META_OPENER]),
    ("In this picture, a cat sleeps.", "prose", 150, None, [g.ISSUE_META_OPENER]),
    ("Here is a description of a cat.", "prose", 150, None, [g.ISSUE_META_OPENER]),
    ("this image shows, 1girl", "tags", 150, None, [g.ISSUE_META_OPENER]),
    ("1girl, solo, smile, red dress, outdoors", "tags", 150, None, []),
    ("A girl in a red dress is smiling outdoors near a lake", "tags", 150, None, [g.ISSUE_NOT_TAGS]), # Prose, not tags
    ("1girl, a girl in a red dress smiling outdoors near the lake, solo", "tags", 150, None, [g.ISSUE_NOT_TAGS]),
    ("A painting in watercolor style of a lake.", "prose", 150, ["watercolor"], [g.ISSUE_OMITTED_TERM]),
    ("1girl, watercolor, lake", "tags", 150, ["watercolor"], [g.ISSUE_OMITTED_TERM]),
])
def test_check_caption(text, style, max_tokens, omit, expected):
    assert g.check_caption(text, style, max_tokens, omit) == expected


def test_retryable_issues_exclude_omitted_terms():
    assert g.ISSUE_OMITTED_TERM not in g.RETRYABLE
    assert {g.ISSUE_EMPTY, g.ISSUE_REFUSAL, g.ISSUE_META_OPENER, g.ISSUE_REPETITION, g.ISSUE_TRUNCATED,
            g.ISSUE_NOT_TAGS} == g.RETRYABLE


@pytest.mark.parametrize("text, expected", [
    ("1girl, solo, smile", False),
    ("long hair, looking at viewer, blue eyes, white background", False),
    ("one two three four five six", True), # A single comma-free stretch of 6+ words
    ("one two three four five", False),
    ("tag, one two three four five six seven eight", True), # 8+ words without a comma
    ("", False),
])
def test_looks_like_sentence(text, expected):
    assert g.looks_like_sentence(text) is expected


# --- repetition / truncation ---

def test_has_repetition_prose():
    loop = "a cat on the sofa " * 4
    assert g.has_repetition(loop, "prose") is True
    assert g.has_repetition("A cat on the sofa and a dog under the table near the window.", "prose") is False
    assert g.has_repetition("a cat a cat a cat", "prose") is False # Too short to judge


def test_has_repetition_tags():
    assert g.has_repetition("cat, cat, cat, cat, dog, cat", "tags") is True
    assert g.has_repetition("cat, dog, sofa, window, lamp, rug", "tags") is False
    assert g.has_repetition("cat, cat, cat", "tags") is False # Fewer than 6 tags


def test_looks_truncated():
    words = " ".join(f"w{i}" for i in range(100)) # ~130 estimated tokens, no repetition
    assert g.looks_truncated(words, "prose", 150) is True # Uses >85% of the budget, no final punctuation
    assert g.looks_truncated(words + ".", "prose", 150) is False
    assert g.looks_truncated(words, "prose", 1000) is False # Far from the budget
    assert g.looks_truncated(words, "tags", 150) is False
    assert g.looks_truncated(words, "prose", None) is False
    assert g.check_caption(words, "prose", 150) == [g.ISSUE_TRUNCATED]


# --- terms ---

def test_find_terms_prose_whole_words_and_phrases():
    text = "An oil-painting of a lake with brush strokes; no oilpainting here."
    found = g.find_terms(text, ["oil painting", "brush strokes", "style", "lake"], "prose")
    assert found == ["oil painting", "brush strokes", "lake"]
    assert g.find_terms("A stylish hat.", ["style"], "prose") == [] # Whole words only
    assert g.find_terms("x", ["  "], "prose") == []


def test_find_terms_tags():
    assert g.find_terms("1girl, Oil_Painting, lake", ["oil painting", "lake", "sky"], "tags") == ["oil painting", "lake"]


def test_remove_tags():
    assert g.remove_tags("1girl, watercolor, Oil_Painting, lake", ["watercolor", "oil painting", " "]) == "1girl, lake"
    assert g.remove_tags("watercolor", ["watercolor"]) == ""


# --- strip_meta_opener / prefix / suffix ---

def test_strip_meta_opener():
    assert g.strip_meta_opener("Caption: \"A cat\"") == "A cat"
    assert g.strip_meta_opener("TAGS:1girl, solo") == "1girl, solo"
    assert g.strip_meta_opener("A cat") == "A cat"
    assert g.strip_meta_opener(None) == ""


@pytest.mark.parametrize("text, prefix, suffix, expected", [
    ("a cat on a sofa", "ohwx style, ", ", film grain", "ohwx style, a cat on a sofa, film grain"),
    ("ohwx style, a cat on a sofa", "ohwx style, ", "", "ohwx style, a cat on a sofa"), # Trigger not doubled
    ("OHWX style a cat", "ohwx style, ", "", "ohwx style, a cat"),
    ("a cat, film grain", "", ", film grain", "a cat, film grain"), # Suffix not doubled
    ("a cat", "", "", "a cat"),
    ("a cat", None, None, "a cat"),
])
def test_apply_prefix_suffix(text, prefix, suffix, expected):
    assert g.apply_prefix_suffix(text, prefix, suffix) == expected
