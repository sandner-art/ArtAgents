# ArtAgent/tests/test_dataset_clip_tokenizer.py
"""
Tests for the offline CLIP token counter and the token-budget helpers.

The reference counts below were computed once with OpenAI/open_clip's SimpleTokenizer
(ftfy + regex, same bpe_simple_vocab_16e6 vocabulary), excluding the start/end tokens,
and are hardcoded so this test needs no network or extra packages.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset.clip_tokenizer import ClipTokenizer, clip_token_count
from core.dataset.tokens import budget_checks, budget_markdown, count_tokens, estimate_tokens

REFERENCE_COUNTS = [
    ("a cat", 2),
    ("A photograph of an astronaut riding a horse on the moon.", 12),
    ("1girl, solo, long_hair, looking_at_viewer, smile, blue_eyes", 20),
    ("score_9, score_8_up, score_7_up, source_anime, rating_safe", 23),
    ("txcl painting, a woman's red umbrella in the rain; she isn't smiling", 17),
    ("It's 3.14 o'clock — we'll see!", 14),
    ("café naïve façade", 7),
    ("“quoted text” and ‘single’", 8),
    ("Tom &amp; Jerry &lt;3", 5),
    ("@artist_name, #hashtag, :) ...", 10),
    ("  lots   of\n\nwhitespace\tand NEWLINES  ", 7),
    ("emoji 🎨🖌️ and kanji 猫", 11),
    ("masterpiece, best quality, newest, absurdres, highres", 12),
    ("an extremely detailed, hyperrealistic, ultra-high-resolution digital illustration of a cyberpunk city at night", 20),
    ("x" * 40, 10),
    ("12345 67890", 10),
    ("", 0),
]


@pytest.mark.parametrize("text, expected", REFERENCE_COUNTS)
def test_clip_token_count_matches_reference(text, expected):
    assert clip_token_count(text) == expected


def test_clip_count_is_case_insensitive_and_whitespace_insensitive():
    assert clip_token_count("A Red CHAIR") == clip_token_count("a red chair")
    assert clip_token_count("a  red\n\tchair") == clip_token_count("a red chair")


def test_clip_count_is_additive_for_simple_words():
    words = "red chair on a beach at sunset"
    assert clip_token_count(words) == sum(clip_token_count(w) for w in words.split())


def test_clean_unescapes_html_and_straightens_quotes():
    assert ClipTokenizer.clean("Tom &amp;amp; Jerry") == "tom & jerry" # Double-unescaped like CLIP's basic_clean
    assert ClipTokenizer.clean("“Hi” ’s") == "\"hi\" 's"
    assert ClipTokenizer.clean(None) == ""


def test_clip_count_none_is_zero():
    assert clip_token_count(None) == 0


def test_tokenizer_cache_gives_same_results():
    tok = ClipTokenizer()
    first = tok.count("hyperrealistic hyperrealistic")
    assert first == tok.count("hyperrealistic hyperrealistic")
    assert first == 2 * tok.count("hyperrealistic")


def test_long_caption_counts_past_one_chunk():
    caption = ", ".join(["detailed background"] * 40)
    assert clip_token_count(caption) > 75


# --- tokens.count_tokens / estimate_tokens ---

def test_count_tokens_exact_for_clip_estimate_otherwise():
    assert count_tokens("a cat", "clip") == (2, True)
    count, exact = count_tokens("a cat on a sofa.", "t5")
    assert exact is False and count > 0
    assert count_tokens("a cat on a sofa.", "llm")[1] is False


def test_estimate_tokens():
    assert estimate_tokens("", "t5") == 0
    # 5 words, 1 punctuation mark
    assert estimate_tokens("a cat on a sofa.", "t5") == 8 # ceil(5*1.35 + 0.6) = ceil(7.35)
    assert estimate_tokens("a cat on a sofa.", "llm") == 8 # ceil(5*1.3 + 0.6) = ceil(7.1)
    assert estimate_tokens("word " * 100, "t5") > estimate_tokens("word " * 100, "llm")


# --- budget_checks ---

CLIP = {"tokenizer": "clip", "budget": 75, "max": 225, "label": "CLIP"}
T5 = {"tokenizer": "t5", "budget": 512, "label": "T5"}
LLM_UNKNOWN = {"tokenizer": "llm", "budget": None, "label": "LLM encoder"}
SOFT_ONLY = {"tokenizer": "clip", "budget": 5, "label": "CLIP (pooled)"}


def test_budget_checks_statuses():
    short = "a cat"
    over_chunk = " ".join(["cat"] * 100) # 100 CLIP tokens: past 75, below 225
    over_max = " ".join(["cat"] * 300)
    assert clip_token_count(over_chunk) == 100 and clip_token_count(over_max) == 300

    assert budget_checks(short, [CLIP])[0] == {"label": "CLIP", "count": 2, "exact": True, "budget": 75, "max": 225, "status": "ok"}
    assert budget_checks(over_chunk, [CLIP])[0]["status"] == "over_budget"
    assert budget_checks(over_max, [CLIP])[0]["status"] == "over_max"
    assert budget_checks(" ".join(["cat"] * 75), [CLIP])[0]["status"] == "ok" # Exactly at the budget
    assert budget_checks(short, [LLM_UNKNOWN])[0]["status"] == "unknown"
    # Without a hard max, passing the budget means truncation
    assert budget_checks("one two three four five six", [SOFT_ONLY])[0]["status"] == "over_max"


def test_budget_checks_several_budgets_and_defaults():
    checks = budget_checks("a cat on a sofa.", [CLIP, T5])
    assert [c["label"] for c in checks] == ["CLIP", "T5"]
    assert checks[0]["exact"] is True and checks[1]["exact"] is False
    assert budget_checks("x", []) == []
    assert budget_checks("x", None) == []
    assert budget_checks("x", [{"tokenizer": "t5", "budget": 10}])[0]["label"] == "t5"


# --- budget_markdown ---

def test_budget_markdown_icons():
    assert budget_markdown("", [CLIP]) == ""
    assert budget_markdown("a cat", [CLIP]) == "CLIP: 2/75 (max 225) ✅"
    over = budget_markdown(" ".join(["cat"] * 100), [CLIP])
    assert over == "CLIP: 100/75 (max 225) ⚠️ uses more than one CLIP chunk"
    truncated = budget_markdown(" ".join(["cat"] * 300), [CLIP])
    assert truncated == "CLIP: 300/75 (max 225) ⛔ will be truncated"


def test_budget_markdown_estimates_and_unknown():
    text = "a cat on a sofa."
    assert budget_markdown(text, [T5]) == "T5: ≈8/512 ✅"
    assert budget_markdown(text, [LLM_UNKNOWN]) == "LLM encoder: ≈8 tokens"
    assert budget_markdown(text, [CLIP, T5]) == "CLIP: 6/75 (max 225) ✅ · T5: ≈8/512 ✅"
