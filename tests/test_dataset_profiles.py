# ArtAgent/tests/test_dataset_profiles.py
"""Tests for core/dataset/profiles.py and the shipped dataset_profiles/*.yaml files."""
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import profiles as p
from core.dataset.tokens import budget_checks

PROFILES_FILE_PATH = "core.dataset.profiles.PROFILES_FILE"
PRESETS_FILE_PATH = "core.dataset.profiles.PRESETS_FILE"
GET_ABS_PATH = "core.dataset.profiles.get_absolute_path"


# --- The shipped YAML files ---

def test_shipped_profiles_are_valid():
    profiles = p.load_profiles()
    assert len(profiles) >= 10
    ids = [prof["id"] for prof in profiles]
    assert len(ids) == len(set(ids)), "profile ids must be unique"
    labels = [prof["label"] for prof in profiles]
    assert len(labels) == len(set(labels)), "profile labels must be unique"
    for prof in profiles:
        assert prof["style"] in (p.STYLE_TAGS, p.STYLE_PROSE), prof["id"]
        assert isinstance(prof.get("budgets"), list) and prof["budgets"], prof["id"]
        for budget in prof["budgets"]:
            assert budget["tokenizer"] in ("clip", "t5", "llm"), prof["id"]
            assert budget.get("label"), prof["id"]
        assert int(prof.get("max_output_tokens") or 0) > 0, prof["id"]
        assert int(prof.get("word_target") or 0) > 0, prof["id"]
        for note in prof.get("notes") or []:
            assert note.get("text"), prof["id"]
            assert note.get("evidence") in p.EVIDENCE_BADGES, (prof["id"], note)


def test_shipped_profile_budgets_are_checkable():
    for prof in p.load_profiles():
        checks = budget_checks("a cat on a sofa", prof["budgets"])
        assert len(checks) == len(prof["budgets"])
        assert all(c["status"] in ("ok", "unknown") for c in checks), prof["id"]


def test_shipped_presets_are_valid():
    presets = p.load_presets()
    ids = [preset["id"] for preset in presets]
    assert {"general", "style", "concept", "character", "object", "function", "multi_concept"} <= set(ids)
    assert len(ids) == len(set(ids))
    for preset in presets:
        assert "{trigger}" in preset.get("prefix", ""), preset["id"]
        assert isinstance(preset.get("omit_terms", []), list), preset["id"]
        for option in preset.get("options", []):
            assert option in p.CAPTION_OPTIONS, (preset["id"], option)
        for item in preset.get("guidance") or []:
            assert item.get("text"), preset["id"]
            assert item.get("evidence") in p.EVIDENCE_BADGES, (preset["id"], item)


def test_known_profile_values():
    pony = p.get_profile("pony_v6")
    assert pony["style"] == p.STYLE_TAGS
    assert pony["quality_prefix"].startswith("score_9, score_8_up")
    assert p.get_profile("flux1")["budgets"][0] == {"tokenizer": "t5", "budget": 512, "label": "T5 (dev)"}
    assert p.get_profile("sd_sdxl")["budgets"][0]["max"] == 225
    assert p.get_profile("anima")["underscores"] is False


# --- Lookup ---

def test_get_profile_by_label_id_and_fallback():
    first = p.load_profiles()[0]
    pony = p.get_profile("pony_v6")
    assert p.get_profile(pony["label"]) == pony
    assert p.get_profile("does-not-exist") == first
    assert p.get_profile(None) == first
    assert p.get_preset("style")["label"].startswith("Style")
    assert p.get_preset(None) == p.load_presets()[0]


def test_find_by_label_empty_list_uses_fallback():
    assert p.find_by_label([], "x", {"id": "fb"}) == {"id": "fb"}


def test_missing_yaml_uses_builtin_defaults(tmp_path, capsys):
    with patch(GET_ABS_PATH, side_effect=lambda rel: str(tmp_path / rel)):
        profiles, presets = p.load_profiles(), p.load_presets()
    assert profiles == [p._DEFAULT_PROFILE]
    assert presets == [p._DEFAULT_PRESET]
    assert "not found" in capsys.readouterr().out


def test_invalid_yaml_uses_builtin_defaults(tmp_path, capsys):
    (tmp_path / "dataset_profiles").mkdir()
    (tmp_path / "dataset_profiles" / "models.yaml").write_text("profiles: [unclosed", encoding="utf-8")
    with patch(GET_ABS_PATH, side_effect=lambda rel: str(tmp_path / rel)):
        assert p.load_profiles() == [p._DEFAULT_PROFILE]
    assert "could not parse" in capsys.readouterr().out


def test_yaml_entries_without_id_or_label_are_skipped(tmp_path):
    (tmp_path / "dataset_profiles").mkdir()
    (tmp_path / "dataset_profiles" / "models.yaml").write_text(
        "profiles:\n  - {id: a, label: A, style: tags}\n  - {label: no id}\n  - just a string\n", encoding="utf-8")
    with patch(GET_ABS_PATH, side_effect=lambda rel: str(tmp_path / rel)):
        assert [prof["id"] for prof in p.load_profiles()] == ["a"]


def test_loaded_defaults_are_copies():
    """Mutating a loaded default profile must not change the module default."""
    with patch(GET_ABS_PATH, side_effect=lambda rel: "/nonexistent/" + rel):
        prof = p.load_profiles()[0]
        prof["label"] = "changed"
    assert p._DEFAULT_PROFILE["label"] == "SD 1.5 / SDXL"


# --- style ---

def test_style_of_and_labels():
    tags_profile = {"style": p.STYLE_TAGS}
    assert p.style_of(tags_profile) == p.STYLE_TAGS
    assert p.style_of(tags_profile, "Natural language") == p.STYLE_PROSE
    assert p.style_of(tags_profile, "Tags") == p.STYLE_TAGS
    assert p.style_of(tags_profile, "unknown label") == p.STYLE_TAGS
    assert p.style_of({}) == p.STYLE_PROSE


def test_style_mismatch_warning():
    pony = p.get_profile("pony_v6")
    assert p.style_mismatch_warning(pony, p.STYLE_TAGS) == ""
    warning = p.style_mismatch_warning(pony, p.STYLE_PROSE)
    assert warning.startswith("⚠️") and "tag captions" in warning and "natural-language captions may train worse" in warning
    assert p.style_mismatch_warning(p.get_profile("sd_sdxl"), p.STYLE_TAGS) == "" # Hybrid: no warning


# --- options / prefix / instruction ---

def test_default_options():
    general = p.get_preset("general")
    assert p.default_options(p.get_profile("sd_sdxl"), general) == [
        "Only clearly visible content", "No 'This image shows…' openers"]
    qwen = p.default_options(p.get_profile("qwen_image"), general)
    assert "Quote visible text exactly" in qwen
    wan = p.default_options(p.get_profile("wan"), p.get_preset("character"))
    assert wan.count("Include camera angle & framing") == 1 # Not duplicated


@pytest.mark.parametrize("preset_id, trigger, class_noun, expected", [
    ("style", "txcl", "painting", "txcl painting, "),
    ("style", "txcl", "", "txcl, "),
    ("style", "", "painting", "painting, "),
    ("concept", "glsk", "glass skin effect", "glsk, "), # Concept prefix has no {class}
    ("general", "  ohwx  ", "", "ohwx, "),
    ("general", "", "", ""),
])
def test_compose_prefix(preset_id, trigger, class_noun, expected):
    assert p.compose_prefix(p.get_preset(preset_id), trigger, class_noun, p.STYLE_PROSE) == expected


def test_build_instruction_prose():
    profile, preset = p.get_profile("sd_sdxl"), p.get_preset("character")
    text = p.build_instruction(profile, preset, p.STYLE_PROSE, ["Only clearly visible content"], 30,
                               ["beautiful", " "], "woman", "Mention the umbrella.")
    assert text.startswith("Write one caption for this image as a single paragraph of plain natural language.")
    assert "Use about 30 words." in text
    assert "Refer to the main subject only as 'the woman'." in text
    assert "Only mention things that are clearly visible; do not guess." in text
    assert "Do not use these words: beautiful." in text
    assert text.endswith("Mention the umbrella.")


def test_build_instruction_tags():
    text = p.build_instruction(p.get_profile("illustrious"), p.get_preset("general"), p.STYLE_TAGS, [], 25, [], "")
    assert text.startswith("Write tags for this image as a single comma-separated line")
    assert "Use Danbooru tags." in text
    assert "Order the tags as: person count" in text
    assert "Use about 25 tags." in text
    assert "Output only the tags." in text
    anima = p.build_instruction(p.get_profile("anima"), p.get_preset("general"), p.STYLE_TAGS, [], 0, [], "")
    assert "Use spaces instead of underscores inside tags." in anima
    assert "Use about" not in anima


def test_build_instruction_class_noun_defaults_to_subject():
    text = p.build_instruction(p.get_profile("sd_sdxl"), p.get_preset("object"), p.STYLE_PROSE, [], 0, [], "  ")
    assert "the subject" in text and "{class}" not in text


def test_build_instruction_ignores_unknown_option_labels():
    text = p.build_instruction(p.get_profile("sd_sdxl"), p.get_preset("general"), p.STYLE_PROSE, ["Bogus option"], 0, [])
    assert "Bogus" not in text


@pytest.mark.parametrize("text, expected", [
    ("", []),
    (None, []),
    ("style, art style\nwatercolor,, ", ["style", "art style", "watercolor"]),
    ("  one  ", ["one"]),
])
def test_parse_terms(text, expected):
    assert p.parse_terms(text) == expected


def test_guidance_markdown_has_badges_and_sections():
    md = p.guidance_markdown(p.get_profile("pony_v6"), p.get_preset("character"))
    assert md.startswith("**Target model: Pony Diffusion V6")
    assert "**Dataset type: Character / person**" in md
    assert "🔵 Docs" in md and "🟢 Measured" in md and "🟠 Anecdotal" in md
    assert "- Images: 15–30 varied images" in md
    assert "- Class noun examples: woman, man" in md
    assert "Most LoRA caption advice is not measured" in md


def test_evidence_badges_cover_all_levels():
    assert set(p.EVIDENCE_BADGES) == {"measured", "docs", "visual-test", "anecdotal", "inference"}
