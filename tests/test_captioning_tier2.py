# ArtAgent/tests/test_captioning_tier2.py
"""Tests for the Tier 2 caption sources (tagger, hints, tagger + VLM, structured) and the OpenAI-compatible backend."""
import json
import os
import sys
from unittest.mock import patch

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import captioning_logic as cl
from core.dataset import provenance, structured

EXECUTE_PATH = "core.captioning_logic.execute_chat_or_team"
ASK_PATH = "core.dataset.vision.ask"
PROBS_PATH = "core.dataset.tagger.tag_probabilities"
DIGEST_PATH = "core.captioning_logic._model_digest"

PROBS = {"general": (0.9, 9), "1girl": (0.97, 0), "solo": (0.9, 0), "smile": (0.6, 0), "hat": (0.3, 0),
         "hatsune_miku_(vocaloid)": (0.92, 4)}
TAGGER_CFG = {"id": "fake", "label": "Fake tagger", "folder": "/nowhere"}


@pytest.fixture(autouse=True)
def _offline():
    with patch(DIGEST_PATH, return_value="digest123456") as digest, \
            patch("core.dataset.tagger.load_taggers", return_value=[dict(TAGGER_CFG)]):
        yield digest


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    Image.new("RGB", (40, 30), "white").save(d / "a.png")
    Image.new("RGB", (40, 30), "black").save(d / "b.png")
    return d


def _agent_returns(text):
    return (text, "session text", None, ["previous", "new"])


def _run(folder, config, name="a.png", agent="(Direct Agent Call)", model="vision-model (VISION)", mode="Overwrite",
         instruction="Write tags for this image.", prefix="", captions=None):
    paths = {n: str(folder / n) for n in ("a.png", "b.png")}
    return cl.generate_captions_for_selected(
        name, agent, model, mode, paths, captions or {}, {}, [{"name": "vision-model", "vision": True}], {}, {}, {},
        [], ["previous"], caption_instruction=instruction, caption_prefix=prefix, caption_config=config)


def _prov(folder, name="a.png"):
    return provenance.read_provenance(str(folder / name))["latest"]


# --- tagger only ---

def test_tagger_only_needs_no_model_or_agent(folder):
    with patch(PROBS_PATH, return_value=(PROBS, False)) as probs, patch(EXECUTE_PATH) as execute, \
            patch(ASK_PATH) as ask:
        status, caps, last, _ = _run(folder, {"source": cl.SOURCE_TAGGER, "profile": "pony_v6",
                                              "tagger": {"label": "Fake tagger", "add_rating": True}},
                                     agent=None, model=None, prefix="txcl, ")
    assert "Status: Success" in status
    assert last == "txcl, rating_safe, 1girl, hatsune miku (vocaloid), solo, smile"
    assert (folder / "a.txt").read_text(encoding="utf-8") == last
    execute.assert_not_called()
    ask.assert_not_called()
    assert probs.call_args[0][0]["id"] == "fake"
    record = _prov(folder)
    assert record["model"] == "tagger:fake" and record["backend"] == "tagger" and record["source"] == "tagger"
    assert record["tagger"] == {"model": "fake", "thresholds": [0.35, 0.85], "cached": False, "tags": 5}
    assert record["instruction"] == "" and record["model_digest"] == "" and record["retries"] == 0
    assert record["agent"] == "(Direct)"


def test_tagger_only_thresholds_and_omit_terms(folder):
    with patch(PROBS_PATH, return_value=(PROBS, True)):
        _, _, last, _ = _run(folder, {"source": cl.SOURCE_TAGGER, "omit_terms": ["smile"],
                                      "tagger": {"label": "Fake tagger", "general_threshold": 0.25, "blacklist": ["solo"]}})
    assert last == "1girl, hatsune miku (vocaloid), hat"


def test_tagger_only_no_tags_is_an_error(folder):
    with patch(PROBS_PATH, return_value=({"general": (0.9, 9)}, False)):
        status, _, last, _ = _run(folder, {"source": cl.SOURCE_TAGGER, "tagger": {"label": "Fake tagger"}})
    assert "Status: Error" in status and "no tags above the thresholds" in status
    assert last == "" and not (folder / "a.txt").exists()


def test_tagger_error_is_reported(folder):
    from core.dataset.tagger import TaggerError
    with patch(PROBS_PATH, side_effect=TaggerError("The tagger needs onnxruntime")):
        status, _, _, _ = _run(folder, {"source": cl.SOURCE_TAGGER, "tagger": {"label": "Fake tagger"}})
    assert "Status: Error" in status and "needs onnxruntime" in status


# --- tagger hints ---

def test_tagger_hints_are_appended_to_the_instruction(folder):
    with patch(PROBS_PATH, return_value=(PROBS, False)), \
            patch(EXECUTE_PATH, return_value=_agent_returns("1girl, solo, smiling")) as execute:
        status, _, last, _ = _run(folder, {"source": cl.SOURCE_TAGGER_HINTS, "style": "tags",
                                           "tagger": {"label": "Fake tagger", "hint_threshold": 0.5}},
                                  agent="Captioner")
    sent = execute.call_args.kwargs["user_input"]
    assert sent.startswith("Write tags for this image.\n\nTags from an automatic image tagger")
    assert "Characters: hatsune miku (vocaloid)" in sent
    assert sent.splitlines()[-1] == "Tags: 1girl, solo, smile"      # 'hat' (0.30) is below the hint threshold
    assert execute.call_args.kwargs["selected_role_or_team"] == "Captioner"   # Agents still work with Ollama
    assert last == "1girl, solo, smiling"
    record = _prov(folder)
    assert record["instruction"] == sent and record["source"] == "tagger_hints" and record["agent"] == "Captioner"


def test_tagger_hints_respect_omit_terms(folder):
    with patch(PROBS_PATH, return_value=(PROBS, False)), \
            patch(EXECUTE_PATH, return_value=_agent_returns("1girl, solo")) as execute:
        _run(folder, {"source": cl.SOURCE_TAGGER_HINTS, "style": "tags", "omit_terms": ["smile"],
                      "tagger": {"label": "Fake tagger"}})
    sent = execute.call_args.kwargs["user_input"]
    hint = sent.split("Tags from an automatic image tagger", 1)[1]
    assert "smile" not in hint


# --- tagger + VLM ---

def test_tagger_plus_prose(folder):
    with patch(PROBS_PATH, return_value=(PROBS, False)), \
            patch(EXECUTE_PATH, return_value=_agent_returns("A girl smiling under a tree.")):
        _, _, last, _ = _run(folder, {"source": cl.SOURCE_TAGGER_PLUS, "style": "prose",
                                      "tagger": {"label": "Fake tagger"}}, instruction="Describe it.")
    assert last == "1girl, hatsune miku (vocaloid), solo, smile. A girl smiling under a tree."


def test_tagger_plus_tags_merge_without_duplicates(folder):
    with patch(PROBS_PATH, return_value=(PROBS, False)), \
            patch(EXECUTE_PATH, return_value=_agent_returns("solo, outdoors, smile, tree")):
        _, _, last, _ = _run(folder, {"source": cl.SOURCE_TAGGER_PLUS, "style": "tags", "tagger": {"label": "Fake tagger"}})
    assert last == "1girl, hatsune miku (vocaloid), solo, smile, outdoors, tree"


# --- structured ---

FIELDS_JSON = json.dumps({"subject": "a girl", "action": "standing", "hair": "red hair", "clothing": "blue dress",
                          "setting": "a park", "extra": "ignored"})


def test_structured_with_ollama_uses_direct_call(folder):
    with patch(ASK_PATH, return_value=FIELDS_JSON) as ask, patch(EXECUTE_PATH) as execute:
        status, _, last, _ = _run(folder, {"source": cl.SOURCE_STRUCTURED, "style": "prose", "seed": 3, "temperature": 0,
                                           "structured": {"fields": ["subject", "action", "hair", "clothing", "setting"]}},
                                  agent="Captioner", prefix="txcl, ")
    execute.assert_not_called()
    target, settings, prompt, images = ask.call_args[0]
    kw = ask.call_args.kwargs
    assert target.backend == "ollama" and target.model == "vision-model (VISION)"
    assert '- "hair": ' in prompt and len(images) == 1
    assert kw["json_schema"]["required"] == ["subject", "action", "hair", "clothing", "setting"]
    assert kw["seed"] == 3 and kw["temperature"] == 0.0 and kw["max_tokens"] >= 400
    assert last == "txcl, A girl standing, with red hair, wearing blue dress, a park."
    stored = structured.load_fields(str(folder / "a.png"))
    assert stored == {"subject": "a girl", "action": "standing", "hair": "red hair", "clothing": "blue dress",
                      "setting": "a park"}
    record = _prov(folder)
    assert record["fields"] == stored and record["agent"] == "(Direct)" and record["source"] == "structured"


def test_structured_template_and_include(folder):
    with patch(ASK_PATH, return_value=FIELDS_JSON):
        _, _, last, _ = _run(folder, {"source": cl.SOURCE_STRUCTURED, "style": "tags",
                                      "structured": {"fields": ["subject", "hair", "setting"],
                                                     "template": "{setting}, {subject}, {hair}", "include": ["subject", "setting"]}})
    assert last == "a park, a girl"


def test_structured_invalid_json_is_retried(folder):
    with patch(ASK_PATH, side_effect=["Sorry, here you go: not json", FIELDS_JSON]) as ask:
        status, _, last, _ = _run(folder, {"source": cl.SOURCE_STRUCTURED, "style": "prose", "retries": 1, "seed": 42,
                                           "structured": {"fields": ["subject"]}})
    assert ask.call_count == 2
    assert ask.call_args_list[1].kwargs["seed"] == 43 and ask.call_args_list[1].kwargs["temperature"] >= 0.3
    assert last == "A girl." and "Retried 1x" in status


def test_retry_reason_is_reported(folder):
    with patch(ASK_PATH, side_effect=["not json", FIELDS_JSON]):
        status, _, _, _ = _run(folder, {"source": cl.SOURCE_STRUCTURED, "retries": 1, "structured": {"fields": ["subject"]}})
    assert "Retried 1x because of: invalid JSON" in status


def test_structured_invalid_json_without_retries_fails(folder):
    with patch(ASK_PATH, return_value="nope"):
        status, _, last, _ = _run(folder, {"source": cl.SOURCE_STRUCTURED, "structured": {"fields": ["subject"]}})
    assert "Status: Error" in status and "empty caption" in status
    assert not (folder / "a.txt").exists()


def test_structured_needs_no_agent(folder):
    with patch(ASK_PATH, return_value=FIELDS_JSON):
        status, _, _, _ = _run(folder, {"source": cl.SOURCE_STRUCTURED, "structured": {"fields": ["subject"]}}, agent=None)
    assert "Status: Success" in status


# --- OpenAI-compatible backend ---

OPENAI = {"base_url": "http://localhost:1234/v1", "model": "joycaption", "api_key": "k"}


def test_openai_backend_uses_vision_ask(folder):
    with patch(ASK_PATH, return_value="A cat on a sofa.") as ask, patch(EXECUTE_PATH) as execute:
        status, _, last, _ = _run(folder, {"backend": "openai", "openai": OPENAI, "style": "prose", "seed": 9},
                                  agent="Captioner", model="ignored-ollama-model (VISION)", instruction="Describe.")
    execute.assert_not_called()
    target, _, prompt, images = ask.call_args[0]
    assert (target.backend, target.model, target.base_url, target.api_key) == ("openai", "joycaption",
                                                                                "http://localhost:1234/v1", "k")
    assert prompt == "Describe." and ask.call_args.kwargs["seed"] == 9
    assert last == "A cat on a sofa."
    record = _prov(folder)
    assert record["backend"] == "openai" and record["model"] == "joycaption" and record["model_digest"] == ""


@pytest.mark.parametrize("oa", [{}, {"base_url": "http://h:1"}, {"model": "m"}])
def test_openai_backend_needs_url_and_model(folder, oa):
    status, caps, last, _ = _run(folder, {"backend": "openai", "openai": oa})
    assert status == "Enter the server URL and model name for the OpenAI-compatible backend."
    assert last == ""


def test_openai_backend_error(folder):
    from core.dataset.vision import VisionError
    with patch(ASK_PATH, side_effect=VisionError("Cannot connect to http://localhost:1234/v1")):
        status, _, _, _ = _run(folder, {"backend": "openai", "openai": OPENAI})
    assert "Status: Error" in status and "Cannot connect" in status


def test_vlm_with_ollama_still_uses_agents(folder):
    with patch(EXECUTE_PATH, return_value=_agent_returns("A cat.")) as execute, patch(ASK_PATH) as ask:
        _run(folder, {"style": "prose"}, agent="Captioner")
    ask.assert_not_called()
    assert execute.call_args.kwargs["selected_role_or_team"] == "Captioner"
    record = _prov(folder)
    assert record["backend"] == "ollama" and record["source"] == "vlm" and record["model_digest"] == "digest123456"
    assert record["tagger"] is None and record["fields"] is None


def test_missing_agent_message_for_vlm(folder):
    status, _, _, _ = _run(folder, {"style": "prose"}, agent=None)
    assert status.startswith("Please select an Agent")


# --- batch ---

def _all(folder, config, agent="(Direct Agent Call)", model="vision-model (VISION)"):
    paths = {n: str(folder / n) for n in ("a.png", "b.png")}
    return list(cl.iter_generate_captions_for_all(paths, {}, agent, model, "Overwrite", {}, [], {}, {}, {}, [], [],
                                                  caption_instruction="x", caption_config=config))


def test_batch_tagger_only_without_model(folder):
    with patch(PROBS_PATH, return_value=(PROBS, False)):
        updates = _all(folder, {"source": cl.SOURCE_TAGGER, "tagger": {"label": "Fake tagger"}}, agent=None, model=None)
    final_status, final_caps = updates[-1][0], updates[-1][1]
    assert set(final_caps) == {"a.png", "b.png"}
    assert (folder / "b.txt").read_text(encoding="utf-8").startswith("1girl")
    assert "2" in final_status


def test_batch_openai_needs_url(folder):
    updates = _all(folder, {"backend": "openai", "openai": {"model": "m"}})
    assert updates == [("Enter the server URL and model name for the OpenAI-compatible backend.", {}, "", [])]


def test_batch_structured_without_agent(folder):
    with patch(ASK_PATH, return_value=FIELDS_JSON):
        updates = _all(folder, {"source": cl.SOURCE_STRUCTURED, "structured": {"fields": ["subject"]}}, agent=None)
    assert set(updates[-1][1]) == {"a.png", "b.png"}


def test_batch_vlm_still_requires_model_and_agent(folder):
    assert _all(folder, {}, model=None)[0][0] == "Please select a Vision Model."
    assert _all(folder, {}, agent=None)[0][0].startswith("Please select an Agent")
