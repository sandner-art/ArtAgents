# ArtAgent/tests/test_dataset_audit.py
"""Tests for core/dataset/audit.py (consistency audit, tagger comparison, cross-model verifier)."""
import json
import os
import re
import subprocess
import sys
from unittest.mock import patch

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.dataset import audit, tagger
from core.dataset.vision import VisionError, VisionTarget

ASK_PATH = "core.dataset.audit.ask"


def _paths(tmp_path, names):
    out = {}
    for n in names:
        p = tmp_path / n
        Image.new("RGB", (8, 8)).save(p)
        out[n] = str(p)
    return out


def _checks(result, name=None):
    return {(img, check) for img, check, _ in result["rows"] if name is None or img == name}


# --- contradictions ---

def test_load_contradictions_from_yaml():
    groups = audit.load_contradictions()
    assert ["indoors", "outdoors"] in groups
    assert any("very long hair" in g for g in groups)
    assert all(len(g) > 1 for g in groups)


def test_load_contradictions_missing_file():
    with patch("core.dataset.audit.get_absolute_path", return_value="/nonexistent/contradictions.yaml"):
        assert audit.load_contradictions() == []


def test_contradictions_tags_and_prose(tmp_path):
    paths = _paths(tmp_path, ["t.png", "p.png", "ok.png", "long.png"])
    caps = {"t.png": "1girl, indoors, outdoors", "p.png": "A woman indoors with short hair and long hair.",
            "ok.png": "1girl, outdoors", "long.png": "A woman with very long hair."}
    tags = audit.audit(caps, {k: paths[k] for k in ("t.png", "ok.png")}, "tags")
    assert ("t.png", audit.CHECK_CONTRADICTION) in _checks(tags)
    assert not _checks(tags, "ok.png")
    detail = [d for i, c, d in tags["rows"] if i == "t.png"][0]
    assert detail == "indoors + outdoors"
    prose = audit.audit(caps, {k: paths[k] for k in ("p.png", "long.png")}, "prose")
    assert ("p.png", audit.CHECK_CONTRADICTION) in _checks(prose)
    # 'very long hair' also contains 'long hair': not a contradiction
    assert not _checks(prose, "long.png")


# --- triggers ---

def test_trigger_checks(tmp_path):
    paths = _paths(tmp_path, ["a.png", "b.png", "c.png", "d.png", "e.png"])
    caps = {"a.png": "txcl, 1girl", "b.png": "1girl, solo", "c.png": "1girl, txcl", "d.png": "txcl, txcl, smile",
            "e.png": "txcl, zkx, group"}
    result = audit.audit(caps, paths, "tags", ["txcl", "zkx"])
    assert not _checks(result, "a.png")
    assert ("b.png", audit.CHECK_NO_TRIGGER) in _checks(result)
    assert ("c.png", audit.CHECK_TRIGGER_POSITION) in _checks(result)
    assert ("d.png", audit.CHECK_TRIGGER_TWICE) in _checks(result)
    assert ("e.png", audit.CHECK_SEVERAL_TRIGGERS) in _checks(result)
    assert ("e.png", audit.CHECK_TRIGGER_POSITION) in _checks(result)   # zkx is not at the start
    assert "- Trigger 'txcl': in 4/5 caption(s)" in result["summary"]
    assert "- Trigger 'zkx': in 1/5 caption(s)" in result["summary"]


def test_trigger_matches_whole_words_only(tmp_path):
    paths = _paths(tmp_path, ["a.png"])
    result = audit.audit({"a.png": "txclx, cat"}, paths, "tags", ["txcl"])
    assert ("a.png", audit.CHECK_NO_TRIGGER) in _checks(result)


def test_only_first_caption_line_is_audited(tmp_path):
    paths = _paths(tmp_path, ["a.png"])
    result = audit.audit({"a.png": "txcl, 1girl\nindoors, outdoors"}, paths, "tags", ["txcl"])
    assert not _checks(result)


def test_omitted_terms(tmp_path):
    paths = _paths(tmp_path, ["a.png", "b.png"])
    caps = {"a.png": "A watercolor painting of a fox.", "b.png": "A fox in snow."}
    result = audit.audit(caps, paths, "prose", omit_terms=["watercolor", "painting"])
    detail = [d for i, c, d in result["rows"] if i == "a.png" and c == audit.CHECK_OMITTED][0]
    assert detail == "watercolor, painting"
    assert not _checks(result, "b.png")


def test_length_outliers_need_five_captions(tmp_path):
    names = [f"{i}.png" for i in range(6)]
    paths = _paths(tmp_path, names)
    caps = {n: "a, b, c, d" for n in names}
    caps["0.png"] = ", ".join("t%d" % i for i in range(12))
    caps["1.png"] = "a"
    result = audit.audit(caps, paths, "tags")
    assert ("0.png", audit.CHECK_LONG) in _checks(result)
    assert ("1.png", audit.CHECK_SHORT) in _checks(result)
    few = audit.audit({n: caps[n] for n in names[:4]}, {n: paths[n] for n in names[:4]}, "tags")
    assert not any(c in (audit.CHECK_LONG, audit.CHECK_SHORT) for _, c, _ in few["rows"])


def test_empty_captions_are_skipped(tmp_path):
    paths = _paths(tmp_path, ["a.png", "b.png"])
    result = audit.audit({"a.png": "", "b.png": "   "}, paths, "tags", ["txcl"])
    assert result["rows"] == []
    assert result["summary"].startswith("**Audited 0 caption(s); 0 with something to check.**")


# --- tagger comparison ---

TAGGER_CFG = {"id": "fake-audit", "label": "Fake audit tagger"}


def _sidecar(image, tags, model="fake-audit"):
    tagger.save_sidecar(image, {"id": model}, tags)


@pytest.fixture
def fake_registry():
    with patch("core.dataset.tagger.load_taggers", return_value=[dict(TAGGER_CFG)]), \
            patch("core.dataset.tagger.vocabulary", return_value={"smile": 0, "hat": 0, "2girls": 0, "1girl": 0}):
        yield


def test_tagger_comparison_rows(tmp_path, fake_registry):
    paths = _paths(tmp_path, ["a.png"])
    _sidecar(paths["a.png"], {"1girl": (0.97, 0), "smile": (0.03, 0), "long_hair": (0.92, 0), "red_dress": (0.88, 0),
                              "general": (0.9, 9), "2girls": (0.02, 0)})
    caps = {"a.png": "txcl, 2girls, smile, hat, unknown tag"}
    result = audit.audit(caps, paths, "tags", ["txcl"], tagger_label="Fake audit tagger")
    rows = {c: d for i, c, d in result["rows"]}
    assert rows[audit.CHECK_TAGGER_DISAGREES] == "2girls (0.02), smile (0.03), hat (<0.02)"
    assert rows[audit.CHECK_COUNT] == "caption: 2girls; tagger: 1girl (0.97)"
    assert rows[audit.CHECK_TAGGER_MISSING] == "long hair (0.92), red dress (0.88)"
    assert "Compared with Fake audit tagger for 1 image(s)." in result["summary"]


def test_tagger_comparison_ignores_other_model_and_prose(tmp_path, fake_registry):
    paths = _paths(tmp_path, ["a.png"])
    _sidecar(paths["a.png"], {"smile": (0.01, 0)}, model="other-model")
    result = audit.audit({"a.png": "smile"}, paths, "tags", tagger_label="Fake audit tagger")
    assert not result["rows"]
    assert "No tagger results for Fake audit tagger yet" in result["summary"]
    prose = audit.audit({"a.png": "smile"}, paths, "prose", tagger_label="Fake audit tagger")
    assert "tagger" not in prose["summary"].lower()


def test_tagger_comparison_without_vocabulary(tmp_path):
    paths = _paths(tmp_path, ["a.png"])
    _sidecar(paths["a.png"], {"smile": (0.03, 0)})
    with patch("core.dataset.tagger.load_taggers", return_value=[dict(TAGGER_CFG)]), \
            patch("core.dataset.tagger.vocabulary", return_value=None):
        result = audit.audit({"a.png": "smile, hat"}, paths, "tags", tagger_label="Fake audit tagger")
    rows = {c: d for i, c, d in result["rows"]}
    assert rows[audit.CHECK_TAGGER_DISAGREES] == "smile (0.03)"   # 'hat' unknown without the vocabulary
    assert "vocabulary not on disk" in result["summary"]


# --- claims ---

def test_claims_from_tags_skip_trigger_and_prefixes():
    claims = audit.claims_from_caption("txcl painting, score_9, source_anime, rating_safe, 1girl, Solo, solo, red hat",
                                       "tags", ["txcl painting"])
    assert claims == ["1girl", "Solo", "red hat"]


def test_claims_from_prose():
    caption = "txcl, A woman sitting on a bench, wearing a red coat. Snow falls; a dog sleeps. Smiling."
    claims = audit.claims_from_caption(caption, "prose", ["txcl"])
    assert claims == ["A woman sitting on a bench", "wearing a red coat", "Snow falls", "a dog sleeps"]


def test_claims_only_first_line_and_limit():
    caption = ", ".join(f"tag{i}" for i in range(30)) + "\nsecond line"
    claims = audit.claims_from_caption(caption, "tags")
    assert len(claims) == audit.MAX_CLAIMS and "second line" not in claims
    assert audit.claims_from_caption("", "tags") == []


def test_claims_are_independent_of_hash_seed():
    code = ("import sys; sys.path.insert(0, %r); from core.dataset import audit; "
            "print(audit.claims_from_caption('txcl painting of a cat on a sofa, a red ball', 'prose', "
            "['txcl', 'painting', 'txcl painting']))" % ROOT)
    results = set()
    for seed in (0, 4):  # Seeds 0 and 4 order the set differently on CPython 3.13
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
        results.add(out.stdout.strip())
    assert len(results) == 1, results


def test_control_claims_deterministic_and_absent():
    a = audit.control_claims("a cat on a sofa", "img1.png")
    assert a == audit.control_claims("a cat on a sofa", "img1.png")
    assert len(a) == 2 and all(c in audit.CONTROL_OBJECTS for c in a)
    caption = "giraffe bicycle umbrella tower sailboat pineapple hydrant violin snowman balloon lighthouse"
    assert audit.control_claims(caption, "x") == ["a cactus"]
    assert audit.control_claims(caption + " cactus", "x") == []
    keys = {tuple(audit.control_claims("a cat", f"img{i}.png")) for i in range(20)}
    assert len(keys) > 1   # Varies per image


def test_verify_prompt_numbering():
    text = audit.verify_prompt(["A cat.", "A dog."])
    assert "1. A cat.\n2. A dog." in text and '{"answers": [...]}' in text


# --- verify ---

def _fake_ask(rules):
    """Answers each numbered statement in the prompt by the first matching rule (substring -> answer)."""
    def ask(target, settings, prompt, images, **kwargs):
        ask.calls.append({"prompt": prompt, "images": images, **kwargs})
        statements = re.findall(r"^\d+\. (.*)$", prompt, re.M)
        answers = []
        for s in statements:
            answers.append(next((a for key, a in rules if key in s.lower()), "yes"))
        return json.dumps({"answers": answers})
    ask.calls = []
    return ask


def test_verify_maps_shuffled_answers_back(tmp_path):
    image = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(image)
    fake = _fake_ask([("giraffe", "no"), ("bicycle", "no"), ("umbrella", "no"), ("tower", "no"), ("sailboat", "no"),
                      ("pineapple", "no"), ("hydrant", "no"), ("violin", "no"), ("snowman", "no"), ("balloon", "no"),
                      ("lighthouse", "no"), ("cactus", "no"), ("red hat", "no"), ("smile", "unsure")])
    caption = "txcl, 1girl, red hat, smile, outdoors"
    with patch(ASK_PATH, side_effect=fake):
        result = audit.verify(str(image), caption, Image.new("RGB", (8, 8)), VisionTarget("ollama", "m"), {}, "tags", ["txcl"])
    assert [c["claim"] for c in result["claims"]] == ["1girl", "red hat", "smile", "outdoors"]
    assert [c["answer"] for c in result["claims"]] == ["yes", "no", "unsure", "yes"]
    assert result["disputed"] == ["red hat"] and result["unsure"] == ["smile"]
    assert result["control_failures"] == 0 and len(result["controls"]) == 2
    call = fake.calls[0]
    assert call["temperature"] == 0.0 and call["seed"] == 7 and call["json_schema"] == audit.ANSWER_SCHEMA
    assert "The image shows red hat." in call["prompt"]
    stored = json.loads(open(audit.verification_path(str(image)), encoding="utf-8").read())
    assert stored["disputed"] == ["red hat"] and stored["model"] == "m"


def test_verify_control_failures(tmp_path):
    image = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(image)
    with patch(ASK_PATH, side_effect=_fake_ask([])):   # Says yes to everything
        result = audit.verify(str(image), "A cat on a sofa.", Image.new("RGB", (8, 8)), VisionTarget("ollama", "m"), {}, "prose")
    assert result["control_failures"] == 2 and result["disputed"] == []


def test_verify_prose_statements_are_clauses(tmp_path):
    image = tmp_path / "a.png"
    Image.new("RGB", (8, 8)).save(image)
    fake = _fake_ask([])
    with patch(ASK_PATH, side_effect=fake):
        audit.verify(str(image), "A cat on a sofa, a red ball nearby.", Image.new("RGB", (8, 8)),
                     VisionTarget("ollama", "m"), {}, "prose")
    statements = re.findall(r"^\d+\. (.*)$", fake.calls[0]["prompt"], re.M)
    assert "A cat on a sofa" in statements and "a red ball nearby" in statements
    assert len(statements) == 4


def test_verify_length_mismatch(tmp_path):
    image = tmp_path / "a.png"
    with patch(ASK_PATH, return_value='{"answers": ["yes"]}'), pytest.raises(VisionError, match="answered 1 of"):
        audit.verify(str(image), "1girl, solo, hat", None, VisionTarget("ollama", "m"), {}, "tags")


def test_verify_invalid_json(tmp_path):
    with patch(ASK_PATH, return_value="I think yes"), pytest.raises(VisionError, match="did not return valid JSON"):
        audit.verify(str(tmp_path / "a.png"), "1girl, solo", None, VisionTarget("ollama", "m"), {}, "tags")


def test_verify_nothing_to_check(tmp_path):
    with pytest.raises(VisionError, match="Nothing to verify"):
        audit.verify(str(tmp_path / "a.png"), "txcl", None, VisionTarget("ollama", "m"), {}, "tags", ["txcl"])


def test_verification_rows(tmp_path):
    paths = _paths(tmp_path, ["a.png", "b.png", "c.png"])
    controls_no = [(o.split()[-1], "no") for o in audit.CONTROL_OBJECTS]
    with patch(ASK_PATH, side_effect=_fake_ask(controls_no + [("shows hat", "no"), ("solo", "unsure")])):
        audit.verify(paths["a.png"], "1girl, hat, solo", None, VisionTarget("ollama", "m"), {}, "tags")
    with patch(ASK_PATH, side_effect=_fake_ask([])):
        audit.verify(paths["b.png"], "1girl", None, VisionTarget("ollama", "m2"), {}, "tags")
    rows = audit.verification_rows(paths, {"a.png": "1girl, hat, solo", "b.png": "1girl, edited", "c.png": "x"})
    assert rows[0] == ["a.png", "hat", 1, 0, "m", ""]
    assert rows[1][0] == "b.png" and rows[1][1] == "—"
    assert "⚠ said yes to an absent object" in rows[1][5] and "caption changed since" in rows[1][5]
    assert len(rows) == 2


def test_load_verification_missing_or_broken(tmp_path):
    image = str(tmp_path / "a.png")
    assert audit.load_verification(image) == {}
    os.makedirs(os.path.dirname(audit.verification_path(image)))
    with open(audit.verification_path(image), "w") as f:
        f.write("[]")
    assert audit.load_verification(image) == {}
