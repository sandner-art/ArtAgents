# ArtAgent/tests/test_lab_metrics_voting.py
"""Tests for core/lab/metrics.py and core/lab/voting.py (no torch needed)."""
import json
import math
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset.vision import VisionTarget
from core.lab import lora_sweep as ls
from core.lab import metrics as m
from core.lab import voting as vt


# ---------- maths ----------

def test_normalize_and_cosine():
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    n = m.normalize(x)
    assert np.allclose(n[0], [0.6, 0.8]) and np.all(np.isfinite(n))
    c = m.pairwise_cosine(np.array([[1.0, 0], [0, 1.0]]), np.array([[1.0, 0], [1.0, 1.0]]))
    assert np.allclose(c, [[1, 1 / math.sqrt(2)], [0, 1 / math.sqrt(2)]])


def test_fidelity_and_adherence():
    gen = np.array([[1.0, 0.0], [0.0, 1.0]])
    ref = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(m.fidelity_scores(gen, ref), [2 / 3, 1 / 3])
    assert m.fidelity_scores(np.zeros((0, 2)), ref).size == 0
    assert m.fidelity_scores(gen, np.zeros((0, 2))).size == 0
    assert np.allclose(m.adherence_scores(gen, np.array([[2.0, 0.0], [1.0, 1.0]])), [1.0, 1 / math.sqrt(2)])


def _reference_cmmd(x, y, sigma=10.0, scale=1000.0):
    """Hand-written transcription of google-research/cmmd distance.py (numpy instead of jax)."""
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    gamma = 1 / (2 * sigma ** 2)
    x_sq, y_sq = np.diag(x @ x.T), np.diag(y @ y.T)
    k_xx = np.mean(np.exp(-gamma * (-2 * x @ x.T + x_sq[:, None] + x_sq[None, :])))
    k_xy = np.mean(np.exp(-gamma * (-2 * x @ y.T + x_sq[:, None] + y_sq[None, :])))
    k_yy = np.mean(np.exp(-gamma * (-2 * y @ y.T + y_sq[:, None] + y_sq[None, :])))
    return scale * (k_xx + k_yy - 2 * k_xy)


def test_cmmd_matches_reference_and_properties():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(15, 16)), rng.normal(size=(9, 16)) + 1.5
    assert m.cmmd(x, y) == pytest.approx(_reference_cmmd(x, y), rel=1e-9)
    assert m.cmmd(x, y) > 0
    assert m.cmmd(x, x) == pytest.approx(0, abs=1e-9)
    assert m.cmmd(x, 5 * x) == pytest.approx(0, abs=1e-9)              # Embeddings are normalized
    far = rng.normal(size=(15, 16)) + 10
    assert m.cmmd(x, far) > m.cmmd(x, x + 0.01)
    assert math.isnan(m.cmmd(x[:1], y)) and math.isnan(m.cmmd(x, y[:1]))


def test_bootstrap_ci():
    mean, lo, hi = m.bootstrap_ci([0.5, 0.6, 0.7, 0.55], n_boot=500, seed=1)
    assert mean == pytest.approx(0.5875) and lo <= mean <= hi and lo >= 0.5 and hi <= 0.7
    assert m.bootstrap_ci([0.5, 0.6, 0.7, 0.55], seed=1) == m.bootstrap_ci([0.5, 0.6, 0.7, 0.55], seed=1)
    assert m.bootstrap_ci([0.3]) == (0.3, 0.3, 0.3)
    assert all(math.isnan(v) for v in m.bootstrap_ci([]))
    assert m.bootstrap_ci([None, float("nan"), 1.0])[0] == 1.0


def test_fmt_ci():
    assert m.fmt_ci((0.5, 0.4, 0.6)) == "0.500 [0.400, 0.600]"
    assert m.fmt_ci((0.5, 0.4, 0.6), 1) == "0.5 [0.4, 0.6]"
    assert m.fmt_ci((float("nan"),) * 3) == "—"


# ---------- embedders ----------

def test_available_embedders_without_torch():
    with patch("core.lab.metrics.importlib.util.find_spec", return_value=None):
        assert m.available_embedders() == []
    assert "torch" in m.install_hint() and "open_clip" in m.install_hint()


def test_available_embedders_with_packages():
    present = {"torch", "open_clip"}
    with patch("core.lab.metrics.importlib.util.find_spec", side_effect=lambda n: object() if n in present else None):
        assert m.available_embedders() == [k for k, v in m.EMBEDDERS.items() if v[0] == "open_clip"]
    present = {"torch", "transformers"}
    with patch("core.lab.metrics.importlib.util.find_spec", side_effect=lambda n: object() if n in present else None):
        assert m.available_embedders() == ["DINOv2 small (transformers)"]


def test_embedder_unknown_and_missing_torch():
    with pytest.raises(m.MetricsError, match="Unknown"):
        m.Embedder("nope")
    e = m.Embedder("DINOv2 small (transformers)")
    assert e.has_text is False and e.embed_texts(["x"]) is None
    assert m.Embedder("CLIP ViT-B/32 (open_clip, LAION-2B)").has_text
    with patch("core.lab.metrics.importlib.util.find_spec", return_value=None):
        with pytest.raises(m.MetricsError, match="PyTorch"):
            e.embed_images([Image.new("RGB", (4, 4))])


class FakeEmbedder:
    """Embeds an image as its mean RGB (normalized later); text as a fixed vector per word count."""

    def __init__(self, label="fake", has_text=True):
        self.label, self.has_text, self.calls = label, has_text, 0

    def embed_images(self, images):
        self.calls += 1
        return np.array([np.asarray(im.convert("RGB"), dtype=float).reshape(-1, 3).mean(0) + 1 for im in images])

    def embed_texts(self, texts):
        return np.array([[1.0, 1.0, 1.0] for _ in texts]) if self.has_text else None


def _img(path, color):
    Image.new("RGB", (8, 8), color).save(path)
    return str(path)


def test_cached_image_embeddings(tmp_path):
    paths = [_img(tmp_path / "a.png", (255, 0, 0)), _img(tmp_path / "b.png", (0, 255, 0))]
    cache = str(tmp_path / "cache" / "emb.npz")
    e = FakeEmbedder()
    first = m.cached_image_embeddings(e, paths, cache)
    assert first.shape == (2, 3) and e.calls == 1 and os.path.exists(cache)
    again = m.cached_image_embeddings(e, paths, cache)
    assert e.calls == 1 and np.allclose(first, again)
    # Changed file → only that one is re-embedded
    os.utime(paths[1], (1, 1))
    _img(tmp_path / "b.png", (0, 0, 255))
    os.utime(paths[1], (5, 5))
    third = m.cached_image_embeddings(e, paths, cache)
    assert e.calls == 2 and np.allclose(third[0], first[0]) and not np.allclose(third[1], first[1])
    # A different model ignores the cache
    other = FakeEmbedder("other")
    m.cached_image_embeddings(other, paths, cache)
    assert other.calls == 1
    assert m.cached_image_embeddings(e, [], cache).shape[0] == 0


def test_cached_embeddings_corrupt_cache(tmp_path):
    paths = [_img(tmp_path / "a.png", (1, 2, 3))]
    cache = tmp_path / "emb.npz"
    cache.write_bytes(b"not a zip")
    e = FakeEmbedder()
    assert m.cached_image_embeddings(e, paths, str(cache)).shape == (1, 3) and e.calls == 1


def _make_run(tmp_path, colors):
    """Run with prompt 0 (held-out) and prompt 1 (trigger-less), conditions baseline + one LoRA, 2 seeds."""
    vset = {"trigger": "t", "seeds": [1, 2], "defaults": {},
            "prompts": [{"bucket": "held-out", "prompt": "[trigger] cat", "negative": "", "flags": ""},
                        {"bucket": "trigger-less", "prompt": "cat", "negative": "", "flags": ""}]}
    jobs = ls.build_jobs(vset, ["style.safetensors"], [1.0])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results = []
    for j in jobs:
        name = f"{j.index}.png"
        _img(run_dir / name, colors(j))
        results.append({"job": j.index, "path": name, "error": None})
    from dataclasses import asdict
    run = {"jobs": [asdict(j) for j in jobs], "results": results}
    return run, str(run_dir)


def test_score_run(tmp_path):
    train = [_img(tmp_path / f"t{i}.png", (255, 0, 0)) for i in range(3)]
    run, run_dir = _make_run(tmp_path, lambda j: (255, 0, 0) if j.lora else (0, 0, 255))
    result = m.score_run(run, run_dir, train, FakeEmbedder())
    rows = {(r[0], r[1]): r for r in result["rows"]}
    assert set(rows) == {(ls.BASELINE, "held-out"), (ls.BASELINE, "trigger-less"),
                         ("style @ 1", "held-out"), ("style @ 1", "trigger-less")}
    assert result["rows"][0][0] == ls.BASELINE and result["rows"][1][0] == ls.BASELINE
    lora = rows[("style @ 1", "held-out")]
    assert lora[2] == 2 and lora[3].startswith("1.000")            # Same colour as training images
    assert float(lora[4]) > 0                                      # Δ vs baseline
    assert rows[(ls.BASELINE, "held-out")][4] == ""                # No Δ on baseline rows
    assert rows[("style @ 1", "trigger-less")][4].startswith("+")
    assert lora[5] != "—" and float(lora[6]) >= 0
    assert "Scored 8 image(s)" in result["summary"]


def test_score_run_without_text_and_without_baseline(tmp_path):
    train = [_img(tmp_path / f"t{i}.png", (255, 0, 0)) for i in range(2)]
    run, run_dir = _make_run(tmp_path, lambda j: (200, 50, 0))
    run["results"] = [r for r, j in zip(run["results"], run["jobs"]) if j["lora"]]
    rows = m.score_run(run, run_dir, train, FakeEmbedder(has_text=False))["rows"]
    assert all(r[0] != ls.BASELINE for r in rows)
    assert all(r[4] == "" for r in rows)                           # No baseline to compare with
    assert all(r[5] == "—" for r in rows)


def test_score_run_errors(tmp_path):
    run, run_dir = _make_run(tmp_path, lambda j: (0, 0, 0))
    with pytest.raises(m.MetricsError, match="training image folder"):
        m.score_run(run, run_dir, [], FakeEmbedder())
    with pytest.raises(m.MetricsError, match="no images"):
        m.score_run({"jobs": run["jobs"], "results": [{"job": 0, "path": None}]}, run_dir, ["x"], FakeEmbedder())


# ---------- voting ----------

def _vote_run(tmp_path):
    vset = {"trigger": "t", "seeds": [1, 2], "defaults": {},
            "prompts": [{"bucket": "held-out", "prompt": "[trigger] cat", "negative": "", "flags": ""}]}
    jobs = ls.build_jobs(vset, ["a.safetensors", "b.safetensors"], [1.0])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    from dataclasses import asdict
    results = []
    for j in jobs:
        _img(run_dir / f"{j.index}.png", (j.index * 20, 0, 0))
        results.append({"job": j.index, "path": f"{j.index}.png", "error": None})
    results.append({"job": 99, "path": "ghost.png"})               # Unknown job: ignored
    return {"jobs": [asdict(j) for j in jobs], "results": results}, str(run_dir)


def test_make_pairs(tmp_path):
    run, run_dir = _vote_run(tmp_path)
    pairs = vt.make_pairs(run, run_dir, seed=1)
    assert len(pairs) == 2 * 3                                     # 2 seeds × C(3, 2)
    for p in pairs:
        assert p["prompt"] == "t cat" and p["bucket"] == "held-out" and p["left"] != p["right"]
    keys = {(p["seed"], vt.pair_key(p)) for p in pairs}
    assert len(keys) == 6
    assert pairs == vt.make_pairs(run, run_dir, seed=1)
    sides = [p["left"] for p in vt.make_pairs(run, run_dir, seed=2)]
    assert sorted(sides) != sorted([p["left"] for p in pairs]) or vt.make_pairs(run, run_dir, seed=2) != pairs
    no_base = vt.make_pairs(run, run_dir, seed=1, include_baseline=False)
    assert len(no_base) == 2 and all(ls.BASELINE not in (p["left"], p["right"]) for p in no_base)
    assert len(vt.make_pairs(run, run_dir, seed=1, limit=3)) == 3


def test_pair_key_order_independent():
    a = {"prompt_index": 0, "seed": 1, "left": "x", "right": "y"}
    b = {"prompt_index": 0, "seed": 1, "left": "y", "right": "x"}
    assert vt.pair_key(a) == vt.pair_key(b) == "0|1|x|y"


def test_record_and_load_votes(tmp_path):
    pair = {"prompt_index": 0, "seed": 1, "prompt": "p", "bucket": "held-out", "left": "A", "right": "B",
            "left_path": "1.png", "right_path": "2.png"}
    v1 = vt.record_vote(str(tmp_path), pair, "left")
    vt.record_vote(str(tmp_path), pair, "right", voter="judge")
    vt.record_vote(str(tmp_path), pair, "tie")
    assert v1["winner"] == "A" and v1["loser"] == "B" and not v1["tie"]
    with open(tmp_path / vt.VOTES_FILE, "a") as f:
        f.write("not json\n")
    assert len(vt.load_votes(str(tmp_path))) == 3
    assert [v["winner"] for v in vt.load_votes(str(tmp_path), "judge")] == ["B"]
    tie = vt.load_votes(str(tmp_path), "user")[1]
    assert tie["tie"] and tie["winner"] is None
    with pytest.raises(ValueError):
        vt.record_vote(str(tmp_path), pair, "both")
    assert vt.load_votes(str(tmp_path / "none")) == []


def _votes(spec):
    out = []
    for a, b, result in spec:
        if result == "tie":
            out.append({"pair": [a, b], "tie": True})
        else:
            w, l = (a, b) if result == "a" else (b, a)
            out.append({"pair": [a, b], "winner": w, "loser": l, "tie": False})
    return out


def test_bradley_terry_ordering_and_symmetry():
    votes = _votes([("A", "B", "a")] * 8 + [("A", "B", "b")] * 2 + [("B", "C", "a")] * 5 + [("A", "C", "tie")])
    s = vt.bradley_terry(votes)
    assert s["A"] > s["B"] > s["C"] and abs(sum(s.values())) < 1e-9
    sym = vt.bradley_terry(_votes([("A", "B", "a"), ("A", "B", "b")]))
    assert sym["A"] == pytest.approx(0) and sym["B"] == pytest.approx(0)
    ties = vt.bradley_terry(_votes([("A", "B", "tie")] * 3))
    assert ties["A"] == pytest.approx(ties["B"])
    unbeaten = vt.bradley_terry(_votes([("A", "B", "a")] * 5))
    assert np.isfinite(unbeaten["A"]) and unbeaten["A"] > 0 > unbeaten["B"]
    assert vt.bradley_terry([]) == {} and vt.bradley_terry(_votes([])) == {}
    assert vt.bradley_terry([{"pair": ["A", None]}]) == {"A": 0.0}


def test_bradley_terry_transitive_without_direct_games():
    s = vt.bradley_terry(_votes([("A", "B", "a")] * 4 + [("B", "C", "a")] * 4))
    assert s["A"] > s["B"] > s["C"]


@pytest.mark.parametrize("wins, n, lo, hi", [(5, 10, 0.2366, 0.7634), (0, 10, 0.0, 0.2775), (10, 10, 0.7225, 1.0),
                                             (81, 263, 0.2553, 0.3662)])
def test_wilson_known_values(wins, n, lo, hi):
    a, b = vt.wilson(wins, n)
    assert a == pytest.approx(lo, abs=1e-3) and b == pytest.approx(hi, abs=1e-3)


def test_wilson_no_games():
    assert vt.wilson(0, 0) == (0.0, 1.0)


def test_results_table():
    votes = _votes([("A", "B", "a")] * 3 + [("A", "B", "tie")])
    rows = vt.results_table(votes)
    assert [r[0] for r in rows] == ["A", "B"]
    assert rows[0][2:5] == [3, 0, 1] and rows[1][2:5] == [0, 3, 1]
    assert rows[0][5].startswith("88%")                            # (3 + 0.5) / 4
    assert vt.results_table([]) == []


def test_condition_of():
    assert vt.condition_of({"lora": None, "strength": 0}) == ls.BASELINE
    assert vt.condition_of({"lora": "x/a.safetensors", "strength": 0.75}) == "a @ 0.75"


# ---------- judge ----------

ASK = "core.dataset.vision.ask"


def _judge_pair(tmp_path):
    _img(tmp_path / "l.png", (255, 0, 0))
    _img(tmp_path / "r.png", (0, 0, 255))
    return {"prompt_index": 0, "seed": 1, "prompt": "a cat", "bucket": "held-out", "left": "A", "right": "B",
            "left_path": "l.png", "right_path": "r.png"}


def test_judge_consistent_choice(tmp_path):
    pair = _judge_pair(tmp_path)
    seen = []

    def ask(target, settings, prompt, images, **kw):
        seen.append(images[0].getpixel((0, 0)))
        assert kw["json_schema"] == vt.JUDGE_SCHEMA and "a cat" in prompt and len(images) == 2
        return json.dumps({"better": "first" if images[0].getpixel((0, 0)) == (255, 0, 0) else "second"})

    with patch(ASK, side_effect=ask):
        result = vt.judge_pair(pair, str(tmp_path), VisionTarget("ollama", "m"), {})
    assert result == {"choice": "left", "consistent": True, "raw": ["first", "second"]}
    assert seen == [(255, 0, 0), (0, 0, 255)]                      # Second call swaps the order


def test_judge_position_bias_and_ties(tmp_path):
    pair = _judge_pair(tmp_path)
    with patch(ASK, return_value='{"better": "first"}'):
        r = vt.judge_pair(pair, str(tmp_path), VisionTarget("ollama", "m"), {})
    assert r["choice"] is None and not r["consistent"]
    with patch(ASK, return_value='{"better": "tie"}'):
        assert vt.judge_pair(pair, str(tmp_path), VisionTarget("ollama", "m"), {})["choice"] == vt.TIE
    with patch(ASK, return_value="no json at all"):
        r = vt.judge_pair(pair, str(tmp_path), VisionTarget("ollama", "m"), {})
    assert r["choice"] is None and r["raw"] == ["", ""]


def test_judge_agreement(tmp_path):
    d = str(tmp_path)
    assert "No pairs" in vt.judge_agreement(d)
    pairs = [{"prompt_index": 0, "seed": s, "left": "A", "right": "B"} for s in range(4)]
    for p in pairs:
        vt.record_vote(d, p, "left", "user")
    for p, choice in zip(pairs, ["left", "left", "right", "tie"]):
        vt.record_vote(d, p, choice, "judge")
    text = vt.judge_agreement(d)
    assert "2/3 pairs" in text and "67%" in text and "50% is chance" in text
