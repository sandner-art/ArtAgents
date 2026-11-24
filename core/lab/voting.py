# ArtAgent/core/lab/voting.py
"""
Blind pairwise voting on LoRA test images.

Each pair shows two images with the same prompt and seed from different conditions
(LoRA / strength / baseline), labels hidden and sides randomized. Votes are fitted with a
Bradley-Terry model (Hunter's MM algorithm, numpy only) and win rates get Wilson intervals.

An optional vision-model judge votes on both orderings of each pair; when its two answers
disagree it is position-biased on that pair and the vote is not counted. Its agreement with
your own votes is reported so you can decide whether to trust it (EXPERIMENTAL).
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from itertools import combinations

import numpy as np

from .lora_sweep import BASELINE, lora_label

VOTES_FILE = "votes.jsonl"
TIE = "tie"


def condition_of(job: dict) -> str:
    return BASELINE if job.get("lora") is None else f"{lora_label(job['lora'])} @ {job['strength']:g}"


def make_pairs(run: dict, run_dir: str, seed: int = 0, include_baseline: bool = True, limit: int | None = None) -> list[dict]:
    """All pairs of conditions within the same prompt and seed, shuffled, with random sides."""
    jobs = {j["index"]: j for j in run.get("jobs", [])}
    groups: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for r in run.get("results", []):
        if not r.get("path") or r["job"] not in jobs:
            continue
        j = jobs[r["job"]]
        cond = condition_of(j)
        if cond == BASELINE and not include_baseline:
            continue
        groups.setdefault((j["prompt_index"], j["seed"]), []).append((cond, r["path"]))
    rng = random.Random(seed)
    pairs = []
    for (pi, s), items in sorted(groups.items()):
        prompt = next(j["prompt"] for j in jobs.values() if j["prompt_index"] == pi)
        bucket = next(j["bucket"] for j in jobs.values() if j["prompt_index"] == pi)
        for a, b in combinations(items, 2):
            if rng.random() < 0.5:
                a, b = b, a
            pairs.append({"prompt_index": pi, "seed": s, "prompt": prompt, "bucket": bucket,
                          "left": a[0], "left_path": a[1], "right": b[0], "right_path": b[1]})
    rng.shuffle(pairs)
    return pairs[:limit] if limit else pairs


def pair_key(pair: dict) -> str:
    a, b = sorted([pair["left"], pair["right"]])
    return f"{pair['prompt_index']}|{pair['seed']}|{a}|{b}"


def record_vote(run_dir: str, pair: dict, choice: str, voter: str = "user", note: str = "") -> dict:
    """choice: 'left', 'right' or 'tie'."""
    if choice not in ("left", "right", TIE):
        raise ValueError("choice must be left, right or tie")
    winner = {"left": pair["left"], "right": pair["right"]}.get(choice)
    loser = {"left": pair["right"], "right": pair["left"]}.get(choice)
    vote = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "voter": voter, "key": pair_key(pair),
            "prompt_index": pair["prompt_index"], "seed": pair["seed"], "bucket": pair.get("bucket", ""),
            "winner": winner, "loser": loser, "tie": choice == TIE, "pair": [pair["left"], pair["right"]], "note": note}
    with open(os.path.join(run_dir, VOTES_FILE), "a", encoding="utf-8") as f:
        f.write(json.dumps(vote, ensure_ascii=False) + "\n")
    return vote


def load_votes(run_dir: str, voter: str | None = None) -> list[dict]:
    path = os.path.join(run_dir, VOTES_FILE)
    if not os.path.exists(path):
        return []
    votes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                v = json.loads(line)
            except ValueError:
                continue
            if voter is None or v.get("voter") == voter:
                votes.append(v)
    return votes


def bradley_terry(votes: list[dict], iterations: int = 500, prior: float = 0.5) -> dict[str, float]:
    """
    Bradley-Terry strengths via Hunter's MM algorithm. Ties count half a win each way; a small
    prior (half a virtual win and loss against an average opponent) keeps unbeaten items finite.
    Returns {item: log-strength}, centred on 0.
    """
    items = sorted({x for v in votes for x in v.get("pair", []) if x})
    if len(items) < 2:
        return {i: 0.0 for i in items}
    idx = {it: k for k, it in enumerate(items)}
    n = len(items)
    wins = np.full((n, n), 0.0)
    for v in votes:
        a, b = v.get("pair", [None, None])
        if a not in idx or b not in idx:
            continue
        if v.get("tie"):
            wins[idx[a], idx[b]] += 0.5
            wins[idx[b], idx[a]] += 0.5
        elif v.get("winner") in idx and v.get("loser") in idx:
            wins[idx[v["winner"]], idx[v["loser"]]] += 1
    games = wins + wins.T
    p = np.ones(n)
    for _ in range(iterations):
        w = wins.sum(axis=1) + prior
        denom = np.array([sum(games[i, j] / (p[i] + p[j]) for j in range(n) if j != i and games[i, j] > 0)
                          + 2 * prior / (p[i] + 1.0) for i in range(n)])
        new = w / denom
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(new - p)) < 1e-9:
            p = new
            break
        p = new
    logs = np.log(p)
    return {it: float(logs[idx[it]] - logs.mean()) for it in items}


def wilson(wins: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    phat = wins / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


RESULT_HEADERS = ["Condition", "Bradley-Terry score", "Wins", "Losses", "Ties", "Win rate [95% Wilson]"]


def results_table(votes: list[dict]) -> list[list]:
    scores = bradley_terry(votes)
    stats = {item: [0, 0, 0] for item in scores}
    for v in votes:
        a, b = v.get("pair", [None, None])
        if v.get("tie"):
            for x in (a, b):
                if x in stats:
                    stats[x][2] += 1
        else:
            if v.get("winner") in stats:
                stats[v["winner"]][0] += 1
            if v.get("loser") in stats:
                stats[v["loser"]][1] += 1
    rows = []
    for item, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        w, l, t = stats[item]
        n = w + l + t
        lo, hi = wilson(w + 0.5 * t, n)
        rate = (w + 0.5 * t) / n if n else 0.0
        rows.append([item, round(score, 3), w, l, t, f"{100 * rate:.0f}% [{100 * lo:.0f}–{100 * hi:.0f}%]" if n else "—"])
    return rows


# ---------- vision-model judge ----------
JUDGE_SCHEMA = {"type": "object", "properties": {"better": {"type": "string", "enum": ["first", "second", "tie"]}},
                "required": ["better"]}


def judge_prompt(prompt: str, criterion: str) -> str:
    return (f"Two images were generated from the same prompt: \"{prompt}\".\n"
            f"Which image is better? Judge: {criterion or 'how well it matches the prompt, and image quality'}.\n"
            'Answer with JSON only: {"better": "first" | "second" | "tie"}.')


def judge_pair(pair: dict, run_dir: str, target, settings: dict, criterion: str = "", seed: int = 7) -> dict:
    """Asks the judge twice with swapped order. Returns {"choice": left/right/tie/None, "consistent": bool}."""
    from ..image_utils import open_image
    from ..dataset.vision import ask, parse_json_object
    left, right = (open_image(os.path.join(run_dir, pair[k])) for k in ("left_path", "right_path"))
    try:
        answers = []
        for order in ((left, right), (right, left)):
            raw = ask(target, settings, judge_prompt(pair["prompt"], criterion), list(order), temperature=0.0, seed=seed,
                      max_tokens=30, json_schema=JUDGE_SCHEMA)
            try:
                answers.append(str(parse_json_object(raw).get("better", "")).lower())
            except ValueError:
                answers.append("")
    finally:
        left.close()
        right.close()
    first = {"first": "left", "second": "right", "tie": TIE}.get(answers[0])
    second = {"first": "right", "second": "left", "tie": TIE}.get(answers[1])  # Order was swapped
    consistent = first is not None and first == second
    return {"choice": first if consistent else None, "consistent": consistent, "raw": answers}


def judge_agreement(run_dir: str) -> str:
    """How often the judge's counted votes match the user's on the same pair (ties excluded)."""
    user = {v["key"]: v for v in load_votes(run_dir, "user") if not v.get("tie")}
    judge = [v for v in load_votes(run_dir, "judge") if not v.get("tie")]
    both = [(user[v["key"]], v) for v in judge if v["key"] in user]
    if not both:
        return "No pairs voted by both you and the judge yet."
    agree = sum(1 for u, j in both if u["winner"] == j["winner"])
    lo, hi = wilson(agree, len(both))
    return (f"Judge agrees with you on {agree}/{len(both)} pairs ({100 * agree / len(both):.0f}%, "
            f"95% interval {100 * lo:.0f}–{100 * hi:.0f}%). 50% is chance.")
