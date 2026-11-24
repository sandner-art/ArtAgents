# ArtAgent/core/lab/metrics.py
"""
Metric scoring for LoRA test runs (EXPERIMENTAL).

Fidelity:   mean cosine similarity of generated images to the training images
            (DreamBooth's CLIP-I / DINO; DINO correlates with human preference at only ~0.32).
Adherence:  CLIP-T, the cosine between prompt and image embeddings (CLIP models only).
Drift:      CMMD between a condition's images and the training set (MMD² with an RBF kernel on
            normalized embeddings, σ=10, ×1000, computed as in google-research/cmmd).
Leakage:    the change in fidelity on trigger-less and unrelated prompts against the no-LoRA baseline.
Every number comes with a bootstrap 95% interval over prompts × seeds; differences smaller than
the interval are noise.

Embedding models are optional: open_clip (pip install open_clip_torch) or DINOv2 via transformers.
They download weights from Hugging Face on first use and need PyTorch.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

CMMD_SIGMA, CMMD_SCALE = 10.0, 1000.0

EMBEDDERS = {
    "CLIP ViT-B/32 (open_clip, LAION-2B)": ("open_clip", "ViT-B-32", "laion2b_s34b_b79k"),
    "CLIP ViT-L/14 (open_clip, OpenAI)": ("open_clip", "ViT-L-14", "openai"),
    "DINOv2 small (transformers)": ("dinov2", "facebook/dinov2-small", None),
}


class MetricsError(Exception):
    pass


# ---------- maths ----------
def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return normalize(a) @ normalize(b).T


def fidelity_scores(generated: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per generated image: mean cosine to all reference images."""
    if len(generated) == 0 or len(reference) == 0:
        return np.zeros(0)
    return pairwise_cosine(generated, reference).mean(axis=1)


def adherence_scores(images: np.ndarray, texts: np.ndarray) -> np.ndarray:
    """Per image: cosine to its own prompt (rows aligned)."""
    return np.sum(normalize(images) * normalize(texts), axis=1)


def cmmd(x: np.ndarray, y: np.ndarray, sigma: float = CMMD_SIGMA, scale: float = CMMD_SCALE) -> float:
    """CMMD (Jayasumana et al. 2024), the reference implementation's estimator; usable at small n, unlike FID."""
    x, y = normalize(x), normalize(y)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    gamma = 1.0 / (2 * sigma ** 2)

    def k(a, b):
        sq = np.sum(a ** 2, 1)[:, None] + np.sum(b ** 2, 1)[None, :] - 2 * a @ b.T
        return np.exp(-gamma * np.maximum(sq, 0))

    # Plain means including the diagonal, exactly as google-research/cmmd distance.py (always ≥ 0)
    return float(scale * (k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()))


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, low, high) percentile bootstrap interval."""
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float(v[0]), float(v[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def fmt_ci(ci: tuple[float, float, float], digits: int = 3) -> str:
    m, lo, hi = ci
    if np.isnan(m):
        return "—"
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# ---------- embedders ----------
def available_embedders() -> list[str]:
    has_torch = importlib.util.find_spec("torch") is not None
    out = []
    for label, (kind, _, _) in EMBEDDERS.items():
        if not has_torch:
            continue
        if kind == "open_clip" and importlib.util.find_spec("open_clip"):
            out.append(label)
        elif kind == "dinov2" and importlib.util.find_spec("transformers"):
            out.append(label)
    return out


def install_hint() -> str:
    return ("Metric scoring needs PyTorch plus open_clip or transformers, e.g. `pip install torch open_clip_torch` "
            "(see pytorch.org for the GPU build). Models download from Hugging Face on first use.")


class Embedder:
    """Wraps one embedding model; images and texts → numpy arrays (rows = items)."""

    def __init__(self, label: str):
        if label not in EMBEDDERS:
            raise MetricsError(f"Unknown embedder: {label}")
        self.label = label
        self.kind, self.model_name, self.pretrained = EMBEDDERS[label]
        self._model = self._preprocess = self._tokenizer = self._processor = None

    @property
    def has_text(self) -> bool:
        return self.kind == "open_clip"

    def _load(self):
        if self._model is not None:
            return
        if importlib.util.find_spec("torch") is None:
            raise MetricsError(install_hint())
        try:
            if self.kind == "open_clip":
                import open_clip
                model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
                self._model, self._preprocess = model.eval(), preprocess
                self._tokenizer = open_clip.get_tokenizer(self.model_name)
            else:
                from transformers import AutoImageProcessor, AutoModel
                self._processor = AutoImageProcessor.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name).eval()
        except ImportError as e:
            raise MetricsError(install_hint()) from e
        except Exception as e:  # Download or load failure
            raise MetricsError(f"Could not load {self.label}: {e.__class__.__name__}: {str(e)[:200]}") from e

    def embed_images(self, images: list, batch: int = 16) -> np.ndarray:
        self._load()
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch):
                chunk = [im.convert("RGB") for im in images[i:i + batch]]
                if self.kind == "open_clip":
                    x = torch.stack([self._preprocess(im) for im in chunk])
                    out.append(self._model.encode_image(x).float().cpu().numpy())
                else:
                    inputs = self._processor(images=chunk, return_tensors="pt")
                    out.append(self._model(**inputs).last_hidden_state[:, 0].float().cpu().numpy())  # CLS token
        return np.concatenate(out) if out else np.zeros((0, 1))

    def embed_texts(self, texts: list[str]) -> np.ndarray | None:
        if not self.has_text:
            return None
        self._load()
        import torch
        with torch.no_grad():
            return self._model.encode_text(self._tokenizer(texts)).float().cpu().numpy()


def _load_images(paths: list[str]):
    from ..image_utils import open_image
    images = []
    for p in paths:
        with open_image(p) as im:
            images.append(im.convert("RGB").copy())
    return images


def cached_image_embeddings(embedder: Embedder, paths: list[str], cache_file: str) -> np.ndarray:
    """Embeddings cached in an .npz keyed by path + mtime + size."""
    keys = [f"{os.path.abspath(p)}|{os.path.getmtime(p)}|{os.path.getsize(p)}" for p in paths]
    cache = {}
    if os.path.exists(cache_file):
        try:
            with np.load(cache_file, allow_pickle=False) as data:
                if str(data["model"]) == embedder.label:
                    cache = dict(zip([str(k) for k in data["keys"]], data["vectors"]))
        except (OSError, ValueError, KeyError):
            cache = {}
    missing = [i for i, k in enumerate(keys) if k not in cache]
    if missing:
        vectors = embedder.embed_images(_load_images([paths[i] for i in missing]))
        for i, v in zip(missing, vectors):
            cache[keys[i]] = v
        all_keys = list(cache)
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        np.savez(cache_file, model=np.array(embedder.label), keys=np.array(all_keys), vectors=np.stack([cache[k] for k in all_keys]))
    return np.stack([cache[k] for k in keys]) if keys else np.zeros((0, 1))


# ---------- scoring a run ----------
SCORE_HEADERS = ["Condition", "Bucket", "Images", "Fidelity (to training)", "Δ vs baseline", "Adherence (CLIP-T)", "CMMD ↓"]


def score_run(run: dict, run_dir: str, train_paths: list[str], embedder: Embedder) -> dict:
    """Returns {"rows": table rows, "summary": markdown}."""
    from .lora_sweep import BASELINE, lora_label
    if not train_paths:
        raise MetricsError("Choose the training image folder (fidelity is measured against it).")
    jobs = {j["index"]: j for j in run.get("jobs", [])}
    results = [(jobs[r["job"]], os.path.join(run_dir, r["path"])) for r in run.get("results", [])
               if r.get("path") and r["job"] in jobs and os.path.exists(os.path.join(run_dir, r["path"]))]
    if not results:
        raise MetricsError("This run has no images yet.")
    train = cached_image_embeddings(embedder, train_paths, os.path.join(run_dir, "embeddings_train.npz"))
    gen = cached_image_embeddings(embedder, [p for _, p in results], os.path.join(run_dir, "embeddings_generated.npz"))
    fid = fidelity_scores(gen, train)
    adh = None
    if embedder.has_text:
        adh = adherence_scores(gen, embedder.embed_texts([j["prompt"] for j, _ in results]))

    def cond(j):
        return BASELINE if j["lora"] is None else f"{lora_label(j['lora'])} @ {j['strength']:g}"

    groups: dict[tuple[str, str], list[int]] = {}
    for i, (j, _) in enumerate(results):
        groups.setdefault((cond(j), j["bucket"]), []).append(i)
    baseline_fid = {b: fid[idx].mean() for (c, b), idx in groups.items() if c == BASELINE}
    rows = []
    for (c, bucket), idx in sorted(groups.items(), key=lambda kv: (kv[0][0] != BASELINE, kv[0])):
        f_ci = bootstrap_ci(fid[idx])
        delta = "" if c == BASELINE or bucket not in baseline_fid else f"{f_ci[0] - baseline_fid[bucket]:+.3f}"
        rows.append([c, bucket, len(idx), fmt_ci(f_ci), delta, fmt_ci(bootstrap_ci(adh[idx])) if adh is not None else "—",
                     f"{cmmd(gen[idx], train):.3f}" if len(idx) > 1 and len(train) > 1 else "—"])
    lines = [f"**Scored {len(results)} image(s) with {embedder.label}** against {len(train_paths)} training image(s).",
             "- Fidelity: mean cosine to the training images, with a bootstrap 95% interval over prompts × seeds.",
             "- Δ vs baseline on **trigger-less** and **unrelated** prompts is leakage: the LoRA changes images "
             "without its trigger (a large Δ there is usually bad).",
             "- Treat differences smaller than the intervals as noise. These are signals, not verdicts: "
             "confirm with blind votes."]
    return {"rows": rows, "summary": "\n".join(lines)}
