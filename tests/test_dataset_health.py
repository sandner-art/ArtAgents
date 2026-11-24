# ArtAgent/tests/test_dataset_health.py
"""Tests for core/dataset/health.py (pixel checks, duplicates, caption checks, cache, quarantine)."""
import os
import shutil
import sys
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dataset import health as hl


def _noise(seed, size=(256, 256)):
    rng = np.random.default_rng(seed)
    # Smooth-ish random pattern (upsampled noise) so perceptual hashes are stable under re-encoding
    small = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    return Image.fromarray(small).resize(size, Image.BICUBIC)


def _sharp(seed, size=(600, 600)):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8))


# --- hashes & measurements ---

def test_phash_identical_reencoded_and_different(tmp_path):
    a = _noise(1)
    a.save(tmp_path / "a.png")
    a.save(tmp_path / "a.jpg", quality=80)
    h_png = hl.phash(Image.open(tmp_path / "a.png").convert("L"))
    h_jpg = hl.phash(Image.open(tmp_path / "a.jpg").convert("L"))
    h_other = hl.phash(_noise(2).convert("L"))
    assert hl.hamming(h_png, h_png) == 0
    assert hl.hamming(h_png, h_jpg) <= hl.NEAR_DUP_HAMMING
    assert hl.hamming(h_png, h_other) > hl.NEAR_DUP_HAMMING
    assert 0 <= h_png < 2 ** 64


def test_hamming():
    assert hl.hamming(0b1011, 0b0001) == 2
    assert hl.hamming(0, 2 ** 64 - 1) == 64


def test_laplacian_variance_sharp_vs_blurred():
    sharp = _sharp(3, (300, 300)).convert("L")
    blurred = sharp.filter(ImageFilter.GaussianBlur(6))
    assert hl.laplacian_variance(sharp) > 20 * hl.laplacian_variance(blurred)
    assert hl.laplacian_variance(Image.new("L", (2, 2))) == 0.0
    assert hl.laplacian_variance(Image.new("L", (50, 50), 128)) == 0.0


def test_laplacian_variance_downscales_large_images():
    big = _sharp(4, (2000, 1000)).convert("L")
    assert hl.laplacian_variance(big) > 0


def test_measure_basic_and_alpha(tmp_path):
    Image.new("RGBA", (40, 20), (255, 0, 0, 255)).save(tmp_path / "opaque.png")
    img = Image.new("RGBA", (40, 20), (255, 0, 0, 255))
    img.putpixel((0, 0), (0, 0, 0, 0))
    img.paste((0, 0, 0, 0), (0, 0, 10, 10))
    img.save(tmp_path / "transparent.png")
    Image.new("RGB", (40, 20), (10, 10, 10)).save(tmp_path / "dark.jpg")
    m = hl.measure(str(tmp_path / "opaque.png"))
    assert (m["width"], m["height"], m["mode"], m["format"]) == (40, 20, "RGBA", "PNG")
    assert m["alpha"] is False
    assert hl.measure(str(tmp_path / "transparent.png"))["alpha"] is True
    dark = hl.measure(str(tmp_path / "dark.jpg"))
    assert dark["alpha"] is False and dark["brightness"] < 25
    assert len(dark["sha256"]) == 64 and len(dark["phash"]) == 16


def test_file_sha256(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert hl.file_sha256(str(p)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


# --- scan ---

@pytest.fixture
def dataset(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    base = _sharp(10)
    base.save(d / "a.png")
    shutil.copy(d / "a.png", d / "b.png")                        # Exact duplicate of a
    base.save(d / "c.jpg", quality=92)                           # Near duplicate of a
    _sharp(11).save(d / "d.png")
    _sharp(12).filter(ImageFilter.GaussianBlur(8)).save(d / "e.png")   # Blurry
    _sharp(13, (100, 100)).save(d / "f.png")                     # Small
    _sharp(14).save(d / "g.png")
    captions = {"a.png": "txcl, cat", "c.jpg": "", "d.png": "txcl, dog\nvariant", "e.png": "dog, park",
                "f.png": "txcl, cat", "g.png": "txcl, " + ", ".join(f"tag{i}" for i in range(40))}
    for name, text in captions.items():
        (d / (os.path.splitext(name)[0] + ".txt")).write_text(text, encoding="utf-8")
    paths = {n: str(d / n) for n in sorted(os.listdir(d)) if n.endswith((".png", ".jpg"))}
    return d, paths, captions


def _issues(result, name):
    return {issue for img, issue, _ in result["issues"] if img == name}


def test_scan_finds_expected_issues(dataset):
    d, paths, captions = dataset
    budgets = [{"tokenizer": "clip", "budget": 75, "max": 225, "label": "CLIP"}]
    result = hl.scan(paths, captions, resolution=512, budgets=budgets, trigger="txcl")
    assert hl.ISSUE_DUP in _issues(result, "b.png")
    assert hl.ISSUE_DUP not in _issues(result, "a.png")              # The first copy is the original
    assert hl.ISSUE_NEAR_DUP in _issues(result, "c.jpg")
    assert hl.ISSUE_BLUR in _issues(result, "e.png")
    assert hl.ISSUE_BLUR not in _issues(result, "d.png")
    assert hl.ISSUE_SMALL in _issues(result, "f.png")
    assert hl.ISSUE_NO_CAPTION in _issues(result, "b.png")
    assert hl.ISSUE_EMPTY in _issues(result, "c.jpg")
    assert hl.ISSUE_MULTILINE in _issues(result, "d.png")
    assert hl.ISSUE_NO_TRIGGER in _issues(result, "e.png")
    assert hl.ISSUE_SAME_CAPTION in _issues(result, "a.png") and hl.ISSUE_SAME_CAPTION in _issues(result, "f.png")
    assert hl.ISSUE_BUDGET in _issues(result, "g.png")
    assert _issues(result, "g.png") == {hl.ISSUE_BUDGET}
    assert result["exact_dupes"] == [["a.png", "b.png"]]
    assert any("c.jpg" in g for g in result["near_dupes"])
    assert set(result["flagged"]) == {"a.png", "b.png", "c.jpg", "d.png", "e.png", "f.png", "g.png"}
    assert result["summary"].startswith("**7 image(s) scanned, 7 with something to check.**")
    assert "nothing is deleted" in result["summary"]


def test_scan_below_training_resolution(dataset):
    _, paths, captions = dataset
    result = hl.scan(paths, captions, resolution=1024)
    detail = [det for img, issue, det in result["issues"] if img == "d.png" and issue == hl.ISSUE_UPSCALE][0]
    assert "600×600 = 0.36 MP" in detail


def test_scan_blur_needs_five_images(tmp_path):
    _sharp(1).save(tmp_path / "a.png")
    _sharp(2).filter(ImageFilter.GaussianBlur(8)).save(tmp_path / "b.png")
    paths = {n: str(tmp_path / n) for n in ("a.png", "b.png")}
    result = hl.scan(paths, {}, resolution=512)
    assert all(issue != hl.ISSUE_BLUR for _, issue, _ in result["issues"])


def test_scan_aspect_mode_alpha_brightness(tmp_path):
    Image.new("RGB", (1600, 500), (128, 128, 128)).save(tmp_path / "wide.png")
    Image.new("CMYK", (600, 600), (0, 0, 0, 0)).save(tmp_path / "cmyk.jpg")
    img = Image.new("RGBA", (600, 600), (128, 100, 90, 255))
    img.paste((0, 0, 0, 0), (0, 0, 100, 100))
    img.save(tmp_path / "alpha.png")
    Image.new("RGB", (600, 600), (250, 250, 250)).save(tmp_path / "bright.png")
    paths = {n: str(tmp_path / n) for n in ("wide.png", "cmyk.jpg", "alpha.png", "bright.png")}
    result = hl.scan(paths, {}, resolution=512)
    assert hl.ISSUE_ASPECT in _issues(result, "wide.png")
    assert hl.ISSUE_MODE in _issues(result, "cmyk.jpg")
    assert hl.ISSUE_ALPHA in _issues(result, "alpha.png")
    assert hl.ISSUE_BRIGHT in _issues(result, "bright.png")
    assert hl.ISSUE_LOW_CONTRAST in _issues(result, "bright.png")


def test_scan_unreadable_file(tmp_path):
    (tmp_path / "broken.png").write_bytes(b"not an image")
    result = hl.scan({"broken.png": str(tmp_path / "broken.png")}, {}, resolution=512)
    assert hl.ISSUE_UNREADABLE in _issues(result, "broken.png")


def test_scan_caches_measurements(dataset):
    _, paths, captions = dataset
    hl.scan(paths, captions, resolution=512)
    cache = os.path.join(os.path.dirname(paths["a.png"]), ".artagents", "cache", "health.json")
    assert os.path.isfile(cache)
    with patch("core.dataset.health.measure", side_effect=AssertionError("should use the cache")):
        again = hl.scan(paths, captions, resolution=512)
    assert hl.ISSUE_DUP in _issues(again, "b.png")


def test_scan_remeasures_changed_files(dataset):
    _, paths, captions = dataset
    hl.scan(paths, captions, resolution=512)
    _sharp(99).save(paths["g.png"])  # Content and size change
    calls = []
    real = hl.measure
    with patch("core.dataset.health.measure", side_effect=lambda p: calls.append(os.path.basename(p)) or real(p)):
        hl.scan(paths, captions, resolution=512)
    assert calls == ["g.png"]


def test_scan_progress_callback(dataset):
    _, paths, captions = dataset
    seen = []
    hl.scan(paths, captions, resolution=512, progress=lambda frac, desc="": seen.append(desc))
    assert len(seen) == len(paths) and seen[0].startswith("Scanning ")


def test_measure_all_empty():
    assert hl.measure_all({}) == {}


def test_sizes_from_stats():
    assert hl.sizes_from_stats({"a": {"width": 3, "height": 4}, "b": {"error": "x"}}) == {"a": (3, 4)}


def test_near_duplicate_groups_transitive():
    stats = {"a": {"phash": "0000000000000000"}, "b": {"phash": "00000000000000ff"},   # 8 bits from a
             "c": {"phash": "000000000000ffff"},                                        # 8 bits from b, 16 from a
             "d": {"phash": "ffffffffffffffff"}, "e": {}}
    assert hl.near_duplicate_groups(stats) == [["a", "b", "c"]]


# --- quarantine ---

def test_quarantine_and_restore(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(img)
    (tmp_path / "a.txt").write_text("cap", encoding="utf-8")
    moved = hl.quarantine(str(img))
    qdir = tmp_path / ".artagents" / "quarantine"
    assert moved == str(qdir / "a.png")
    assert not img.exists() and not (tmp_path / "a.txt").exists()
    assert (qdir / "a.png").exists() and (qdir / "a.txt").read_text(encoding="utf-8") == "cap"
    assert hl.restore_quarantine(str(tmp_path)) == ["a.png", "a.txt"]
    assert img.exists() and (tmp_path / "a.txt").exists()


def test_quarantine_twice_refuses(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(img)
    hl.quarantine(str(img))
    Image.new("RGB", (4, 4)).save(img)
    with pytest.raises(FileExistsError):
        hl.quarantine(str(img))
    assert img.exists()


def test_restore_never_overwrites(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(img)
    hl.quarantine(str(img))
    Image.new("RGB", (4, 4), (0, 255, 0)).save(img)      # A new file with the same name
    assert hl.restore_quarantine(str(tmp_path)) == []
    assert Image.open(img).getpixel((0, 0)) == (0, 255, 0)
    assert (tmp_path / ".artagents" / "quarantine" / "a.png").exists()


def test_restore_nothing(tmp_path):
    assert hl.restore_quarantine(str(tmp_path)) == []
