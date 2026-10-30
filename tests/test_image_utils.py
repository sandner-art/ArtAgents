# ArtAgent/tests/test_image_utils.py

import base64
import io
import os
import sys

import pytest

# --- Adjust import path ---
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir)
sys.path.insert(0, project_root)

from PIL import Image, UnidentifiedImageError

from core.image_utils import (
    ALLOWED_IMAGE_FORMATS,
    THUMBNAIL_MAX_SIDE,
    encode_image_for_ollama,
    is_image_filename,
    make_thumbnail,
    open_image,
    pillow_security_warning,
    pillow_version_tuple,
)

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _decode(b64: str) -> bytes:
    return base64.b64decode(b64)


# --- open_image: decoder whitelist ---

@pytest.mark.parametrize("fmt, ext", [("PNG", "png"), ("JPEG", "jpg"), ("WEBP", "webp"), ("GIF", "gif"), ("BMP", "bmp")])
def test_open_image_accepts_supported_formats(tmp_path, fmt, ext):
    path = tmp_path / f"ok.{ext}"
    Image.new("RGB", (8, 8), "blue").save(path, format=fmt)
    with open_image(str(path)) as img:
        assert img.format == fmt


def test_open_image_rejects_tiff_named_png(tmp_path):
    """Pillow sniffs content, not extensions: a TIFF renamed to .png must not reach the TIFF decoder."""
    path = tmp_path / "x.png"
    Image.new("RGB", (8, 8)).save(path, format="TIFF")
    with Image.open(path) as unrestricted:
        assert unrestricted.format == "TIFF" # Plain Image.open would happily decode it
    with pytest.raises(UnidentifiedImageError):
        open_image(str(path))


def test_open_image_rejects_xbm_named_png(tmp_path):
    """XBM (regex-parsed header, ReDoS advisory) is not in the whitelist."""
    path = tmp_path / "x.png"
    path.write_bytes(b"#define im_width 8\n#define im_height 1\nstatic char im_bits[] = {\n0xff,\n};\n")
    with Image.open(path) as unrestricted:
        assert unrestricted.format == "XBM"
    with pytest.raises(UnidentifiedImageError):
        open_image(str(path))


def test_open_image_rejects_eps_named_jpg(tmp_path):
    """EPS would shell out to Ghostscript; it must be refused before any decoding happens."""
    path = tmp_path / "x.jpg"
    path.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 8 8\nshowpage\n")
    with pytest.raises(UnidentifiedImageError):
        open_image(str(path))


def test_allowed_formats_are_the_expected_set():
    assert set(ALLOWED_IMAGE_FORMATS) == {"PNG", "JPEG", "BMP", "GIF", "WEBP"}


@pytest.mark.parametrize("name, expected", [
    ("a.png", True), ("b.JPG", True), ("c.jpeg", True), ("d.webp", True), ("e.gif", True), ("f.bmp", True),
    ("g.tif", False), ("h.eps", False), ("i.txt", False), ("noext", False),
])
def test_is_image_filename(name, expected):
    assert is_image_filename(name) is expected


# --- make_thumbnail ---

def test_make_thumbnail_bounds_size_and_keeps_aspect(tmp_path):
    path = tmp_path / "big.jpg"
    Image.new("RGB", (3000, 1500), "red").save(path, format="JPEG")
    thumb = make_thumbnail(str(path))
    assert max(thumb.size) <= THUMBNAIL_MAX_SIDE
    assert thumb.size[0] == 2 * thumb.size[1] or abs(thumb.size[0] / thumb.size[1] - 2) < 0.02
    assert thumb.mode == "RGB"


def test_make_thumbnail_small_image_not_upscaled(tmp_path):
    path = tmp_path / "small.png"
    Image.new("RGB", (40, 30)).save(path)
    assert make_thumbnail(str(path)).size == (40, 30)


@pytest.mark.parametrize("mode, kwargs, expected_mode", [
    ("RGBA", {}, "RGBA"),
    ("LA", {}, "RGBA"),
    ("L", {}, "RGB"),
    ("P", {"transparency": 0}, "RGBA"),
    ("P", {}, "RGB"),
])
def test_make_thumbnail_returns_rgb_or_rgba(tmp_path, mode, kwargs, expected_mode):
    path = tmp_path / "img.png"
    Image.new(mode, (20, 20)).save(path, **kwargs)
    thumb = make_thumbnail(str(path))
    assert thumb.mode == expected_mode


def test_make_thumbnail_is_detached_from_file(tmp_path):
    """The thumbnail must be usable after the source file is gone (Gradio saves it later)."""
    path = tmp_path / "gone.png"
    Image.new("RGB", (20, 20), "green").save(path)
    thumb = make_thumbnail(str(path))
    os.remove(path)
    assert thumb.getpixel((0, 0)) == (0, 128, 0)


def test_make_thumbnail_rejects_disguised_file(tmp_path):
    path = tmp_path / "x.png"
    Image.new("RGB", (8, 8)).save(path, format="TIFF")
    with pytest.raises(UnidentifiedImageError):
        make_thumbnail(str(path))


# --- encode_image_for_ollama ---

@pytest.mark.parametrize("mode", ["RGB", "L", "CMYK", "I;16", "1"])
def test_encode_opaque_modes_as_jpeg(mode):
    data = _decode(encode_image_for_ollama(Image.new(mode, (8, 8))))
    assert data.startswith(JPEG_MAGIC)
    with Image.open(io.BytesIO(data)) as decoded:
        assert decoded.mode in ("RGB", "L")


@pytest.mark.parametrize("mode", ["RGBA", "LA"])
def test_encode_alpha_modes_as_png(mode):
    data = _decode(encode_image_for_ollama(Image.new(mode, (8, 8))))
    assert data.startswith(PNG_MAGIC)
    with Image.open(io.BytesIO(data)) as decoded:
        assert decoded.mode == "RGBA"


def test_encode_palette_gif_without_transparency(tmp_path):
    """GIF frames are mode 'P'; saving those as JPEG used to raise 'cannot write mode P as JPEG'."""
    path = tmp_path / "anim.gif"
    Image.new("P", (8, 8), 5).save(path)
    with Image.open(path) as gif:
        assert gif.mode == "P"
        data = _decode(encode_image_for_ollama(gif))
    assert data.startswith(JPEG_MAGIC)


def test_encode_palette_with_transparency_as_png(tmp_path):
    path = tmp_path / "transparent.gif"
    Image.new("P", (8, 8), 0).save(path, transparency=0)
    with Image.open(path) as gif:
        assert "transparency" in gif.info
        data = _decode(encode_image_for_ollama(gif))
    assert data.startswith(PNG_MAGIC)


def test_encode_does_not_modify_input():
    img = Image.new("P", (8, 8), 1)
    encode_image_for_ollama(img)
    assert img.mode == "P"


# --- Pillow version guard ---

def test_pillow_security_warning_for_patched_versions():
    assert pillow_security_warning("13.0.0") is None
    assert pillow_security_warning("13.1.2") is None
    assert pillow_security_warning("14.0.0.dev0") is None


def test_pillow_security_warning_for_old_versions():
    msg = pillow_security_warning("12.3.0")
    assert msg is not None
    assert "12.3.0" in msg and "13.0.0" in msg
    assert "requirements-pillow.txt" in msg
    assert pillow_security_warning("9.5.0") is not None


def test_pillow_version_tuple_parsing():
    assert pillow_version_tuple("13.0.0") == (13, 0, 0)
    assert pillow_version_tuple("9.5.0") == (9, 5, 0)
    assert pillow_version_tuple("14.0.0.dev0")[:3] == (14, 0, 0)



def test_pillow_version_tuple_ignores_prerelease_suffix():
    assert pillow_version_tuple("12.3.0rc1")[:3] == (12, 3, 0)
    assert pillow_version_tuple("13.0.0b2")[:3] == (13, 0, 0)


def test_installed_pillow_is_patched():
    """The test environment itself should run the patched Pillow."""
    assert pillow_security_warning() is None, pillow_security_warning()
