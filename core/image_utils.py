# ArtAgent/core/image_utils.py
"""
Safe image helpers shared by chat, captioning and the Ollama agent.

Pillow detects a file's format from its *content*, not its extension, so a file
named "photo.png" that is really an EPS, TIFF, PDF, MSP or McIdas file would be
handed to that (much riskier) decoder -- EPS even shells out to Ghostscript.
Every image ArtAgents reads from disk goes through `open_image`, which only
allows the formats the app actually supports.
"""
import base64
import io
import os
import re

from PIL import Image, ImageOps

# Extensions shown/processed by the app, and the Pillow decoders allowed to open them.
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
ALLOWED_IMAGE_FORMATS = ("PNG", "JPEG", "BMP", "GIF", "WEBP")

# Gallery thumbnails: big enough for the preview pane, small enough to load fast.
THUMBNAIL_MAX_SIDE = 768

MIN_PILLOW_VERSION = (13, 0, 0)


def pillow_version_tuple(version: str | None = None) -> tuple[int, ...]:
    """Parses a Pillow version string such as '13.0.0' into a tuple of ints."""
    version = version or Image.__version__
    parts = []
    for piece in version.split("."):
        digits = re.match(r"\d+", piece) # Leading digits only: "0rc1" -> 0
        if not digits:
            break
        parts.append(int(digits.group()))
    return tuple(parts)


def pillow_security_warning(version: str | None = None) -> str | None:
    """Returns a console warning if the installed Pillow predates the security fixes."""
    if pillow_version_tuple(version) >= MIN_PILLOW_VERSION:
        return None
    required = ".".join(str(p) for p in MIN_PILLOW_VERSION)
    return (
        f"WARNING: Pillow {version or Image.__version__} is installed; ArtAgents needs Pillow {required}+ "
        "for current security fixes.\n"
        "         Run: pip install --no-deps -r requirements-pillow.txt"
    )


def is_image_filename(filename: str) -> bool:
    """True if the filename has one of the supported image extensions."""
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def open_image(path: str) -> Image.Image:
    """
    Opens an image restricted to the supported formats.

    Raises PIL.UnidentifiedImageError if the content is not PNG/JPEG/BMP/GIF/WEBP,
    and Pillow's DecompressionBombError for absurdly large images.
    """
    return Image.open(path, formats=ALLOWED_IMAGE_FORMATS)


def make_thumbnail(path: str, max_side: int = THUMBNAIL_MAX_SIDE) -> Image.Image:
    """
    Loads an image from disk and returns a small RGB(A) copy for display.

    Returning an in-memory thumbnail (instead of the file path) lets Gradio serve
    the gallery from its own cache, so the app never has to expose the user's
    folders through `allowed_paths`.
    """
    with open_image(path) as img:
        img.draft("RGB", (max_side, max_side))  # Fast JPEG downscale; no-op for other formats
        img = ImageOps.exif_transpose(img)
        img.thumbnail((max_side, max_side))
        if img.mode not in ("RGB", "RGBA"):
            bands = img.getbands()
            has_alpha = "A" in bands or "a" in bands or "transparency" in img.info
            img = img.convert("RGBA" if has_alpha else "RGB")
        return img.copy()


def encode_image_for_ollama(img: Image.Image) -> str:
    """
    Encodes a PIL image as base64 for the Ollama `images` field.

    Images with transparency are sent as PNG, everything else as JPEG. Palette
    (GIF), 16-bit, CMYK and other modes are converted first; saving those
    straight to JPEG raises "cannot write mode P as JPEG".
    """
    bands = img.getbands()
    has_alpha = "A" in bands or "a" in bands or (img.mode == "P" and "transparency" in img.info)
    if img.mode == "La":  # Premultiplied grayscale+alpha has no direct RGBA conversion
        img = img.convert("LA")
    if has_alpha:
        prepared, save_format = (img if img.mode == "RGBA" else img.convert("RGBA")), "PNG"
    else:
        prepared, save_format = (img if img.mode in ("RGB", "L") else img.convert("RGB")), "JPEG"
    buffered = io.BytesIO()
    prepared.save(buffered, format=save_format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
