# ArtAgent/tests/test_captioning_logic.py

import pytest
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch, mock_open, MagicMock

# --- Adjust import path ---
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir)
sys.path.insert(0, project_root)

try:
    from PIL import Image
    from core.captioning_logic import (
        load_images_and_captions,
        update_caption_display_from_gallery,
        save_caption,
        batch_edit_captions,
        generate_captions_for_selected,
        generate_captions_for_all,
    )
    from core.image_utils import THUMBNAIL_MAX_SIDE
except ImportError as e:
    pytest.skip(f"Skipping captioning_logic tests, modules not found: {e}", allow_module_level=True)

EXECUTE_PATH = 'core.captioning_logic.execute_chat_or_team'
FIXED_PROMPT_START = "Describe this image concisely."


# --- Helpers / Fixtures ---

def _make_image(path, size=(32, 24), mode="RGB", fmt=None):
    Image.new(mode, size, "white" if mode == "RGB" else 0).save(path, format=fmt)
    return path


@pytest.fixture
def caption_test_folder(tmp_path):
    """Creates a temporary folder with real (tiny) images and caption files."""
    d = tmp_path / "caption_folder"
    d.mkdir()
    _make_image(d / "image1.png")
    _make_image(d / "image2.jpg")
    _make_image(d / "image3.webp")
    _make_image(d / "image_without_caption.jpeg", fmt="JPEG")
    (d / "subfolder").mkdir() # Should be ignored
    (d / "not_an_image.txt").write_text("ignore me", encoding="utf-8") # Should be ignored

    (d / "image1.txt").write_text("Caption for image 1.", encoding='utf-8')
    (d / "image2.txt").write_text("Caption for image 2, with comma.", encoding='utf-8')
    # image3.txt and image_without_caption.txt intentionally missing
    return d


def _agent_returns(text):
    """Return value of execute_chat_or_team: (response, session_text, model_state, session_list)."""
    return (text, "session text", None, ["previous", "new entry"])


COMMON_ARGS = dict(
    agent_or_team_display_name="Captioner",
    selected_model_display_name="vision-model (VISION)",
    settings={}, models_data=[{"name": "vision-model", "vision": True}],
    limiters_data={}, teams_data={}, file_agents={}, history_list=[], session_history=["previous"],
)


# --- Tests for load_images_and_captions ---

def test_load_images_and_captions_success(caption_test_folder):
    """Gallery gets (thumbnail, filename) pairs; paths and captions are keyed by filename."""
    folder_path = str(caption_test_folder)
    gallery, img_paths, captions, status, first_img, first_cap, first_file_disp = load_images_and_captions(folder_path)

    expected_filenames = sorted(["image1.png", "image2.jpg", "image_without_caption.jpeg", "image3.webp"])
    assert status == "Loaded 4 image(s)."
    assert [label for _, label in gallery] == expected_filenames
    # In-memory thumbnails, not file paths (so the app never needs allowed_paths)
    assert all(isinstance(img, Image.Image) for img, _ in gallery)
    assert list(img_paths.keys()) == expected_filenames

    assert img_paths["image1.png"] == os.path.join(folder_path, "image1.png")
    assert captions["image1.png"] == "Caption for image 1."
    assert captions["image2.jpg"] == "Caption for image 2, with comma."
    assert captions["image3.webp"] == ""
    assert captions["image_without_caption.jpeg"] == ""

    assert first_img == "image1.png"
    assert first_cap == "Caption for image 1."
    assert first_file_disp == "image1.png"


def test_load_images_and_captions_thumbnails_are_bounded(tmp_path):
    """Large images are downscaled for the gallery."""
    _make_image(tmp_path / "big.png", size=(2000, 1000))
    gallery, *_ = load_images_and_captions(str(tmp_path))
    thumb, label = gallery[0]
    assert label == "big.png"
    assert max(thumb.size) <= THUMBNAIL_MAX_SIDE
    assert thumb.mode in ("RGB", "RGBA")


def test_load_images_and_captions_skips_disguised_and_corrupt_files(tmp_path):
    """Files with an image extension but other content (TIFF, junk) are skipped, not decoded."""
    _make_image(tmp_path / "good.png")
    _make_image(tmp_path / "really_a_tiff.png", fmt="TIFF")
    (tmp_path / "junk.jpg").write_bytes(b"this is not an image")

    gallery, img_paths, captions, status, *_ = load_images_and_captions(str(tmp_path))

    assert [label for _, label in gallery] == ["good.png"]
    assert list(img_paths) == ["good.png"]
    assert "Loaded 1 image(s)." in status
    assert "Skipped 2 unreadable or unsupported file(s)" in status


def test_load_images_and_captions_empty_folder(tmp_path):
    """Test loading from an empty folder."""
    d = tmp_path / "empty_folder"
    d.mkdir()
    gallery, img_paths, captions, status, first_img, first_cap, first_file_disp = load_images_and_captions(str(d))
    assert status == "No supported image files found."
    assert gallery == []
    assert img_paths == {}
    assert captions == {}
    assert first_img is None
    assert first_cap == ""
    assert first_file_disp is None


def test_load_images_and_captions_no_images(tmp_path):
    """Test loading from a folder with no supported images."""
    (tmp_path / "document.txt").touch()
    (tmp_path / "archive.zip").touch()
    gallery, *_rest = load_images_and_captions(str(tmp_path))
    assert _rest[2] == "No supported image files found."
    assert gallery == []


def test_load_images_and_captions_invalid_path():
    """Test loading with an invalid folder path."""
    gallery, img_paths, captions, status, *_ = load_images_and_captions("path/that/does/not/exist")
    assert status.startswith("Error: Invalid folder path provided:")
    assert gallery == []


def test_load_images_and_captions_no_path():
    """Test loading with no folder path provided."""
    gallery, img_paths, captions, status, *_ = load_images_and_captions("")
    assert status == "Error: No folder specified."
    assert gallery == []


def test_load_images_and_captions_read_error(tmp_path, capsys):
    """An unreadable caption file yields an empty caption and a console warning."""
    _make_image(tmp_path / "image1.png")
    _make_image(tmp_path / "image2.jpg")
    (tmp_path / "image1.txt").mkdir() # "exists" but cannot be opened as a file
    (tmp_path / "image2.txt").write_text("Caption 2", encoding="utf-8")

    _, img_paths, captions, *_ = load_images_and_captions(str(tmp_path))

    assert "image1.png" in img_paths
    assert captions["image1.png"] == ""
    assert captions["image2.jpg"] == "Caption 2"
    assert "Warning: Could not read caption file image1.txt" in capsys.readouterr().out


# --- Tests for update_caption_display_from_gallery ---

SAMPLE_CAPTIONS = {"img1.png": "Cap1", "img2.jpg": "Cap2"}
SAMPLE_PATHS = {"img1.png": "/fake/path/to/img1.png", "img2.jpg": "/fake/path/to/img2.jpg"}


def _select_event(caption, index=0):
    """Gradio 4+ Gallery select data: value is {'image': {...}, 'caption': label}."""
    return SimpleNamespace(value={"image": {"path": "/tmp/gradio/thumb.webp"}, "caption": caption}, index=index)


def test_gallery_select_shows_caption():
    caption, selected, display = update_caption_display_from_gallery(_select_event("img2.jpg", 1), SAMPLE_CAPTIONS, SAMPLE_PATHS)
    assert caption == "Cap2"
    assert selected == "img2.jpg"
    assert display == "img2.jpg"


def test_gallery_select_unknown_filename():
    caption, selected, display = update_caption_display_from_gallery(_select_event("img3.gif"), SAMPLE_CAPTIONS, SAMPLE_PATHS)
    assert "Caption data not found" in caption
    assert selected == "img3.gif"
    # Saving still refuses it, because the filename is not in the loaded image paths
    status, _ = save_caption(selected, "x", SAMPLE_PATHS, SAMPLE_CAPTIONS)
    assert "Could not find original path" in status


def test_gallery_select_invalid_event_data():
    evt = SimpleNamespace(value="not-a-dict", index=0)
    caption, selected, display = update_caption_display_from_gallery(evt, SAMPLE_CAPTIONS, SAMPLE_PATHS)
    assert caption == "Error: Invalid event data."
    assert selected is None
    assert display == "Error"


def test_gallery_select_no_event():
    assert update_caption_display_from_gallery(None, SAMPLE_CAPTIONS, SAMPLE_PATHS) == ("", None, None)


# --- Tests for save_caption ---

def test_save_caption_success(caption_test_folder):
    """Saving writes <image base name>.txt next to the image and updates the state."""
    img_paths = {"image1.png": str(caption_test_folder / "image1.png")}
    caption_data = {"image1.png": "Old caption."}

    status, updated_data = save_caption("image1.png", "Updated caption for image 1.", img_paths, caption_data)

    assert "successfully" in status.lower()
    assert "image1.txt" in status
    assert (caption_test_folder / "image1.txt").read_text(encoding="utf-8") == "Updated caption for image 1."
    assert updated_data == {"image1.png": "Updated caption for image 1."}
    assert caption_data == {"image1.png": "Old caption."} # Input state not mutated


def test_save_caption_no_selection():
    status, updated_data = save_caption(None, "caption", {}, {})
    assert "No image selected" in status
    assert updated_data == {}


def test_save_caption_path_missing():
    status, updated_data = save_caption("img1.png", "caption", {}, {"img1.png": "old"})
    assert "Could not find original path" in status
    assert updated_data == {"img1.png": "old"}


@patch("builtins.open", new_callable=mock_open)
def test_save_caption_write_error(mock_file_write, caption_test_folder):
    """Test handling errors during file writing."""
    mock_file_write.side_effect = IOError("Disk full")
    img_paths = {"image1.png": str(caption_test_folder / "image1.png")}
    caption_data = {"image1.png": "Old caption."}

    status, updated_data = save_caption("image1.png", "Cannot write this.", img_paths, caption_data)

    assert "Error saving caption" in status
    assert "Disk full" in status
    assert str(caption_test_folder / "image1.txt") in status
    assert updated_data == caption_data


# --- batch_edit_captions (disabled in the UI until multi-select exists) ---

def test_batch_edit_is_disabled_placeholder():
    caption_data = {"img.png": "cap"}
    status, updated_data = batch_edit_captions(["img.png"], "text", "Append", {"img.png": "path"}, caption_data)
    assert "needs update" in status
    assert updated_data == caption_data


# --- Tests for generate_captions_for_selected ---

def test_generate_caption_validation_errors(caption_test_folder):
    paths = {"image1.png": str(caption_test_folder / "image1.png")}
    args = {**COMMON_ARGS, "image_paths": paths, "current_captions": {}, "generate_mode": "Overwrite"}
    with patch(EXECUTE_PATH) as mock_execute:
        assert "No image selected" in generate_captions_for_selected(None, **args)[0]
        assert "valid Agent or Team" in generate_captions_for_selected("image1.png", **{**args, "agent_or_team_display_name": "(Direct Agent Call)"})[0]
        assert "Vision Model" in generate_captions_for_selected("image1.png", **{**args, "selected_model_display_name": ""})[0]
        assert "Image path not found" in generate_captions_for_selected("missing.png", **args)[0]
    mock_execute.assert_not_called()


@patch(EXECUTE_PATH, return_value=_agent_returns("  A white square.  "))
def test_generate_caption_overwrite_success(mock_execute, caption_test_folder):
    paths = {"image1.png": str(caption_test_folder / "image1.png")}
    status, captions, last_caption, session = generate_captions_for_selected(
        "image1.png", generate_mode="Overwrite", image_paths=paths, current_captions={"image1.png": "Old"}, **COMMON_ARGS)

    assert "Status: Success." in status
    assert "Overwritten" in status
    assert last_caption == "A white square."
    assert captions["image1.png"] == "A white square."
    assert (caption_test_folder / "image1.txt").read_text(encoding="utf-8") == "A white square."
    assert session == ["previous", "new entry"]

    kwargs = mock_execute.call_args.kwargs
    assert isinstance(kwargs["single_image_input"], Image.Image)
    assert kwargs["user_input"].startswith(FIXED_PROMPT_START)
    assert kwargs["selected_role_or_team"] == "Captioner"
    assert kwargs["model_with_vision"] == "vision-model (VISION)"
    assert kwargs["clean_artifacts_flag"] is True


@pytest.mark.parametrize("mode, expected", [
    ("Append", "Caption for image 1.\n\n---\n\nNew caption"),
    ("Prepend", "New caption\n\n---\n\nCaption for image 1."),
])
@patch(EXECUTE_PATH, return_value=_agent_returns("New caption"))
def test_generate_caption_append_prepend(mock_execute, mode, expected, caption_test_folder):
    paths = {"image1.png": str(caption_test_folder / "image1.png")}
    status, captions, _, _ = generate_captions_for_selected(
        "image1.png", generate_mode=mode, image_paths=paths,
        current_captions={"image1.png": "Caption for image 1."}, **COMMON_ARGS)
    assert "Status: Success." in status
    assert captions["image1.png"] == expected
    assert (caption_test_folder / "image1.txt").read_text(encoding="utf-8") == expected


@patch(EXECUTE_PATH)
def test_generate_caption_skip_existing(mock_execute, caption_test_folder):
    paths = {"image1.png": str(caption_test_folder / "image1.png")}
    status, captions, last_caption, _ = generate_captions_for_selected(
        "image1.png", generate_mode="Skip", image_paths=paths, current_captions={"image1.png": "Caption for image 1."}, **COMMON_ARGS)
    assert "Skipped image1.png" in status
    assert last_caption == ""
    mock_execute.assert_not_called()
    assert (caption_test_folder / "image1.txt").read_text(encoding="utf-8") == "Caption for image 1."


@patch(EXECUTE_PATH, return_value=_agent_returns("⚠️ Error: Could not connect to Ollama"))
def test_generate_caption_agent_error_leaves_file_untouched(mock_execute, caption_test_folder):
    paths = {"image1.png": str(caption_test_folder / "image1.png")}
    status, captions, last_caption, _ = generate_captions_for_selected(
        "image1.png", generate_mode="Overwrite", image_paths=paths, current_captions={"image1.png": "Caption for image 1."}, **COMMON_ARGS)
    assert "Status: Error." in status
    assert "Could not connect to Ollama" in status
    assert last_caption == ""
    assert captions["image1.png"] == "Caption for image 1."
    assert (caption_test_folder / "image1.txt").read_text(encoding="utf-8") == "Caption for image 1."


@patch(EXECUTE_PATH)
def test_generate_caption_rejects_disguised_image(mock_execute, tmp_path):
    """A TIFF renamed to .png is not decoded or sent to the model."""
    _make_image(tmp_path / "sneaky.png", fmt="TIFF")
    paths = {"sneaky.png": str(tmp_path / "sneaky.png")}
    status, *_ = generate_captions_for_selected(
        "sneaky.png", generate_mode="Overwrite", image_paths=paths, current_captions={}, **COMMON_ARGS)
    assert "Status: Error." in status
    mock_execute.assert_not_called()
    assert not (tmp_path / "sneaky.txt").exists()


# --- Tests for generate_captions_for_all ---

@patch(EXECUTE_PATH, side_effect=[_agent_returns("Cap A"), _agent_returns("Cap B")])
def test_generate_captions_for_all_success(mock_execute, tmp_path):
    _make_image(tmp_path / "a.png")
    _make_image(tmp_path / "b.png")
    paths = {"a.png": str(tmp_path / "a.png"), "b.png": str(tmp_path / "b.png")}

    status, captions, last_caption, _ = generate_captions_for_all(
        paths, {}, "Captioner", "vision-model (VISION)", "Overwrite",
        {}, [], {}, {}, {}, [], [])

    assert "Processed=2, Errors=0, Skipped=0" in status
    assert captions == {"a.png": "Cap A", "b.png": "Cap B"}
    assert last_caption == "Cap B"
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "Cap A"


def test_generate_captions_for_all_validation():
    assert generate_captions_for_all({}, {}, "Captioner", "m", "Overwrite", {}, [], {}, {}, {}, [], [])[0] == "No images loaded."
    assert "Vision Model" in generate_captions_for_all({"a": "p"}, {}, "Captioner", "", "Overwrite", {}, [], {}, {}, {}, [], [])[0]



@patch(EXECUTE_PATH, return_value=_agent_returns("Cap B"))
def test_generate_captions_for_all_counts_skipped(mock_execute, tmp_path):
    _make_image(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("existing", encoding="utf-8")
    _make_image(tmp_path / "b.png")
    paths = {"a.png": str(tmp_path / "a.png"), "b.png": str(tmp_path / "b.png")}

    status, *_ = generate_captions_for_all(
        paths, {"a.png": "existing"}, "Captioner", "vision-model (VISION)", "Skip",
        {}, [], {}, {}, {}, [], [])

    assert "Processed=1, Errors=0, Skipped=1" in status
