# ArtAgent/tests/test_captioning_logic.py

import pytest
import os
import sys
# CORRECTED: Import 'call' from unittest.mock
from unittest.mock import patch, mock_open, call, MagicMock

# --- Adjust import path ---
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir)
sys.path.insert(0, project_root)

try:
    from core.captioning_logic import (
        load_images_and_captions,
        update_caption_display,
        save_caption,
        batch_edit_captions,
        IMAGE_EXTENSIONS # Import for reference if needed
    )
    # Mock PIL Image needed if functions check image validity, but they currently don't seem to.
    # If added later, mock PIL similar to test_agent.py
except ImportError as e:
    pytest.skip(f"Skipping captioning_logic tests, modules not found: {e}", allow_module_level=True)


# --- Fixtures ---

@pytest.fixture
def caption_test_folder(tmp_path):
    """Creates a temporary folder structure for caption tests."""
    d = tmp_path / "caption_folder"
    d.mkdir()
    # Image files
    (d / "image1.png").touch()
    (d / "image2.jpg").touch()
    (d / "image3.webp").touch()
    (d / "subfolder").mkdir() # Should be ignored
    (d / "not_an_image.txt").touch() # Should be ignored
    (d / "image_without_caption.jpeg").touch()

    # Corresponding caption files
    (d / "image1.txt").write_text("Caption for image 1.", encoding='utf-8')
    (d / "image2.txt").write_text("Caption for image 2, with comma.", encoding='utf-8')
    # image3.txt intentionally missing
    # image_without_caption.txt missing

    return d

# --- Tests for load_images_and_captions ---

def test_load_images_and_captions_success(caption_test_folder):
    """Test loading images and captions successfully."""
    folder_path = str(caption_test_folder)
    filenames, img_paths, captions, status, first_img, first_cap, first_file_disp = load_images_and_captions(folder_path)

    assert status == "Loaded 4 image(s)."
    assert len(filenames) == 4
    assert len(img_paths) == 4
    assert len(captions) == 4

    # Check sorted order (based on os.listdir)
    expected_filenames = sorted(["image1.png", "image2.jpg", "image_without_caption.jpeg", "image3.webp"])
    assert filenames == expected_filenames

    # Check paths
    assert img_paths["image1.png"] == os.path.join(folder_path, "image1.png")
    assert img_paths["image_without_caption.jpeg"] == os.path.join(folder_path, "image_without_caption.jpeg")

    # Check captions
    assert captions["image1.png"] == "Caption for image 1."
    assert captions["image2.jpg"] == "Caption for image 2, with comma."
    assert captions["image3.webp"] == "" # Missing caption file -> empty string
    assert captions["image_without_caption.jpeg"] == "" # Missing caption file -> empty string

    # Check first item outputs (depends on sorted order)
    assert first_img == expected_filenames[0] # Should be image1.png
    assert first_cap == "Caption for image 1."
    assert first_file_disp == expected_filenames[0]


def test_load_images_and_captions_empty_folder(tmp_path):
    """Test loading from an empty folder."""
    d = tmp_path / "empty_folder"
    d.mkdir()
    filenames, img_paths, captions, status, first_img, first_cap, first_file_disp = load_images_and_captions(str(d))
    assert status == "No supported image files found in the specified folder."
    assert filenames == []
    assert img_paths == {}
    assert captions == {}
    assert first_img is None
    assert first_cap == "" # Default empty string
    assert first_file_disp is None

def test_load_images_and_captions_no_images(tmp_path):
    """Test loading from a folder with no supported images."""
    d = tmp_path / "no_images_folder"
    d.mkdir()
    (d / "document.txt").touch()
    (d / "archive.zip").touch()
    filenames, img_paths, captions, status, first_img, first_cap, first_file_disp = load_images_and_captions(str(d))
    assert status == "No supported image files found in the specified folder."
    assert filenames == []

def test_load_images_and_captions_invalid_path():
    """Test loading with an invalid folder path."""
    invalid_path = "path/that/does/not/exist"
    filenames, img_paths, captions, status, first_img, first_cap, first_file_disp = load_images_and_captions(invalid_path)
    assert status.startswith("Error: Invalid folder path provided:")
    assert filenames == []

def test_load_images_and_captions_no_path():
    """Test loading with no folder path provided."""
    filenames, img_paths, captions, status, first_img, first_cap, first_file_disp = load_images_and_captions("")
    assert status == "Error: No folder specified."
    assert filenames == []

# Mock open to simulate read error
@patch("builtins.open", new_callable=mock_open) # Mock open globally for this test
@patch("os.path.exists") # Also mock exists
def test_load_images_and_captions_read_error(mock_exists, mock_file_open, caption_test_folder, capsys):
    """Test handling of read errors for caption files."""
    folder_path = str(caption_test_folder)
    image1_txt_path = os.path.join(folder_path, "image1.txt")
    image2_txt_path = os.path.join(folder_path, "image2.txt")

    # Simulate exists returning True for relevant txt files
    def exists_side_effect(path):
        if path == image1_txt_path or path == image2_txt_path:
            return True
        return False # Assume other txt files don't exist
    mock_exists.side_effect = exists_side_effect

    # Make open raise error only for image1.txt
    original_open = open # Keep reference if needed, but mock_open provides handle
    def open_side_effect(path, mode='r', *args, **kwargs):
        if path == image1_txt_path and mode == 'r':
            raise IOError("Permission denied")
        elif path == image2_txt_path and mode == 'r':
            # Use mock_open's default behavior for successful reads
             m = mock_open(read_data="Caption for image 2, with comma.")
             return m(path, mode, *args, **kwargs)
        else:
            # For other calls (like isfile checks on images maybe?), allow failure or mock specifically
            # Or raise an error if unexpected file is opened
             raise FileNotFoundError(f"Unexpected open call in test: {path}")
    mock_file_open.side_effect = open_side_effect


    filenames, img_paths, captions, status, _, _, _ = load_images_and_captions(folder_path)

    assert "image1.png" in filenames
    assert captions["image1.png"] == "" # Should default to empty on read error
    assert captions["image2.jpg"] == "Caption for image 2, with comma." # Should load fine

    captured = capsys.readouterr()
    assert "Warning: Could not read caption file image1.txt" in captured.out

# --- Tests for update_caption_display ---

SAMPLE_CAPTIONS = {"img1.png": "Cap1", "img2.jpg": "Cap2"}
# Use realistic paths for isfile check later
SAMPLE_PATHS = {"img1.png": "/fake/path/to/img1.png", "img2.jpg": "/fake/path/to/img2.jpg"}

@patch("os.path.isfile", return_value=True) # Assume files exist for these tests
def test_update_caption_display_single_selection(mock_isfile):
    """Test updating display with a single selection string."""
    caption, state_sel, display_sel, preview_path = update_caption_display("img1.png", SAMPLE_CAPTIONS, SAMPLE_PATHS)
    assert caption == "Cap1"
    assert state_sel == "img1.png"
    assert display_sel == "img1.png"
    assert preview_path == "/fake/path/to/img1.png"
    mock_isfile.assert_called_once_with("/fake/path/to/img1.png")

@patch("os.path.isfile", return_value=True)
def test_update_caption_display_list_selection_single(mock_isfile):
    """Test updating display with a list containing one selection."""
    caption, state_sel, display_sel, preview_path = update_caption_display(["img2.jpg"], SAMPLE_CAPTIONS, SAMPLE_PATHS)
    assert caption == "Cap2"
    assert state_sel == "img2.jpg"
    assert display_sel == "img2.jpg"
    assert preview_path == "/fake/path/to/img2.jpg"
    mock_isfile.assert_called_once_with("/fake/path/to/img2.jpg")

@patch("os.path.isfile", return_value=True)
def test_update_caption_display_list_selection_multiple(mock_isfile):
    """Test updating display with multiple selections (uses first)."""
    caption, state_sel, display_sel, preview_path = update_caption_display(["img1.png", "img2.jpg"], SAMPLE_CAPTIONS, SAMPLE_PATHS)
    assert caption == "Cap1" # Uses first item
    assert state_sel == "img1.png"
    assert display_sel == "img1.png"
    assert preview_path == "/fake/path/to/img1.png"
    mock_isfile.assert_called_once_with("/fake/path/to/img1.png")

@patch("os.path.isfile") # No need to mock return value as it won't be called
def test_update_caption_display_no_selection(mock_isfile):
    """Test updating display with no selection."""
    caption, state_sel, display_sel, preview_path = update_caption_display(None, SAMPLE_CAPTIONS, SAMPLE_PATHS)
    assert caption == ""; assert state_sel is None; assert display_sel is None; assert preview_path is None
    mock_isfile.assert_not_called()

    caption, state_sel, display_sel, preview_path = update_caption_display([], SAMPLE_CAPTIONS, SAMPLE_PATHS)
    assert caption == ""; assert state_sel is None; assert display_sel is None; assert preview_path is None
    mock_isfile.assert_not_called()

@patch("os.path.isfile", return_value=False) # Simulate path not found/invalid
def test_update_caption_display_selection_not_in_data(mock_isfile):
    """Test updating display when selection doesn't exist in path data."""
    caption, state_sel, display_sel, preview_path = update_caption_display("img3.gif", SAMPLE_CAPTIONS, SAMPLE_PATHS)
    assert "Caption data not found" in caption # Caption missing
    assert state_sel == "img3.gif"
    assert display_sel == "img3.gif" # Still display filename
    assert preview_path is None # Path missing from SAMPLE_PATHS
    mock_isfile.assert_not_called() # isfile not called if path not found in dict

@patch("os.path.isfile", return_value=False) # Simulate path not found/invalid
def test_update_caption_display_path_missing_or_invalid(mock_isfile):
    """Test updating display when image path exists in dict but os.path.isfile returns False."""
    captions = {"img1.png": "Cap1"}
    paths = {"img1.png": "/fake/path/non_existent_img1.png"}
    caption, state_sel, display_sel, preview_path = update_caption_display(["img1.png"], captions, paths)
    assert caption == "Cap1"
    assert state_sel == "img1.png"
    assert "Image Path Error" in display_sel # Check for error indication
    assert preview_path is None # Preview path should be None
    mock_isfile.assert_called_once_with("/fake/path/non_existent_img1.png")


# --- Tests for save_caption ---

@patch("builtins.open", new_callable=mock_open) # Mock file writing
def test_save_caption_success(mock_file_write, caption_test_folder):
    """Test successfully saving a caption."""
    selected_file = "image1.png"
    new_caption = "Updated caption for image 1."
    img_paths = {"image1.png": os.path.join(str(caption_test_folder), "image1.png")}
    caption_data = {"image1.png": "Old caption."}
    expected_txt_path = os.path.join(str(caption_test_folder), "image1.txt")

    status, updated_data = save_caption(selected_file, new_caption, img_paths, caption_data)

    # Check status message content (case-insensitive and content check)
    assert "successfully" in status.lower()
    assert "image1.txt" in status # Check base filename is mentioned

    mock_file_write.assert_called_once_with(expected_txt_path, 'w', encoding='utf-8')
    handle = mock_file_write()
    handle.write.assert_called_once_with(new_caption)
    assert updated_data == {"image1.png": new_caption}

def test_save_caption_no_selection():
    """Test saving with no file selected."""
    status, updated_data = save_caption(None, "caption", {}, {})
    assert "No image selected" in status
    assert updated_data == {}

def test_save_caption_path_missing():
    """Test saving when the image path is missing in the dictionary."""
    status, updated_data = save_caption("img1.png", "caption", {}, {"img1.png": "old"})
    assert "Could not find original path" in status
    assert updated_data == {"img1.png": "old"} # Data should be unchanged

@patch("builtins.open", new_callable=mock_open)
def test_save_caption_write_error(mock_file_write, caption_test_folder):
    """Test handling errors during file writing."""
    mock_file_write.side_effect = IOError("Disk full")
    selected_file = "image1.png"
    new_caption = "Cannot write this."
    img_paths = {"image1.png": os.path.join(str(caption_test_folder), "image1.png")}
    caption_data = {"image1.png": "Old caption."}
    expected_txt_path = os.path.join(str(caption_test_folder), "image1.txt")

    status, updated_data = save_caption(selected_file, new_caption, img_paths, caption_data)

    assert "Error saving caption" in status
    assert "Disk full" in status
    assert expected_txt_path in status
    assert updated_data == caption_data # State should remain unchanged on error

# --- Tests for batch_edit_captions ---

@patch("builtins.open", new_callable=mock_open)
def test_batch_edit_append_success(mock_file_batch_write, caption_test_folder):
    """Test successfully appending text to multiple captions."""
    selected = ["image1.png", "image2.jpg"]
    text_to_add = ", appended text"
    img_paths = {"image1.png": os.path.join(str(caption_test_folder), "image1.png"), "image2.jpg": os.path.join(str(caption_test_folder), "image2.jpg"), "image3.webp": os.path.join(str(caption_test_folder), "image3.webp")}
    caption_data = {"image1.png": "Cap1", "image2.jpg": "Cap2", "image3.webp": "Cap3"}
    expected_path1 = os.path.join(str(caption_test_folder), "image1.txt")
    expected_path2 = os.path.join(str(caption_test_folder), "image2.txt")
    expected_content1 = "Cap1, appended text"; expected_content2 = "Cap2, appended text"

    status, updated_data = batch_edit_captions(selected, text_to_add, "Append", img_paths, caption_data)

    # Check status message (case-insensitive)
    assert "batch append complete" in status.lower()
    assert "Processed: 2" in status; assert "Errors: 0" in status; assert "Skipped: 0" in status

    assert mock_file_batch_write.call_count == 2
    calls = mock_file_batch_write.call_args_list
    assert call(expected_path1, 'w', encoding='utf-8') in calls
    assert call(expected_path2, 'w', encoding='utf-8') in calls
    assert updated_data["image1.png"] == expected_content1
    assert updated_data["image2.jpg"] == expected_content2
    assert updated_data["image3.webp"] == "Cap3" # Unselected unchanged

@patch("builtins.open", new_callable=mock_open)
def test_batch_edit_prepend_success(mock_file_batch_write, caption_test_folder):
    """Test successfully prepending text to multiple captions."""
    selected = ["image1.png", "image2.jpg"]; text_to_add = "Prepended text: "
    img_paths = {"image1.png": os.path.join(str(caption_test_folder), "image1.png"), "image2.jpg": os.path.join(str(caption_test_folder), "image2.jpg")}
    caption_data = {"image1.png": "Cap1", "image2.jpg": "Cap2"}
    expected_content1 = "Prepended text: Cap1"; expected_content2 = "Prepended text: Cap2"

    status, updated_data = batch_edit_captions(selected, text_to_add, "Prepend", img_paths, caption_data)

    assert "batch prepend complete" in status.lower() # Case-insensitive check
    assert "Processed: 2" in status
    assert updated_data["image1.png"] == expected_content1
    assert updated_data["image2.jpg"] == expected_content2

def test_batch_edit_no_selection():
    status, updated_data = batch_edit_captions(None, "text", "Append", {}, {}); assert "No images selected" in status; assert updated_data == {}
    status, updated_data = batch_edit_captions([], "text", "Append", {}, {}); assert "No images selected" in status; assert updated_data == {}

def test_batch_edit_invalid_mode():
    status, updated_data = batch_edit_captions(["img.png"], "text", "InvalidMode", {"img.png":"path"}, {"img.png":"cap"})
    assert "Invalid batch mode: InvalidMode" in status; assert updated_data == {"img.png":"cap"}

@patch("builtins.open", new_callable=mock_open)
def test_batch_edit_some_errors(mock_file_batch_write, caption_test_folder):
    """Test batch edit where some files cause write errors."""
    mock_handle = mock_file_batch_write(); mock_handle.write.side_effect = [None, IOError("Write failed for image2")]
    selected = ["image1.png", "image2.jpg", "image_missing.png"]
    text_to_add = ", added"
    img_paths = {"image1.png": os.path.join(str(caption_test_folder), "image1.png"), "image2.jpg": os.path.join(str(caption_test_folder), "image2.jpg")}
    caption_data = {"image1.png": "Cap1", "image2.jpg": "Cap2"}

    status, updated_data = batch_edit_captions(selected, text_to_add, "Append", img_paths, caption_data)

    assert "batch append complete" in status.lower() # Case-insensitive
    assert "Processed: 1" in status; assert "Errors: 1" in status; assert "Skipped: 1" in status
    assert "- Skipped image_missing.png: Path not found." in status
    assert "- Error appending image2.jpg: Write failed for image2" in status
    assert updated_data["image1.png"] == "Cap1, added"
    assert updated_data["image2.jpg"] == "Cap2"
    assert "image_missing.png" not in updated_data