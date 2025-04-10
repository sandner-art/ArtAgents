# ArtAgent/tests/test_captioning_logic.py

import pytest
import os
import sys
# CORRECTED: Import 'call' from unittest.mock
from unittest.mock import patch, mock_open, call

# --- Adjust import path ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    assert filenames == ["image1.png", "image2.jpg", "image_without_caption.jpeg", "image3.webp"]

    # Check paths
    assert img_paths["image1.png"] == os.path.join(folder_path, "image1.png")
    assert img_paths["image_without_caption.jpeg"] == os.path.join(folder_path, "image_without_caption.jpeg")

    # Check captions
    assert captions["image1.png"] == "Caption for image 1."
    assert captions["image2.jpg"] == "Caption for image 2, with comma."
    assert captions["image3.webp"] == "" # Missing caption file -> empty string
    assert captions["image_without_caption.jpeg"] == "" # Missing caption file -> empty string

    # Check first item outputs
    assert first_img == "image1.png"
    assert first_cap == "Caption for image 1."
    assert first_file_disp == "image1.png"


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
@patch("builtins.open", mock_open()) # Mock open globally for this test
def test_load_images_and_captions_read_error(caption_test_folder, capsys):
    """Test handling of read errors for caption files."""
    # Setup the mock_open to raise an error when reading image1.txt
    mock_open().side_effect = [
        mock_open(read_data="").return_value, # Simulate successful open for directory listing check? No, listdir is separate.
        IOError("Permission denied") # Error when trying to read image1.txt
    ]
    # This mocking strategy might be fragile. It depends on the order of file access.
    # A more robust way might be to patch os.path.exists and then raise in open only for specific file.

    # Let's try patching os.path.exists and open specifically
    original_exists = os.path.exists
    original_open = open

    def mock_exists(path):
        # Let exists work normally except for the problematic file's txt
        # if path.endswith("image1.txt"):
        #     return True # Say it exists
        return original_exists(path)

    def mock_open_specific(*args, **kwargs):
        if args[0].endswith("image1.txt") and args[1] == 'r':
            raise IOError("Cannot read file")
        else:
            # Important: call the original open for other files
            return original_open(*args, **kwargs)

    with patch("os.path.exists", mock_exists), \
         patch("builtins.open", mock_open_specific):
        filenames, img_paths, captions, status, _, _, _ = load_images_and_captions(str(caption_test_folder))

        assert "image1.png" in filenames
        assert captions["image1.png"] == "" # Should default to empty on read error
        assert captions["image2.jpg"] == "Caption for image 2, with comma." # Others should load fine

        captured = capsys.readouterr()
        assert "Warning: Could not read caption file image1.txt" in captured.out

# --- Tests for update_caption_display ---

SAMPLE_CAPTIONS = {"img1.png": "Cap1", "img2.jpg": "Cap2"}
SAMPLE_PATHS = {"img1.png": "/path/to/img1.png", "img2.jpg": "/path/to/img2.jpg"}

def test_update_caption_display_single_selection():
    """Test updating display with a single selection string."""
    caption, state_sel, display_sel, preview_path = update_caption_display("img1.png", SAMPLE_CAPTIONS, SAMPLE_PATHS)
    assert caption == "Cap1"
    assert state_sel == "img1.png"
    assert display_sel == "img1.png"
    assert preview_path == "/path/to/img1.png"

def test_update_caption_display_list_selection_single():
    """Test updating display with a list containing one selection."""
    caption, state_sel, display_sel, preview_path = update_caption_display(["img2.jpg"], SAMPLE_CAPTIONS, SAMPLE_PATHS)
    assert caption == "Cap2"
    assert state_sel == "img2.jpg"
    assert display_sel == "img2.jpg"
    assert preview_path == "/path/to/img2.jpg"

def test_update_caption_display_list_selection_multiple():
    """Test updating display with multiple selections (uses first)."""
    caption, state_sel, display_sel, preview_path = update_caption_display(["img1.png", "img2.jpg"], SAMPLE_CAPTIONS, SAMPLE_PATHS)
    assert caption == "Cap1" # Uses first item
    assert state_sel == "img1.png"
    assert display_sel == "img1.png"
    assert preview_path == "/path/to/img1.png"

def test_update_caption_display_no_selection():
    """Test updating display with no selection."""
    caption, state_sel, display_sel, preview_path = update_caption_display(None, SAMPLE_CAPTIONS, SAMPLE_PATHS)
    assert caption == ""
    assert state_sel is None
    assert display_sel is None
    assert preview_path is None

    caption, state_sel, display_sel, preview_path = update_caption_display([], SAMPLE_CAPTIONS, SAMPLE_PATHS)
    assert caption == ""
    assert state_sel is None
    assert display_sel is None
    assert preview_path is None

def test_update_caption_display_selection_not_in_data():
    """Test updating display when selection doesn't exist in data."""
    caption, state_sel, display_sel, preview_path = update_caption_display("img3.gif", SAMPLE_CAPTIONS, SAMPLE_PATHS)
    assert "Caption data not found" in caption
    assert state_sel == "img3.gif"
    assert display_sel == "img3.gif" # Displays filename even if data missing
    assert preview_path is None # Path also not found

def test_update_caption_display_path_missing(tmp_path):
    """Test updating display when image path is missing or invalid."""
    captions = {"img1.png": "Cap1"}
    paths = {"img1.png": str(tmp_path / "non_existent_img1.png")} # Path exists in dict but not on disk
    caption, state_sel, display_sel, preview_path = update_caption_display(["img1.png"], captions, paths)
    assert caption == "Cap1"
    assert state_sel == "img1.png"
    assert "Image Path Error" in display_sel # Check for error indication
    assert preview_path is None # Preview path should be None


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

    assert "successfully" in status
    assert expected_txt_path in status
    # Check that mock_open was called correctly to write the file
    mock_file_write.assert_called_once_with(expected_txt_path, 'w', encoding='utf-8')
    # Check that the correct content was written
    handle = mock_file_write()
    handle.write.assert_called_once_with(new_caption)
    # Check state update
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

# Use mock_open to intercept multiple file writes
@patch("builtins.open", new_callable=mock_open)
def test_batch_edit_append_success(mock_file_batch_write, caption_test_folder):
    """Test successfully appending text to multiple captions."""
    selected = ["image1.png", "image2.jpg"]
    text_to_add = ", appended text"
    img_paths = {
        "image1.png": os.path.join(str(caption_test_folder), "image1.png"),
        "image2.jpg": os.path.join(str(caption_test_folder), "image2.jpg"),
        "image3.webp": os.path.join(str(caption_test_folder), "image3.webp"), # Unselected
    }
    caption_data = {
        "image1.png": "Cap1",
        "image2.jpg": "Cap2",
        "image3.webp": "Cap3",
    }
    expected_path1 = os.path.join(str(caption_test_folder), "image1.txt")
    expected_path2 = os.path.join(str(caption_test_folder), "image2.txt")
    expected_content1 = "Cap1, appended text"
    expected_content2 = "Cap2, appended text"

    status, updated_data = batch_edit_captions(selected, text_to_add, "Append", img_paths, caption_data)

    assert "Batch Append complete." in status
    assert "Processed: 2" in status
    assert "Errors: 0" in status

    # Check mock_open calls (tricky with multiple calls)
    assert mock_file_batch_write.call_count == 2
    calls = mock_file_batch_write.call_args_list
    # Check paths called - order might vary depending on iteration
    # Use call object imported from unittest.mock
    assert call(expected_path1, 'w', encoding='utf-8') in calls
    assert call(expected_path2, 'w', encoding='utf-8') in calls

    # Check content written (requires inspecting handles from mock calls)
    # This gets complex. A simpler check is the final state dict.
    assert updated_data["image1.png"] == expected_content1
    assert updated_data["image2.jpg"] == expected_content2
    assert updated_data["image3.webp"] == "Cap3" # Unselected unchanged

@patch("builtins.open", new_callable=mock_open)
def test_batch_edit_prepend_success(mock_file_batch_write, caption_test_folder):
    """Test successfully prepending text to multiple captions."""
    selected = ["image1.png", "image2.jpg"]
    text_to_add = "Prepended text: "
    img_paths = {
        "image1.png": os.path.join(str(caption_test_folder), "image1.png"),
        "image2.jpg": os.path.join(str(caption_test_folder), "image2.jpg"),
    }
    caption_data = {"image1.png": "Cap1", "image2.jpg": "Cap2"}
    expected_content1 = "Prepended text: Cap1"
    expected_content2 = "Prepended text: Cap2"

    status, updated_data = batch_edit_captions(selected, text_to_add, "Prepend", img_paths, caption_data)

    assert "Batch Prepend complete." in status
    assert "Processed: 2" in status
    assert updated_data["image1.png"] == expected_content1
    assert updated_data["image2.jpg"] == expected_content2

def test_batch_edit_no_selection():
    status, updated_data = batch_edit_captions(None, "text", "Append", {}, {})
    assert "No images selected" in status
    assert updated_data == {}

    status, updated_data = batch_edit_captions([], "text", "Append", {}, {})
    assert "No images selected" in status
    assert updated_data == {}

def test_batch_edit_invalid_mode():
    status, updated_data = batch_edit_captions(["img.png"], "text", "InvalidMode", {"img.png":"path"}, {"img.png":"cap"})
    assert "Invalid batch mode: InvalidMode" in status
    assert updated_data == {"img.png":"cap"} # Unchanged

@patch("builtins.open", new_callable=mock_open)
def test_batch_edit_some_errors(mock_file_batch_write, caption_test_folder):
    """Test batch edit where some files cause write errors."""
    # Mock open globally and make write fail on second call.
    mock_handle = mock_file_batch_write()
    mock_handle.write.side_effect = [
        None, # First write (image1) succeeds
        IOError("Write failed for image2") # Second write (image2) fails
    ]

    selected = ["image1.png", "image2.jpg", "image_missing.png"] # Include one with missing path
    text_to_add = ", added"
    img_paths = {
        "image1.png": os.path.join(str(caption_test_folder), "image1.png"),
        "image2.jpg": os.path.join(str(caption_test_folder), "image2.jpg"),
        # image_missing.png path deliberately missing
    }
    caption_data = {"image1.png": "Cap1", "image2.jpg": "Cap2"}
    expected_path1 = os.path.join(str(caption_test_folder), "image1.txt")
    expected_path2 = os.path.join(str(caption_test_folder), "image2.txt") # Write will be attempted

    status, updated_data = batch_edit_captions(selected, text_to_add, "Append", img_paths, caption_data)

    assert "Batch Append complete." in status
    assert "Processed: 1" in status # Only image1 succeeded fully
    assert "Errors: 1" in status # image2 write error
    assert "Skipped: 1" in status # image_missing.png skipped due to path

    assert "- Skipped image_missing.png: Path not found." in status
    assert "- Error Appending image2.jpg: Write failed for image2" in status

    # Check final data state
    assert updated_data["image1.png"] == "Cap1, added" # Successfully updated
    assert updated_data["image2.jpg"] == "Cap2" # Failed write, state not updated for this one
    assert "image_missing.png" not in updated_data # Was never in original data