# ArtAgent/tests/test_history_manager.py

import pytest
import os
import json
import sys
from unittest.mock import patch, mock_open

# --- Adjust import path ---
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir)
sys.path.insert(0, project_root)

try:
    # Import functions and constants to be tested/mocked
    from core.history_manager import (
        load_history,
        save_history,
        add_to_history,
        MAX_HISTORY_ENTRIES,
        HISTORY_FILE as REAL_HISTORY_FILE # Keep original name for reference
    )
    # Import separately for mocking if needed, or assume history_manager imports them
    # from core.utils import get_absolute_path
except ImportError as e:
    pytest.skip(f"Skipping history_manager tests, core module not found: {e}", allow_module_level=True)

# --- Constants ---
LOAD_JSON_PATH = 'core.history_manager.load_json'
SAVE_HISTORY_PATH = 'core.history_manager.save_history' # To mock save called by add
GET_ABS_PATH = 'core.history_manager.get_absolute_path'
BUILTINS_OPEN_PATH = 'builtins.open' # Used by save_history
JSON_DUMP_PATH = 'core.history_manager.json.dump' # Used by save_history


# --- Fixtures ---
# No autouse fixture needed with the new mocking strategy for load_history

@pytest.fixture
def temp_history_file(tmp_path):
    """Provides a temporary file path, mainly for save_history tests."""
    return tmp_path / "test_history.json"


# --- Tests for load_history ---

@patch(LOAD_JSON_PATH) # Mock load_json used by history_manager
def test_load_history_valid(mock_load_json):
    """Test loading a valid history list via mocked load_json."""
    test_data = ["entry1", "entry2"]
    mock_load_json.return_value = test_data # Simulate load_json returning list
    history = load_history()
    assert history == test_data
    mock_load_json.assert_called_once_with(REAL_HISTORY_FILE, is_relative=True)

@patch(LOAD_JSON_PATH, return_value={}) # Simulate load_json returning {} for not found/empty/error
def test_load_history_non_existent(mock_load_json):
    """Test loading when file doesn't exist (load_json returns {})."""
    history = load_history()
    assert history == [] # load_history converts {} to []
    mock_load_json.assert_called_once_with(REAL_HISTORY_FILE, is_relative=True)

@patch(LOAD_JSON_PATH, return_value={})
def test_load_history_empty_file(mock_load_json):
    """Test loading an empty history file (load_json returns {})."""
    history = load_history()
    assert history == []
    mock_load_json.assert_called_once_with(REAL_HISTORY_FILE, is_relative=True)

@patch(LOAD_JSON_PATH)
def test_load_history_invalid_json(mock_load_json):
     # Simulate load_json returning {} after internal error (no warning check needed here)
    mock_load_json.return_value = {}
    history = load_history()
    assert history == []
    mock_load_json.assert_called_once_with(REAL_HISTORY_FILE, is_relative=True)

@patch(LOAD_JSON_PATH)
def test_load_history_not_a_list(mock_load_json, capsys):
    """Test loading a file containing a dict (load_json returns dict)."""
    mock_load_json.return_value = {"key": "value"} # Simulate load_json returning dict
    history = load_history()
    assert history == []
    captured = capsys.readouterr()
    # Check for the specific warning from load_history itself
    assert "did not contain a valid list" in captured.out or "did not contain a valid list" in captured.err
    mock_load_json.assert_called_once_with(REAL_HISTORY_FILE, is_relative=True)


# --- Tests for save_history ---

@patch(JSON_DUMP_PATH) # Mock json.dump
@patch(BUILTINS_OPEN_PATH, new_callable=mock_open) # Mock file open
@patch(GET_ABS_PATH) # Mock get_absolute_path used by save_history
def test_save_history_basic(mock_get_abs, mock_file_open, mock_json_dump, temp_history_file):
    """Test saving a simple history list."""
    # Configure mock get_absolute_path to return the temp file path
    mock_get_abs.return_value = str(temp_history_file)
    test_data = ["save_entry1", "save_entry2"]

    save_history(test_data) # Call the function

    mock_get_abs.assert_called_once_with(REAL_HISTORY_FILE) # Check abs path called correctly
    mock_file_open.assert_called_once_with(str(temp_history_file), 'w', encoding='utf-8') # Check file opened
    mock_json_dump.assert_called_once_with(test_data, mock_file_open(), indent=4) # Check data dumped

@patch(JSON_DUMP_PATH)
@patch(BUILTINS_OPEN_PATH, new_callable=mock_open)
@patch(GET_ABS_PATH)
def test_save_history_empty(mock_get_abs, mock_file_open, mock_json_dump, temp_history_file):
    """Test saving an empty history list."""
    mock_get_abs.return_value = str(temp_history_file)
    save_history([])
    mock_get_abs.assert_called_once_with(REAL_HISTORY_FILE)
    mock_file_open.assert_called_once_with(str(temp_history_file), 'w', encoding='utf-8')
    mock_json_dump.assert_called_once_with([], mock_file_open(), indent=4)


# --- Tests for add_to_history ---

@patch(SAVE_HISTORY_PATH) # Mock save_history called by add_to_history
def test_add_to_history_append(mock_save_history_func):
    """Test adding a new entry to an existing history."""
    initial_history = ["entry_A"]
    updated_history = add_to_history(initial_history, "entry_B") # Pass list directly
    assert updated_history == ["entry_A", "entry_B"]
    # Verify save_history was called with the updated list
    mock_save_history_func.assert_called_once_with(["entry_A", "entry_B"])

@patch(SAVE_HISTORY_PATH)
def test_add_to_history_from_empty(mock_save_history_func):
    """Test adding an entry when history starts empty."""
    initial_history = []
    updated_history = add_to_history(initial_history, "first_entry")
    assert updated_history == ["first_entry"]
    mock_save_history_func.assert_called_once_with(["first_entry"])

@patch(SAVE_HISTORY_PATH)
def test_add_to_history_duplicate_skip(mock_save_history_func):
    """Test that adding an exact duplicate entry is skipped."""
    initial_history = ["duplicate_entry"]
    updated_history = add_to_history(initial_history, "duplicate_entry")
    assert updated_history == ["duplicate_entry"] # Should not change
    # Save should still be called, but with the original list
    mock_save_history_func.assert_called_once_with(["duplicate_entry"])


@patch(SAVE_HISTORY_PATH) # Mock save_history
@patch('core.history_manager.MAX_HISTORY_ENTRIES', 3) # Mock constant within the module
def test_add_to_history_max_entries(mock_save_history_func):
    """Test that history is truncated when MAX_HISTORY_ENTRIES is exceeded."""
    # Note: We test the list manipulation logic here.
    # The mock_save_history allows us to check what *would* be saved.

    # Start below limit
    h0 = ["entry1", "entry2"]
    # Add entry 3 (reaches limit)
    h1 = add_to_history(h0.copy(), "entry3")
    assert h1 == ["entry1", "entry2", "entry3"]
    mock_save_history_func.assert_called_with(["entry1", "entry2", "entry3"]) # Check last call

    # Add entry 4 (exceeds limit)
    h2 = add_to_history(h1.copy(), "entry4")
    assert h2 == ["entry2", "entry3", "entry4"] # entry1 should be dropped
    mock_save_history_func.assert_called_with(["entry2", "entry3", "entry4"])

    # Add entry 5
    h3 = add_to_history(h2.copy(), "entry5")
    assert h3 == ["entry3", "entry4", "entry5"] # entry2 should be dropped
    mock_save_history_func.assert_called_with(["entry3", "entry4", "entry5"])


@patch(SAVE_HISTORY_PATH) # Mock save_history
def test_add_to_history_invalid_input_type(mock_save_history_func, capsys):
    """Test adding to history when the input 'history' is not a list."""
    initial_history = {"not": "a list"} # Invalid input
    updated_history = add_to_history(initial_history, "new_entry")
    assert updated_history == ["new_entry"] # Should reset and add the entry
    # Check save was called with the corrected list
    mock_save_history_func.assert_called_once_with(["new_entry"])
    captured = capsys.readouterr()
    assert "Warning: History is not a list" in captured.out or "Warning: History is not a list" in captured.err