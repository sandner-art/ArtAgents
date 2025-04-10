# ArtAgent/tests/test_history_manager.py

import pytest
import os
import json
import sys
from unittest.mock import patch

# --- Adjust import path ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    from core.utils import get_absolute_path # Needed if we test interaction with it
except ImportError as e:
    pytest.skip(f"Skipping history_manager tests, core module not found: {e}", allow_module_level=True)


# --- Fixtures ---

@pytest.fixture
def temp_history_file(tmp_path):
    """Creates a temporary file path for history tests."""
    return tmp_path / "test_history.json"

@pytest.fixture(autouse=True) # Apply this mock automatically to all tests in this file
def mock_history_file_path(monkeypatch, temp_history_file):
    """Mocks the HISTORY_FILE constant to use the temporary file path."""
    monkeypatch.setattr('core.history_manager.HISTORY_FILE', str(temp_history_file))
    # Also mock the get_absolute_path to just return the temp path directly
    # This prevents tests from relying on the real project structure for path resolution
    # when dealing with the mocked HISTORY_FILE.
    monkeypatch.setattr('core.history_manager.get_absolute_path', lambda x: str(temp_history_file))


# --- Tests for load_history ---

def test_load_history_valid(temp_history_file):
    """Test loading a valid history list from the mocked file."""
    test_data = ["entry1", "entry2"]
    temp_history_file.write_text(json.dumps(test_data), encoding='utf-8')
    history = load_history()
    assert history == test_data

def test_load_history_non_existent(temp_history_file):
    """Test loading when the history file doesn't exist (should return [])."""
    # Ensure file does not exist (tmp_path is empty initially)
    assert not temp_history_file.exists()
    history = load_history()
    assert history == []

def test_load_history_empty_file(temp_history_file):
    """Test loading an empty history file (should return [])."""
    temp_history_file.write_text("", encoding='utf-8')
    history = load_history()
    # The underlying load_json returns {} for empty, but load_history checks type
    assert history == [] # Expected empty list for history context

def test_load_history_invalid_json(temp_history_file, capsys):
    """Test loading a file with invalid JSON (should return [] and warn)."""
    temp_history_file.write_text("[invalid json", encoding='utf-8')
    history = load_history()
    assert history == []
    captured = capsys.readouterr()
    # Check if the underlying load_json warning is printed
    assert "Error decoding JSON" in captured.out or "Error decoding JSON" in captured.err

def test_load_history_not_a_list(temp_history_file, capsys):
    """Test loading a file containing a dict instead of a list (should return [])."""
    test_data = {"key": "value"}
    temp_history_file.write_text(json.dumps(test_data), encoding='utf-8')
    history = load_history()
    assert history == []
    captured = capsys.readouterr()
    # Check for the specific warning from load_history itself
    assert "did not contain a valid list" in captured.out or "did not contain a valid list" in captured.err


# --- Tests for save_history ---

def test_save_history_basic(temp_history_file):
    """Test saving a simple history list."""
    test_data = ["save_entry1", "save_entry2"]
    save_history(test_data)
    assert temp_history_file.exists()
    read_data = json.loads(temp_history_file.read_text(encoding='utf-8'))
    assert read_data == test_data

def test_save_history_empty(temp_history_file):
    """Test saving an empty history list."""
    save_history([])
    assert temp_history_file.exists()
    read_data = json.loads(temp_history_file.read_text(encoding='utf-8'))
    assert read_data == []

# --- Tests for add_to_history ---

def test_add_to_history_append(temp_history_file):
    """Test adding a new entry to an existing history."""
    initial_history = ["entry_A"]
    temp_history_file.write_text(json.dumps(initial_history), encoding='utf-8')
    updated_history = add_to_history(initial_history, "entry_B") # Pass list directly
    assert updated_history == ["entry_A", "entry_B"]
    # Verify file was saved
    read_data = json.loads(temp_history_file.read_text(encoding='utf-8'))
    assert read_data == ["entry_A", "entry_B"]

def test_add_to_history_from_empty(temp_history_file):
    """Test adding an entry when history starts empty."""
    initial_history = []
    assert not temp_history_file.exists() # Starts non-existent
    updated_history = add_to_history(initial_history, "first_entry")
    assert updated_history == ["first_entry"]
    # Verify file was saved
    assert temp_history_file.exists()
    read_data = json.loads(temp_history_file.read_text(encoding='utf-8'))
    assert read_data == ["first_entry"]

def test_add_to_history_duplicate_skip(temp_history_file):
    """Test that adding an exact duplicate entry is skipped."""
    initial_history = ["duplicate_entry"]
    temp_history_file.write_text(json.dumps(initial_history), encoding='utf-8')
    updated_history = add_to_history(initial_history, "duplicate_entry")
    assert updated_history == ["duplicate_entry"] # Should not change
    read_data = json.loads(temp_history_file.read_text(encoding='utf-8'))
    assert read_data == ["duplicate_entry"] # File should reflect no change

# Mock MAX_HISTORY_ENTRIES for this specific test
@patch('core.history_manager.MAX_HISTORY_ENTRIES', 3) # Temporarily set limit to 3
def test_add_to_history_max_entries(temp_history_file):
    """Test that history is truncated when MAX_HISTORY_ENTRIES is exceeded."""
    initial_history = ["entry1", "entry2"]
    temp_history_file.write_text(json.dumps(initial_history), encoding='utf-8')

    # Add entry 3 (reaches limit)
    h1 = add_to_history(initial_history.copy(), "entry3")
    assert h1 == ["entry1", "entry2", "entry3"]
    read_data1 = json.loads(temp_history_file.read_text(encoding='utf-8'))
    assert read_data1 == ["entry1", "entry2", "entry3"]

    # Add entry 4 (exceeds limit)
    h2 = add_to_history(h1.copy(), "entry4")
    assert h2 == ["entry2", "entry3", "entry4"] # entry1 should be dropped
    read_data2 = json.loads(temp_history_file.read_text(encoding='utf-8'))
    assert read_data2 == ["entry2", "entry3", "entry4"]

    # Add entry 5
    h3 = add_to_history(h2.copy(), "entry5")
    assert h3 == ["entry3", "entry4", "entry5"] # entry2 should be dropped
    read_data3 = json.loads(temp_history_file.read_text(encoding='utf-8'))
    assert read_data3 == ["entry3", "entry4", "entry5"]


def test_add_to_history_invalid_input_type(temp_history_file, capsys):
    """Test adding to history when the input 'history' is not a list."""
    initial_history = {"not": "a list"} # Invalid input
    updated_history = add_to_history(initial_history, "new_entry")
    assert updated_history == ["new_entry"] # Should reset and add the entry
    read_data = json.loads(temp_history_file.read_text(encoding='utf-8'))
    assert read_data == ["new_entry"]
    captured = capsys.readouterr()
    assert "Warning: History is not a list" in captured.out or "Warning: History is not a list" in captured.err