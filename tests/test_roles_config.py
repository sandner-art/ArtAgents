# ArtAgent/tests/test_roles_config.py

import pytest
import os
import json
import sys
from unittest.mock import patch

# --- Adjust import path ---
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir)
sys.path.insert(0, project_root)

try:
    from agents.roles_config import (
        load_all_roles,
        get_role_display_name,
        get_actual_role_name,
        DEFAULT_ROLES_FILE,
        CUSTOM_ROLES_FILE
    )
    # Also need load_json from utils because load_all_roles uses it internally
    from core.utils import load_json
except ImportError as e:
    pytest.skip(f"Skipping roles_config tests, modules not found: {e}", allow_module_level=True)

# --- Test Data ---

DEFAULT_ROLES_DATA = {
    "DefaultDesigner": {"description": "Designs things by default."},
    "SharedRole": {"description": "Default shared description."}
}

CUSTOM_ROLES_DATA = {
    "CustomArtist": {"description": "Creates custom art."},
    "SharedRole": {"description": "Custom shared description OVERRIDE."}
}

FILE_AGENTS_DATA = {
    "FileAgent": {"description": "Loaded from file."},
    "DefaultDesigner": {"description": "File overrides default designer."}
}

# Settings variations
SETTINGS_DEFAULT_ONLY = {"using_default_agents": True, "using_custom_agents": False}
SETTINGS_CUSTOM_ONLY = {"using_default_agents": False, "using_custom_agents": True}
SETTINGS_BOTH = {"using_default_agents": True, "using_custom_agents": True}
SETTINGS_NEITHER = {"using_default_agents": False, "using_custom_agents": False}


# --- Fixtures ---

@pytest.fixture
def temp_roles_files(tmp_path):
    """Creates temporary default and custom roles files."""
    default_file = tmp_path / "temp_default_roles.json"
    custom_file = tmp_path / "temp_custom_roles.json"
    default_file.write_text(json.dumps(DEFAULT_ROLES_DATA), encoding='utf-8')
    custom_file.write_text(json.dumps(CUSTOM_ROLES_DATA), encoding='utf-8')
    return default_file, custom_file

@pytest.fixture(autouse=True)
def mock_roles_file_paths(monkeypatch, temp_roles_files):
    """Mocks the file path constants and load_json to use temporary files."""
    default_file, custom_file = temp_roles_files
    # Mock the constants used within the roles_config module
    monkeypatch.setattr('agents.roles_config.DEFAULT_ROLES_FILE', str(default_file))
    monkeypatch.setattr('agents.roles_config.CUSTOM_ROLES_FILE', str(custom_file))

    # Mock load_json used *by* roles_config to read from the temp files
    original_load_json = load_json # Keep a reference if needed outside mock

    def mock_load_json_for_roles(file_path, is_relative=True):
        # Use the mocked paths (which are absolute temp paths now)
        path_to_check = file_path
        # Load based on the mocked constants
        if path_to_check == str(default_file):
            if default_file.exists():
                return json.loads(default_file.read_text(encoding='utf-8'))
            else:
                return {} # Simulate file not found
        elif path_to_check == str(custom_file):
            if custom_file.exists():
                return json.loads(custom_file.read_text(encoding='utf-8'))
            else:
                return {} # Simulate file not found
        else:
             # Fallback for any other path if roles_config calls load_json unexpectedly
             # print(f"Warning: Unexpected load_json call in roles_config test: {file_path}")
             return {}

    monkeypatch.setattr('agents.roles_config.load_json', mock_load_json_for_roles)


# --- Tests for load_all_roles ---

def test_load_all_roles_default_only(temp_roles_files):
    roles = load_all_roles(SETTINGS_DEFAULT_ONLY)
    assert "DefaultDesigner" in roles
    assert "CustomArtist" not in roles
    assert roles["SharedRole"]["description"] == "Default shared description."
    assert len(roles) == 2

def test_load_all_roles_custom_only(temp_roles_files):
    roles = load_all_roles(SETTINGS_CUSTOM_ONLY)
    assert "DefaultDesigner" not in roles
    assert "CustomArtist" in roles
    assert roles["SharedRole"]["description"] == "Custom shared description OVERRIDE."
    assert len(roles) == 2

def test_load_all_roles_both(temp_roles_files):
    """Test custom roles override default roles when both are loaded."""
    roles = load_all_roles(SETTINGS_BOTH)
    assert "DefaultDesigner" in roles
    assert "CustomArtist" in roles
    # Check override
    assert roles["SharedRole"]["description"] == "Custom shared description OVERRIDE."
    assert len(roles) == 3 # DefaultDesigner, CustomArtist, SharedRole (custom version)

def test_load_all_roles_neither(temp_roles_files):
    roles = load_all_roles(SETTINGS_NEITHER)
    assert not roles # Expect empty dictionary

def test_load_all_roles_with_file_agents_default_only(temp_roles_files):
    """Test file agents override default when only default is enabled."""
    roles = load_all_roles(SETTINGS_DEFAULT_ONLY, file_agents=FILE_AGENTS_DATA)
    assert "FileAgent" in roles
    assert "CustomArtist" not in roles
    # Check file override of default
    assert roles["DefaultDesigner"]["description"] == "File overrides default designer."
    # Check default role not overridden by file
    assert roles["SharedRole"]["description"] == "Default shared description."
    assert len(roles) == 3 # DefaultDesigner (file), SharedRole (default), FileAgent

def test_load_all_roles_with_file_agents_custom_only(temp_roles_files):
    """Test file agents merge with custom when only custom is enabled."""
    roles = load_all_roles(SETTINGS_CUSTOM_ONLY, file_agents=FILE_AGENTS_DATA)
    assert "FileAgent" in roles
    assert "CustomArtist" in roles
    # Check file agent exists alongside custom
    assert roles["DefaultDesigner"]["description"] == "File overrides default designer."
    # Check custom override role is present
    assert roles["SharedRole"]["description"] == "Custom shared description OVERRIDE."
    assert len(roles) == 4 # FileAgent, DefaultDesigner (file), CustomArtist, SharedRole (custom)

def test_load_all_roles_with_file_agents_both(temp_roles_files):
    """Test file agents override both default and custom."""
    roles = load_all_roles(SETTINGS_BOTH, file_agents=FILE_AGENTS_DATA)
    assert "FileAgent" in roles
    assert "CustomArtist" in roles
    # Check file override of default (which would have been overridden by custom anyway)
    assert roles["DefaultDesigner"]["description"] == "File overrides default designer."
    # Custom should still override default shared role if not overridden by file
    assert roles["SharedRole"]["description"] == "Custom shared description OVERRIDE."
    assert len(roles) == 4 # FileAgent, DefaultDesigner (file), CustomArtist, SharedRole (custom)

def test_load_all_roles_with_file_agents_neither(temp_roles_files):
    """Test only file agents load when default/custom disabled."""
    roles = load_all_roles(SETTINGS_NEITHER, file_agents=FILE_AGENTS_DATA)
    assert "FileAgent" in roles
    assert "DefaultDesigner" in roles # The version from the file
    assert "CustomArtist" not in roles
    assert "SharedRole" not in roles
    assert len(roles) == 2 # Only the file agents

@patch('agents.roles_config.load_json') # Mock load_json directly here
def test_load_all_roles_missing_files(mock_load_json_local, capsys):
    """Test behavior when role files don't exist (load_json returns {})."""
    # Make the mocked load_json return {} always for this test
    mock_load_json_local.return_value = {}

    roles = load_all_roles(SETTINGS_BOTH) # Try loading both
    assert roles == {} # Expect empty dict

    # The function load_all_roles itself might print warnings now or in future
    captured = capsys.readouterr()
    # Check for specific warnings printed BY load_all_roles if they exist,
    # otherwise, this test just verifies the empty dict return.
    # Example potential warning check:
    # assert "Warning: Failed to load or parse default roles" in captured.out


# --- Tests for get_role_display_name ---

FILE_KEYS = list(FILE_AGENTS_DATA.keys())

def test_get_role_display_name_normal():
    assert get_role_display_name("CustomArtist") == "CustomArtist"
    assert get_role_display_name("CustomArtist", []) == "CustomArtist"
    assert get_role_display_name("CustomArtist", None) == "CustomArtist"

def test_get_role_display_name_from_file():
    assert get_role_display_name("FileAgent", FILE_KEYS) == "[File] FileAgent"
    # Check the one that overrides default
    assert get_role_display_name("DefaultDesigner", FILE_KEYS) == "[File] DefaultDesigner"

def test_get_role_display_name_not_from_file():
    assert get_role_display_name("CustomArtist", FILE_KEYS) == "CustomArtist"
    assert get_role_display_name("SharedRole", FILE_KEYS) == "SharedRole" # Even though shared, not in file keys

# --- Tests for get_actual_role_name ---

def test_get_actual_role_name_normal():
    assert get_actual_role_name("CustomArtist") == "CustomArtist"

def test_get_actual_role_name_from_display():
    assert get_actual_role_name("[File] FileAgent") == "FileAgent"
    assert get_actual_role_name("[File] DefaultDesigner") == "DefaultDesigner"

def test_get_actual_role_name_no_prefix():
    assert get_actual_role_name("NotAFilePrefix Role") == "NotAFilePrefix Role"
    # Test case where it coincidentally starts similarly but without space
    assert get_actual_role_name("[File]AgentNoSpace") == "[File]AgentNoSpace"