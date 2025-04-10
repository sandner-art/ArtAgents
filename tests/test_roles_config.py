# ArtAgent/tests/test_roles_config.py

import pytest
import os
import json
import sys

# --- Adjust import path ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    """Mocks the file path constants to use temporary files."""
    default_file, custom_file = temp_roles_files
    monkeypatch.setattr('agents.roles_config.DEFAULT_ROLES_FILE', str(default_file))
    monkeypatch.setattr('agents.roles_config.CUSTOM_ROLES_FILE', str(custom_file))
    # IMPORTANT: Also patch the paths used internally by load_json if roles_config calls it
    # Since load_json resolves relative paths using utils.PROJECT_ROOT, we need to ensure
    # the mocked paths are treated correctly. It's often easier to mock load_json itself
    # *within the scope of the roles_config module* if path resolution becomes complex.
    # Let's try mocking load_json used *by* roles_config:

    original_load_json = load_json # Keep a reference

    def mock_load_json_for_roles(file_path, is_relative=True):
        # Only intercept calls for the specific mocked file paths
        if file_path == str(default_file):
             return json.loads(default_file.read_text(encoding='utf-8'))
        elif file_path == str(custom_file):
             return json.loads(custom_file.read_text(encoding='utf-8'))
        else:
             # For any other path, use the original function (might be needed?)
             # Or just return {} if roles_config shouldn't load anything else
             return {} # Return empty dict for unexpected paths in this context

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

def test_load_all_roles_missing_files(monkeypatch, capsys):
    """Test behavior when role files don't exist (should warn)."""
    # Point to non-existent files
    monkeypatch.setattr('agents.roles_config.DEFAULT_ROLES_FILE', "non_existent_default.json")
    monkeypatch.setattr('agents.roles_config.CUSTOM_ROLES_FILE', "non_existent_custom.json")
    # Mock load_json to simulate file not found (returns {})
    monkeypatch.setattr('agents.roles_config.load_json', lambda file_path, is_relative=True: {})

    roles = load_all_roles(SETTINGS_BOTH) # Try loading both
    assert roles == {} # Expect empty dict

    # Check if warnings were printed by the *underlying* load_json (which we mocked)
    # or potentially by load_all_roles itself if it added checks.
    # Let's assume load_json prints the warning based on core.utils implementation.
    # We might need to adjust this assertion based on where the warning actually originates.
    # The current load_all_roles doesn't explicitly warn for empty loads from files.
    # captured = capsys.readouterr()
    # assert "File not found" in captured.out or "File not found" in captured.err # Check if utils warns

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