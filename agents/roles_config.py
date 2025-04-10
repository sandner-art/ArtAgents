# ArtAgent/agents/roles_config.py
import os
from core.utils import load_json # Use utility function

DEFAULT_ROLES_FILE = 'agents/agent_roles.json' # Relative to project root
CUSTOM_ROLES_FILE = 'agents/custom_agent_roles.json' # Relative to project root

def load_all_roles(settings):
    """
    Loads default and custom roles based on settings flags.

    Args:
        settings (dict): The application settings dictionary.

    Returns:
        dict: A dictionary containing the merged agent roles.
    """
    roles = {}
    use_custom = settings.get("using_custom_agents", False)
    use_default = settings.get("using_default_agents", True) # Default to True maybe?

    # Load default roles first if enabled
    if use_default:
        default_roles = load_json(DEFAULT_ROLES_FILE, is_relative=True)
        if isinstance(default_roles, dict):
             roles.update(default_roles)
        else:
            print(f"Warning: Failed to load or parse default roles from {DEFAULT_ROLES_FILE}")


    # Load custom roles if enabled, potentially overriding defaults
    if use_custom:
        custom_roles = load_json(CUSTOM_ROLES_FILE, is_relative=True)
        if isinstance(custom_roles, dict):
             roles.update(custom_roles) # Custom roles override default ones with same name
        else:
            print(f"Warning: Failed to load or parse custom roles from {CUSTOM_ROLES_FILE}")


    if not roles and (use_default or use_custom):
         print("Warning: No agent roles were loaded. Check JSON files and settings.")
    elif not use_default and not use_custom:
         print("Info: Both default and custom agents are disabled in settings.")


    return roles