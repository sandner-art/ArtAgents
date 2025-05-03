# ArtAgent/agents/roles_config.py
import os
from core.utils import load_json # Use utility function

DEFAULT_ROLES_FILE = 'agents/agent_roles.json'
CUSTOM_ROLES_FILE = 'agents/custom_agent_roles.json'

def load_all_roles(settings, file_agents: dict = None):
    """
    Loads default, custom, and optionally file-loaded roles based on settings flags.

    Args:
        settings (dict): The application settings dictionary.
        file_agents (dict, optional): Agents loaded from a user-provided file. Defaults to None.

    Returns:
        dict: A dictionary containing the merged agent roles.
    """
    roles = {}
    use_custom = settings.get("using_custom_agents", False)
    use_default = settings.get("using_default_agents", True)

    # 1. Load default roles first if enabled
    if use_default:
        default_roles = load_json(DEFAULT_ROLES_FILE, is_relative=True)
        if isinstance(default_roles, dict):
             roles.update(default_roles)
        else:
            print(f"Warning: Failed to load or parse default roles from {DEFAULT_ROLES_FILE}")

    # 2. Load custom roles if enabled, potentially overriding defaults
    if use_custom:
        custom_roles = load_json(CUSTOM_ROLES_FILE, is_relative=True)
        if isinstance(custom_roles, dict):
             roles.update(custom_roles) # Custom roles override defaults
        else:
            print(f"Warning: Failed to load or parse custom roles from {CUSTOM_ROLES_FILE}")

    # 3. Load roles from file if provided, potentially overriding default/custom
    if file_agents and isinstance(file_agents, dict):
        print(f"Merging {len(file_agents)} agent(s) loaded from file...")
        # Add prefix to keys from file_agents to avoid name collisions in the final dict?
        # Or just let them override? Let's let them override for simplicity now.
        roles.update(file_agents)
        # If adding prefix:
        # for key, value in file_agents.items():
        #     roles[f"[File] {key}"] = value


    if not roles and (use_default or use_custom or file_agents):
         print("Warning: No agent roles ended up being loaded. Check JSON files and settings.")
    elif not use_default and not use_custom and not file_agents:
         print("Info: All agent sources (default, custom, file) are disabled or none provided.")

    return roles

def get_role_display_name(role_name: str, file_agent_keys: list = None) -> str:
    """Adds a prefix to role names loaded from file for display purposes."""
    if file_agent_keys and role_name in file_agent_keys:
        return f"[File] {role_name}"
    return role_name

def get_actual_role_name(display_name: str) -> str:
    """Removes the prefix from a display name to get the actual key."""
    prefix = "[File] "
    if display_name.startswith(prefix):
        return display_name[len(prefix):]
    return display_name