# ArtAgent/core/history_manager.py
import json
import os
from .utils import load_json, get_absolute_path # Import from sibling module

HISTORY_FILE = 'core/history.json' # Path relative to project root
MAX_HISTORY_ENTRIES = 150

def load_history():
    """Loads history from the JSON file."""
    # load_json from utils now handles path resolution and errors
    history_data = load_json(HISTORY_FILE, is_relative=True)
    if isinstance(history_data, list):
        return history_data
    else:
        # Handle case where file exists but isn't a list (or load_json returned dict)
        print(f"Warning: History file '{HISTORY_FILE}' did not contain a valid list. Initializing empty history.")
        return []


def save_history(history):
    """Saves the history list to the JSON file."""
    full_path = get_absolute_path(HISTORY_FILE)
    try:
        with open(full_path, 'w', encoding='utf-8') as file:
            json.dump(history, file, indent=4)
    except Exception as e:
        print(f"Error saving history to {full_path}: {e}")


def add_to_history(history, entry):
    """Adds an entry to history, manages size, and saves."""
    if not isinstance(history, list):
        print("Warning: History is not a list. Cannot add entry.")
        history = [] # Reset if corrupted

    if entry not in history: # Avoid exact duplicates if desired
        history.append(entry)

    if len(history) > MAX_HISTORY_ENTRIES:
        # Keep the most recent entries
        history = history[-MAX_HISTORY_ENTRIES:]

    save_history(history)
    return history