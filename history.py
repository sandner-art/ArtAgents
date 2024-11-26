# history.py
import json
import os

HISTORY_FILE = 'history.json'
MAX_HISTORY_ENTRIES = 150

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as file:
            try:
                # Read the file content
                content = file.read()
                # Check if the file is empty
                if content.strip():
                    return json.load(file)
                else:
                    # Return an empty list if the file is empty
                    return []
            except json.JSONDecodeError:
                # Handle the case where the file contains invalid JSON
                print(f"Warning: {HISTORY_FILE} contains invalid JSON. Initializing an empty history.")
                return []
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w') as file:
        json.dump(history, file, indent=4)

def add_to_history(history, entry):
    if entry not in history:  # Ensure the entry is not already in history
        history.append(entry)
    if len(history) > MAX_HISTORY_ENTRIES:
        history = history[-MAX_HISTORY_ENTRIES:]
    save_history(history)
    return history