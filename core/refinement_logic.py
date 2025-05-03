# ArtAgent/core/refinement_logic.py

import time
# Import necessary functions/classes from sibling modules or agents
from .utils import load_json # Assuming utils handles JSON loading if needed elsewhere
from agents.roles_config import load_all_roles # To load roles if needed by agent call
from agents.ollama_agent import get_llm_response # To call the LLM
from . import history_manager as history # To log to history

def comment_logic(
    # Input parameters from UI/State
    llm_response_text: str,           # Text currently in the response box
    comment: str,                     # User's refinement instruction
    max_tokens_ui: int,               # Max tokens setting from UI
    use_ollama_api_options: bool,     # Flag from UI
    model_with_vision: str | None,    # Model selected in the main dropdown
    current_settings: dict,           # Current application settings state
    file_agents_dict: dict,           # State of agents loaded from file
    history_list_state: list,         # Persistent history list state
    session_history_list_state: list  # Current session history list state
    ) -> tuple[str, str, list]:       # Return: new_response, new_session_history_text, new_session_list
    """
    Handles the commenting/refinement logic using the currently selected model
    and a structured prompt.
    """
    # Use copies of mutable state lists
    history_list = list(history_list_state)
    current_session_history = list(session_history_list_state) # This is the list we modify

    # Validate inputs
    if not comment or not model_with_vision:
        # Updated print message for clarity
        print(f"Comment ignored: No comment text provided or no model selected (Comment: '{comment}', Model: '{model_with_vision}').")
        # Return unchanged values, matching output signature (3 values)
        return llm_response_text, "\n---\n".join(current_session_history), current_session_history

    # Determine actual model name (strip vision tag if present)
    # This logic might need refinement if models_data_state is needed, but typically dropdown value is sufficient
    actual_model_name = model_with_vision.replace(" (VISION)", "")

    # Load roles needed for the agent call
    roles_data_current = load_all_roles(current_settings, file_agents=file_agents_dict)

    print(f"Processing refinement using model {actual_model_name}...")

    # Construct the structured refinement prompt
    refiner_prompt = f"""**Role:** You are an AI assistant specialized in refining and modifying existing text based on user instructions.
**Goal:** Modify the 'Original Text' below according to the 'User's Refinement Instruction'. Maintain the core essence of the original text unless the instruction explicitly asks for a fundamental change. Output only the revised text.

**Original Text:**
---
{llm_response_text}
---

**User's Refinement Instruction:**
---
{comment}
---

**Revised Text:**
"""
    # Define a generic role for the agent call itself
    # We are embedding the persona in the prompt above
    agent_call_role = "Refinement Assistant"
    agent_ollama_options = {} # Agent function handles merging

    # Call the LLM agent
    response = get_llm_response(
        role=agent_call_role,           # Generic role for the call
        prompt=refiner_prompt,          # Use the structured prompt
        model=actual_model_name,        # Use the currently selected model
        settings=current_settings,
        roles_data=roles_data_current,  # Pass roles data for potential option merging
        images=None,                    # Comments don't process images directly
        max_tokens=max_tokens_ui,       # Respect UI setting
        ollama_api_options=agent_ollama_options
    )

    # Add entry to history logs
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    # Log the actual model used and the specific action
    entry = f"Timestamp: {timestamp}\nAction: Refine Text (Comment)\nModel: {actual_model_name}\nRefinement Instruction: {comment}\nContext: Previous response text\nResponse:\n{response}\n---\n"
    history_list = history.add_to_history(history_list, entry) # Update persistent history
    current_session_history.append(entry) # Update session history copy

    # Return new response text, new session history text, and new session history list
    # Cleaning is generally not applied to comment responses unless specifically desired
    return response, "\n---\n".join(current_session_history), current_session_history