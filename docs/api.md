# ArtAgents Core API Reference

## 1. Introduction

This document provides a reference for the core Python functions and modules within the ArtAgents project. It is intended for developers looking to understand the internal workings, data flow, and key logic components of the application. This is not an exhaustive list of every function but focuses on the primary interfaces between different parts of the system.

## 2. Core Modules Overview

*   **`agents/ollama_agent.py`:** Handles all direct communication with the Ollama API.
*   **`core/agent_manager.py`:** Orchestrates the execution of multi-agent "Team" workflows.
*   **`core/app_logic.py`:** Contains main Gradio UI callbacks and the central router (`execute_chat_or_team`) for user requests.
*   **`core/refinement_logic.py`:** Contains the logic for the "Comment/Refine" feature.
*   **`core/captioning_logic.py`:** Manages loading, editing, and generating image captions.
*   **`core/sweep_manager.py`:** Executes experiment sweeps across different configurations.
*   **`core/utils.py`:** Provides shared utility functions (config loading, path handling, text cleaning).
*   **`core/history_manager.py`:** Handles persistent logging.
*   **`agents/roles_config.py`:** Loads and manages agent role definitions.

## 3. Key Function Reference

### 3.1. Ollama Interaction

**`agents.ollama_agent.get_llm_response(role, prompt, model, settings, roles_data, images=None, max_tokens=1500, ollama_api_options=None) -> str`**

*   **Purpose:** The primary function for sending requests to the Ollama API and receiving responses.
*   **Key Parameters:**
    *   `role` (str): The name of the agent role being used (used primarily for merging API options).
    *   `prompt` (str): The fully constructed prompt text to send to the LLM.
    *   `model` (str): The name of the Ollama model to use (e.g., `llama3:latest`, `llava:latest`).
    *   `settings` (dict): The current application settings dictionary (provides Ollama URL, global API options).
    *   `roles_data` (dict): Dictionary of all loaded agent roles (used for role-specific API options).
    *   `images` (list[PIL.Image], optional): A list containing PIL Image objects to be sent with the request (requires a vision model). Images are automatically base64 encoded.
    *   `max_tokens` (int): Fallback value for `num_predict` if not otherwise specified in options.
    *   `ollama_api_options` (dict, optional): Allows direct override of API options for this specific call.
*   **Returns:** (str) The text response from the LLM, or an error message string prefixed with `⚠️ Error:` if an issue occurred during image processing, connection, request execution, or stream decoding.
*   **Notes:** Handles merging of API options (direct > role-specific > global). Uses streaming API endpoint. Includes error handling for common issues.

### 3.2. Workflow Management

**`core.agent_manager.run_team_workflow(team_name, team_definition, user_input, initial_settings, all_roles_data, history_list, worker_model_name, single_image_input=None, return_intermediate_steps=False) -> tuple[str, list, dict | None]`**

*   **Purpose:** Executes a multi-step agent team workflow as defined in `agent_teams.json`.
*   **Key Parameters:**
    *   `team_name` (str): The name of the team being executed.
    *   `team_definition` (dict): The dictionary defining the team's steps and assembly strategy.
    *   `user_input` (str): The initial user request text.
    *   `initial_settings` (dict): Current application settings.
    *   `all_roles_data` (dict): Dictionary of all available agent role definitions.
    *   `history_list` (list): The current persistent history list (used for logging steps).
    *   `worker_model_name` (str): The Ollama model name to be used by *all* agents within this workflow run.
    *   `single_image_input` (PIL.Image, optional): A PIL image passed down from the UI/router, potentially used by steps if the `worker_model_name` supports vision.
    *   `return_intermediate_steps` (bool): If `True`, the third element of the return tuple will contain a dictionary of outputs/errors per step.
*   **Returns:** (tuple)
    1.  `final_output` (str): The final text result assembled according to the team's `assembly_strategy`. Prefixed with `Error:` on failure.
    2.  `updated_history_list` (list): The history list updated with logs from the workflow execution.
    3.  `step_outputs_dict | None` (dict | None): Dictionary of intermediate results if requested, otherwise `None`.
*   **Notes:** Executes steps sequentially, passing context (user input + previous outputs) to each step. Calls `ollama_agent.get_llm_response` for each step. Implements assembly strategies: `concatenate`, `refine_last`, `summarize_all`, `structured_concatenate`. Logs start, steps, errors, and end to persistent history.

### 3.3. UI Logic / Routing

**`core.app_logic.execute_chat_or_team(..., clean_artifacts_flag: bool, ...) -> tuple[str, str, str | None, list]`**

*   **Purpose:** Acts as the central router for the main "Generate Response" button click in the Chat tab. Determines whether to execute a single agent call or a team workflow based on the selected dropdown item. Also applies optional artifact cleaning.
*   **Key Parameters:** Takes numerous parameters directly from UI component values (dropdowns, textboxes, checkboxes) and Gradio state objects (settings, models, teams, history, etc.).
    *   `selected_role_or_team` (str): The value from the main agent/team selection dropdown.
    *   `clean_artifacts_flag` (bool): Whether to apply artifact cleaning to the final result.
    *   `single_image_input` (numpy.ndarray | PIL.Image | None): Image data from the UI or passed from captioning.
    *   Other parameters include `user_input`, `model_with_vision`, UI options, and various state dictionaries.
*   **Returns:** (tuple) Formatted for direct output binding in `app.py`:
    1.  `response_text` (str): The final (potentially cleaned) text response to display.
    2.  `session_history_text` (str): The formatted string for the session history display.
    3.  `model_name_used_state` (str | None): The name of the model used for this specific execution (used by `comment_logic`).
    4.  `new_session_history_list` (list): The updated session history list state.
*   **Notes:** Calls either `app_logic.chat_logic` (for single agents) or `agent_manager.run_team_workflow` (for teams). Applies `utils.clean_agent_artifacts` if the flag is set before returning the response text.

**`core.refinement_logic.comment_logic(llm_response_text, comment, max_tokens_ui, use_ollama_api_options, model_with_vision, current_settings, file_agents_dict, history_list_state, session_history_list_state) -> tuple[str, str, list]`**

*   **Purpose:** Handles the "Comment/Refine" button logic. Takes existing text and a user comment, then uses the *currently selected* model to generate a refined response based on a structured prompt.
*   **Key Parameters:**
    *   `llm_response_text` (str): The text currently in the main response display box.
    *   `comment` (str): The user's refinement instruction from the comment input box.
    *   `model_with_vision` (str | None): The model *currently selected* in the main model dropdown UI component.
    *   Other parameters include UI options and state dictionaries.
*   **Returns:** (tuple) Formatted for direct output binding in `app.py`:
    1.  `new_response_text` (str): The refined text generated by the LLM.
    2.  `new_session_history_text` (str): Formatted session history including this refinement step.
    3.  `new_session_history_list` (list): Updated session history list state.
*   **Notes:** Constructs a specific prompt instructing the LLM to act as a refiner. Calls `ollama_agent.get_llm_response` using the selected model. Logs the refinement action to history.

### 3.4. Feature Logic Modules

**`core.captioning_logic`**

*   Contains functions like `load_images_and_captions`, `update_caption_display_from_gallery`, `save_caption`, `generate_captions_for_selected`, `generate_captions_for_all`.
*   Handles file system interaction (finding images, reading/writing `.txt` caption files).
*   Manages state related to loaded images and captions.
*   Calls `app_logic.execute_chat_or_team` (passing PIL image data) to leverage agents/teams for caption generation.

**`core.sweep_manager.run_sweep(...) -> str`**

*   **Purpose:** Executes the Experiment Sweep functionality.
*   **Key Parameters:** Takes base prompts, selected team names, selected model names, output folder name, logging flag, and state dictionaries.
*   **Functionality:**
    *   Creates a timestamped output directory in `sweep_runs/`.
    *   Loops through **Models -> Prompts -> Teams**.
    *   Calls `agent_manager.run_team_workflow` for each configuration.
    *   Saves a detailed JSON protocol file for each individual run.
    *   Saves the final (cleaned) text output from successful runs to separate `.txt` files, one per model tested (`prompts_[model].txt`), with one prompt per line.
*   **Returns:** (str) A final summary status message for the UI.

### 3.5. Utilities and Configuration

*   **`core.utils`:** Provides `load_json`, `save_json`, `get_absolute_path`, `clean_agent_artifacts`, `format_json_to_html_table`, `get_theme_object`.
*   **`agents.roles_config`:** Provides `load_all_roles` (merges default/custom/file roles based on settings), `get_role_display_name`, `get_actual_role_name`.

## 4. Conclusion

This document outlines the primary functions and modules driving ArtAgents. Understanding these core components and their interactions is key to modifying or extending the application's functionality. Refer to the specific module files for detailed implementation logic.
