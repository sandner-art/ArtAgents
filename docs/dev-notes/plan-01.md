**I. Current Project Structure (Refactored)**

This reflects the structure after implementing the modules, API profiles, themes, file agent loading, and the Agent Team *execution* logic (but not the editor).

```
ArtAgent/
│
├── app.py                  # Main Gradio App: UI Structure, Event Wiring, State Mgmt
├── requirements.txt        # Python Dependencies (gradio, requests, Pillow, numpy)
├── settings.json           # App Config: Ollama URL, defaults, global API opts, theme
├── models.json             # Ollama models known to the app (name, vision)
├── limiters.json           # Prompt style limiters (name, tokens, format string)
├── ollama_profiles.json    # Presets for Ollama API options (Fast, Balanced, etc.)
├── agent_teams.json        # Stores PREDEFINED Agent Team/Workflow definitions
│
├── agents/                 # --- Agent Logic & Definitions ---
│   ├── __init__.py         # Makes 'agents' a Python package
│   ├── roles_config.py     # Logic to load/merge default, custom, file roles
│   ├── ollama_agent.py     # Core get_llm_response function (interacts with Ollama)
│   ├── agent_roles.json    # Default agent definitions (persona, instructions)
│   └── custom_agent_roles.json # User's custom persistent agents + options
│
├── core/                   # --- Core Logic & Utilities ---
│   ├── __init__.py         # Makes 'core' a Python package
│   ├── app_logic.py        # Callback logic functions (chat, comment, settings, teams)
│   ├── agent_manager.py    # Orchestrates Agent Team Workflows (run_team_workflow)
│   ├── history_manager.py  # Loads/saves persistent history (history.json)
│   ├── ollama_checker.py   # Ollama startup check logic (OllamaStatusChecker)
│   ├── ollama_manager.py   # Ollama specific actions (release_model, release_all)
│   ├── utils.py            # Common utilities (load_json, get_absolute_path, etc.)
│   ├── help_content.py     # Stores help text (tooltips, markdown) for UI
│   └── history.json        # Persistent history data file
│
├── ui/                     # --- UI Tab Definitions ---
│   ├── __init__.py         # Makes 'ui' a Python package
│   ├── chat_tab.py         # Builds the main Chat interface components
│   ├── app_settings_tab.py # Builds the App Settings interface components
│   ├── team_editor_tab.py  # Builds the Agent Team Editor interface components
│   ├── roles_tab.py        # Builds the Default/Custom Roles display tabs
│   ├── history_tab.py      # Builds the Full History display tab
│   └── common_ui_elements.py # Common UI parts (e.g., footer)
│
├── scripts/                # --- Utility & Setup Scripts ---
│   ├── setup.bat           # Windows: Initial Ollama model setup helper
│   ├── setupvenv.bat       # Windows: Setup Python virtual environment & install reqs
│   ├── go.bat              # Windows: Run app directly
│   ├── govenv.bat          # Windows: Run app within venv
│   └── full_project_creator.py # Script to create single project dump file
│   └── (Optional: .sh equivalents for Linux/macOS)
│
├── tests/                  # --- Automated Tests (Planned) ---
│   ├── __init__.py         # Makes 'tests' a Python package
│   └── test_agent.py       # Example test file (partially implemented)
│   └── (Planned: test_app_logic.py, test_agent_manager.py, test_utils.py, etc.)
│
├── .gitignore              # Git ignore rules (venv, pycache, logs, etc.)
└── README.md               # Project documentation (setup, usage, structure)
```

**II. Progress Report & Implemented Features**

*   **Core Refactoring:** Completed. Logic is separated into `core`, `agents`, `ui` modules. `app.py` handles UI structure, state, and event wiring.
*   **Ollama Interaction:** Centralized in `agents/ollama_agent.py`. Handles API calls, streaming, error handling.
*   **Ollama Startup Check:** Implemented in `core/ollama_checker.py`, provides console feedback on launch.
*   **Agent Role System:** Loads roles from default, custom, and temporary files (`core/utils.py`, `agents/roles_config.py`). Dropdowns update dynamically.
*   **API Option Profiles:** Implemented (`ollama_profiles.json`, UI in `app_settings_tab`, logic in `app_logic`). Allows loading presets onto settings UI (requires manual save).
*   **Theme Selection:** Implemented (`core/utils.py`, UI in `app_settings_tab`, logic in `app_logic`). Allows selecting built-in themes (requires app restart).
*   **Contextual Help:** Implemented (`core/help_content.py`, integrated into `ui/*.py` files via `info=` and `gr.Markdown`).
*   **Agent Team Execution:**
    *   Data structure defined (`agent_teams.json`).
    *   Orchestration logic implemented (`core/agent_manager.py::run_team_workflow`). Handles sequential execution, context passing, result assembly ("concatenate", "refine_last"), and history logging.
    *   Integrated into Chat Tab (`ui/chat_tab.py`, `core/app_logic.py::execute_chat_or_team`). Users can select and run predefined teams from the dropdown.
*   **Basic Application Stability:** Resolved major errors related to imports, function signatures, state handling, and syntax.

**III. Planned Feature: Agent Team Editor (Phase 2)**

*   **Goal:** Allow users to visually create, modify, save, and delete their own Agent Teams/Workflows directly within the application.
*   **User Interface (`ui/team_editor_tab.py`):**
    *   Load existing teams into a dropdown (`team_select_dropdown`).
    *   Display/Edit Team Name and Description (`team_name_textbox`, `team_description_textbox`).
    *   Display the steps of the loaded/current team (using `gr.JSON` initially for simplicity: `steps_display_json`).
    *   Provide controls to add new steps:
        *   Dropdown (`agent_to_add_dropdown`) populated with *all* available agents (Default, Custom, File-Loaded).
        *   Button (`add_step_button`) to append the selected agent to the current team's steps list.
    *   Provide controls to remove steps:
        *   Number input (`step_index_to_remove`) to specify the step number (1-based).
        *   Button (`remove_step_button`) to remove the specified step.
    *   Select the final assembly strategy (`assembly_strategy_radio`).
    *   Buttons for core actions: "Load Selected Team", "Save Current Team Definition", "Clear Editor / New Team", "Delete Selected Team".
    *   Status display (`save_status_textbox`).
*   **Backend Logic (`core/app_logic.py`):**
    *   Implement the callback functions:
        *   `load_team_for_editing`: Loads selected team data into UI components and editor state.
        *   `clear_team_editor`: Resets UI components and editor state.
        *   `add_step_to_editor`: Modifies the `current_team_editor_state` by adding a new step definition. Updates the `steps_display_json`.
        *   `remove_step_from_editor`: Modifies the `current_team_editor_state` by removing a step. Updates the `steps_display_json`.
        *   `save_team_from_editor`: Constructs the team dictionary from UI/state, saves it to `agent_teams.json` (using `save_teams_to_file` helper), updates `teams_data_state`, and refreshes relevant dropdowns (editor's load list, chat tab's agent/team list).
        *   `delete_team_logic`: Removes the selected team from `teams_data_state`, saves to file, updates dropdowns, and clears the editor.
*   **State Management (`app.py`):**
    *   Introduce `current_team_editor_state` to hold the team being edited.
    *   Ensure `teams_data_state` is correctly updated by save/delete operations.
    *   Ensure the dropdown refresh logic (`refresh_agent_team_dropdown_wrapper`, `refresh_available_agents_for_editor_wrapper`) correctly updates the necessary dropdowns when teams change.

**IV. Future Development Brainstorming (Post-Editor)**

*   **Advanced Team Editor UI:** Replace `gr.JSON` step display with dynamic rows, allowing drag-and-drop reordering, inline goal editing per step.
*   **Advanced Workflow Strategies:**
    *   **Conditional Steps:** Allow Manager agent to decide *if* a step runs based on context.
    *   **Parallel Execution:** Run multiple agents concurrently (if outputs are independent) and merge results.
    *   **Feedback Loops:** Allow output of a later agent to refine the input for an earlier agent in an iterative process.
*   **Manager Agent Implementation (Design 2/3):** Allow selecting a "Manager" role that dynamically plans the workflow based on user input instead of using predefined steps. Requires robust prompt engineering and output parsing.
*   **Model Selection per Step:** Allow defining specific Ollama models for individual steps within a team definition.
*   **Persistent File Agents:** Allow saving agents loaded from files more permanently into `custom_agent_roles.json`.
*   **Unit Testing:** Flesh out the `tests/` directory with comprehensive tests for all core logic modules (`utils`, `history_manager`, `roles_config`, `ollama_agent`, `agent_manager`, `app_logic` callbacks, `team_editor_logic`).
*   **Captioning Tab:** Re-integrate or implement image captioning/editing features if desired.
*   **Gradio 4 Upgrade:** Plan and execute the upgrade for access to newer features, accepting the need for code modifications.

This plan provides a clear path forward, focusing on delivering the user-configurable Agent Team editor next, while keeping the more advanced hierarchical agent concepts as future enhancements built upon this foundation.