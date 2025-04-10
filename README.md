# ArtAgents: Agent-Based Creative Toolkit

ArtAgents is a prototype framework designed for artists, designers, and creators to experiment with LLM-based prompt engineering and creative content generation. It leverages Ollama for local model serving, allowing users to interact with various text and multimodal models through specialized AI 'agents' and structured workflows.

[![artagents-github](https://github.com/user-attachments/assets/9350bb3a-9e19-4818-b109-983c5a6b0bb1)]() <!-- Consider updating screenshot -->

## Overview

Select predefined agents, load custom agents from files, or utilize multi-agent "Teams" (workflows) to generate detailed prompts, descriptions, or other text outputs. Provide text instructions and optionally images (single or folder) as input. Fine-tune generation using Ollama API parameters, prompt style limiters, and agent presets. Experiment with different models and configurations systematically using the Sweep feature.

## Key Features

**Core Functionality:**

*   **Ollama Integration:** Connects to a running Ollama instance to utilize locally served LLMs (text & multimodal).
*   **Agent System:** Interact via specialized agents (Designer, Photographer, Styler, etc.) defined with unique instructions and optional API parameter overrides.
*   **Multimodal Input:** Supports single image upload or processing images within a specified folder.
*   **Chat Interface:** Main tab for direct interaction with selected agents or teams. Includes session history display.
*   **Persistent History:** Logs all interactions (including workflow steps) to `core/history.json`, viewable in the "Full History" tab.
*   **Configuration Management:** External JSON files for settings (`settings.json`), models (`models.json`), limiters (`limiters.json`), roles (`agent_roles.json`, `custom_agent_roles.json`), API profiles (`ollama_profiles.json`), and agent teams (`agent_teams.json`).
*   **App Settings UI:** Dedicated tab to configure Ollama URL, agent loading preferences, default behaviors, UI theme, and default Ollama API parameters.

**Advanced Features:**

*   **Agent Team / Workflow Execution:** Define and run multi-step agent sequences ("Teams") from `agent_teams.json`. Supports sequential execution with context passing and different result assembly strategies ("concatenate", "refine_last").
*   **Custom Agent File Loading:** Load temporary agent definitions from user-provided `.json` files directly in the Chat tab for quick experiments.
*   **Ollama API Profiles:** Load predefined sets of Ollama API parameters (e.g., "Fast", "Balanced", "Creative") onto the settings UI via the "App Settings" tab (requires saving settings to apply).
*   **Experiment Sweeps (Basic):** Dedicated "Experiment Sweep" tab to systematically run base prompts across multiple selected Agent Teams and Worker Models. Saves detailed JSON protocol files for each run in `sweep_runs/`.
*   **Contextual Help:** Integrated tooltips and markdown explanations throughout the UI for better usability.
*   **Theme Selection:** Choose from built-in Gradio themes (requires app restart).

**Utilities:**

*   **Model Release:** Option to unload models from Ollama VRAM (manually per model change or via button in settings).
*   **Modular Codebase:** Refactored structure (`core`, `agents`, `ui`) for better organization and maintainability.
*   **Setup Scripts:** Batch files included for easier environment setup and model downloading on Windows.

## Project Structure

```
ArtAgent/
│
├── app.py                  # Main Gradio App: UI Structure, Event Wiring, State Mgmt
├── requirements.txt        # Python Dependencies
├── settings.json           # App Config: Ollama URL, defaults, global API opts, theme
├── models.json             # Ollama models known to the app (name, vision)
├── limiters.json           # Prompt style limiters (name, tokens, format string)
├── ollama_profiles.json    # Presets for Ollama API options
├── agent_teams.json        # Stores PREDEFINED Agent Team/Workflow definitions
│
├── agents/                 # Agent Logic & Definitions
│   ├── __init__.py
│   ├── roles_config.py     # Logic to load/merge roles
│   ├── ollama_agent.py     # Interacts with Ollama API (get_llm_response)
│   ├── agent_roles.json    # Default agent definitions
│   ├── custom_agent_roles.json # User's custom persistent agents
│   └── (Example agent files: *.json) # e.g., agents_architecture.json
│
├── core/                   # Core Logic & Utilities
│   ├── __init__.py
│   ├── app_logic.py        # Callback logic functions
│   ├── agent_manager.py    # Orchestrates Agent Team Workflows
│   ├── history_manager.py  # Loads/saves persistent history
│   ├── ollama_checker.py   # Ollama startup check logic
│   ├── ollama_manager.py   # Ollama model release logic
│   ├── sweep_manager.py    # Logic for running experiment sweeps
│   ├── utils.py            # Common utilities (JSON loading etc.)
│   ├── help_content.py     # Stores help text for UI
│   └── history.json        # Persistent history data file
│
├── ui/                     # UI Tab Definitions (Gradio components)
│   ├── __init__.py
│   ├── chat_tab.py
│   ├── app_settings_tab.py
│   ├── team_editor_tab.py  # UI for editing teams
│   ├── sweep_tab.py        # UI for experiment sweeps
│   ├── roles_tab.py
│   ├── history_tab.py
│   └── common_ui_elements.py
│
├── scripts/                # Utility & Setup Scripts
│   ├── (Batch files: setup.bat, setupvenv.bat, go.bat, govenv.bat)
│   └── full_project_creator.py
│
├── sweep_runs/             # Default Output folder for Sweep Protocols
│
├── tests/                  # Automated Tests
│   ├── __init__.py
│   └── test_agent.py       # Example tests (Partially Implemented)
│
├── .gitignore
└── README.md
```

## Installation & Setup

1.  **Install Ollama:** Get from [ollama.com](https://ollama.com/) and ensure `ollama` command works.
2.  **Clone Repository:** `git clone <your-repository-url>` & `cd ArtAgent`.
3.  **(Recommended) Setup Venv:** Run `.\scripts\setupvenv.bat` (Windows) or equivalent `.sh` script (Linux/macOS).
4.  **Setup Ollama Models:** Run `.\scripts\setup.bat` (Windows) or equivalent `.sh` script to check Ollama and pull recommended models (or use `ollama pull <model_name>` manually). See `models.json`.
5.  **Configure (Optional):** Edit `settings.json`, `models.json`, `agent_teams.json`, etc. as needed.

## Running the Application

1.  **Start Ollama Service:** Ensure Ollama is running (e.g., `ollama serve` or Desktop App).
2.  **Run ArtAgents:** Use `.\scripts\govenv.bat` (if using venv) or `.\scripts\go.bat` (Windows), or equivalent `.sh` scripts.
3.  **Access UI:** Open the local URL (usually `http://127.0.0.1:7860`) in your browser.

## Development Status & Plan

**Implemented Features:**

*   Core Ollama Interaction & Agent System
*   Multimodal Input (Single/Folder)
*   Persistent & Session History
*   Configurable Settings, Models, Limiters via JSON & UI
*   Agent Role Loading (Default, Custom, File)
*   Ollama API Option Profiles (Loadable via UI)
*   UI Theme Selection (Requires Restart)
*   Contextual Help Tooltips & Markdown
*   Agent Team / Workflow **Execution** (Predefined in `agent_teams.json`)
*   Basic Experiment Sweep **Execution** (Teams & Models) with JSON protocol output

**Work In Progress / Immediate Next Steps:**

1.  **Implement Agent Team Editor UI & Logic (High Priority):**
    *   **Goal:** Allow users to create, edit, save, delete Agent Teams via the dedicated UI tab.
    *   **Status:** UI components defined (`ui/team_editor_tab.py`), backend logic functions defined (`core/app_logic.py`), event wiring in `app.py` implemented. Needs thorough testing and potentially UI refinement (especially the step editor).
2.  **Unit Testing (High Priority - Parallel Task):**
    *   **Goal:** Develop comprehensive unit tests using `pytest` and `pytest-mock`.
    *   **Status:** Basic structure (`tests/`) and example agent test (`test_agent.py`) exist. Needs significant expansion to cover core logic (`utils`, `history`, `roles`, `agent_manager`, `app_logic` callbacks), focusing on mocking external dependencies (Ollama API, file system).

**Future / Planned Enhancements:**

*   **Advanced Agent Teams / Hierarchical Agents:** Implement manager agents for dynamic planning, conditional execution, or iterative refinement loops.
*   **Refine Sweep Feature:** Add parameter sweeping, per-step model selection, potentially a protocol viewer. (If upgrading Gradio) Add progress bars.
*   **UI/UX Improvements:** Enhance the Team Editor UI (e.g., drag-and-drop), improve status reporting, potentially allow custom CSS.
*   **Captioning Tab:** Re-integrate caption viewing/editing functionality.
*   **Gradio 4.x Upgrade:** Evaluate and execute upgrade for newer features, accepting potential code refactoring needs.

---
ArtAgents by Daniel Sandner © 2024 - 2025. Adapt and use creatively. No guarantees provided.
[AI/ML Articles](https://sandner.art/)
