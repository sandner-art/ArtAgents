# ArtAgents: Agent-Based Creative Tool

ArtAgents is a framework designed for artists, designers, and creators to experiment with LLM-based prompt engineering and creative content generation, especially focused on visual concepts. It leverages Ollama for local model serving, allowing users to interact with various text and multimodal models through specialized AI 'agents', configurable multi-agent "Teams" (workflows), and includes tools for managing image captions.

## Overview

Select predefined agents, load custom agents from files, or utilize multi-agent "Teams" to generate detailed prompts, descriptions, image captions, or other text outputs. Provide text instructions and optionally images (single or folder) as input. Fine-tune generation using Ollama API parameters, prompt style limiters, and agent presets. Experiment with different models and configurations systematically using the Sweep feature. Manage and edit image caption (`.txt`) files directly within the application.

## Key Features

**Core Functionality:**

*   **Ollama Integration:** Connects to a running Ollama instance to utilize locally served LLMs (text & multimodal). Startup check included.
*   **Agent System:** Interact via specialized agents (Designer, Photographer, etc.) defined with unique instructions and optional API parameter overrides.
*   **Agent Loading:** Supports loading from default (`agents/agent_roles.json`), custom (`agents/custom_agent_roles.json`), and temporary user-uploaded (`.json`) files.
*   **Multimodal Input:** Supports single image upload or processing images within a specified folder for chat/captioning context.
*   **Chat Interface:** Main tab for direct interaction with selected agents or teams. Includes session history display.
*   **Agent Team / Workflow Execution:** Define (`agent_teams.json`) and run multi-step agent sequences ("Teams"). Supports sequential execution with context passing and different result assembly strategies ("concatenate", "refine_last"). Workflow steps are logged to history.
*   **Persistent History:** Logs all single interactions and detailed workflow steps to `core/history.json`, viewable and clearable in the "Full History" tab.
*   **Configuration Management:** External JSON files for settings (`settings.json`), models (`models.json`), limiters (`limiters.json`), roles (default, custom), API profiles (`ollama_profiles.json`), and agent teams (`agent_teams.json`).
*   **App Settings UI:** Dedicated tab to configure Ollama URL, agent loading preferences, default UI states, UI theme, and default Ollama API parameters.
*   **Ollama API Profiles:** Load predefined sets of Ollama API parameters (e.g., "Fast", "Balanced", "Creative") onto the settings UI.
*   **Theme Selection:** Choose from built-in Gradio themes (requires app restart).
*   **Contextual Help:** Integrated tooltips (`info=`) and markdown explanations throughout the UI.
*   **Model Release Utility:** Option to unload models from Ollama VRAM.
*   **Modular Codebase:** Refactored structure (`core`, `agents`, `ui`) for better organization.

**Image Captioning Features:**

*   **Caption Editor Tab:** Dedicated UI to load images from a folder, view/edit associated `.txt` caption files, and save changes. Includes image preview.
*   **Batch Caption Editing:** Append or prepend text to captions of multiple selected images.
*   **(WIP / Planned) Agent-Driven Captioning:** Generate captions for single or multiple images using selected Agents or Agent Teams via multimodal LLMs.

**Experimentation Features:**

*   **Experiment Sweep Tab (Basic):** Systematically run base prompts across multiple selected Agent Teams and Worker Models. Saves detailed JSON protocol files for each run in `sweep_runs/`.

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
├── agent_teams.json        # Stores PREDEFINED & USER-SAVED Agent Team/Workflow definitions
│
├── agents/                 # --- Agent Logic & Definitions ---
│   ├── __init__.py
│   ├── roles_config.py     # Logic to load/merge default, custom, file roles
│   ├── ollama_agent.py     # Interacts with Ollama API (get_llm_response)
│   ├── agent_roles.json    # Default agent definitions
│   ├── custom_agent_roles.json # User's custom persistent agents
│   └── examples/           # --- Optional: Example Agent Files (Loaded via UI) ---
│       ├── agents_architecture.json
│       ├── agents_fashion_design.json
│       ├── agents_industrial_design.json
│       └── agents_scifi_robots.json
│
├── core/                   # --- Core Logic & Utilities ---
│   ├── __init__.py
│   ├── app_logic.py        # Callback logic functions (router, chat, comment, settings, teams, etc.)
│   ├── agent_manager.py    # Orchestrates Agent Team Workflows (run_team_workflow)
│   ├── captioning_logic.py # Logic for loading, saving, editing captions
│   ├── history_manager.py  # Loads/saves persistent history (history.json)
│   ├── ollama_checker.py   # Ollama startup check logic (OllamaStatusChecker)
│   ├── ollama_manager.py   # Ollama specific actions (release_model, release_all)
│   ├── sweep_manager.py    # Logic for running experiment sweeps (run_sweep)
│   ├── utils.py            # Common utilities (JSON loading, path handling, theme map etc.)
│   ├── help_content.py     # Stores help text (tooltips, markdown) for UI
│   └── history.json        # Persistent history data file
│
├── ui/                     # --- UI Tab Definitions ---
│   ├── __init__.py
│   ├── chat_tab.py
│   ├── app_settings_tab.py
│   ├── team_editor_tab.py  # UI for editing teams
│   ├── sweep_tab.py        # UI for experiment sweeps
│   ├── captions_tab.py     # UI for caption editing & generation
│   ├── roles_tab.py
│   ├── history_tab.py
│   └── common_ui_elements.py
│
├── scripts/                # --- Utility & Setup Scripts ---
│   ├── (Batch files: setup.bat, setupvenv.bat, go.bat, govenv.bat)
│   └── full_project_creator.py
│   └── (Optional: .sh equivalents)
│
├── sweep_runs/             # Default Output folder for Sweep Protocols (add to .gitignore)
│
├── tests/                  # --- Automated Tests ---
│   ├── __init__.py
│   └── test_agent.py       # Example tests (Partially Implemented)
│   └── (Planned: test_app_logic.py, test_agent_manager.py, test_captioning_logic.py, etc.)
│
├── .gitignore              # Should include venv/, __pycache__/, sweep_runs/ etc.
└── README.md               # This file
```

## Development Status & Plan

**Implemented:**

*   Core Refactoring (Modular Structure)
*   Ollama Integration (API Call, Startup Check, Error Handling)
*   Agent System (Default, Custom, File Loading)
*   Multimodal Input Handling (Single/Folder in Chat Logic)
*   Configuration System (JSON files + Settings UI)
*   API Option Profiles (Loadable via UI)
*   UI Theme Selection (Requires Restart)
*   Contextual Help System
*   History System (Persistent & Session)
*   Agent Team / Workflow **Execution** (Sequential, Basic Assembly)
*   Basic Experiment Sweep **Execution** (Teams & Models)
*   Basic Image Caption Editor Tab (Load, View, Edit, Save, Batch Append/Prepend)

**Work In Progress / Immediate Next Steps:**

1.  **Implement Agent-Driven Captioning (High Priority):**
    *   **Goal:** Integrate agent/team execution into the Captions tab to generate captions for single/multiple images using multimodal models.
    *   **Status:** Basic caption editor tab UI and logic exist. Need to add agent selection, generate buttons, and the backend logic (`generate_caption_with_agent`, `batch_generate_captions` in `core/captioning_logic.py`) which calls the `execute_chat_or_team` router with the image context. Requires careful state passing and model selection (using chat model or dedicated caption model).
2.  **Implement Agent Team Editor (High Priority):**
    *   **Goal:** Allow users to visually create, edit, save, delete Agent Teams via the UI.
    *   **Status:** UI defined (`ui/team_editor_tab.py`), backend logic functions defined (`core/app_logic.py`). Needs thorough testing, potential UI refinement for step editing/reordering. Ensure save/delete correctly updates all relevant dropdowns.
3.  **Unit Testing (High Priority - Parallel Task):**
    *   **Goal:** Develop comprehensive unit tests for core logic modules.
    *   **Status:** Minimal implementation. Needs significant expansion using `pytest` and `pytest-mock` covering `utils`, `history`, `roles`, `agent_manager`, `captioning_logic`, `sweep_manager`, `app_logic` callbacks.

**Future / Planned Enhancements:**

*   **Advanced Agent Teams / Hierarchical Agents:** Implement manager agents for dynamic planning, conditional execution, or iterative refinement loops.
*   **Refine Sweep Feature:** Add parameter sweeping, per-step model selection, progress bars (requires Gradio upgrade or alternative handling).
*   **UI/UX Improvements:** Enhance Team Editor, improve status reporting, more help content.
*   **Gradio 4.x Upgrade:** Evaluate and execute upgrade for newer features.

---
ArtAgents by Daniel Sandner ©2024. Adapt and use creatively. No guarantees provided.
[AI/ML Articles](https://sandner.art/)
