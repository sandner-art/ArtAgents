# ArtAgents: Agent-Based Creative Toolkit

ArtAgents is a prototype framework designed for artists, designers, and creators to experiment with LLM-based prompt engineering and creative content generation. It leverages Ollama for local model serving, allowing users to interact with various text and multimodal models through specialized AI 'agents' and structured, configurable workflows ("Teams").

[![artagents-github](https://github.com/user-attachments/assets/9350bb3a-9e19-4818-b109-983c5a6b0bb1)]() <!-- Consider updating screenshot to reflect current UI -->

## Overview

Select predefined agents, load custom agents, or utilize multi-agent "Teams" to generate detailed prompts, descriptions, image captions, or other text outputs. Provide text instructions and optionally images as input. Fine-tune generation using Ollama API parameters, prompt style limiters, and agent presets. Experiment systematically using the Sweep feature and manage image captions directly within the application.

## Key Features

**Core Functionality:**

*   **Ollama Integration:** Connects to a running Ollama instance to utilize locally served LLMs (text & multimodal) with startup check.
*   **Agent System:** Define and use specialized agents (Designer, Photographer, Styler, etc.) with unique instructions and optional API overrides (`agent_roles.json`, `custom_agent_roles.json`).
*   **Agent Team / Workflow Execution:** Define (`agent_teams.json`) and run multi-step agent sequences ("Teams"). Supports sequential execution with context passing and multiple result assembly strategies (`concatenate`, `refine_last`, `summarize_all`, `structured_concatenate`).
*   **Team Editor:** Create, edit, save, and delete Agent Teams via a dedicated UI tab.
*   **Chat Interface:** Main tab for direct interaction with selected agents or teams, including session history and response refinement.
*   **Multimodal Input:** Supports single image upload or processing images within a specified folder for chat or captioning context.
*   **Image Captioning:** Dedicated tab to load images from a folder, view/edit associated `.txt` caption files, save changes, and generate captions using selected agents/teams and vision models.
*   **Experiment Sweeps:** Systematically run base prompts across multiple selected Agent Teams and Worker Models. Saves detailed JSON protocol files for each run and separate `.txt` files containing the raw generated prompts per model.
*   **Configuration Management:** External JSON files for easy customization of settings, models, limiters, API profiles, agent roles, and agent teams.
*   **App Settings UI:** Dedicated tab to configure Ollama URL, agent loading preferences, default behaviors, UI theme, and detailed Ollama API parameters (with loadable profiles).
*   **Persistent History:** Logs all single interactions and detailed workflow steps to `core/history.json`, viewable and clearable in the "Full History" tab.
*   **Utilities:** Copy-to-clipboard for responses, optional prompt artifact cleaning, model release functions, contextual help tooltips, setup scripts.
*   **Modular Codebase:** Organized structure (`core`, `agents`, `ui`) for maintainability.
   
![01-git2](https://github.com/user-attachments/assets/234496b9-b816-4053-bff1-a50518e619d1)

## Project Structure

```
ArtAgent/
│
├── app.py                  # Main Gradio App: UI Structure, Event Wiring, State Mgmt
├── requirements.txt        # Python Dependencies (Consider migrating to pyproject.toml/Poetry)
├── settings.json           # App Config: Ollama URL, defaults, global API opts, theme
├── models.json             # Ollama models known to the app (name, vision)
├── limiters.json           # Prompt style limiters (name, tokens, format string)
├── ollama_profiles.json    # Presets for Ollama API options
├── agent_teams.json        # Stores PREDEFINED & USER-SAVED Agent Team/Workflow definitions
│
├── agents/                 # --- Agent Logic & Definitions ---
│   ├── __init__.py
│   ├── roles_config.py     # Logic to load/merge roles
│   ├── ollama_agent.py     # Interacts with Ollama API (get_llm_response)
│   ├── agent_roles.json    # Default agent definitions
│   ├── custom_agent_roles.json # User's custom persistent agents
│   └── examples/           # --- Optional: Example Agent Files ---
│       └── *.json
│
├── core/                   # --- Core Logic & Utilities ---
│   ├── __init__.py
│   ├── app_logic.py        # Callback logic functions (router, UI callbacks)
│   ├── refinement_logic.py # Logic for comment/refinement feature
│   ├── agent_manager.py    # Orchestrates Agent Team Workflows
│   ├── captioning_logic.py # Logic for caption editing & generation
│   ├── history_manager.py  # Loads/saves persistent history
│   ├── ollama_checker.py   # Ollama startup check logic
│   ├── ollama_manager.py   # Ollama model release logic
│   ├── sweep_manager.py    # Logic for running experiment sweeps
│   ├── utils.py            # Common utilities (JSON loading, cleaning etc.)
│   ├── help_content.py     # Stores help text for UI
│   └── history.json        # Persistent history data file
│
├── ui/                     # --- UI Tab Definitions (Gradio components) ---
│   ├── __init__.py
│   ├── chat_tab.py
│   ├── captions_tab.py     # UI for caption editing & generation
│   ├── team_editor_tab.py  # UI for editing teams
│   ├── sweep_tab.py        # UI for experiment sweeps
│   ├── history_tab.py
│   ├── info_tab.py         # Consolidated info tab (replaces roles_tab.py)
│   ├── app_settings_tab.py
│   └── common_ui_elements.py
│
├── scripts/                # --- Utility & Setup Scripts ---
│   ├── (Batch files: setup.bat, setupvenv.bat, go.bat, govenv.bat)
│   └── full_project_creator.py
│   └── (Optional: .sh equivalents)
│
├── docs/                   # --- Detailed Documentation ---
│   ├── index.md            # Overview (Placeholder)
│   ├── user-guide.md       # User manual (Placeholder)
│   ├── architecture.md     # System design (Placeholder)
│   └── api.md              # Core function details (Placeholder, Optional)
│
├── sweep_runs/             # Default Output folder for Sweep Protocols (add to .gitignore)
│
├── tests/                  # --- Automated Tests ---
│   ├── __init__.py
│   └── test_agent.py       # Example tests (Needs Expansion)
│   └── (Placeholder: other test files)
│
├── .gitignore
└── README.md               # This file
```

## Installation & Setup

1.  **Install Ollama:** Download and install from [ollama.com](https://ollama.com/). Ensure the `ollama` command is available in your terminal.
2.  **Clone Repository:** `git clone <your-repository-url>` and navigate into the `ArtAgent` directory (`cd ArtAgent`).
3.  **Setup Python Environment (Recommended):**
    *   **Using Venv (Manual):** Create and activate a virtual environment (Python 3.9+ recommended, 3.10+ required for potential Gradio 5 upgrade).
        ```bash
        python -m venv venv
        # On Windows: .\venv\Scripts\activate
        # On Linux/macOS: source venv/bin/activate
        ```
        Then install requirements:
        ```bash
        pip install --upgrade pip
        pip install -r requirements.txt
        ```
    *   **(Alternative) Using Scripts:** Run `.\scripts\setupvenv.bat` (Windows) or equivalent `.sh` script to automate venv creation and `pip install`.
    *   **(Future) Using Poetry:** If Poetry is implemented, replace step 3 with `poetry install`.
4.  **Setup Ollama Models:** Run `.\scripts\setup.bat` (Windows) or equivalent `.sh` script. This checks Ollama connectivity and downloads recommended models listed in `models.json`. Alternatively, use `ollama pull <model_name>` manually for desired models.
5.  **Configure (Optional):** Review and edit JSON files (`settings.json`, `models.json`, `agent_teams.json`, etc.) to customize the application.

## Running the Application

1.  **Start Ollama Service:** Ensure the Ollama service is running (e.g., launch the Ollama Desktop application or run `ollama serve` in a separate terminal).
2.  **Activate Environment:** If using a virtual environment, activate it (`source venv/bin/activate` or `.\venv\Scripts\activate`).
3.  **Run ArtAgents:**
    *   If using venv: `python app.py`
    *   Using Scripts: `.\scripts\govenv.bat` (Windows) or equivalent `.sh` script.
    *   (Future) Using Poetry: `poetry run python app.py`
4.  **Access UI:** Open the local URL provided in the console (usually `http://127.0.0.1:7860`) in your web browser.

## Documentation

For more detailed information, please refer to the [documents](docs/index.md) in the `/docs` directory:
*   `/docs/user-guide.md` 
*   `/docs/architecture.md` 

## Development Status & Plan

**Phase 0: Stabilization & Core Refinement (Complete)**

*   Agent Captioning functionality stabilized.
*   Agent Team Editor implemented and stabilized.
*   Core assembly strategies (`concatenate`, `refine_last`, `summarize_all`, `structured_concatenate`) tested.
*   Sweep output format implemented (per-model `.txt` prompt files + JSON protocols).
*   Optional prompt artifact cleaner added.
*   Copy-to-clipboard button added.
*   Consolidated "Info" tab implemented.
*   Error handling reviewed and improved.

**Phase 1: Foundational Expansion & Modernization (Current Focus)**

*   **Gradio 5.x Upgrade:** Evaluate and execute upgrade from Gradio 3.x.
*   **Hydra Integration:** Migrate `.json` configurations to Hydra (`.yaml`) for improved experiment management.
*   **Implement Select Novel Synthesis Strategies:** Add 2-3 creative strategies (e.g., Metaphorical Synthesis, Conceptual Blending) to `agent_manager.py` and Team Editor UI.
*   **NLP Library Integration (`nlpaug`):** Integrate for noise/synonym capabilities within strategies or as agent steps.
*   **Unit Testing Expansion:** Write comprehensive `pytest` tests for core logic and new features.

**Future / Planned Enhancements (Phase 2+):**

*   Advanced Agent Teams (Hierarchical agents, conditional logic, feedback loops).
*   Advanced Experimentation (Parameter sweeping via Hydra, potentially MLFlow integration).
*   Direct Image Generation API Integration (e.g., ComfyUI, A1111).
*   Workflow Visualization.
*   Enhanced UI/UX (Improved Team Editor, potential Gradio custom components).
*   Explainability / XAI Features.
*   More Novel Synthesis Strategies & NLP features.

## Contributing

Contributions are welcome! Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting issues, suggesting features, or submitting pull requests.

## License

ArtAgents by Daniel Sandner © 2024 - 2025. Adapt and use creatively. No guarantees provided. [MIT LICENSE](LICENSE).

---
[sandner.art | AI/ML Articles](https://sandner.art/)
