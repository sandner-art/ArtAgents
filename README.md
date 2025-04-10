# ArtAgents: Agent-Based Chat with Ollama

Prototype framework for LLM based captioning and prompt engineering experiments for artists and designers. Uses Ollama for local model serving.

[![artagents-github](https://github.com/user-attachments/assets/9350bb3a-9e19-4818-b109-983c5a6b0bb1)]() <!-- Update screenshot link if needed -->

## Overview

Select an agent, an Ollama model, and provide text or image input to generate descriptive prompts or other text outputs. Experiment with different agent personalities ("roles") and prompt styling ("limiters").

- **Multimodal Input:** Use a single image upload or point to a folder of images.
- **Agent Roles:** Define different personas (Designer, Photographer, etc.) with specific instructions and API parameter overrides (`agents/agent_roles.json`, `agents/custom_agent_roles.json`).
- **Limiters:** Apply stylistic constraints to the output prompt (`limiters.json`).
- **Configuration:** Manage Ollama settings, agent usage, and API parameters via `settings.json` and the "App Settings" tab.
- **History:** Session history and persistent history (`core/history.json`) are tracked.

## Project Structure

```
ArtAgent/
│
├── app.py                  # Main Gradio UI & Application Logic
├── requirements.txt        # Dependencies
├── settings.json           # Core application settings
├── models.json             # Ollama models configuration
├── limiters.json           # Prompt style limiters
│
├── agents/                 # Agent logic & configuration
│   ├── __init__.py
│   ├── roles_config.py     # Loads agent roles
│   ├── ollama_agent.py     # Interacts with Ollama API
│   ├── agent_roles.json    # Default roles
│   └── custom_agent_roles.json # Custom roles
│
├── core/                   # Core utilities
│   ├── __init__.py
│   ├── history_manager.py  # History persistence
│   ├── ollama_checker.py   # Ollama startup check
│   └── utils.py            # Common utilities (JSON loading etc.)
│   └── history.json        # History data file
│
├── scripts/                # Setup & run scripts
│   ├── setup.bat
│   ├── setupvenv.bat
│   ├── go.bat
│   ├── govenv.bat
│   └── full_project_creator.py # Generates project dump
│
├── tests/                  # Unit tests (using pytest)
│   ├── __init__.py
│   └── test_agent.py       # Example tests
│
├── .gitignore
└── README.md
```

## Installation & Setup

1.  **Install Ollama:** Download and install from [ollama.com](https://ollama.com/). Ensure the `ollama` command is available in your system's PATH.
2.  **Clone Repository:** Get the ArtAgents code:
    ```bash
    git clone <your-repository-url>
    cd ArtAgent
    ```
3.  **(Recommended) Setup Virtual Environment:**
    *   Open a terminal/command prompt in the `ArtAgent` directory.
    *   Run the setup script: `.\scripts\setupvenv.bat` (Windows) or `bash ./scripts/setupvenv.sh` (Linux/macOS - *you'll need to create this .sh script*).
    *   This creates a `venv` folder and installs dependencies from `requirements.txt`.
4.  **Setup Ollama Models:**
    *   Run the setup script: `.\scripts\setup.bat` (Windows) or `bash ./scripts/setup.sh` (*create this*).
    *   This script checks for Ollama and prompts you to download recommended models (`llava`, `llama3`, etc.). You can manage models manually using the `ollama pull <model_name>` command.
5.  **Configure Settings (Optional):**
    *   Edit `settings.json` to change the Ollama API URL if it's not running on the default `http://localhost:11434`.
    *   Adjust default API parameters or application behavior flags.
    *   Edit `models.json` to list the Ollama models you have available locally. Set `"vision": true` for multimodal models.

## Running the Application

1.  **Start Ollama Service:** Ensure the Ollama service/application is running in the background. You can usually start it by running `ollama serve` in a separate terminal.
2.  **Run ArtAgents:**
    *   **If using virtual environment (Recommended):**
        *   Open a terminal in the `ArtAgent` directory.
        *   Run: `.\scripts\govenv.bat` (Windows) or `bash ./scripts/govenv.sh` (*create this*).
    *   **If not using virtual environment:**
        *   Open a terminal in the `ArtAgent` directory.
        *   Run: `.\scripts\go.bat` (Windows) or `bash ./scripts/go.sh` (*create this*).
3.  **Access UI:** Open the local URL provided in the terminal (usually `http://127.0.0.1:7860`) in your web browser.

## Notes

- If there is no folder path or image inserted, the prompt is created based only on the User Input text and selected Agent.
- The "Captions" tab functionality from the original version needs to be re-integrated if desired (potentially in `ui/captions_tab.py` and `app.py`).
- Edit `agents/custom_agent_roles.json` to add your own specialized agents.

## Development

- **Testing:** Run `pytest` from the project root directory to execute unit tests (requires `pytest` and `pytest-mock`).
- **Refactoring:** The code is structured into modules (`core`, `agents`, `ui`) for better organization.

---
ArtAgents by Daniel Sandner ©2024. Adapt and use creatively. No guarantees provided.
[AI/ML Articles](https://sandner.art/)
```
--- END OF FILE ArtAgent/README.md ---

This completes the file dump based on the proposed structure. Remember to create the empty `__init__.py` files in the `agents`, `core`, and `tests` directories. You'll also need to ensure the `.bat` files correctly reference paths relative to their location in the `scripts/` folder (e.g., using `..\app.py` to point to the parent directory).