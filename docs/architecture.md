# ArtAgents System Architecture

## 1. Overview

ArtAgents is designed as a Python application with a web-based user interface provided by the Gradio library. Its core purpose is to facilitate interaction with local language models (LLMs) served by Ollama, enabling users to leverage specialized AI "Agents" and multi-step "Team" workflows for creative text generation, image captioning, and experimentation.

The architecture emphasizes **modularity**, **local processing**, and **configurability**.

## 2. Core Principles

*   **Local First:** All primary AI processing relies on a locally running Ollama instance. No external cloud APIs are required for core LLM interactions.
*   **Modularity:** The codebase is divided into distinct functional areas:
    *   `ui/`: Defines Gradio components for each tab/section.
    *   `core/`: Contains backend logic, utilities, and managers for core functionalities.
    *   `agents/`: Manages agent definitions and interaction with the Ollama API.
*   **Configuration via Files:** Key aspects like application settings, available models, prompt limiters, agent definitions (roles), API parameter presets, and team workflows are stored in external JSON files, allowing for easy customization without code changes.
*   **Agent-Based Abstraction:** Interactions are framed around "Agents" with specific roles or multi-agent "Teams", providing a higher-level abstraction over direct LLM prompting.

## 3. Component Breakdown

```
+---------------------+      +-------------------------+      +---------------------+
|      UI Layer       |----->|  Backend Logic Layer    |<---->|   Agent Subsystem   |
| (Gradio Interface)  |      | (app.py, core/*.py)     |      | (agents/, core/*)   |
| - ui/*.py           |<-----|                         |----->|                     |
| - app.py (Wiring)   |      +-------------------------+      +----------+----------+
+---------------------+                |                             |
        ^                            |                             |
        |                            v                             v
+---------------------+      +-------------------------+      +---------------------+
|       User          |      | Configuration Files     |      |   Ollama Service    |
|   (Web Browser)     |      | (*.json)                |      | (Local LLMs)        |
+---------------------+      +-------------------------+      +---------------------+
```

*   **UI Layer (`ui/`, `app.py`)**
    *   **Framework:** Built entirely using the [Gradio](https://www.gradio.app/) library.
    *   **Structure:** Defined modularly in files within the `ui/` directory (e.g., `chat_tab.py`, `captions_tab.py`). Each file typically contains a function that creates the Gradio components for a specific tab.
    *   **Orchestration (`app.py`):** Initializes the Gradio `Blocks` interface, calls the UI creation functions from `ui/`, defines application-level `gr.State` variables, and wires UI component events (`.click()`, `.change()`, etc.) to backend callback functions located primarily in `core/app_logic.py` and `core/refinement_logic.py`.

*   **Backend Logic Layer (`core/`)**
    *   **`app_logic.py`:** Contains most of the Python callback functions directly triggered by Gradio UI events (button clicks, dropdown changes). It acts as a router (`execute_chat_or_team`) directing requests to either single-agent logic or the team manager. It also handles UI updates, state management interactions, settings logic, and team editor actions.
    *   **`refinement_logic.py`:** Contains the `comment_logic` specifically for handling the "Comment/Refine" feature, constructing a specialized prompt for text modification.
    *   **`utils.py`:** Provides common utility functions for tasks like loading/saving JSON, resolving file paths, cleaning text artifacts, and formatting data for display.
    *   **`history_manager.py`:** Manages reading from and writing to the persistent `history.json` log file.
    *   **Feature-Specific Logic:** Modules like `captioning_logic.py` and `sweep_manager.py` encapsulate the backend logic for the Image Captions and Experiment Sweep tabs, respectively.

*   **Agent Subsystem (`agents/`, `core/agent_manager.py`)**
    *   **Agent Definitions (`agents/*.json`):** JSON files define the available agent "roles". Each role typically includes a description (used in the system prompt) and optional default Ollama API parameters. `agent_roles.json` holds defaults, `custom_agent_roles.json` allows user overrides.
    *   **Role Loading (`agents/roles_config.py`):** Logic to load and merge agent roles from default, custom, and optionally user-uploaded files based on application settings.
    *   **Ollama Interaction (`agents/ollama_agent.py`):** Contains the critical `get_llm_response` function. This function:
        *   Constructs the final prompt based on role, context, and user input.
        *   Merges Ollama API options from settings, role definitions, and direct overrides.
        *   Handles encoding of optional image inputs (PIL Images) into base64 format.
        *   Sends the request payload (prompt, model, options, images) to the configured Ollama API endpoint (`/api/generate`).
        *   Processes the streaming response from Ollama.
        *   Includes error handling for connection issues, timeouts, HTTP errors, and stream decoding problems.
    *   **Team Orchestration (`core/agent_manager.py`):** Contains the `run_team_workflow` function. This function:
        *   Parses team definitions from `agent_teams.json`.
        *   Executes agent steps sequentially.
        *   Manages the passing of context (previous step outputs) to subsequent agents.
        *   Calls `agents/ollama_agent.py` for each step.
        *   Handles different final output assembly strategies (`concatenate`, `refine_last`, `summarize_all`, `structured_concatenate`).
        *   Logs workflow steps to history.

*   **Ollama Service Interaction (`core/ollama_*.py`)**
    *   **`ollama_checker.py`:** Performs a simple check on application startup to see if the Ollama service is responsive at its base URL.
    *   **`ollama_manager.py`:** Provides functions (`release_model`, `release_all_models_logic`) to interact with Ollama for unloading models from memory (VRAM), typically by sending a request with `keep_alive: 0`.

*   **Configuration Files (`*.json`)**
    *   Central to customizing the application's behavior without code changes. Files include `settings.json`, `models.json`, `limiters.json`, `ollama_profiles.json`, `agent_roles.json`, `custom_agent_roles.json`, `agent_teams.json`. Loaded via `core/utils.py`.

## 4. Key Workflows / Data Flow

*   **Single Agent Chat (Text):**
    1.  User Input (UI) -> `app.py` (Event Wiring)
    2.  -> `app_logic.execute_chat_or_team` (Router) -> `app_logic.chat_logic`
    3.  -> `ollama_agent.get_llm_response` (Formats prompt, options)
    4.  -> Ollama API (`/api/generate` POST request with JSON payload)
    5.  <- Ollama API (Streaming response chunks)
    6.  <- `ollama_agent.get_llm_response` (Assembles response text or error)
    7.  <- `app_logic` (Applies optional cleaning) -> `app.py` -> UI Output (Textbox update)

*   **Single Agent Chat (Image):**
    1.  User Input + Image (UI) -> `app.py`
    2.  -> `app_logic.execute_chat_or_team` -> `app_logic.chat_logic`
    3.  -> `ollama_agent.get_llm_response` (Formats prompt, options, **encodes image to base64**)
    4.  -> Ollama API (POST request with JSON payload **including "images" list**)
    5.  <- Ollama API (Streaming response)
    6.  <- `ollama_agent` -> `app_logic` -> `app.py` -> UI Output

*   **Team Workflow Execution:**
    1.  User Input (UI) -> `app.py`
    2.  -> `app_logic.execute_chat_or_team` (Router identifies Team)
    3.  -> `agent_manager.run_team_workflow` (Parses team definition)
    4.  -> **Loop:** For each step:
        *   `agent_manager` builds context -> `ollama_agent.get_llm_response` -> Ollama API -> Ollama Response -> `agent_manager` updates context.
    5.  -> `agent_manager` assembles final output based on `assembly_strategy`.
    6.  <- `app_logic` (Applies optional cleaning) -> `app.py` -> UI Output

*   **Caption Generation:**
    1.  User selects image/triggers action (UI) -> `app.py`
    2.  -> `captioning_logic.generate_captions_for_selected` (Loads image via PIL)
    3.  -> `app_logic.execute_chat_or_team` (Called with image data)
    4.  -> (Follows Single Agent or Team workflow path, passing image data down)
    5.  <- Response received by `captioning_logic`
    6.  -> `captioning_logic` saves response to `.txt` file and updates UI/state.

## 5. Current Limitations / Future Directions

*   **Configuration:** Primarily JSON-based. Migration to Hydra planned for Phase 1 for more powerful configuration management and experiment tracking.
*   **UI Framework:** Built on Gradio 3.x. An upgrade to Gradio 5.x is planned for Phase 1 to potentially leverage newer components and features, but will require careful migration.
*   **Workflow Logic:** Currently sequential. Future phases may introduce more complex logic (conditional steps, loops, hierarchical agents).
*   **Error Handling:** Basic error handling exists, but could be made more robust and user-friendly in specific scenarios.
*   **Testing:** Unit test coverage is currently minimal and needs significant expansion (planned).

This architecture provides a flexible foundation for experimenting with agent-based interactions and workflows using local LLMs.
