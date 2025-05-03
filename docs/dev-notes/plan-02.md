**I. Current Project Structure (Refactored)**

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
│   ├── custom_agent_roles.json # User's custom persistent agents + options
│   │                         # --- Example Agent Files (Loaded via UI) ---
│   ├── agents_architecture.json
│   ├── agents_fashion_design.json
│   ├── agents_industrial_design.json
│   └── agents_scifi_robots.json
│
├── core/                   # --- Core Logic & Utilities ---
│   ├── __init__.py         # Makes 'core' a Python package
│   ├── app_logic.py        # Callback logic functions (router, chat, comment, settings, teams, etc.)
│   ├── agent_manager.py    # Orchestrates Agent Team Workflows (run_team_workflow)
│   ├── history_manager.py  # Loads/saves persistent history (history.json)
│   ├── ollama_checker.py   # Ollama startup check logic (OllamaStatusChecker)
│   ├── ollama_manager.py   # Ollama specific actions (release_model, release_all)
│   ├── sweep_manager.py    # Logic for running experiment sweeps (run_sweep)
│   ├── utils.py            # Common utilities (load_json, get_absolute_path, theme map etc.)
│   ├── help_content.py     # Stores help text (tooltips, markdown) for UI
│   └── history.json        # Persistent history data file
│
├── ui/                     # --- UI Tab Definitions ---
│   ├── __init__.py         # Makes 'ui' a Python package
│   ├── chat_tab.py         # Builds the main Chat interface components
│   ├── app_settings_tab.py # Builds the App Settings interface components
│   ├── team_editor_tab.py  # Builds the Agent Team Editor interface components
│   ├── sweep_tab.py        # Builds the Experiment Sweep interface components
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
├── sweep_runs/             # --- Output folder for Sweep Protocols (Created by sweep_manager) ---
│   └── YYYYMMDD_HHMMSS_FolderName/ # Subfolder for each sweep run
│       └── protocol_file_1.json
│       └── protocol_file_2.json
│       └── ...
│
├── tests/                  # --- Automated Tests (Partially Implemented) ---
│   ├── __init__.py         # Makes 'tests' a Python package
│   └── test_agent.py       # Tests for ollama_agent.py (using mocks)
│   └── (Planned: test_app_logic.py, test_agent_manager.py, test_utils.py, etc.)
│
├── .gitignore              # Git ignore rules (venv, pycache, logs, sweep_runs etc.)
└── README.md               # Project documentation (setup, usage, structure)

```
*(Note: `sweep_runs/` added to show where output goes; needs to be added to `.gitignore`)*
*(Note: Example agent files like `agents_architecture.json` are placed in `agents/` for simplicity, could be in `agents/examples/`)*

**II. Development Plan**

**What's Done / Implemented:**

1.  **Core Refactoring:** Codebase organized into `core`, `agents`, `ui` modules. `app.py` handles UI assembly and event wiring.
2.  **Stable Single Agent Chat:** Core chat and comment functionality using selected agents and models.
3.  **Ollama Integration:** Connection via API, error handling for connection issues during requests (`agents/ollama_agent.py`), startup check (`core/ollama_checker.py`), model release utilities (`core/ollama_manager.py`).
4.  **Configuration System:** Settings (`settings.json`), Models (`models.json`), Limiters (`limiters.json`), Profiles (`ollama_profiles.json`), Agent Roles (default, custom), Agent Teams (`agent_teams.json`) are loaded from external JSON files.
5.  **UI Features:**
    *   App Settings Tab: Allows configuration of most settings and Ollama API options.
    *   API Option Profiles: Users can load presets (Fast, Balanced, Creative) onto the settings UI (requires manual save).
    *   Theme Selection: Users can select built-in Gradio themes (requires app restart).
    *   Agent File Loading: Users can load `.json` agent definitions temporarily for the current session via the Chat tab.
    *   Contextual Help: Tooltips and markdown explanations added to many UI elements (`core/help_content.py`).
    *   History Tracking: Persistent history (`core/history_manager.py`, History Tab) and Session history display (Chat Tab).
6.  **Agent Team / Workflow EXECUTION:**
    *   Agent teams can be defined in `agent_teams.json`.
    *   `core/agent_manager.py` orchestrates sequential execution based on team definitions.
    *   Supports "concatenate" and "refine_last" assembly strategies.
    *   Logs workflow steps to persistent history.
    *   Users can select and run predefined teams via the Chat tab dropdown.
7.  **Experiment Sweep (Basic - Teams & Models):**
    *   Dedicated "Experiment Sweep" tab (`ui/sweep_tab.py`).
    *   Users can select multiple prompts, teams, and worker models.
    *   `core/sweep_manager.py` iterates through combinations, calls the agent manager, and saves structured JSON protocol files for each run in `sweep_runs/`.
    *   Basic status reporting to the UI textbox (no progress bar due to Gradio 3.x limitations without queue).

**Planned Next Steps / Future Development:**

1.  **Implement Agent Team Editor (Phase 2 - Highest Priority):**
    *   **Goal:** Allow users to create, edit, save, and delete Agent Teams via the "Agent Team Editor" tab UI.
    *   **Description:** Implement the remaining callbacks in `core/app_logic.py` (`load_team_for_editing`, `clear_team_editor`, `add_step_to_editor`, `remove_step_from_editor`, `save_team_from_editor`, `delete_team_logic`). Ensure saving/deleting updates `agent_teams.json` and refreshes the relevant dropdowns in the Chat and Editor tabs. May require refining the dynamic step editor UI beyond simple JSON display for better UX.
2.  **Unit Testing (High Priority - Parallel Task):**
    *   **Goal:** Create robust unit tests for core logic modules.
    *   **Description:** Flesh out `tests/` directory. Write `pytest` tests using `pytest-mock` for:
        *   `core/utils.py` (file loading, path handling).
        *   `core/history_manager.py` (loading, saving, adding).
        *   `core/agent_manager.py` (workflow execution logic, context building, assembly - mocking `get_llm_response`).
        *   `core/app_logic.py` (callback logic, especially routing and state manipulation - may require mocking Gradio components/updates or focusing on pure logic parts).
        *   `agents/roles_config.py` (role loading and merging).
3.  **Refine Sweep Feature:**
    *   **Goal:** Improve usability and capabilities.
    *   **Description:**
        *   **(If Upgrading Gradio):** Re-implement `gr.Progress` for visual feedback.
        *   Add ability to sweep specific Ollama API parameters (requires UI changes and modification to `sweep_manager` and `agent_manager`).
        *   Add option to specify different models *per step* within a team definition.
        *   Improve protocol file content/structure.
        *   Potentially add a simple protocol viewer tab.
4.  **Advanced Agent Team / Hierarchical Agents:**
    *   **Goal:** Implement more dynamic workflow orchestration (Designs 2 or 3).
    *   **Description:** Introduce a "Manager" agent role. Implement logic in `agent_manager` for the Manager to parse user input, generate a plan (sequence of worker agents), and potentially modify execution based on intermediate results. Requires significant prompt engineering for the Manager and robust parsing logic in Python.
5.  **UI/UX Enhancements:**
    *   **Goal:** Improve usability and visual appeal.
    *   **Description:** Refine the Team Editor UI (e.g., drag-and-drop steps). Improve status reporting. Add more granular contextual help. Potentially allow custom CSS themes.
6.  **Captioning Tab:**
    *   **Goal:** Re-integrate image caption viewing/editing functionality.
    *   **Description:** Create `ui/captions_tab.py` and port/adapt logic from the original `previewcaption.py` or the initial `app.py` implementation.
7.  **Gradio 4.x Upgrade:**
    *   **Goal:** Move to the latest Gradio version.
    *   **Description:** Undertake as a dedicated task *after* current features are stable. Requires code modifications according to Gradio 4 API changes.

This plan focuses on completing the user-configurable teams feature next, alongside building a testing foundation, before moving onto more advanced workflow capabilities or major library upgrades.

--- 

# **Advanced Workflow Capabilities & Hierarchical Agents (Future Goals)**

The core idea is to move beyond fixed, predefined sequences and introduce dynamism, decision-making, and potentially feedback loops within the agent workflows. This usually involves one or more "Manager" or "Controller" agents.

**1. Manager-Planned Workflows (Dynamic Sequencing)**

*   **Concept:** Instead of the user selecting a fixed team like "Detailed Object Design", they might select a high-level goal or a "Planning Manager" agent. This manager analyzes the user's request and dynamically determines the sequence of specialized "Worker" agents needed.
*   **Workflow:**
    1.  **User Input:** "Generate a photorealistic image prompt for a cozy reading nook in a spaceship, featuring a large window overlooking nebula X."
    2.  **UI Selection:** User selects "Generate Scene Prompt" workflow type, which triggers the `PlanningManager` agent.
    3.  **Orchestrator (`agent_manager.py`):** Calls `PlanningManager`.
    4.  **`PlanningManager` Prompt:** "User Request: '[User Input]'. Available Worker Agents: [Interior_Designer, SciFi_Concept_Artist, Detailer, Photographer, Styler]. Generate a step-by-step plan using the available agents to create a final, detailed image prompt. Output the plan as a numbered list or JSON."
    5.  **`PlanningManager` Output (Plan):**
        ```json
        [
          {"step": 1, "role": "Interior_Designer", "goal": "Design the basic layout, furniture, and cozy elements of the reading nook."},
          {"step": 2, "role": "SciFi_Concept_Artist", "goal": "Integrate spaceship elements, the nebula view through the window, and futuristic details."},
          {"step": 3, "role": "Detailer", "goal": "Add specific textures (fabrics, metals), lighting details, and small objects (books, mug)."},
          {"step": 4, "role": "Photographer", "goal": "Define camera angle, lens, depth of field, and overall cinematic lighting for the final shot."},
          {"step": 5, "role": "Universal Prompter", "goal": "Combine all descriptions into a coherent, effective image prompt."}
        ]
        ```
    6.  **Orchestrator (`agent_manager.py`):** Parses the plan (JSON is easiest) and executes the steps sequentially, passing context like before.
*   **Implementation Challenges:**
    *   **Prompt Engineering:** Crafting the `PlanningManager` prompt is critical. It needs clear instructions, access to the list of available worker agents and their capabilities (descriptions).
    *   **Output Parsing:** Reliably parsing the Manager's plan (JSON is preferred over natural language lists).
    *   **LLM Capability:** Requires a sufficiently powerful LLM for the Manager role to perform effective planning.

**2. Conditional Execution & Agent Selection**

*   **Concept:** A Manager agent doesn't just plan the sequence but makes decisions *during* the workflow based on intermediate results.
*   **Workflow Example:**
    1.  **User Input:** "Design a cyberpunk character - maybe a hacker or a street samurai."
    2.  **Call `PlanningManager`:** It might output a plan like: `[CharacterCore, SelectArchetype, DetailOutfit, FinalizePrompt]`.
    3.  **Execute Step 1 (`CharacterCore`):** Output might be "Slightly built figure, agile, augmented eyes, neutral expression."
    4.  **Execute Step 2 (`SelectArchetype` - A Decision Agent):**
        *   *Input:* User Request, Core Concept Output.
        *   *Prompt:* "User wants a cyberpunk hacker OR street samurai. Core concept is '[Output from Step 1]'. Based on this, which archetype fits better, or should we ask the user for clarification? Output ONLY 'Hacker', 'Samurai', or 'Clarify'."
        *   *Output:* `Hacker`
    5.  **Orchestrator:** Receives "Hacker". It *conditionally* calls the `DetailOutfit` agent with instructions tailored to a hacker archetype (e.g., focus on tech wear, datajack ports). If the output was "Samurai", it would call `DetailOutfit` with different instructions (e.g., focus on augmented katana, armored clothing). If "Clarify", it would pause and prompt the user in the UI.
*   **Implementation Challenges:**
    *   More complex Manager prompts for decision-making.
    *   Orchestrator needs conditional logic (`if/else`) based on agent outputs.
    *   Handling clarification requests back to the user via the UI.

**3. Iterative Refinement & Feedback Loops**

*   **Concept:** Allow the output of one agent (or the assembled output) to be reviewed, critiqued, or refined by another agent (or even the same agent again) in a loop until a quality threshold or iteration limit is met.
*   **Workflow Example:**
    1.  Run a standard sequential workflow (e.g., Style -> Form -> Detail -> Assemble).
    2.  **Call `CritiqueAgent`:**
        *   *Input:* Assembled Prompt from previous steps.
        *   *Prompt:* "Critique the following image prompt for clarity, detail, consistency, and potential for generating a high-quality image: '[Assembled Prompt]'. List specific areas for improvement."
        *   *Output:* "Critique: Good detail on materials, but the lighting description is vague. The pose could be more dynamic. Consider adding atmospheric effects."
    3.  **Call `RefinementAgent` (or loop back to specific earlier agents):**
        *   *Input:* Assembled Prompt + Critique Output.
        *   *Prompt:* "Revise the following prompt based on the critique provided. Original: '[Assembled Prompt]'. Critique: '[Critique Output]'."
        *   *Output:* Revised Prompt.
    4.  **Orchestrator:** Could loop this critique/refinement cycle 1-2 times or based on some condition.
*   **Implementation Challenges:**
    *   Designing effective `CritiqueAgent` and `RefinementAgent` prompts.
    *   Implementing the loop logic in the orchestrator.
    *   Defining stopping conditions for the loop.
    *   Managing context length as it grows with each iteration.

**4. Parallel Execution (for Independent Sub-tasks)**

*   **Concept:** If a request involves distinct components that can be designed independently, run specialist agents for those components in parallel and then combine the results.
*   **Workflow Example:**
    1.  **User Input:** "Design a sci-fi scene with a sleek exploration rover parked next to a strange crystalline alien plant."
    2.  **`PlanningManager` Output (Plan):**
        ```json
        [
          {"task": "RoverDesign", "role": "Robot_Designer_HardSurface", "goal": "Design the rover."},
          {"task": "PlantDesign", "role": "Alien_Creature_Designer", "goal": "Design the crystalline plant."},
          {"task": "SceneComposition", "role": "Photographer", "goal": "Describe how rover and plant are arranged, lighting, environment."}
        ]
        ```
        *(Note: Manager identifies potentially parallelizable tasks)*
    3.  **Orchestrator:**
        *   Calls RoverDesign agent -> Gets Rover Description.
        *   Calls PlantDesign agent -> Gets Plant Description. (These two *could* potentially run in parallel if using async execution).
        *   Calls SceneComposition agent, providing *both* the Rover and Plant descriptions as context -> Gets Composition/Lighting details.
    4.  **Final Assembly:** Combine Rover Desc + Plant Desc + Scene Comp details into the final prompt.
*   **Implementation Challenges:**
    *   Manager needs to identify tasks suitable for parallel execution.
    *   Orchestrator needs logic to potentially run calls concurrently (using `asyncio` or threading, adding complexity) or just sequentially.
    *   Combining the parallel outputs coherently in the final step.

**Common Needs for Advanced Workflows:**

*   **Robust Orchestrator (`core/agent_manager.py`):** Needs to handle different strategies, parse plans/decisions, manage context effectively, potentially handle loops/conditionals.
*   **Capable Manager LLM:** An LLM with strong reasoning and instruction-following is key for dynamic planning and decision-making.
*   **Clear Agent Definitions:** Worker agents need well-defined roles and potentially structured input/output formats if the manager relies on parsing specific information.
*   **Context Management:** Careful handling of the growing context passed between steps is vital to avoid exceeding model limits and keep agents focused. Techniques like summarizing previous steps might be needed.
*   **Error Handling:** More complex workflows have more potential failure points. The orchestrator needs robust error handling for each step.
*   **Testing:** Mocking becomes even more critical to test the orchestrator's logic without numerous LLM calls.

Implementing these advanced features transforms ArtAgents from a prompt generation helper into a more sophisticated, automated creative workflow tool. Start simple (fixed pipelines), test thoroughly, and incrementally add complexity like manager agents and conditional logic.