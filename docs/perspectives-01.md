**Proposed Additions/Enhancements for ArtAgents Experimental Features**

The goal is to enable systematic research into image transformation workflows (Alpha -> Omega) driven by agentic text synthesis, allowing quantitative evaluation of outcomes and component contributions.

**1. New Workflow Type: "Image-to-Image Synthesis" Team**

*   **Concept:** Define a specialized "Agent Team" type in `agent_teams.json` specifically for the Alpha-to-Omega workflow.
*   **Structure Definition:** This team type would have a predefined structure:
    *   **Input:** Takes an Image Path (Alpha) as primary input.
    *   **Stage 1: Parallel Agent Description:** Executes multiple specified agents (e.g., "Photographer", "Art Critic", "Color Analyst") in parallel, each taking the Alpha image (potentially via a multimodal LLM specified for that agent) and generating a text description based on their role/instructions.
    *   **Stage 2: Manager Synthesis:** Collects the text descriptions from Stage 1 agents. Applies a specified "Synthesis Strategy" (see point 2) to combine them into a single, unified text prompt. This stage might use a dedicated "Manager" agent (another LLM call) or predefined logic.
    *   **Stage 3: Image Generation:** Sends the synthesized prompt to a specified Image Generation backend (see point 3) to generate the Omega image.
    *   **Output:** Returns the path to the generated Omega image and potentially intermediate data (agent descriptions, final prompt).
*   **Configuration (`agent_teams.json`):**
    *   `"team_type": "image_to_image_synthesis"`
    *   `"input_mode": "image"`
    *   `"stage1_agents": ["agent_id_1", "agent_id_2", ...]` (references agents defined elsewhere)
    *   `"stage2_synthesis_strategy": "strategy_name"` (e.g., "concatenate_structured", "manager_agent_refine", "comparative_summary")
    *   `"stage2_manager_agent": "optional_manager_agent_id"` (if strategy uses an LLM)
    *   `"stage3_image_generator": "generator_config_id"` (references config in a new file, e.g., `image_generators.json`)

**2. Enhanced Synthesis Strategies**

*   **Concept:** Formalize and expand the ways agent outputs are combined in Stage 2 of the new team type (and potentially applicable to existing text workflows).
*   **Implementation:** Define strategies beyond simple "concatenate" or "refine_last":
    *   `concatenate_structured`: Concatenates with clear labels (e.g., "Perspective A: [...], Perspective B: [...]").
    *   `comparative_summary`: An LLM call (using the specified `stage2_manager_agent`) prompted to explicitly compare/contrast agent inputs and summarize key points/differences.
    *   `thematic_integration`: An LLM call prompted to identify common themes and weave agent inputs into a narrative or descriptive text based on those themes.
    *   `weighted_synthesis`: (Advanced) Allow assigning weights to agents, influencing the Manager LLM's synthesis focus.
*   **Configuration:** Strategies could be simple keywords mapped to internal logic or complex prompts stored in a `synthesis_strategies.json` file, referenced by name.

**3. Image Generation Backend Integration**

*   **Concept:** Abstract the image generation step to support different methods.
*   **Implementation:**
    *   Create `image_generators.json` to define configurations (e.g., "StableDiffusion_API", "Local_ComfyUI_Workflow", "Future_Ollama_Image_Model").
    *   Each config specifies: method (API call, command line execution), endpoint/path, model identifier, default generation parameters (steps, CFG scale, sampler, seed).
    *   The core app needs code to interact with these different backends based on the selected configuration.

**4. Experiment Sweep Enhancements**

*   **Concept:** Make the "Experiment Sweep" tab the primary interface for running Alpha-Omega research batches.
*   **UI/Functionality Changes:**
    *   **Input:** Allow selecting an Alpha image (or folder of images) as the sweep input.
    *   **Sweep Variables:** Significantly expand the axes for sweeping, allowing selection of:
        *   `Image Generator Configs` (from `image_generators.json`)
        *   `Image Generation Parameters` (e.g., list of Seeds, CFG Scales, Steps)
        *   `Agent Teams` (filtering for `image_to_image_synthesis` type)
        *   `Synthesis Strategies` (for Stage 2)
        *   *Specific Agent Roles/Instructions* (Allow swapping Agent A's role within a fixed team structure across runs).
        *   *Specific Agent Models* (Allow changing the LLM used by Agent A, B, or the Manager across runs).
    *   **Metric Selection:** Add a checklist/multi-select UI for users to choose which Alpha-Omega evaluation metrics (see point 5) should be calculated and logged automatically after each successful run.
    *   **Output Management:** Define base output directory for sweep runs; automatically create subfolders for each run containing Omega image, logs, intermediate texts, and metrics file.

**5. Evaluation Metric Calculation Module**

*   **Concept:** Integrate automated calculation of key Alpha-Omega comparison metrics.
*   **Implementation:**
    *   A dedicated Python module (`evaluation_metrics.py`?) triggered after Omega generation within a sweep run.
    *   Input: Alpha image path, Omega image path, potentially the synthesized prompt.
    *   Output: A JSON object/file containing calculated metric values.
    *   **Metrics Library:** Include functions/wrappers for:
        *   Pixel-based: PSNR, SSIM (using libraries like `scikit-image`).
        *   Perceptual: LPIPS (requires `lpips` library & model download).
        *   Semantic: CLIP Score (Image-Image: `CLIP(Alpha, Omega)`), CLIP Score (Text-Image Alignment: `CLIP(Prompt, Omega)`). Requires `transformers` or `clip-openai`.
        *   (Optional/Advanced): Hooks for aesthetic models (NIMA), object detection comparison, bias metrics if feasible locally.
    *   **Configuration:** Allow users to specify paths to metric models (CLIP, LPIPS) in App Settings.

**6. Enhanced Logging and Results Visualization**

*   **Concept:** Ensure all relevant data for analysis is captured and easily accessible.
*   **Implementation:**
    *   **Sweep Protocol Files (`sweep_runs/run_xyz.json`):** Expand to meticulously log:
        *   Full run configuration (team used, specific agent roles/models, synthesis strategy, GenAI settings, seed, Ollama params).
        *   Paths to Alpha and Omega images.
        *   Full text output from each Stage 1 agent.
        *   The final synthesized prompt sent to Stage 3.
        *   The calculated evaluation metrics dictionary.
        *   Timestamps and any errors.
    *   **Results Viewer UI:** (Could be a new Tab or enhance "Full History")
        *   Load and parse sweep protocol files.
        *   Display results in a table, sortable/filterable by configuration parameters and metric values.
        *   Show Alpha and Omega images side-by-side for selected runs.
        *   Allow easy viewing of intermediate agent texts and the synthesized prompt for a selected run.

**Strategic Approach Summary:**

By defining a dedicated `image_to_image_synthesis` workflow, integrating image generation, formalizing synthesis strategies, vastly expanding the `Experiment Sweep` capabilities to handle image I/O and multi-dimensional parameter variations, and building in automated `Evaluation Metric` calculation and logging, ArtAgents can transform into a powerful, systematic research tool for exploring the creative potential and evaluating the performance of agent-based image transformation pipelines. The focus is on modularity (separate configs, metric module) and leveraging the sweep feature as the central research engine.