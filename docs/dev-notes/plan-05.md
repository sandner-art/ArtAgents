### **`plan-05.md` - Updated Development Plan**

## ArtAgents Development Plan v0.5

This document outlines the strategic development plan for ArtAgents, reflecting the completion of the initial stabilization phase and focusing on expanding the tool's capabilities as a creative workflow orchestration engine for artists and researchers.

### Current State: Phase 0 (Stabilization) Complete

The application has achieved a stable and functional baseline:
*   **Gradio 4+ Migration:** The UI has been successfully upgraded, resolving major breaking changes and modernizing the technical foundation.
*   **Core Feature Stability:** The Chat, Image Captioning, and Team Editor tabs are functional.
*   **Novel Synthesis Strategies:** The initial implementation of dynamic, creative strategies (`metaphorical_synthesis`, `conceptual_blend`, `stylistic_mashup`) is complete and integrated into the `agent_manager`.
*   **Agent Team Functionality:** The core system for defining and executing multi-agent workflows is robust.

### Strategic Vision: From Prompt Generator to Creative Engine

The project's focus will now shift from being a tool that *generates text prompts* to a comprehensive engine that **orchestrates and executes entire creative visual workflows**. This involves integrating directly with image generation models and building out the framework to support systematic, research-oriented experimentation for artists.

---

### **Phase 1: Foundational Expansion & Integration**

**Goal:** Integrate direct image generation capabilities and make the core configuration more robust and scalable.

1.  **Direct Image Generation Integration (Top Priority):**
    *   **Architecture:** Create an abstracted backend system for image generators. A new `image_generators.json` file will define connection details (API endpoints, types, model names).
    *   **Initial Implementation (ComfyUI):** Develop a client in `core/` to connect to a local ComfyUI instance via its API. This client will be responsible for sending prompts and receiving generated images.
    *   **UI Enhancements:**
        *   Add a "Generate Image" button to the Chat tab.
        *   Add an image display component to the Chat tab to show the final generated image.
        *   The `execute_chat_or_team` function will be updated to optionally trigger this image generation step after the prompt is created.

2.  **Refine Creative Strategies with User Input:**
    *   Enhance the `metaphorical_synthesis` and `stylistic_mashup` strategies.
    *   Add an optional textbox to the **Team Editor UI** that becomes visible when these strategies are selected, allowing the user to provide their own metaphor or style (e.g., "Use my metaphor: 'a library of whispers'").
    *   The `agent_manager` will be updated to use this user-provided text if available, falling back to the dynamic LLM-based extraction if it's empty.

3.  **Hydra Integration for Configuration:**
    *   Following the original plan, migrate all `.json` configuration files (`settings.json`, `models.json`, etc.) to a `conf/` directory using Hydra's `.yaml` format.
    *   This will dramatically improve the management of complex experimental setups, especially for the Sweep tab.

4.  **Unit Testing Expansion:**
    *   Develop a suite of `pytest` tests for the new image generation clients and the refined creative strategies to ensure stability.

---

### **Phase 2: Research & Experimental Features (The Alpha-Omega Workflow)**

**Goal:** Build the full "Image-to-Image Synthesis" workflow as envisioned in the research plans, turning the Sweep tab into a powerful research tool.

1.  **Implement Image-to-Image Team Type:**
    *   Update `agent_teams.json` and the `agent_manager` to support the new workflow:
        1.  **Stage 1 (Analysis):** An initial image (Alpha) is described by multiple parallel agents.
        2.  **Stage 2 (Synthesis):** The text descriptions are synthesized into a new prompt using a selected strategy.
        3.  **Stage 3 (Generation):** The new prompt is sent to the integrated image generator to create the final image (Omega).

2.  **Major Sweep Tab Overhaul:**
    *   Update the **Sweep UI** to be the primary interface for these research workflows.
    *   Allow sweeping across multiple axes: Alpha images, Agent Teams, Synthesis Strategies, and key image generation parameters (Seed, CFG Scale, Sampler Steps).

3.  **Automated Evaluation Metrics:**
    *   Create a new `core/evaluation_metrics.py` module.
    *   Integrate libraries like `scikit-image`, `lpips`, and `clip` to calculate metrics (SSIM, LPIPS, CLIP Score) comparing Alpha and Omega images.
    *   These metrics will be automatically calculated and saved in the sweep protocol files.

---

### **Phase 3: Usability, Extensibility & Future Vision**

**Goal:** Lower the barrier to entry for artists and researchers, and expand the tool's reach.

1.  **Advanced UI/UX - Results Viewer:**
    *   Create a new "Results Viewer" tab designed to load, parse, and display the output from Sweep runs.
    *   It will feature a filterable table of results and a side-by-side Alpha/Omega image viewer with access to all intermediate text and metrics.

2.  **Advanced Agentics:**
    *   Implement a "Manager Agent" that can dynamically plan a sequence of worker agents based on a high-level user goal, as outlined in `paper-plan.md`.

3.  **External API & Cloud Model Integration:**
    *   Expand the image generation backend to support cloud-based APIs like Gemini or DALL-E, allowing users without powerful local hardware to leverage the agentic workflows.
    *   Explore creating a simple API for ArtAgents itself, allowing other tools like ComfyUI to call an ArtAgents workflow to generate a prompt and receive the text back.

4.  **Extensibility & Community:**
    *   Develop a clear contribution guide for adding new `agents`, `synthesis_strategies`, and `evaluation_metrics`, turning the tool into an extensible framework.

