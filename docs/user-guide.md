# ArtAgents User Guide

Welcome to ArtAgents! This guide will help you get started with using the application for creative exploration with local language models.

## 1. Introduction

**What is ArtAgents?**

ArtAgents is a desktop tool designed for experimenting with AI agents and multi-agent workflows ("Teams") to generate creative text, especially prompts for image generation models. It runs **locally** on your machine using [Ollama](https://ollama.com/) to serve various open-source language models (LLMs), including multimodal (vision) models.

**Who is it for?**

Artists, designers, prompt engineers, researchers, and anyone interested in:
*   Exploring different AI agent personas.
*   Building sequential workflows for complex prompt generation.
*   Comparing outputs across different local LLMs.
*   Generating image captions.
*   Managing local configurations for Ollama models.

## 2. Getting Started

Follow these steps to set up and run ArtAgents:

1.  **Install Ollama:** Download and install from [ollama.com](https://ollama.com/). Make sure the Ollama service can be started (either via the Desktop app or `ollama serve` in a terminal).
2.  **Get ArtAgents:** Clone or download the ArtAgents repository from GitHub.
3.  **Setup Environment:** Navigate to the `ArtAgent` directory. It's highly recommended to use a Python virtual environment (Python 3.9+). Use the provided setup scripts (e.g., `.\scripts\setupvenv.bat` on Windows) or manually create an environment and install dependencies using `pip install -r requirements.txt`.
4.  **Download Models:** Run the setup script (e.g., `.\scripts\setup.bat` on Windows) or use `ollama pull <model_name>` to download the LLMs you want to use (see `models.json` for suggestions like `llama3`, `llava`). At least one text model and one vision model (like LLaVA) are recommended.
5.  **Run ArtAgents:** Ensure Ollama is running. Activate your virtual environment (if used) and run `python app.py` or use the provided `go*.bat`/`.sh` scripts.
6.  **Access UI:** Open the local URL shown in your terminal (usually `http://127.0.0.1:7860`) in your web browser.

*(Refer to the main `README.md` for more detailed installation options.)*

## 3. Core Concepts

*   **Agents:** Think of these as specialized AI assistants. Each agent has a unique "role" defined by specific instructions (its persona, task, expertise - see `agent_roles.json`, `custom_agent_roles.json`, or load from file). You select an agent to perform a task based on your input.
*   **Teams (Workflows):** These are sequences of agents defined in `agent_teams.json`. When you run a team:
    *   Each agent runs **sequentially**.
    *   The output of one agent becomes part of the **context** for the next agent.
    *   The final output depends on the team's **Assembly Strategy** (e.g., joining all outputs, using only the last one, summarizing).
    *   Teams allow for more complex, multi-step generation processes.

## 4. Using the Application Tabs

### 4.1. Chat Tab (Main Interaction)

This is where you primarily interact with agents and teams.

1.  **Select Agent/Team:** Choose either a single Agent (e.g., `Designer`, `Artist`) or a predefined workflow (`[Team] Detailed Object Design`) from the dropdown. `(Direct Agent Call)` ignores this selection.
2.  **Select Model:** Choose the Ollama LLM to process your request. Models marked `(VISION)` can process images.
3.  **Provide Input:**
    *   **User Input:** Type your main instructions or base prompt.
    *   **Image Input (Optional):** Upload a single image OR provide a folder path. The selected model must be a vision model to process images.
4.  **Configure (Optional - Advanced Accordion):**
    *   **Prompt Style Limiter:** Apply constraints for specific output styles (e.g., SDXL tags).
    *   **Max Tokens:** Adjust the approximate maximum length of the response.
    *   **Use Advanced Ollama API Options:** Toggle using the detailed parameters set in the "App Settings" tab.
    *   **Clean Prompt Artifacts:** Check this to remove headers like `--- Output from Agent... ---` from the final response (useful if the raw output is needed for other tools).
    *   **Load Agents from File:** Temporarily load agent definitions from a local `.json` file for the current session.
5.  **Generate:** Click **"✨ Generate Response"**.
6.  **Refine (Optional):** After a response appears, type follow-up instructions in the "Enter Comment / Refinement" box and click **"💬 Comment/Refine"**. This uses the **currently selected model** in the dropdown to modify the text shown in the response box based on your comment.
7.  **Copy:** Click the **"📋 Copy"** button next to the response header to copy the response text to your clipboard.
8.  **History:** The "Session History" panel shows interactions from the current session. "Full History" tab shows persistent logs.

### 4.2. Image Captions Tab

View, edit, and generate captions for images in a folder.

1.  **Load Folder:** Enter the path to a folder containing images and click "Load Folder". Supported images will appear in the gallery. Existing `.txt` files with the same base name as images are loaded as captions.
2.  **Select Image:** Click an image in the gallery to view/edit its caption in the text box on the right.
3.  **Edit/Save:** Manually edit the caption in the text box and click "Save THIS Caption" to save changes to the corresponding `.txt` file.
4.  **Generate Caption:**
    *   Select a **Vision Model** and an **Agent/Team** suitable for captioning (e.g., `llava`, `Bespoke Captioner`).
    *   Choose the **File Handling** mode (Overwrite, Skip, etc.).
    *   Click **"Generate Caption for SELECTED Image"** to generate and save a caption for the image currently selected in the gallery.
    *   Click **"Generate Captions for ALL Loaded Images"** to process all images in the loaded folder.

### 4.3. Agent Team Editor Tab

Create, view, modify, and delete multi-agent workflows (Teams).

1.  **Load/Select:** Choose an existing team from the "Select Team" dropdown and click "Load Selected Team" to view its details and steps.
2.  **Create New:** Click "Clear Editor / New Team".
3.  **Edit Details:** Modify the "Team Name", "Team Description", and select the desired "Final Output Strategy" (`concatenate`, `refine_last`, `summarize_all`, `structured_concatenate`).
4.  **Edit Steps:**
    *   **Add:** Select an agent from the "Select Agent Role" dropdown and click "Add Selected Agent as Step". Steps execute in the order listed.
    *   **Remove:** Enter the step number (starting from 1) in "Step Number to Remove" and click "Remove Step #".
5.  **Save:** Click "Save Current Team Definition". The saved team will be available in the dropdowns on the Chat and Captions tabs.
6.  **Delete:** Select a team from the dropdown and click "Delete Selected Team".

### 4.4. Experiment Sweep Tab

Run systematic experiments comparing different prompts, teams, and models.

1.  **Enter Prompts:** Add one base prompt per line in the "Base User Prompt(s)" box.
2.  **Select Teams/Models:** Check the Agent Teams and Worker Models you want to test from the checkbox groups.
3.  **Configure Output:** Provide an "Output Subfolder Name". Check "Log Intermediate Agent Steps?" if you want detailed step outputs in the JSON protocol.
4.  **Run:** Click "🚀 Start Sweep Run".
5.  **Results:** Check the `sweep_runs/[TIMESTAMP]_[YourFolderName]` directory for:
    *   A `.json` protocol file for *each combination* run, containing configuration, intermediate steps (if logged), and the final output.
    *   A separate `.txt` file for *each model* tested (e.g., `prompts_llama3-latest.txt`), containing all the final generated prompts from successful runs using that model, one prompt per line (cleaned for direct use).

### 4.5. Info Tab

View application information and agent definitions.

*   The main page shows basic app info and version.
*   Use the sub-tabs ("Default Roles", "Custom Roles") to see the instructions and settings defined for the agents loaded by the application (read-only view).

### 4.6. Full History Tab

View the persistent log (`core/history.json`) of all interactions (chat, comments, workflow steps, errors) across sessions. Use the "Clear Full History File..." button to permanently delete the log.

### 4.7. App Settings Tab

Configure global application behavior and default Ollama API parameters.

*   Set the **Ollama URL**.
*   Configure **Agent Loading** preferences (use default/custom roles).
*   Set default UI states (e.g., enable advanced API options by default).
*   Load **Ollama API Profiles** (presets for Temperature, Top K, etc.).
*   Manually edit **Default Ollama API Options**.
*   Select the **UI Theme**.
*   **Save Settings** to make changes persistent (`settings.json`).
*   Use **"Release All Ollama Models Now"** to attempt unloading models from VRAM.

## 5. Tips for Effective Use

*   **Experiment!** Try different Agents, Models, Teams, and Ollama API settings (via Profiles or App Settings) to see how they affect the output.
*   **Check Context:** Remember that in Teams, the context builds sequentially. The prompt for later agents includes the output from earlier ones.
*   **Vision Models:** Use `(VISION)` models when providing image input.
*   **Debug:** Check the console where you launched `app.py` for logs and error messages. The "Full History" tab also logs detailed workflow steps and errors.
*   **Custom Agents/Teams:** Define your own agents in `custom_agent_roles.json` and build custom workflows in the Team Editor for tailored results.

---

*ArtAgents is currently a prototype. Use creatively, but no guarantees are provided.*
