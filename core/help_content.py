# ArtAgent/core/help_content.py
import json # Needed only if formatting complex help

# Using dictionaries keyed by logical names or UI element identifiers

TOOLTIPS = {
    # === Chat Tab ===
    "folder_path": "Optional: Full path to a folder with images (.png, .jpg). Processed image-by-image.",
    "single_image": "Optional: Upload one image. Overrides folder path if provided.",
    "file_handling": "Action for existing .txt files when processing a folder.",
    "role_dropdown": "Select the AI agent's persona or expertise.",
    "model_dropdown": "Choose the Ollama model. '(VISION)' = handles images.",
    "user_input": "Your main instructions or prompt content for the agent.",
    "limiter": "Apply stylistic constraints & token limits for specific models (e.g., SDXL). 'Off' = no constraints.",
    "max_tokens": "Approximate max response length (tokens). Limiters may override this.",
    "use_advanced_options": "Apply the detailed Ollama settings from the 'App Settings' tab.",
    "release_model": "Unload previous model from VRAM when switching. Frees memory but slows switching.",
    "agent_file_upload": "Load agent definitions from a local .json file for this session.",
    "comment_input": "Provide follow-up instructions to refine the last generated response.",
    "submit_button": "Generate response based on current inputs.",
    "comment_button": "Refine the last response based on your comment.",
    "clear_session_button": "Clear the history log displayed for the current session.",

    # === App Settings Tab ===
    "ollama_url": "Full URL for Ollama's generate API (e.g., http://localhost:11434/api/generate).",
    "max_tokens_range": "Sets the upper limit for the 'Max Tokens' slider on the Chat tab.",
    "log_api": "Print request details (model, options, prompt start) to the launch console.",
    "load_default_agents": "Include standard agents from agents/agent_roles.json.",
    "load_custom_agents": "Include custom agents from agents/custom_agent_roles.json (overrides defaults).",
    "profile_load": "Select a preset collection of Ollama API options below.",
    "load_profile_button": "Apply the selected profile's values to the options below.",
    "theme_select": "Change the UI appearance. Requires app restart.",
    "release_all_button": "Attempt to unload all models in models.json from Ollama memory.",
    "save_settings_button": "Save all settings on this tab to settings.json.",

    # === Default Ollama API Options (Concise Help) ===
    # --- Sampling ---
    "opt_temperature": "Randomness (0=deterministic, >1=very random). Higher=creative/varied, Lower=focused/safer. Affects quality/coherence.",
    "opt_top_k": "Consider only top 'k' likely words (0=disable). Lower=safer/less diverse, Higher=more diverse/risky. Affects quality/speed slightly.",
    "opt_top_p": "Cumulative probability threshold (0-1). Considers words until probability sum >= 'p'. Lower=safer/less diverse. Affects quality/speed slightly.",
    "opt_min_p": "Minimum probability threshold (0-1). Discards words below this probability, applied after other sampling. Higher = more filtering/safer.",
    "opt_tfs_z": "Tail Free Sampling 'z' (1=disable). Cuts off low-probability tail. Higher=more filtering/less diversity. Affects quality.",
    "opt_typical_p": "Locally Typical Sampling 'p' (1=disable). Favors words typical in context. Lower=more 'typical'/less surprising. Affects quality.",
    "opt_mirostat": "Enable Mirostat sampling (0=disable, 1=v1, 2=v2). Aims for target perplexity/surprise. Can improve coherence but may affect speed.",
    "opt_mirostat_tau": "Mirostat 'tau' (target surprise). Controls coherence vs diversity trade-off. Used if mirostat > 0.",
    "opt_mirostat_eta": "Mirostat 'eta' (learning rate). Controls how quickly sampler adapts. Used if mirostat > 0.",
    # --- Prediction / Penalties ---
    "opt_num_predict": "Max number of tokens to generate (-1=infinite, -2=fill context). Directly affects response length and generation time (speed).",
    "opt_repeat_last_n": "Look back 'n' tokens for penalty checks (0=disable, -1=context size). Affects memory/speed slightly.",
    "opt_repeat_penalty": "Penalty for repeating sequences (>1 discourages). Higher=less repetition/potentially less natural. Affects quality.",
    "opt_presence_penalty": "Penalty for repeating *any* token (>0 discourages). Affects quality.",
    "opt_frequency_penalty": "Penalty based on token frequency (>0 discourages frequent tokens). Affects quality.",
    "opt_penalize_newline": "Penalize generation of newline characters. Useful for single-line outputs.",
    # --- Model / Context ---
    "opt_seed": "Random seed for reproducibility (int). Same seed + same params = same output.",
    "opt_num_ctx": "Context window size (tokens) model considers. Max depends on model. Larger=more context/memory use, potentially slower.",
    # --- Resource Management (Advanced) ---
    "opt_num_keep": "Number of tokens from prompt to always keep in context (-1=all).",
    "opt_num_batch": "Batch size for prompt processing. Adjust based on VRAM/model. May affect speed.",
    "opt_num_gpu": "Number of GPU layers (-1 = auto). Requires Ollama built with GPU support. Affects speed/VRAM.",
    "opt_main_gpu": "Index of primary GPU if multiple used.",
    "opt_low_vram": "Optimize for low VRAM GPUs. May reduce speed or context.",
    "opt_f16_kv": "Use 16-bit floats for KV cache. Saves memory, might affect precision slightly.",
    "opt_numa": "Attempt NUMA optimizations (requires specific hardware/OS setup).",
    "opt_vocab_only": "Load only model vocabulary (for testing).",
    "opt_use_mmap": "Use memory mapping for model loading. Usually recommended.",
    "opt_use_mlock": "Force model to stay in RAM (requires permissions). Prevents swapping but uses more RAM.",
    "opt_num_thread": "Number of CPU threads for prompt processing/generation. Adjust based on CPU cores. Affects CPU usage/speed.",

    # === History Tab ===
    "clear_history_button": "Permanently delete all entries from the persistent history file (core/history.json).",

}

# Could also store longer markdown content if needed later
MARKDOWN_HELP = {
    "api_options_overview": """
### Understanding Ollama API Options
These parameters fine-tune how the language model generates text. Experiment to find optimal settings for your models and tasks. Key areas:
*   **Sampling (`temperature`, `top_k`, `top_p`, `mirostat`, etc.):** Control creativity vs. predictability.
*   **Prediction (`num_predict`, Penalties):** Control response length and prevent repetition.
*   **Context (`num_ctx`):** How much past text the model remembers (limited by model & memory).
*   **Resource Mgmt (`num_gpu`, `num_thread`, etc.):** Advanced hardware utilization settings.

*Refer to the [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion) for full technical details.*
""",
    "agent_file_format": """
### Agent File Format (`.json`)
Structure must be a dictionary where keys are agent names and values are dictionaries containing:
1.  `"description"`: (Required) String explaining the agent's role.
2.  `"ollama_api_options"`: (Optional) Dictionary specifying API parameters to override defaults for this agent.
```json
{
  "MyAgentName": {
    "description": "Agent does X.",
    "ollama_api_options": {
      "temperature": 0.9,
      "num_predict": 800
    }
  },
  "AnotherAgent": {
    "description": "Agent does Y."
  }
}
```
"""
}

def get_tooltip(key: str) -> str:
    """Helper function to retrieve tooltip text."""
    return TOOLTIPS.get(key, "") # Return empty string if key not found

def get_markdown(key: str) -> str:
    """Helper function to retrieve markdown help text."""
    return MARKDOWN_HELP.get(key, "")

# Simple test
if __name__ == "__main__":
    print("Testing help content access:")
    print(f"Tooltip for 'role_dropdown': {get_tooltip('role_dropdown')}")
    print(f"Tooltip for 'opt_temperature': {get_tooltip('opt_temperature')}")
    print(f"Markdown for 'api_options_overview':\n{get_markdown('api_options_overview')}")
    print(f"Tooltip for non-existent key: '{get_tooltip('invalid_key')}'")

