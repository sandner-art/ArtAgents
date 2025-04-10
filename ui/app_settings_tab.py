# ArtAgent/ui/app_settings_tab.py
import gradio as gr
from core.utils import load_json, AVAILABLE_THEMES # Import theme map

OLLAMA_PROFILES_FILE = 'ollama_profiles.json'

def create_app_settings_tab(initial_settings):
    """Creates the Gradio components for the App Settings Tab."""

    profile_data = load_json(OLLAMA_PROFILES_FILE, is_relative=True)
    profile_names = list(profile_data.keys())
    theme_names = list(AVAILABLE_THEMES.keys())

    with gr.Tab("App Settings"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### General Application Settings")
                with gr.Group():
                    settings_ollama_url = gr.Textbox(
                        label="Ollama URL (e.g., http://host:port/api/generate)",
                        value=initial_settings.get("ollama_url", "")
                    )
                    settings_max_tokens_slider_range = gr.Slider(
                        label="Max Tokens Slider Range (Chat Tab)", minimum=512, maximum=16384, step=256,
                        value=initial_settings.get("max_tokens_slider", 4096)
                    )
                    settings_api_to_console = gr.Checkbox(
                        label="Log API Request Details to Console",
                        value=initial_settings.get("ollama_api_prompt_to_console", True)
                    )

                gr.Markdown("### Agent Loading")
                with gr.Group():
                     settings_use_default = gr.Checkbox(
                         label="Load Default Agents (agents/agent_roles.json)",
                         value=initial_settings.get("using_default_agents", True)
                     )
                     settings_use_custom = gr.Checkbox(
                         label="Load Custom Agents (agents/custom_agent_roles.json)",
                         value=initial_settings.get("using_custom_agents", False)
                     )

                gr.Markdown("### Default UI States")
                with gr.Group():
                     settings_use_ollama_opts_default = gr.Checkbox(
                         label="Enable 'Use Advanced Ollama API Options' by Default",
                         value=initial_settings.get("use_ollama_api_options", False)
                     )
                     settings_release_model_default = gr.Checkbox(
                         label="Enable 'Unload Previous Model' by Default",
                         value=initial_settings.get("release_model_on_change", False)
                     )

                gr.Markdown("### Appearance")
                with gr.Group():
                    settings_theme_select = gr.Dropdown(
                         theme_names, label="UI Theme (Requires App Restart)",
                         value=initial_settings.get("gradio_theme", "Default")
                    )

                gr.Markdown("### Ollama Actions")
                with gr.Group():
                    release_models_button = gr.Button(
                         "Release All Ollama Models Now", variant="stop",
                         info="Attempts to unload models listed in models.json."
                    )
                    release_status_display = gr.Textbox(label="Release Status", interactive=False, lines=2)


            with gr.Column(scale=1):
                gr.Markdown("### Default Ollama API Options")
                gr.Markdown("Configure default parameters sent via the API 'options'.")
                 # --- Profile Loading ---
                with gr.Group():
                    with gr.Row():
                        profile_select = gr.Dropdown(profile_names, label="Load API Profile Preset", scale=3)
                        load_profile_button = gr.Button("Load Profile", scale=1)
                # --- Dynamic components for Ollama options ---
                ollama_options_ui_elements = {} # Store UI elements keyed by option name
                initial_ollama_options = initial_settings.get("ollama_api_options", {})
                with gr.Group():
                    # Sort for consistent order
                    for key in sorted(initial_ollama_options.keys()):
                        value = initial_ollama_options[key]
                        comp_args = {"label": key, "value": value, "interactive": True}
                        if isinstance(value, bool):
                            comp = gr.Checkbox(**comp_args)
                        elif key == "seed" and isinstance(value, int):
                             # Allow -1 for seed
                             comp = gr.Number(**comp_args, minimum=-1, precision=0)
                        elif isinstance(value, int):
                            max_val = 32768 if "ctx" in key else (100 if key == "top_k" else (128 if "repeat" in key else 10000))
                            min_val = 0 if key not in ["repeat_last_n", "seed"] else -1
                            step = 1024 if "ctx" in key else (512 if "batch" in key else 1)
                            comp = gr.Slider(minimum=min_val, maximum=max_val, step=step, **comp_args)
                        elif isinstance(value, float):
                            max_f = 2.0 if key == "temperature" or "penalty" in key else 1.0 # Default max 1.0
                            min_f = 0.0
                            step_f = 0.01 if max_f == 1.0 else 0.05
                            comp = gr.Slider(minimum=min_f, maximum=max_f, step=step_f, **comp_args)
                        else:
                            comp = gr.Textbox(**comp_args)
                        ollama_options_ui_elements[key] = comp # Store component

        settings_save_button = gr.Button("Save All App Settings", variant="primary")
        save_status_display = gr.Textbox(label="Save Status", interactive=False)


    # Return components needed for wiring
    return {
        # General Settings
        "settings_ollama_url": settings_ollama_url,
        "settings_max_tokens_slider_range": settings_max_tokens_slider_range,
        "settings_api_to_console": settings_api_to_console,
        "settings_use_default": settings_use_default,
        "settings_use_custom": settings_use_custom,
        "settings_use_ollama_opts_default": settings_use_ollama_opts_default,
        "settings_release_model_default": settings_release_model_default,
        "settings_theme_select": settings_theme_select,
        # Ollama Actions
        "release_models_button": release_models_button,
        "release_status_display": release_status_display,
        # API Options & Profiles
        "profile_select": profile_select,
        "load_profile_button": load_profile_button,
        "ollama_options_ui_elements": ollama_options_ui_elements, # Dict of components
        # Save Action
        "settings_save_button": settings_save_button,
        "save_status_display": save_status_display,
        # Data needed for callbacks
        "profile_data": profile_data,
    }