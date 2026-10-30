# ArtAgent/ui/app_settings_tab.py
import gradio as gr
from core.utils import load_json, AVAILABLE_THEMES # Import theme map
from core.ollama_client import base_url_from_settings, VALID_OPTIONS
from core.image_gen_settings import image_gen_settings
# Import help content functions/data
from core.help_content import get_tooltip, get_markdown

OLLAMA_PROFILES_FILE = 'ollama_profiles.json' # Relative to root

def create_app_settings_tab(initial_settings):
    """Creates the Gradio components for the App Settings Tab."""

    # Load profile names for dropdown
    # Use load_json from core.utils via its own definition for robustness
    profile_data = load_json(OLLAMA_PROFILES_FILE, is_relative=True)
    profile_names = list(profile_data.keys()) if isinstance(profile_data, dict) else []

    theme_names = list(AVAILABLE_THEMES.keys())

    with gr.Tab("App Settings"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### General Application Settings")
                with gr.Group():
                    settings_ollama_url = gr.Textbox(
                        label="Ollama Server URL",
                        value=base_url_from_settings(initial_settings),
                        placeholder="http://localhost:11434"
                    )
                    settings_think_mode = gr.Radio(
                        ["off", "auto", "on"], value=initial_settings.get("ollama_think", "off"),
                        label="Reasoning ('thinking') models",
                        info="off = answer directly (fastest, best for prompts); auto = model default; on = always reason first. Reasoning text is never put into the answer."
                    )
                    with gr.Row():
                        settings_refresh_models_button = gr.Button("🔄 Test Connection & Refresh Model List")
                    settings_models_status = gr.Markdown("")
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
                         theme_names, label="UI Theme",
                         value=initial_settings.get("gradio_theme", "Default")
                    )

                gr.Markdown("### Image Generation Servers")
                img_settings = image_gen_settings(initial_settings)
                with gr.Group():
                    settings_comfyui_url = gr.Textbox(label="ComfyUI URL", value=img_settings["comfyui_url"])
                    settings_sdwebui_url = gr.Textbox(
                        label="Forge / A1111 / SD.Next URL", value=img_settings["sdwebui_url"],
                        info="Start the WebUI with --api. If it uses port 7860, ArtAgents moves to the next free port."
                    )
                    settings_ollama_images_url = gr.Textbox(label="Ollama (image models) URL", value=img_settings["ollama_images_url"])
                    settings_image_output_dir = gr.Textbox(
                        label="Image Output Folder", value=img_settings["output_dir"],
                        info="Relative paths are inside the ArtAgents folder. Images are saved into a dated subfolder."
                    )
                    settings_save_prompt_txt = gr.Checkbox(
                        label="Save the prompt as a .txt next to each generated image (dataset-ready)",
                        value=img_settings["save_prompt_txt"]
                    )

                gr.Markdown("### Ollama Actions")
                with gr.Group():
                    release_models_button = gr.Button(
                         "Release All Ollama Models Now", variant="stop"
                    )
                    release_status_display = gr.Textbox(label="Release Status", interactive=False, lines=2)


            with gr.Column(scale=1):
                gr.Markdown("### Default Ollama API Options")
                # Add markdown overview help
                gr.Markdown(get_markdown("api_options_overview"))
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
                    for key in sorted(k for k in initial_ollama_options.keys() if k in VALID_OPTIONS):
                        value = initial_ollama_options[key]
                        comp_args = {
                            "label": key,
                            "value": value,
                            "interactive": True,
                        }
                        # Determine component type based on value type or key name
                        if isinstance(value, bool):
                            comp = gr.Checkbox(**comp_args)
                        elif key == "seed" and isinstance(value, int):
                             comp = gr.Number(minimum=-1, precision=0, **comp_args) # Allow -1 for seed
                        elif isinstance(value, int):
                            max_val = 32768 if "ctx" in key else (256 if "repeat_last_n" in key else (100 if key == "top_k" else 10000))
                            min_val = -1 if key in ["repeat_last_n", "seed", "num_predict"] else 0 # Allow -1 where appropriate
                            step = 1024 if "ctx" in key else (512 if "batch" in key else 1)
                            comp = gr.Slider(minimum=min_val, maximum=max_val, step=step, **comp_args)
                        elif isinstance(value, float):
                            max_f = 2.0 if key == "temperature" or "penalty" in key else (1.0 if "p" in key or key == "tfs_z" else 10.0) # Adjust ranges
                            min_f = 0.0
                            step_f = 0.01 if max_f <= 1.0 else 0.05
                            comp = gr.Slider(minimum=min_f, maximum=max_f, step=step_f, **comp_args)
                        else:
                            # Fallback for strings (e.g., stop sequences) or other types
                            comp = gr.Textbox(**comp_args)
                        ollama_options_ui_elements[key] = comp # Store component

        settings_save_button = gr.Button("Save All App Settings", variant="primary")
        save_status_display = gr.Textbox(label="Save Status", interactive=False)


    # Return dictionary of components needed by app.py for wiring events
    return {
        # General Settings
        "settings_ollama_url": settings_ollama_url,
        "settings_think_mode": settings_think_mode,
        "settings_refresh_models_button": settings_refresh_models_button,
        "settings_models_status": settings_models_status,
        "settings_comfyui_url": settings_comfyui_url,
        "settings_sdwebui_url": settings_sdwebui_url,
        "settings_ollama_images_url": settings_ollama_images_url,
        "settings_image_output_dir": settings_image_output_dir,
        "settings_save_prompt_txt": settings_save_prompt_txt,
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
        "ollama_options_ui_elements": ollama_options_ui_elements, # Dict of components keyed by name
        # Save Action
        "settings_save_button": settings_save_button,
        "save_status_display": save_status_display,
    }