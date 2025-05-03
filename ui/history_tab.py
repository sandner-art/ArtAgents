# ArtAgent/ui/history_tab.py
import gradio as gr
# Import help content functions/data
from core.help_content import get_tooltip

def create_history_tab(initial_history_list):
    """Creates the Gradio components for the History Tab."""

    history_display_text = "\n---\n".join(initial_history_list) if initial_history_list else "History is empty."

    with gr.Tab("Full History"):
        gr.Markdown("### Interaction History (`core/history.json`)")
        gr.Markdown("This log shows all interactions saved across sessions (up to a limit).")
        full_history_display = gr.Textbox(
            label="Full History Log", lines=30,
            value=history_display_text, interactive=False
        )
        # Confirmation components for clearing history
        with gr.Group(visible=False) as confirm_clear_group:
             with gr.Row():
                  gr.Markdown("❓ **Clear entire history file permanently? This cannot be undone.**", scale=3)
                  yes_clear_button = gr.Button("Yes, Clear History", variant="stop", scale=1)
                  no_clear_button = gr.Button("No, Cancel", scale=1)
        clear_history_button = gr.Button(
            "Clear Full History File...",
            info=get_tooltip("clear_history_button") # Add tooltip
        )

    return {
        "full_history_display": full_history_display,
        "confirm_clear_group": confirm_clear_group,
        "yes_clear_button": yes_clear_button,
        "no_clear_button": no_clear_button,
        "clear_history_button": clear_history_button,
    }