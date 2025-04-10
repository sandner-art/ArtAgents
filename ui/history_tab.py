# ArtAgent/ui/history_tab.py
import gradio as gr

def create_history_tab(initial_history_list):
    """Creates the Gradio components for the History Tab."""

    with gr.Tab("Full History"):
        gr.Markdown("### Interaction History (`core/history.json`)")
        full_history_display = gr.Textbox(
            label="Full History Log", lines=30,
            value="\n---\n".join(initial_history_list), interactive=False
        )
        # Confirmation components for clearing history
        with gr.Group(visible=False) as confirm_clear_group:
             # confirm_msg = gr.Markdown("❓ **Are you sure you want to permanently clear the entire history file?**")
             with gr.Row():
                  gr.Markdown("❓ **Clear entire history file permanently?**", scale=3)
                  yes_clear_button = gr.Button("Yes, Clear History", variant="stop", scale=1)
                  no_clear_button = gr.Button("No, Cancel", scale=1)
        clear_history_button = gr.Button("Clear Full History File...")

    return {
        "full_history_display": full_history_display,
        "confirm_clear_group": confirm_clear_group,
        "yes_clear_button": yes_clear_button,
        "no_clear_button": no_clear_button,
        "clear_history_button": clear_history_button,
    }