# ArtAgent/ui/common_ui_elements.py
import gradio as gr

def create_footer():
    """Creates a standard footer markdown element."""
    # Using HTML for slightly better control over spacing and links if needed
    return gr.HTML("""
        <hr>
        <p style='text-align: center; font-size: 0.9em; color: grey;'>
            ArtAgents | <a href='https://sandner.art/' target='_blank'>sandner.art</a> |
            <a href='https://github.com/sandner-art' target='_blank'>Creative AI/ML Research</a>
        </p>
    """)