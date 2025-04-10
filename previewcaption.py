import os
import gradio as gr
from PIL import Image

def load_images_and_texts(folder_path):
    images = []
    texts = []
    filenames = []
    thumbnails = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            text_filename = os.path.splitext(filename)[0] + ".txt"
            text_path = os.path.join(folder_path, text_filename)
            
            if os.path.exists(text_path):
                with open(text_path, 'r', encoding='utf-8') as file:
                    text_content = file.read()
            else:
                text_content = ""
            
            image = Image.open(image_path)
            images.append(image)
            texts.append(text_content)
            filenames.append(filename)
            
            # Create a thumbnail
            thumbnail = image.copy()
            thumbnail.thumbnail((150, 150))  # Resize to 150x150 pixels for better visibility
            thumbnails.append(thumbnail)
    
    return images, texts, filenames, thumbnails

def update_image_text(index, images, texts, filenames):
    if index < 0 or index >= len(images):
        return None, "", ""
    image = images[index]
    text = texts[index]
    filename = filenames[index]
    return image, text, filename

def save_text(filename, text_content, folder_path):
    text_filename = os.path.splitext(filename)[0] + ".txt"
    text_path = os.path.join(folder_path, text_filename)
    with open(text_path, 'w', encoding='utf-8') as file:
        file.write(text_content)

def load_images(folder_path):
    images, texts, filenames, thumbnails = load_images_and_texts(folder_path)
    
    if not images:
        return "No images found in the folder.", "", "", 0, [], [], [], []
    
    initial_image, initial_text, initial_filename = update_image_text(0, images, texts, filenames)
    return thumbnails, initial_image, initial_text, initial_filename, 0, images, texts, filenames

def update_components(index, images, texts, filenames):
    return update_image_text(index, images, texts, filenames)

def save_and_update(text_content, index, folder_path, images, texts, filenames):
    save_text(filenames[index], text_content, folder_path)
    return update_image_text(index, images, texts, filenames)

def launch_app():
    with gr.Blocks() as demo:
        folder_path_input = gr.Textbox(label="Folder Path")
        launch_button = gr.Button("Load Images")
        
        gallery = gr.Gallery(label="Image Gallery", height="auto")
        image_component = gr.Image(type="pil", label="Image Viewer")
        text_component = gr.Textbox(label="Caption", interactive=True)
        filename_component = gr.Textbox(label="Filename", interactive=False)
        image_index = gr.Slider(0, 0, step=1, label="Image Index", visible=False)
        save_button = gr.Button("Save Caption")
        
        def on_gallery_select(evt, images, texts, filenames):
            if evt is None:
                return None, "", "", 0
            index = evt.index
            image, text, filename = update_image_text(index, images, texts, filenames)
            image_index.value = index
            return image, text, filename, index
        
        def update_slider_max(images):
            return len(images) - 1 if images else 0
        
        def on_launch(folder_path):
            thumbnails, initial_image, initial_text, initial_filename, initial_index, images, texts, filenames = load_images(folder_path)
            return thumbnails, initial_image, initial_text, initial_filename, initial_index, images, texts, filenames, len(images) - 1 if images else 0
        
        launch_button.click(
            on_launch, 
            inputs=[folder_path_input], 
            outputs=[gallery, image_component, text_component, filename_component, image_index, gr.State(), gr.State(), gr.State(), image_index]
        )
        
        gallery.select(
            fn=on_gallery_select, 
            inputs=[gallery, gr.State(), gr.State(), gr.State()], 
            outputs=[image_component, text_component, filename_component, image_index]
        )
        
        image_index.change(
            update_components, 
            inputs=[image_index, gr.State(), gr.State(), gr.State()], 
            outputs=[image_component, text_component, filename_component]
        )
        
        save_button.click(
            save_and_update, 
            inputs=[text_component, image_index, folder_path_input, gr.State(), gr.State(), gr.State()], 
            outputs=[image_component, text_component, filename_component]
        )
    
    return demo

demo = launch_app()
demo.launch()