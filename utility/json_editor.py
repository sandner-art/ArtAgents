import gradio as gr
import json

def load_json(file_path):
    try:
        with open(file_path.name, 'r') as file:
            data = json.load(file)
        flattened_data = flatten_json(data)
        return data, "File loaded successfully", list(flattened_data.keys()), ""
    except json.JSONDecodeError:
        return {}, "Error loading file: Invalid JSON format", [], ""
    except Exception as e:
        return {}, f"Error loading file: {str(e)}", [], ""

def update_value(data, selected_key):
    try:
        # Navigate through the JSON structure based on the selected key
        keys = selected_key.split('.')
        value = data
        for key in keys:
            value = value[key]
        return value
    except Exception as e:
        return f"Error updating value: {str(e)}"

def save_json(file_path, data):
    try:
        with open(file_path.name, 'w') as file:
            json.dump(data, file, indent=4)
        return f"File saved successfully: {file_path.name}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

def update_json(data, selected_key, value):
    try:
        keys = selected_key.split('.')
        obj = data
        for key in keys[:-1]:
            obj = obj[key]
        obj[keys[-1]] = value
        return data, f"Updated JSON with key: {selected_key} and value: {value}"
    except Exception as e:
        return data, f"Error updating JSON: {str(e)}"

def flatten_json(y):
    out = {}
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '.')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '.')
                i += 1
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

def unflatten_json(flattened_data):
    unflattened_data = {}
    for key, value in flattened_data.items():
        parts = key.split('.')
        sub_dict = unflattened_data
        for part in parts[:-1]:
            if part not in sub_dict:
                sub_dict[part] = {}
            sub_dict = sub_dict[part]
        sub_dict[parts[-1]] = value
    return unflattened_data

def download_json(data):
    try:
        json_str = json.dumps(data, indent=4)
        with open("edited_file.json", "w") as file:
            file.write(json_str)
        return "edited_file.json"
    except Exception as e:
        return f"Error downloading file: {str(e)}"

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# JSON File Editor")

        with gr.Row():
            file_input = gr.File(label="Upload JSON File", file_types=[".json"])
            file_output = gr.File(label="Download Edited JSON File", interactive=False)

        with gr.Row():
            json_display = gr.JSON(label="JSON Content")

        with gr.Row():
            key_input = gr.Dropdown(label="Key", choices=[], allow_custom_value=True)
            value_input = gr.Textbox(label="Value")

        with gr.Row():
            load_button = gr.Button("Load JSON")
            save_button = gr.Button("Save JSON")
            update_button = gr.Button("Update JSON")
            download_button = gr.Button("Download JSON")

        with gr.Row():
            message = gr.Textbox(label="Message", interactive=False)

        # Load JSON and update UI
        load_button.click(
            fn=lambda file_path: load_json(file_path),
            inputs=file_input,
            outputs=[json_display, message, key_input, value_input]
        )

        # Update value based on selected key
        key_input.change(
            fn=lambda data, selected_key: update_value(data, selected_key),
            inputs=[json_display, key_input],
            outputs=value_input
        )

        # Save JSON
        save_button.click(
            fn=lambda file_path, data: save_json(file_path, data),
            inputs=[file_input, json_display],
            outputs=message
        )

        # Update JSON content
        update_button.click(
            fn=lambda data, selected_key, value: update_json(data, selected_key, value),
            inputs=[json_display, key_input, value_input],
            outputs=[json_display, message]
        )

        # Download JSON
        download_button.click(
            fn=lambda data: (download_json(data), "Downloaded"),
            inputs=json_display,
            outputs=[file_output, message]
        )

    demo.launch(share=True, debug=True)

if __name__ == "__main__":
    main()