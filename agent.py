import json

def load_roles(role_file):
    with open(role_file, 'r') as file:
        return json.load(file)

def get_llm_response(role, user_input, images=None):
    # Simulate LLM response based on role and user input
    roles = load_roles('agent_roles.json')
    role_description = roles.get(role, "Unknown Role")
    
    if images:
        # Simulate multimodal LLM response
        image_captions = [f"Caption for image {i+1}: This is a description of the uploaded image {i+1}." for i in range(len(images))]
        response = f"Role: {role}\nDescription: {role_description}\nUser Input: {user_input}\nImage Captions: {', '.join(image_captions)}"
    else:
        response = f"Role: {role}\nDescription: {role_description}\nUser Input: {user_input}"
    
    return response