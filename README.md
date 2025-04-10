# ArtAgents: Agent-Based Chat with Ollama
Prototype framework for LLM based captioning and prompt engineering experiments for artists and designers

![artagents-github](https://github.com/user-attachments/assets/9350bb3a-9e19-4818-b109-983c5a6b0bb1)

## Overview
Select an agent, a model, and provide input to get a response from any model served by Ollama. You can provide a folder path of images for multimodal input. Create prompts and combine the text with visuals for captioning or image generation (ArtAgents app is producing *prompts* for image or video generation models). 

Use either a folder of images (1) or one image (2) to experiment with text or multimodal LLMs. Fine-tune  your text prompt output on one image or caption folders using agents in dedicated roles and user input (3). Comment (4) on the output to change the result color scheme, materials, or a setting. Try limiters to adjust the prompts even more. 

![artagents-numbers](https://github.com/user-attachments/assets/ea0f8d00-646b-4a73-97ca-e1938b534d2d)

## Installation
1. Install Ollama https://ollama.com/
2. Download/clone ArtAgents
3. (optional) Run setupvenv.bat. (you may then run ArtAgents in venv with **govenv.bat**)
4. (optional) Run setup.bat to setup ollama models

## Manual Setup of Models You Already Use
1. Run ollama in terminal, use ```olama list``` to see full names of models you have locally (you need to enter the full name as displayed in Ollama!)
2. Enter your models into models.json. Set "vision": true for multimodal LLMs
3. Run ArtAgents with **go.bat** or **govenv.bat**

## Example
1. ```olama pull llava```
2. ```olama list``` shows ollama:latest name. The model has VISION, so put it into models.json (this model is already there actually).

## Running the Program
1. The setting should work with http://localhost:11434/api/generate (the setting is in agent.py), ArtAgents runs via API
2. Run ```ollama serve``` or ollama for desktop
3. Run go.bat or govenv.bat (you may run the program also with ```python app.py``` )
4. Select an Agent depending on what scene you want to focus on
5. Select model from the list, enter your user prompt
6. Have fun

## Notes
- If there is no folder path or image inserted, the prompt is created by User and Agents without any image just based on User input. It is meant for quick sketches of prompts and for work with text only LLMs  
- The goal was to create a tool for unusual captioning style experiments and prompt engineering tests for generative models 

## In Development

- [x] Working prototype
- [x] Editable Default Agents
- [ ] Custom Agents Properties
- [ ] Custom Model Limiters and Agents (for a specific LLM)
- [ ] Use Ollama API Options (full functionality)
- [ ] API parameters finetune, profiles
- [x] Comment Input
- [x] Chat History
- [ ] Multimodal Chat History
- [ ] Agent Training 


## sandner.art | [AI/ML Articles](https://sandner.art/)
ArtAgents by Daniel Sandner © 2024-25. You may use the software or adapt it for your creative work any way you choose. No guarantees.
