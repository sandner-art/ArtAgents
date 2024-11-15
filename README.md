# ArtAgents: Agent-Based Chat with Ollama
Framework for LLM based captioning and prompt engineering experiments

## Overview
Select an agent, a model, and provide input to get a response from Ollama. You can provide a folder path of images for multimodal input. Create prompts and combine the text with visuals for captioning or image generation.

Use either a folder of images (1) or one image (2) to experiment with text or multimodal LLMs. Finetune your text prompt output or caption using agents in dedicated roles and user input (3). Try limiters to adjust the prompts even more. 

![art-agent-screen](https://github.com/user-attachments/assets/b56ba489-c3e3-4a9a-a4b1-6ccb2d89e23d)

## Installation
1. Install Ollama https://ollama.com/
2. Download/clone ArtAgents
3. (optional) Run setupvenv.bat. (you may then run ArtAgents in venv with govenv.bat)
4. (optional) Run setup.bat to setup ollama models

## Manual Setup of Models You Already Use
1. Run ollama in terminal, use ```olama list``` to see full names of models you have locally (you need to enter the full name as displayed in Ollama!)
2. Enter your models into models.json. If the model has vision set "vision": true
3. Run ArtAgents with go.bat or govenv.bat

## Example
1. ```olama pull llava```
2. ```olama list``` shows ollama:latest name. The model has VISION, so put it into models.json (this model is already there actually).
3. The setting should work with http://localhost:11434/api/generate (the setting is in agent.py), ArtAgents runs via API
4. Select an Agent depending on what scene you want to focus on
5. Have fun

## Notes
- If there is no folder path or image inserted, the prompt is created by User and Agents without any image just based on User input 

## sandner.art | [AI/ML Articles](https://sandner.art/)
