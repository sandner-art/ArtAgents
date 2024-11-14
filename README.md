# ArtAgent: Agent-Based Chat with Ollama
Framework for LLM based captioning and prompt engineering experiments

Overview: Select an agent, model, and provide input to get a response from Ollama. You can provide a folder path of images for multimodal input. Create prompts and combine the text with visuals for captioning or image generation.

## Installation
1. Install Ollama https://ollama.com/
2. Install LLM Models
Run locally in cmd terminal with ```olama pull model-name```, ```olama run model-name```
3. Download ArtAgent

## Setup
1. Run ollama in terminal, use ```olama list``` to see full names of models you have locally (you need to enter the full name as displayed in Ollama!)
2. Enter your models into models.json. Set if the model has vision by using "vision": true/false
3. Run ArtAgent

## Example
1. ```olama pull llava```
2. ```olama list``` shows ollama:latest name. The model has VISION, so put it into models.json (this model is already there actually).
3. The setting should work with http://localhost:11434/api/generate (the setting is in agent.py), ArtAgent runs via API
4. Select an Agent depending on what scene you want to focus on
5. Have fun

## Notes
- If there is no folder path inserted, the prompt is created by User and Agents without any image

## sandner.art | [AI/ML Articles](https://sandner.art/)
