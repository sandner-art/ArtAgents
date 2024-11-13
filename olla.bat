@echo off
curl -X POST http://localhost:11434/api/generate ^
     -H "Content-Type: application/json" ^
     -d "{\"model\": \"impactframes/llama3_ifai_sd_prompt_mkr_q4km:latest\", \"prompt\": \"What is the day before saturay?\", \"max_tokens\": 200}"