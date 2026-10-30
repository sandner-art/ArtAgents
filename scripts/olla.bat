@echo off
:: Quick check that Ollama is reachable and which models are installed.
echo Checking Ollama at http://localhost:11434 ...
curl -s http://localhost:11434/api/version
if %errorlevel% neq 0 (
    echo.
    echo Ollama is not reachable. Start the Ollama app or run "ollama serve" in another terminal.
    pause
    exit /b 1
)
echo.
echo.
echo Installed models:
ollama list
echo.
pause
