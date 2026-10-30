@echo off
setlocal
echo Starting ArtAgents (system Python, no virtual environment)...
echo IMPORTANT: Ensure the Ollama service is running separately!
echo.

:: Always work from the project root (the folder above this script)
cd /d "%~dp0.."

:: Check if python is available
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python does not seem to be installed or added to PATH.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo ERROR: app.py not found in "%CD%".
    pause
    exit /b 1
)

echo Running app.py from %CD% ...
python app.py

:: Pause allows user to see output/errors before window closes
echo.
echo ArtAgents has finished or encountered an error.
pause
endlocal
