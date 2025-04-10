@echo off
echo Starting ArtAgents...
echo IMPORTANT: Ensure the Ollama service is running separately!
echo.

:: Check if python is available
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python does not seem to be installed or added to PATH.
    pause
    exit /b
)

:: Check if app.py exists in parent directory
if not exist ..\app.py (
    echo ERROR: app.py not found in the parent directory.
    pause
    exit /b
)

:: Set the Python path to include the parent directory (project root)
:: This helps Python find modules in core/, agents/ etc. when running from scripts/
set PYTHONPATH=%~dp0..;%PYTHONPATH%
echo Running app.py from project root...
cd ..
python app.py
cd scripts

:: Pause allows user to see output/errors before window closes
echo.
echo ArtAgents has finished or encountered an error.
pause