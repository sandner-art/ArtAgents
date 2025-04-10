@echo off
echo Starting ArtAgents using virtual environment...
echo IMPORTANT: Ensure the Ollama service is running separately!
echo.

:: Check if venv exists in parent directory
if not exist ..\venv\Scripts\activate.bat (
    echo ERROR: Virtual environment 'venv' not found or incomplete in the parent directory.
    echo Please run 'setupvenv.bat' first.
    pause
    exit /b
)

:: Activate the virtual environment
echo Activating virtual environment...
call ..\venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment.
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
set PYTHONPATH=%~dp0..;%PYTHONPATH%
echo Running app.py from project root...
cd ..
python app.py
cd scripts


:: Deactivate might not run if app crashes hard, but good practice
echo Deactivating virtual environment...
call ..\venv\Scripts\deactivate.bat

:: Pause allows user to see output/errors before window closes
echo.
echo ArtAgents has finished or encountered an error.
pause