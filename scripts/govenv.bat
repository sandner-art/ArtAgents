@echo off
setlocal
echo Starting ArtAgents using virtual environment...
echo IMPORTANT: Ensure the Ollama service is running separately!
echo.

:: Always work from the project root (the folder above this script), so this
:: works from any folder: double-click, cmd, or PowerShell (.\scripts\govenv.bat)
cd /d "%~dp0.."

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at "%CD%\venv".
    echo Please run scripts\setupvenv.bat first.
    pause
    exit /b 1
)
if not exist "app.py" (
    echo ERROR: app.py not found in "%CD%".
    pause
    exit /b 1
)

echo Activating virtual environment...
call "venv\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate the virtual environment.
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
