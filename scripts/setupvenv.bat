@echo off
echo Setting up the Python virtual environment...

:: Check if python is available
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python does not seem to be installed or added to PATH.
    echo Please install Python 3.8+ and ensure it's in your PATH.
    pause
    exit /b
)

:: Create a virtual environment named 'venv' in the project root
echo Creating virtual environment 'venv'...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b
)

:: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment. Check if venv was created correctly.
    pause
    exit /b
)


:: Check if requirements.txt exists
if not exist ..\requirements.txt (
   echo ERROR: requirements.txt not found in the parent directory. Cannot install packages.
   pause
   exit /b
)

:: Install the required packages from requirements.txt (in parent dir)
echo Installing required packages from requirements.txt...
pip install -r ..\requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install packages from requirements.txt. Check the file and your internet connection.
    pause
    exit /b
)

echo.
echo Environment setup complete!
echo You can now run the application using 'govenv.bat'.
echo To deactivate the environment manually, type 'deactivate'.
pause