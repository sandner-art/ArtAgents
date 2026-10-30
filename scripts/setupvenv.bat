@echo off
setlocal
echo Setting up the Python virtual environment...

:: Always work from the project root (the folder above this script), so the
:: venv lands in the project root, where govenv.bat looks for it, no matter where this is run from.
cd /d "%~dp0.."

:: Check if python is available
python --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python does not seem to be installed or added to PATH.
    echo Please install Python 3.11+ and ensure it's in your PATH.
    pause
    exit /b 1
)

:: Show exactly which Python is used (helps when several are installed)
python -c "import sys, struct; print('Using Python', sys.version.split()[0], str(struct.calcsize('P') * 8) + '-bit', 'at', sys.executable)"

:: Pillow 13 (security fixes) requires Python 3.11 or newer
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if %errorlevel% neq 0 (
    echo ERROR: Python 3.11 or newer is required. Found:
    python --version
    pause
    exit /b 1
)

:: Create a virtual environment named 'venv' in the project root
echo Creating virtual environment 'venv'...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

:: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment. Check if venv was created correctly.
    pause
    exit /b 1
)

:: Check if requirements files exist
if not exist requirements.txt (
   echo ERROR: requirements.txt not found in the project root. Cannot install packages.
   pause
   exit /b 1
)
if not exist requirements-pillow.txt (
   echo ERROR: requirements-pillow.txt not found in the project root. Cannot install packages.
   pause
   exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip

:: Step 1: main dependencies
echo Installing required packages from requirements.txt...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install packages from requirements.txt. Check the file and your internet connection.
    pause
    exit /b 1
)

:: Step 2: patched Pillow (gradio still declares pillow<13, so this is installed on its own)
echo Installing patched Pillow from requirements-pillow.txt...
python -m pip install --no-deps --only-binary=:all: -r requirements-pillow.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Pillow 13 from requirements-pillow.txt.
    echo        Pillow 13 has ready-made Windows packages for Python 3.11 to 3.15, so this is
    echo        usually a network/proxy problem. Check the error above, then run this script again.
    pause
    exit /b 1
)

echo.
echo Environment setup complete!
echo You can now run the application using 'govenv.bat'.
echo To deactivate the environment manually, type 'deactivate'.
pause
endlocal
