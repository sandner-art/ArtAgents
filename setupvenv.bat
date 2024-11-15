@echo off
echo Setting up the environment...

:: Create a virtual environment (optional but recommended)
python -m venv venv

:: Activate the virtual environment
call venv\Scripts\activate

:: Install the required packages
pip install -r requirements.txt

echo Environment setup complete!
pause