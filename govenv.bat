@echo off
echo Starting the application...

:: Activate the virtual environment
call venv\Scripts\activate

:: Run the Gradio application
python app.py

:: Deactivate the virtual environment after the application exits
deactivate

pause