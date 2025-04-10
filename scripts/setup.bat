@echo off
setlocal

echo.
echo ==============================================================================
echo =                                                                            =
echo =               Setup ArtAgents: Agent-Based Chat with Ollama                =
echo =                                                                            =
echo ==============================================================================
echo.
pause

:: Function to check if ollama is installed
:check_ollama
where ollama >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo ==================================================================================
    echo =                                                                                =
    echo =  Ollama is not installed or not in PATH.                                       =
    echo =  Please download and install it from https://ollama.com/download               =
    echo =                                                                                =
    echo ==================================================================================
    echo.
    pause
    exit /b
) else (
    echo Ollama found.
)

:: Run ollama list to check installed models
echo.
echo ==============================================================================
echo =                     Checking locally installed models...                   =
echo ==============================================================================
echo.
ollama list

:: Ask user if they want to pull models (referencing models.json implicitly)
echo.
echo ==============================================================================
echo =                                                                            =
echo =  Models needed are listed in models.json (llava, llama3 recommended).      =
echo =  Each model can be several GBs.                                            =
echo =                                                                            =
echo =  Do you want to attempt pulling recommended models now?                    =
echo =  1. Pull 'llava:latest' (Vision Model)                                     =
echo =  2. Pull 'llama3:latest' (Text Model)                                      =
echo =  3. Pull 'impactframes/ifai_sd_prompt_mkr_q4km:latest' (Text Model)        =
echo =  4. Pull ALL recommended above                                             =
echo =  5. Pull NONE (you will manage models manually)                            =
echo =                                                                            =
echo ==============================================================================
echo.
set /p choice="Enter your choice (1-5): "

:: Pull models based on user choice
if "%choice%"=="1" (
    echo Pulling llava:latest...
    ollama pull llava:latest
) else if "%choice%"=="2" (
    echo Pulling llama3:latest...
    ollama pull llama3:latest
) else if "%choice%"=="3" (
    echo Pulling impactframes/ifai_sd_prompt_mkr_q4km:latest...
    ollama pull impactframes/ifai_sd_prompt_mkr_q4km:latest
) else if "%choice%"=="4" (
    echo Pulling llava:latest...
    ollama pull llava:latest
    echo Pulling llama3:latest...
    ollama pull llama3:latest
    echo Pulling impactframes/ifai_sd_prompt_mkr_q4km:latest...
    ollama pull impactframes/ifai_sd_prompt_mkr_q4km:latest
) else if "%choice%"=="5" (
    echo No models will be pulled automatically.
) else (
    echo Invalid choice. No models pulled.
)

:: Removed automatic run option - user should run go.bat/govenv.bat separately
echo.
echo ==============================================================================
echo =  Setup regarding models is complete.                                       =
echo =  Ensure Ollama service is running before starting the app.                 =
echo =  Use 'go.bat' or 'govenv.bat' (if using venv) to run ArtAgents.            =
echo ==============================================================================
echo.
pause
endlocal