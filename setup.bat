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
    echo =  Ollama is not installed. Please download it from https://ollama.com/download  =
    echo =                                                                                =
    echo ==================================================================================
    echo.
    pause
    exit /b
)

:: Run ollama list to check installed models
echo.
echo ==============================================================================
echo =                                                                            =
echo =                     Checking installed models...                           =
echo =                                                                            =
echo ==============================================================================
echo.
ollama list

:: Ask user if they want to pull models
echo.
echo ==============================================================================
echo =                                                                            =
echo =  Each model is around 5GB.                                                 =
echo =                                                                            =
echo =  Do you want to pull:                                                      =
echo =  1. llava VISION model                                                     =
echo =  2. impactframes/llama3_ifai_sd_prompt_mkr_q4km:latest TEXT model          =
echo =  3. Both                                                                   =
echo =  4. None                                                                   =
echo =                                                                            =
echo ==============================================================================
echo.
set /p choice=Enter your choice (1, 2, 3, or 4):

:: Pull models based on user choice
if "%choice%"=="1" (
    echo.
    echo ==============================================================================
    echo =                                                                            =
    echo =  Pulling llava VISION model...                                             =
    echo =                                                                            =
    echo ==============================================================================
    echo.
    ollama pull llava
) else if "%choice%"=="2" (
    echo.
    echo ==============================================================================
    echo =                                                                            =
    echo =  Pulling impactframes/llama3_ifai_sd_prompt_mkr_q4km:latest TEXT model...  =
    echo =                                                                            =
    echo ==============================================================================
    echo.
    ollama pull impactframes/llama3_ifai_sd_prompt_mkr_q4km:latest
) else if "%choice%"=="3" (
    echo.
    echo ==============================================================================
    echo =                                                                            =
    echo =  Pulling llava VISION model...                                             =
    echo =                                                                            =
    echo ==============================================================================
    echo.
    ollama pull llava
    echo.
    echo ==============================================================================
    echo =                                                                            =
    echo =  Pulling impactframes/llama3_ifai_sd_prompt_mkr_q4km:latest TEXT model...  =
    echo =                                                                            =
    echo ==============================================================================
    echo.
    ollama pull impactframes/llama3_ifai_sd_prompt_mkr_q4km:latest
) else if "%choice%"=="4" (
    echo.
    echo ==============================================================================
    echo =                                                                            =
    echo =  No models will be pulled.                                                 =
    echo =                                                                            =
    echo ==============================================================================
    echo.
) else (
    echo.
    echo ==============================================================================
    echo =                                                                            =
    echo =  Invalid choice. Exiting...                                                =
    echo =                                                                            =
    echo ==============================================================================
    echo.
    pause
    exit /b
)

:: Ask user if they want to run ArtAgent
echo.
echo ==============================================================================
echo =                                                                            =
echo =  Do you want to run ArtAgents now? (yes/no):                               =
echo =                                                                            =
echo ==============================================================================
echo.
set /p run_artagent=Enter your choice (yes/no):

:: Run go.bat if user chooses to run ArtAgent
if /i "%run_artagent%"=="yes" (
    echo.
    echo ==============================================================================
    echo =                                                                            =
    echo =  Running ArtAgents...                                                      =
    echo =                                                                            =
    echo ==============================================================================
    echo.
    call go.bat
) else (
    echo.
    echo ==============================================================================
    echo =                                                                            =
    echo =  ArtAgents will not be run right away. Run it later with "go.bat".         =
    echo =                                                                            =
    echo ==============================================================================
    echo.
)

echo.
echo ==============================================================================
echo =                                                                            =
echo =  Setup complete.                                                           =
echo =                                                                            =
echo ==============================================================================
echo.
pause
