@echo off
echo ===========================================================
echo   Plant Disease Detection -- Gradio UI
echo ===========================================================
echo.

cd /d "%~dp0"

REM --- Check Python ---
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Install Python 3.11+ and try again.
    pause
    exit /b 1
)

REM --- Check if API is running ---
echo Checking if Plant Disease Detection API is running...
curl -s http://127.0.0.1:8003/health >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Plant Disease Detection API is not running on port 8003.
    echo.
    echo Please start the API first:
    echo   start_disease.bat
    echo.
    echo Or run both API and UI together:
    echo   start_disease.bat
    echo   start_disease_ui.bat
    echo.
    pause
)

REM --- Install dependencies if needed ---
echo Installing dependencies...
python -m pip install --upgrade pip -q
python -m pip install gradio requests pillow -q

REM --- Start Gradio UI ---
echo.
echo Starting Gradio UI...
echo Local URL: http://127.0.0.1:7860
echo.

REM Open browser to Gradio UI
start  http://127.0.0.1:7860

python gradio_app.py
