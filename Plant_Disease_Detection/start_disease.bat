@echo off
setlocal EnableDelayedExpansion
title Plant Disease Detection - Start API
color 0A

echo ============================================================
echo   Plant Disease Detection - Starting API Server
echo ============================================================
echo.
echo   ML Model: Trained on plant leaf disease images
echo   Port: 8003 (FastAPI)
echo   Optional: 7860 (Gradio UI)
echo.

cd /d "%~dp0"

REM --- 0. Check if API Gateway is running ---
echo [0/3] Checking API Gateway availability...
netstat -ano | findstr ":8080 " | findstr "LISTENING" >nul
if %ERRORLEVEL% NEQ 0 (
    echo   [WARN] API Gateway not detected on port 8080
    echo         Make sure to run 'start_all.bat' from root or start it separately
)
echo.

REM --- 1. Check Python ---
echo [1/3] Checking Python and venv...
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] No venv found.
    echo.
    echo   Create venv with:
    echo     python -m venv venv
    echo     venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo   venv found.
echo.

REM --- 2. Check for Gradio UI script ---
echo [2/3] Checking for UI components...
if exist "gradio_app.py" (
    echo   Found: gradio_app.py - UI will be available on port 7860
    set "HAS_GRADIO=1"
) else (
    echo   No Gradio UI script found - API only
    set "HAS_GRADIO=0"
)
echo.

REM --- 3. Start API Server ---
echo [3/3] Starting Plant Disease Detection API ^(port 8003^)...
echo.
echo   API URLs:
echo     Local:  http://127.0.0.1:8003/docs
echo     Public: https://agroaiapp.me/api/disease
echo.

if "!HAS_GRADIO!"=="1" (
    echo   UI URLs:
    echo     Local:  http://127.0.0.1:7860
    echo     Public: https://agroaiapp.me/plant-disease/dashboard/
    echo.
)
echo   Opening browser windows...
timeout /t 2 /nobreak >nul
start http://127.0.0.1:8003/docs
timeout /t 1 /nobreak >nul

if "!HAS_GRADIO!"=="1" (
    REM Start Gradio UI in background
    echo   Starting Gradio UI in background...
    start "Disease UI" cmd /k "venv\Scripts\python.exe gradio_app.py"
    timeout /t 2 /nobreak >nul
    start http://127.0.0.1:7860
)

echo.
echo   Starting API server in this window...
echo   Press Ctrl+C to stop.
echo.

REM --- Run the API server ---
venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8003 --reload

echo.
echo ===========================================================
echo   Plant Disease Detection API stopped.
echo ===========================================================
echo.

endlocal
