@echo off
setlocal EnableDelayedExpansion
title Auth Service - Start API
color 0A

echo ============================================================
echo   Auth Service - Starting API Server
echo ============================================================
echo.
echo   Prerequisites:
echo     - PostgreSQL running on localhost:5432
echo     - Database 'agrodb' created
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

REM --- 2. Start API Server ---
echo [2/3] Starting Auth Service API Server ^(port 8002^)...
echo.
echo   Local URLs:
echo     API Docs:  http://127.0.0.1:8002/docs
echo     Health:    http://127.0.0.1:8002/health
echo.
echo   Public URLs ^(via Cloudflare^):
echo     API:       https://agroaiapp.me/services/auth
echo.

REM --- 3. Open browser and run server ---
echo [3/3] Opening browser windows...
echo.
timeout /t 2 /nobreak >nul
start http://127.0.0.1:8002/docs
timeout /t 1 /nobreak >nul
start https://agroaiapp.me/services/auth
timeout /t 1 /nobreak >nul

echo.
echo   Starting API server in this window...
echo   Press Ctrl+C to stop.
echo.

REM --- Run the server ---
venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8002 --reload

echo.
echo ===========================================================
echo   Auth Service stopped.
echo ===========================================================
echo.

endlocal
