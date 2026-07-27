@echo off
setlocal EnableDelayedExpansion
title Crop Recommendation Engine -- Cloudflare Edition
color 0A
echo =====================================================
echo   Crop Recommendation Engine -- Cloudflare Edition
echo =====================================================
echo.

cd /d "%~dp0"

REM --- Check Python ---
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found. Install Python 3.11+ and try again.
    pause
    exit /b 1
)

REM --- Kill stale processes ---
echo [0/3] Cleaning up stale processes...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8001 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8501 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)
taskkill /f /im ngrok.exe >nul 2>&1
taskkill /f /im cloudflared.exe >nul 2>&1
timeout /t 2 /nobreak >nul
echo   Done.
echo.

REM --- 1. Start FastAPI Server ---
echo [1/3] Starting FastAPI API server on port 8001...
if exist "%~dp0venv\Scripts\python.exe" (
    start "CropRec API Server" /d "%~dp0" /min cmd /c "venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8001"
) else (
    echo   [WARN] No venv at Crop_Recommendation_Engine\venv — using system Python ^(may fail if deps mismatch^).
    start "CropRec API Server" /d "%~dp0" /min cmd /c "python -m uvicorn api:app --host 127.0.0.1 --port 8001"
)
echo   API server starting in background...
echo   Health check: http://127.0.0.1:8001/health
echo   Swagger docs: http://127.0.0.1:8001/docs
echo.

REM --- Wait for API to be ready ---
echo   Waiting for API to be ready...
set /a TRIES=0
:WAIT_API
timeout /t 2 /nobreak >nul
set /a TRIES+=1
python -c "import requests; r=requests.get('http://127.0.0.1:8001/health',timeout=3); exit(0 if r.status_code==200 else 1)" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   API is ready!
    echo.
) else (
    if %TRIES% LSS 10 goto WAIT_API
    echo   [WARNING] API not responding after 20s. Continuing anyway...
    echo.
)

REM --- 2. Start Streamlit Dashboard ---
echo [2/3] Starting Streamlit dashboard on port 8501...
if exist "%~dp0venv\Scripts\python.exe" (
    start "CropRec Streamlit" /d "%~dp0" /min cmd /c "venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.headless true"
) else (
    start "CropRec Streamlit" /d "%~dp0" /min cmd /c "streamlit run app.py --server.port 8501 --server.headless true"
)
echo   Dashboard starting in background...
echo   Local URL: http://127.0.0.1:8501
echo.
timeout /t 3 /nobreak >nul

REM --- 3. Start Cloudflare Quick Tunnel ---
if not exist "%~dp0..\tools\cloudflared.exe" (
    echo [3/3] cloudflared.exe not found in tools folder.
    echo   Please copy it to: AgroModules\tools\cloudflared.exe
    echo.
    echo   Skipping external tunnel. App is available locally only.
    goto DONE
)

echo [3/3] Starting Cloudflare quick tunnel for external access...
start "CropRec Tunnel" /d "%~dp0..\tools" /min cmd /c "cloudflared.exe tunnel --url http://localhost:8501 > "%TEMP%\cf_crop_tunnel.log" 2>&1"
echo   Tunnel starting in background...
echo   Fetching public URL...
set TRIES=0
:WAIT_TUNNEL
timeout /t 2 /nobreak >nul
set /a TRIES+=1
findstr /C:"https://" "%TEMP%\cf_crop_tunnel.log" 2>nul | findstr /C:"trycloudflare.com" > "%TEMP%\cf_crop_extracted.txt" 2>nul
for /f "tokens=4 delims= " %%u in ('type "%TEMP%\cf_crop_extracted.txt" 2^>nul') do set CF_URL=%%u

if defined CF_URL (
    echo.
    echo =====================================================
    echo   EXTERNAL LINK:
    echo.
    echo      !CF_URL!
    echo.
    echo =====================================================
    goto DONE
)
if !TRIES! LSS 8 goto WAIT_TUNNEL

echo   [INFO] Tunnel taking longer than expected to initialize.
echo          Please check %TEMP%\cf_crop_tunnel.log manually.
echo.

:DONE
echo =====================================================
echo   SERVICES RUNNING:
echo.
echo   API Server    :  http://127.0.0.1:8001
echo   Dashboard     :  http://127.0.0.1:8501
echo   Docs          :  http://127.0.0.1:8001/docs
echo.
echo   To stop: run  stop_crop.bat
echo =====================================================
echo.

REM Open browser to dashboard
start "" http://127.0.0.1:8501
echo Press any key to keep services running (or close window)...
pause >nul
