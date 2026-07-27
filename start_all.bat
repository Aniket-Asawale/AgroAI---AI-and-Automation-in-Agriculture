@echo off
setlocal EnableDelayedExpansion
title AgroModules - Start ALL Services with Cloudflare Tunnel
color 0A

echo ===========================================================
echo   AgroModules - Starting All Backend Services
echo ===========================================================
echo.
echo   Services to start:
echo     - Cloudflare Tunnel  (agromodules-api)
echo     - Auth Service       (port 8002)
echo     - AgroSensor API     (port 8000)
echo     - Crop API           (port 8001)
echo     - Plant Disease      (port 8003)
echo     - Sensor Dashboard   (port 8502)
echo     - Disease Dashboard  (port 7860)
echo     - API Gateway        (port 8080)
echo     - Homepage           (port 8505)
echo.
echo   [NOTE] Crop Dashboard is on Streamlit Cloud
echo          https://croprecommendationengine.streamlit.app
echo.

cd /d "%~dp0"
set "ROOT=%~dp0"

REM --- 0. Start Cloudflare Tunnel ---
echo [0/8] Starting Cloudflare Tunnel...
tasklist /FI "IMAGENAME eq cloudflared.exe" /FO CSV | find /I "cloudflared" >nul
if %ERRORLEVEL% NEQ 0 (
    set "CLOUDFLARE_API_TOKEN=your_cloudflare_api_token_here"
    start "Cloudflare Tunnel" /d "%ROOT%" /min "%ROOT%tools\cloudflared.exe" tunnel --no-autoupdate --config cloudflare-tunnel-config.yaml run agromodules-api
    timeout /t 3 /nobreak >nul
    echo   Tunnel started. Domains: https://agroaiapp.me + https://api.agroaiapp.me
) else (
    echo   Tunnel already running.
)
echo.

REM --- 1. Auth Service (port 8002) ---
echo [1/8] Auth Service ^(port 8002^)...
if exist "%ROOT%Auth\venv\Scripts\python.exe" (
    start "Auth Service" /d "%ROOT%Auth" cmd /k "venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8002"
    echo   Started. http://127.0.0.1:8002
) else (
    echo   [SKIP] No venv - install manually
)
timeout /t 1 /nobreak >nul
echo.

REM --- 2. AgroSensor API (port 8000) ---
echo [2/8] AgroSensor API ^(port 8000^)...
if exist "%ROOT%AgroSensor\venv\Scripts\python.exe" (
    start "AgroSensor API" /d "%ROOT%AgroSensor" cmd /k "venv\Scripts\python.exe main.py"
    echo   Started. http://127.0.0.1:8000
) else (
    echo   [SKIP] No venv - install manually
)
timeout /t 1 /nobreak >nul
echo.

REM --- 3. Crop Recommendation API (port 8001) ---
echo [3/8] Crop Recommendation API ^(port 8001^)...
if exist "%ROOT%Crop_Recommendation_Engine\venv\Scripts\python.exe" (
    start "Crop API" /d "%ROOT%Crop_Recommendation_Engine" cmd /k "venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8001"
    echo   Started. http://127.0.0.1:8001
) else (
    echo   [SKIP] No venv - install manually
)
timeout /t 1 /nobreak >nul
echo.

REM --- 4. Plant Disease Detection (port 8003) ---
echo [4/8] Plant Disease Detection ^(port 8003^)...
if exist "%ROOT%Plant_Disease_Detection\venv\Scripts\python.exe" (
    start "Disease Detection" /d "%ROOT%Plant_Disease_Detection" cmd /k "venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8003"
    echo   Started. http://127.0.0.1:8003
) else (
    echo   [SKIP] No venv - install manually
)
timeout /t 1 /nobreak >nul
echo.

REM --- 5. Sensor Dashboard (port 8502) ---
echo [5/8] Sensor Dashboard ^(port 8502^)...
if exist "%ROOT%AgroSensor\venv\Scripts\python.exe" (
    start "Sensor Dashboard" /d "%ROOT%AgroSensor" cmd /k "venv\Scripts\python.exe -m http.server 8502 --directory dashboard"
    echo   Started. http://127.0.0.1:8502
) else (
    echo   [SKIP] No venv - install manually
)
timeout /t 1 /nobreak >nul
echo.

REM --- 6. Disease Dashboard - Gradio (port 7860) ---
echo [6/8] Disease Dashboard ^(port 7860^)...
if exist "%ROOT%Plant_Disease_Detection\venv\Scripts\python.exe" (
    start "Disease Dashboard" /d "%ROOT%Plant_Disease_Detection" cmd /k "venv\Scripts\python.exe gradio_app.py"
    echo   Started. http://127.0.0.1:7860
) else (
    echo   [SKIP] No venv - install manually
)
timeout /t 1 /nobreak >nul
echo.

REM --- 7. API Gateway (port 8080) ---
echo [7/8] API Gateway ^(port 8080^)...
if exist "%ROOT%ApiGateway\venv\Scripts\python.exe" (
    start "API Gateway" /d "%ROOT%ApiGateway" cmd /k "venv\Scripts\python.exe main.py"
    echo   Started. http://127.0.0.1:8080/docs
) else (
    echo   [SKIP] No venv - install manually
)
timeout /t 1 /nobreak >nul
echo.

REM --- 8. Homepage (port 8505) ---
echo [8/8] Homepage ^(port 8505^)...
start "Homepage" /d "%ROOT%web" cmd /k "python serve.py"
echo   Started. http://127.0.0.1:8505
timeout /t 2 /nobreak >nul
echo.

REM --- Open browser ---
echo Opening dashboards...
timeout /t 3 /nobreak >nul
start http://127.0.0.1:8505
timeout /t 1 /nobreak >nul

echo.
echo ===========================================================
echo   ALL SERVICES STARTED
echo ===========================================================
echo.
echo   LOCAL ACCESS:
echo     Homepage:            http://127.0.0.1:8505
echo     API Docs:            http://127.0.0.1:8080/docs
echo     Sensor API:          http://127.0.0.1:8000
echo     Sensor Dashboard:    http://127.0.0.1:8502
echo     Crop API:            http://127.0.0.1:8001/docs
echo     Disease API:         http://127.0.0.1:8003/docs
echo     Disease Dashboard:   http://127.0.0.1:7860
echo     Auth API:            http://127.0.0.1:8002/docs
echo.
echo   PUBLIC ACCESS ^(via Cloudflare^):
echo     Homepage:            https://agroaiapp.me
echo     API Docs:            https://api.agroaiapp.me/docs
echo     Crop Dashboard:      https://croprecommendationengine.streamlit.app
echo     Sensor Dashboard:    https://sensor-dashboard.agroaiapp.me
echo     Disease Dashboard:   https://disease-dashboard.agroaiapp.me
echo.
echo   Check individual console windows in taskbar for logs.
echo ===========================================================
echo.

endlocal
pause
