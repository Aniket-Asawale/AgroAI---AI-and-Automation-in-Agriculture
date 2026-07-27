@echo off
setlocal EnableDelayedExpansion
title AgroSensor - Start API and Dashboard
color 0A

echo ============================================================
echo   AgroSensor - Starting All Services
echo ============================================================
echo.
echo   Components:
echo     - MQTT Publisher ^(background, if enabled^)
echo     - Sensor Bridge ^(background, if using COM port^)
echo     - API Server ^(port 8000 - visible^)
echo     - Dashboard ^(port 8502 - static^)
echo.

cd /d "%~dp0"

REM --- 0. Check if API Gateway is running ---
echo [0/4] Checking API Gateway availability...
netstat -ano | findstr ":8080 " | findstr "LISTENING" >nul
if %ERRORLEVEL% NEQ 0 (
    echo   [WARN] API Gateway not detected on port 8080
    echo         Make sure to run 'start_all.bat' from root or start it separately
)
echo.

REM --- 1. Detect and start mode-specific services ---
echo [1/4] Detecting operation mode...
if exist "config.py" (
    python -c "from config import settings; exit(0 if settings.MQTT_ENABLED else 1)" 2>nul
    if !ERRORLEVEL! EQU 0 (
        echo   Mode: MQTT Cloud ^(HiveMQ^)
        echo.
        echo   Starting MQTT publisher in background...
        if exist "venv\Scripts\python.exe" (
            start "AgroSensor MQTT Publisher" /min cmd /c "venv\Scripts\python.exe tools\mqtt_publisher.py --interval 300"
        ) else (
            start "AgroSensor MQTT Publisher" /min cmd /c "python tools\mqtt_publisher.py --interval 300"
        )
        echo   MQTT publisher started.
        timeout /t 2 /nobreak >nul
    ) else (
        echo   Mode: Legacy COM Port ^(Modbus RTU^)
        echo.
        echo   Starting sensor bridge in background...
        echo   ^(Make sure VSPE is running with COM9 ^<-^> COM10 pair active^)
        if exist "_open_browser.vbs" (
            start "" /min wscript "%~dp0start_sensor.vbs" 2>nul
        )
        echo   Sensor bridge started.
        timeout /t 1 /nobreak >nul
    )
) else (
    echo   Could not detect mode. Assuming MQTT enabled.
)
echo.

REM --- 2. Start Sensor Dashboard (static) ---
echo [2/4] Starting Sensor Dashboard ^(port 8502^)...
if exist "venv\Scripts\python.exe" (
    start "Sensor Dashboard" cmd /k "venv\Scripts\python.exe -m http.server 8502 --directory dashboard"
    echo   Started. http://127.0.0.1:8502
) else (
    start "Sensor Dashboard" cmd /k "python -m http.server 8502 --directory dashboard"
    echo   Started (system python). http://127.0.0.1:8502
)
timeout /t 1 /nobreak >nul
echo.

REM --- 3. Start API Server ---
echo [3/4] Starting AgroSensor API Server ^(port 8000^)...
echo.
echo   Local Access:
echo     API:       http://127.0.0.1:8000
echo     Health:    http://127.0.0.1:8000/api/health
echo     Dashboard: http://127.0.0.1:8502
echo.
echo   Public Access ^(via Cloudflare^):
echo     API:       https://agroaiapp.me/api/sensor
echo     Health:    https://agroaiapp.me/api/sensor/health
echo     Dashboard: https://agroaiapp.me/sensor/dashboard/
echo.

REM --- 4. Open browser ---
echo [4/4] Opening browser windows...
if exist "venv\Scripts\python.exe" (
    timeout /t 2 /nobreak >nul
    start http://127.0.0.1:8502
    timeout /t 1 /nobreak >nul

    REM --- Run API server in this window ---
    venv\Scripts\python.exe main.py
) else (
    echo [ERROR] No Python venv found.
    echo.
    echo   Setup ^(one-time^):
    echo     python -m venv venv
    echo     venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo ===========================================================
echo   AgroSensor API stopped.
echo ===========================================================
echo.

endlocal
