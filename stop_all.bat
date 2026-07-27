@echo off
setlocal EnableDelayedExpansion
title AgroModules - Stop All Services
color 0C
echo ===========================================================
echo   AgroModules - Stopping All Backend Services
echo   (Cloudflare Tunnel will remain ACTIVE)
echo ===========================================================
echo.

REM --- 0. Cloudflare Tunnel Status ---
echo [0/9] Cloudflare Tunnel Status...
tasklist /FI "IMAGENAME eq cloudflared.exe" /FO CSV | find /I "cloudflared" >nul
if %ERRORLEVEL% EQU 0 (
    echo   [ACTIVE] Tunnel will stay running for public access
) else (
    echo   [INACTIVE] Tunnel is not running
)
echo.

REM --- 1. API Gateway (port 8080) ---
echo [1/9] Stopping API Gateway (port 8080)...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8080 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)
echo   Done.

REM --- 2. Auth Service (port 8002) ---
echo [2/9] Stopping Auth Service (port 8002)...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8002 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)
echo   Done.

REM --- 3. AgroSensor (port 8000) ---
echo [3/9] Stopping AgroSensor (port 8000)...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8000 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)
REM Also kill background sensor processes
wmic process where "commandline like '%%mqtt_publisher%%'" call terminate >nul 2>&1
echo   Done.

REM --- 4. Crop Recommendation API (port 8001) ---
echo [4/9] Stopping Crop API (port 8001)...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8001 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)
echo   Done.

REM --- 5. Plant Disease Detection (port 8003) ---
echo [5/9] Stopping Disease Detection (port 8003)...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8003 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)
echo   Done.

REM --- 6. Sensor Dashboard (port 8502) ---
echo [6/9] Stopping Sensor Dashboard (port 8502)...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8502 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)
echo   Done.

REM --- 7. Disease Dashboard (port 7860) ---
echo [7/9] Stopping Disease Dashboard (port 7860)...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":7860 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)
echo   Done.

REM --- 8. Homepage (port 8505) ---
echo [8/9] Stopping Homepage (port 8505)...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8505 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)
echo   Done.

REM --- 9. Cleanup ---
echo [9/9] Cleanup...
timeout /t 1 /nobreak >nul
echo   Done.

echo.
echo ===========================================================
echo   [OK] All AgroModules services stopped.
echo ===========================================================
echo   [ACTIVE] Cloudflare Tunnel remains running
echo   [ACTIVE] Homepage https://agroaiapp.me is accessible
echo   [ACTIVE] API https://api.agroaiapp.me is accessible
echo   [NOTE]  Crop Dashboard on Streamlit Cloud is always available
echo   [TIP]   Run AgroManager.bat [1] to restart all services
echo ===========================================================
echo.
pause
