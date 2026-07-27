@echo off
title Crop Recommendation Engine -- Stop All
color 0C
echo =====================================================
echo   Crop Recommendation Engine -- Stopping All Services
echo =====================================================
echo.

REM --- 1. Stop cloudflared tunnel ---
echo [1/3] Stopping cloudflared tunnel...
taskkill /f /im cloudflared.exe >nul 2>&1
echo   Done.

REM --- 2. Stop Streamlit (port 8501) ---
echo [2/3] Stopping Streamlit dashboard...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8501 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)
echo   Done.

REM --- 3. Stop FastAPI (port 8001) ---
echo [3/3] Stopping API server...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8001 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)
echo   Done.

echo.
echo =====================================================
echo   All Crop Recommendation services stopped.
echo =====================================================
echo.
pause
