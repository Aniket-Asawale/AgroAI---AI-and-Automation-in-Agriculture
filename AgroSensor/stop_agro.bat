@echo off
title AgroSensor - Stop All
echo ============================================
echo   AgroSensor - Stopping All Services
echo ============================================
echo.

REM --- Kill background sensor simulator (pythonw.exe running sensor_conn.py) ---
echo [1/2] Stopping sensor bridge...
taskkill /f /im pythonw.exe /fi "WINDOWTITLE eq *" 2>nul
REM Also kill any visible python running sensor_conn.py
wmic process where "commandline like '%%sensor_conn%%'" call terminate >nul 2>&1
echo   Done.

REM --- Kill main.py (uvicorn) ---
echo [2/2] Stopping API server...
wmic process where "commandline like '%%main.py%%'" call terminate >nul 2>&1
wmic process where "commandline like '%%uvicorn%%'" call terminate >nul 2>&1
echo   Done.

echo.
echo All AgroSensor processes stopped.
pause

