@echo off
echo ===========================================
echo   Auth Service - Stopping (Port 8002)
echo ===========================================
echo.

echo Stopping Auth service on port 8002...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8002 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)
echo   Done.
echo.
pause

