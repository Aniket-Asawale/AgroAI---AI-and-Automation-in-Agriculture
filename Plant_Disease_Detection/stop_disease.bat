@echo off
echo ===========================================
echo   Plant Disease Detection - Stopping (Port 8003)
echo ===========================================
echo.

echo Stopping Plant Disease Detection on port 8003...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8003 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)
echo   Done.
echo.
pause

