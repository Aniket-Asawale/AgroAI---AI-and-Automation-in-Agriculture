@echo off
echo ===========================================
echo   API Gateway - Stopping (Port 8080)
echo ===========================================
echo.

echo Stopping API Gateway on port 8080...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8080 " ^| findstr "LISTENING" 2^>nul') do (
    taskkill /PID %%p /F >nul 2>&1
)
echo   Done.
echo.
pause

