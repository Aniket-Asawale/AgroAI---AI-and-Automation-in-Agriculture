@echo off
setlocal
title AgroSensor - Cloud Tunnel (Cloudflare)
echo ============================================
echo   AgroSensor - Cloud Access Setup
echo   (Using Cloudflare Quick Tunnel - Free)
echo ============================================
echo.
echo This exposes your local AgroSensor to the internet.
echo Share the generated trycloudflare.com URL to access from any device.
echo.

REM --- Find cloudflared in tools ---
if exist "%~dp0..\tools\cloudflared.exe" goto :run_cloudflared

echo [ERROR] cloudflared.exe is NOT found in the 'tools' folder.
echo.
echo Please copy cloudflared.exe to:
echo   AgroModules\tools\cloudflared.exe
echo.
pause
exit /b 1

:run_cloudflared
echo cloudflared found in tools folder.
echo.
echo Make sure AgroSensor server is running first (start_agro.bat).
echo The tunnel URL will appear below (look for trycloudflare.com).
echo.
echo Starting quick tunnel on port 8000...
echo.
"%~dp0..\tools\cloudflared.exe" tunnel --url http://localhost:8000
goto :done

:done
echo.
echo cloudflared has stopped.
pause
