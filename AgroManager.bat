@echo off
REM =========================================================================
REM   AGROMODULES CENTRAL MANAGER
REM   Unified service launcher for all AgroModules microservices
REM   Tunnel: Cloudflare (agroaiapp.me) - Always Persistent
REM   Crop Dashboard: Streamlit Cloud (croprecommendationengine.streamlit.app)
REM =========================================================================

setlocal EnableDelayedExpansion
title AgroModules Central Manager
color 0E

set "ROOT=%~dp0"
set "CLOUDFLARED=%ROOT%tools\cloudflared.exe"

REM =========================================================================
REM AUTO-START ON LAUNCH — starts all services automatically on first run
REM =========================================================================
if not defined AUTOSTARTED (
    set "AUTOSTARTED=1"
    echo.
    echo =================================================================
    echo   AUTO-START: Launching ALL services on startup...
    echo =================================================================
    echo.
    call :start_all_services
)

REM =========================================================================
REM MAIN MENU
REM =========================================================================

:main_menu
cls
echo.
echo =================================================================
echo                    AGROMODULES CENTRAL HUB
echo =================================================================
echo.
echo   [1] Start ALL Microservices ^(including Cloudflare Tunnel^)
echo   [2] Stop ALL Microservices cleanly
echo   [3] Start Specific Service manually...
echo   [4] Check Port Status ^(Who's listening?^)
echo.
echo   [5] [WWW] Open Homepage
echo   [6] [WWW] Open Crop Dashboard ^(Streamlit Cloud^)
echo   [7] [API] Open API Swagger Docs
echo   [8] View System Status
echo.
echo   [9] Initialize PostgreSQL Databases
echo   [0] Exit Manager
echo.
echo =================================================================
echo   PUBLIC ACCESS ^(when tunnel is active^):
echo     https://agroaiapp.me                            - Homepage
echo     https://api.agroaiapp.me/docs                   - API Docs
echo     https://croprecommendationengine.streamlit.app  - Crop Dashboard
echo     https://sensor-dashboard.agroaiapp.me            - Sensor Dashboard
echo     https://disease-dashboard.agroaiapp.me           - Disease Detection
echo =================================================================
echo.

set "CHOICE="
set /p CHOICE="> Select an option: "

if "%CHOICE%"=="1" ( call :start_all_services & goto main_menu )
if "%CHOICE%"=="2" ( call :stop_all_services & goto main_menu )
if "%CHOICE%"=="3" ( call :start_specific_service & goto main_menu )
if "%CHOICE%"=="4" ( call :check_port_status & goto main_menu )
if "%CHOICE%"=="5" ( call :open_homepage & goto main_menu )
if "%CHOICE%"=="6" ( call :open_crop_dashboard & goto main_menu )
if "%CHOICE%"=="7" ( call :open_api_docs & goto main_menu )
if "%CHOICE%"=="8" ( call :view_system_status & goto main_menu )
if "%CHOICE%"=="9" ( call :init_databases & goto main_menu )
if "%CHOICE%"=="0" (
    echo.
    echo [INFO] Exiting AgroManager. Tunnel remains active for website access.
    echo.
    timeout /t 2 /nobreak >nul
    exit /b 0
)

echo.
echo [ERROR] Invalid selection. Please try again.
timeout /t 2 /nobreak >nul
goto main_menu


REM =========================================================================
REM [1] START ALL MICROSERVICES
REM =========================================================================

:start_all_services
cls
echo ===========================================================
echo   Starting ALL AgroModules Services
echo ===========================================================
echo.
echo   [*] Services will run in BACKGROUND (no extra windows)
echo   [*] Logs saved to logs\service_*.txt
echo   [*] This may take 10-15 seconds...
echo.

REM [CLEANUP] Kill conflicting ports (8000-8003, 8501-8502, 8505, 7860 only)
echo [CLEANUP] Terminating any processes on service ports...
echo            (Skipping 8080 gateway and tunnel - keeper is managing those)
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":8000 \|:8001 \|:8002 \|:8003 \|:8501 \|:8502 \|:8505 \|:7860 "') do (
    taskkill /PID %%P /F >nul 2>&1
)
timeout /t 2 /nobreak >nul

REM Create logs directory
if not exist "%ROOT%logs" mkdir "%ROOT%logs"

REM Clear old log files
for %%F in (gateway auth sensor crop_api crop_dashboard disease sensor_dashboard disease_dashboard homepage) do (
    if exist "%ROOT%logs\%%F.txt" del "%ROOT%logs\%%F.txt" 2>nul
)

REM --- [1/9] API Gateway ---
echo [1/9] API Gateway ^(port 8080^)...
netstat -ano 2>nul | findstr ":8080 " | findstr "LISTENING" >nul
if %ERRORLEVEL% EQU 0 (
    echo       ALREADY RUNNING ^(keeper managing^)
) else (
    echo       Starting fresh...
    pushd "%ROOT%ApiGateway"
    start /B "Gateway" cmd /c "venv\Scripts\python.exe main.py > ..\logs\gateway.txt 2>&1"
    popd
    timeout /t 2 /nobreak >nul
    echo       [OK] Gateway started.
)
timeout /t 1 /nobreak >nul

REM --- [2/9] Cloudflare Tunnel ---
echo [2/9] Cloudflare Tunnel ^(agromodules-api^)...
tasklist /FI "IMAGENAME eq cloudflared.exe" /FO CSV 2>nul | find /I "cloudflared" >nul
if %ERRORLEVEL% EQU 0 (
    echo       ALREADY RUNNING ^(keeper managing^)
) else (
    echo       Starting tunnel...
    set "CLOUDFLARE_API_TOKEN=your_cloudflare_api_token_here"
    start /B /min "Cloudflare Tunnel" "%CLOUDFLARED%" tunnel --no-autoupdate --config "%ROOT%cloudflare-tunnel-config.yaml" run agromodules-api
    echo       [OK] Tunnel starting. Domains: https://agroaiapp.me + https://api.agroaiapp.me
    timeout /t 2 /nobreak >nul
)
timeout /t 1 /nobreak >nul

REM --- [3/9] Auth Service ---
echo [3/9] Auth Service ^(port 8002^)...
pushd "%ROOT%Auth"
start /B "Auth" cmd /c "venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8002 > ..\logs\auth.txt 2>&1"
popd
timeout /t 1 /nobreak >nul

REM --- [4/9] AgroSensor API ---
echo [4/9] AgroSensor API ^(port 8000^)...
pushd "%ROOT%AgroSensor"
start /B "AgroSensor" cmd /c "venv\Scripts\python.exe main.py > ..\logs\sensor.txt 2>&1"
popd
timeout /t 1 /nobreak >nul

REM --- [5/9] Crop Recommendation API ---
echo [5/9] Crop Recommendation API ^(port 8001^)...
pushd "%ROOT%Crop_Recommendation_Engine"
start /B "CropAPI" cmd /c "venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8001 > ..\logs\crop_api.txt 2>&1"
popd
timeout /t 1 /nobreak >nul

REM --- [6/9] Plant Disease Detection API ---
echo [6/9] Plant Disease Detection ^(port 8003^)...
pushd "%ROOT%Plant_Disease_Detection"
start /B "Disease" cmd /c "venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8003 > ..\logs\disease.txt 2>&1"
popd
timeout /t 1 /nobreak >nul

REM --- [7/9] Sensor Dashboard (static) ---
echo [7/9] Sensor Dashboard ^(port 8502^)...
pushd "%ROOT%AgroSensor"
start /B "SensorDash" cmd /c "venv\Scripts\python.exe -m http.server 8502 --directory dashboard > ..\logs\sensor_dashboard.txt 2>&1"
popd
timeout /t 1 /nobreak >nul

REM --- [8/9] Disease Dashboard (Gradio) ---
echo [8/9] Disease Dashboard - Gradio ^(port 7860^)...
pushd "%ROOT%Plant_Disease_Detection"
start /B "DiseaseDash" cmd /c "venv\Scripts\python.exe gradio_app.py > ..\logs\disease_dashboard.txt 2>&1"
popd
timeout /t 1 /nobreak >nul

REM --- [9/9] Homepage ---
echo [9/9] Homepage ^(port 8505^)...
pushd "%ROOT%web"
start /B "Homepage" cmd /c "python serve.py > ..\logs\homepage.txt 2>&1"
popd
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
echo   LOGS: logs\ directory ^(gateway.txt, auth.txt, sensor.txt, etc.^)
echo.
echo   Waiting 8 seconds before opening dashboards...
timeout /t 8 /nobreak >nul

REM Open key dashboards
echo   [*] Opening Homepage...
start https://agroaiapp.me
timeout /t 2 /nobreak >nul

echo   [*] Opening API Docs...
start https://api.agroaiapp.me/docs
timeout /t 2 /nobreak >nul

echo   [*] Opening Crop Dashboard ^(Streamlit Cloud^)...
start https://croprecommendationengine.streamlit.app
timeout /t 2 /nobreak >nul

echo.
echo ===========================================================
echo   Startup Complete. Returning to menu in 5 seconds...
echo ===========================================================
timeout /t 5 /nobreak >nul
exit /b


REM =========================================================================
REM [2] STOP ALL MICROSERVICES
REM =========================================================================

:stop_all_services
cls
echo ===========================================================
echo   Stopping ALL Backend Services
echo   ^(Cloudflare Tunnel will remain active^)
echo ===========================================================
echo.

echo [*] Killing all services by port...

echo [1/7] Stopping AgroSensor API ^(port 8000^)...
for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":8000 "') do taskkill /PID %%P /F >nul 2>&1
echo       Done.

echo [2/7] Stopping Crop API ^(port 8001^)...
for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":8001 "') do taskkill /PID %%P /F >nul 2>&1
echo       Done.

echo [3/7] Stopping Auth Service ^(port 8002^)...
for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":8002 "') do taskkill /PID %%P /F >nul 2>&1
echo       Done.

echo [4/7] Stopping Disease Detection ^(port 8003^)...
for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":8003 "') do taskkill /PID %%P /F >nul 2>&1
echo       Done.

echo [5/7] Stopping Sensor Dashboard ^(port 8502^)...
for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":8502 "') do taskkill /PID %%P /F >nul 2>&1
echo       Done.

echo [6/7] Stopping Disease Dashboard ^(port 7860^)...
for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":7860 "') do taskkill /PID %%P /F >nul 2>&1
echo       Done.

echo [7/7] Cleanup...
timeout /t 1 /nobreak >nul
echo       Done.

echo.
echo ===========================================================
echo   ALL BACKEND SERVICES STOPPED
echo ===========================================================
echo   [ACTIVE] Cloudflare Tunnel remains running
echo   [ACTIVE] API Gateway remains running ^(keeper managed^)
echo   [NOTE]  Crop Dashboard is on Streamlit Cloud ^(always available^)
echo ===========================================================
echo.
pause >nul
exit /b


REM =========================================================================
REM [3] START SPECIFIC SERVICE
REM =========================================================================

:start_specific_service
cls
echo ===========================================================
echo                 START SPECIFIC SERVICE
echo ===========================================================
echo   [1] Auth Service              ^(port 8002^)
echo   [2] AgroSensor API            ^(port 8000^)
echo   [3] Crop Recommendation API   ^(port 8001^)
echo   [4] Sensor Dashboard          ^(port 8502^)
echo   [5] Plant Disease Detection   ^(port 8003^)
echo   [6] Disease Dashboard ^(Gradio^)  ^(port 7860^)
echo   [7] API Gateway               ^(port 8080 - keeper managed^)
echo   [8] Cloudflare Tunnel ^(keeper managed^)
echo   [0] Back to Menu
echo ===========================================================
echo.

set "SUBCHOICE="
set /p SUBCHOICE="> Select service: "

if "%SUBCHOICE%"=="1" (
    for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":8002 "') do taskkill /PID %%P /F >nul 2>&1
    timeout /t 1 /nobreak >nul
    if not exist "%ROOT%logs" mkdir "%ROOT%logs"
    pushd "%ROOT%Auth"
    start /B "Auth" cmd /c "venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8002 > ..\logs\auth.txt 2>&1"
    popd
    echo [OK] Auth Service started. Log: logs\auth.txt
    echo      http://127.0.0.1:8002/docs
    timeout /t 2 /nobreak >nul
    exit /b
)
if "%SUBCHOICE%"=="2" (
    for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":8000 "') do taskkill /PID %%P /F >nul 2>&1
    timeout /t 1 /nobreak >nul
    if not exist "%ROOT%logs" mkdir "%ROOT%logs"
    pushd "%ROOT%AgroSensor"
    start /B "AgroSensor" cmd /c "venv\Scripts\python.exe main.py > ..\logs\sensor.txt 2>&1"
    popd
    echo [OK] AgroSensor API started. Log: logs\sensor.txt
    echo      http://127.0.0.1:8000
    timeout /t 2 /nobreak >nul
    exit /b
)
if "%SUBCHOICE%"=="3" (
    for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":8001 "') do taskkill /PID %%P /F >nul 2>&1
    timeout /t 1 /nobreak >nul
    if not exist "%ROOT%logs" mkdir "%ROOT%logs"
    pushd "%ROOT%Crop_Recommendation_Engine"
    start /B "CropAPI" cmd /c "venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8001 > ..\logs\crop_api.txt 2>&1"
    popd
    echo [OK] Crop API started. Log: logs\crop_api.txt
    echo      http://127.0.0.1:8001/docs
    timeout /t 2 /nobreak >nul
    exit /b
)
if "%SUBCHOICE%"=="4" (
    for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":8502 "') do taskkill /PID %%P /F >nul 2>&1
    timeout /t 1 /nobreak >nul
    if not exist "%ROOT%logs" mkdir "%ROOT%logs"
    pushd "%ROOT%AgroSensor"
    start /B "SensorDash" cmd /c "venv\Scripts\python.exe -m http.server 8502 --directory dashboard > ..\logs\sensor_dashboard.txt 2>&1"
    popd
    echo [OK] Sensor Dashboard started.
    echo      Local:  http://127.0.0.1:8502
    echo      Public: https://agroaiapp.me/sensor/dashboard/
    timeout /t 3 /nobreak >nul
    start http://127.0.0.1:8502
    timeout /t 2 /nobreak >nul
    exit /b
)
if "%SUBCHOICE%"=="5" (
    for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":8003 "') do taskkill /PID %%P /F >nul 2>&1
    timeout /t 1 /nobreak >nul
    if not exist "%ROOT%logs" mkdir "%ROOT%logs"
    pushd "%ROOT%Plant_Disease_Detection"
    start /B "Disease" cmd /c "venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8003 > ..\logs\disease.txt 2>&1"
    popd
    echo [OK] Plant Disease Detection API started. Log: logs\disease.txt
    echo      http://127.0.0.1:8003/docs
    timeout /t 2 /nobreak >nul
    exit /b
)
if "%SUBCHOICE%"=="6" (
    for /f "tokens=5" %%P in ('netstat -ano 2^>nul ^| findstr ":7860 "') do taskkill /PID %%P /F >nul 2>&1
    timeout /t 1 /nobreak >nul
    if not exist "%ROOT%logs" mkdir "%ROOT%logs"
    pushd "%ROOT%Plant_Disease_Detection"
    start /B "DiseaseDash" cmd /c "venv\Scripts\python.exe gradio_app.py > ..\logs\disease_dashboard.txt 2>&1"
    popd
    echo [OK] Disease Dashboard ^(Gradio^) started. Log: logs\disease_dashboard.txt
    echo      Local:  http://127.0.0.1:7860
    echo      Public: https://agroaiapp.me/plant-disease/dashboard/
    timeout /t 3 /nobreak >nul
    start http://127.0.0.1:7860
    timeout /t 2 /nobreak >nul
    exit /b
)
if "%SUBCHOICE%"=="7" (
    echo [INFO] API Gateway ^(port 8080^) is managed by keep_tunnel_gateway_alive.bat
    echo        To restart: Close keeper window and run it again
    timeout /t 2 /nobreak >nul
    exit /b
)
if "%SUBCHOICE%"=="8" (
    echo [INFO] Cloudflare Tunnel is managed by keep_tunnel_gateway_alive.bat
    echo        To restart: Close keeper window and run it again
    timeout /t 2 /nobreak >nul
    exit /b
)
if "%SUBCHOICE%"=="0" exit /b

echo [ERROR] Invalid selection.
timeout /t 2 /nobreak >nul
goto start_specific_service


REM =========================================================================
REM [4] CHECK PORT STATUS
REM =========================================================================

:check_port_status
cls
echo ===========================================================
echo                   ACTIVE PORT STATUS
echo ===========================================================
echo.

for %%P in (8000,8001,8002,8003,8080,8502,8505,7860) do (
    netstat -ano | findstr ":%%P " | findstr "LISTENING" >nul
    if !ERRORLEVEL! EQU 0 (
        echo   [OK]   Port %%P is LISTENING
    ) else (
        echo   [ ]   Port %%P is free
    )
)

echo.
echo ===========================================================
echo   Port 8000 = AgroSensor API
echo   Port 8001 = Crop Recommendation API
echo   Port 8002 = Auth Service
echo   Port 8003 = Plant Disease Detection
echo   Port 8080 = API Gateway
echo   Port 8502 = Sensor Dashboard ^(Static^)
echo   Port 8505 = Homepage ^(Landing Page^)
echo   Port 7860 = Disease Dashboard ^(Gradio^)
echo ===========================================================
echo   [NOTE] Crop Dashboard is on Streamlit Cloud ^(no local port^)
echo ===========================================================
echo.
pause >nul
exit /b


REM =========================================================================
REM [5] OPEN CROP DASHBOARD
REM =========================================================================

:open_homepage
cls
echo Opening Homepage...
echo.
echo   Local:  http://127.0.0.1:8505
echo   Public: https://agroaiapp.me
echo.
timeout /t 2 /nobreak >nul
start http://127.0.0.1:8505
exit /b


REM =========================================================================
REM [6] OPEN CROP DASHBOARD
REM =========================================================================

:open_crop_dashboard
cls
echo Opening Crop Dashboard ^(Streamlit Cloud^)...
echo.
echo   Public: https://croprecommendationengine.streamlit.app
echo.
timeout /t 2 /nobreak >nul
start https://croprecommendationengine.streamlit.app
exit /b


REM =========================================================================
REM [7] OPEN API DOCS
REM =========================================================================

:open_api_docs
cls
echo Opening API Documentation...
echo.
echo   Local:  http://127.0.0.1:8080/docs
echo   Public: https://api.agroaiapp.me/docs
echo.
timeout /t 2 /nobreak >nul
start http://127.0.0.1:8080/docs
exit /b


REM =========================================================================
REM [8] VIEW SYSTEM STATUS
REM =========================================================================

:view_system_status
cls
echo ===========================================================
echo                    SYSTEM STATUS DASHBOARD
echo ===========================================================
echo.

REM Check tunnel
tasklist /FI "IMAGENAME eq cloudflared.exe" /FO CSV | find /I "cloudflared" >nul
if %ERRORLEVEL% EQU 0 (
    echo [OK] Cloudflare Tunnel (agromodules-api)
    echo      Domain: https://agroaiapp.me
) else (
    echo [XX] Cloudflare Tunnel - NOT RUNNING
)
echo.

echo   SERVICE STATUS:
echo.
for %%P in (8000,8001,8002,8003,8080,8502,8505,7860) do (
    netstat -ano | findstr ":%%P " | findstr "LISTENING" >nul
    if !ERRORLEVEL! EQU 0 (
        echo   [OK]   Port %%P is LISTENING
    ) else (
        echo   [XX]   Port %%P is offline
    )
)

echo.
echo ===========================================================
echo   ALL DASHBOARDS ^& APIS:
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
echo   SERVICE LOGS ^(in logs\ directory^):
echo     Gateway:    logs\gateway.txt
echo     Auth:       logs\auth.txt
echo     Sensor:     logs\sensor.txt
echo     Crop API:   logs\crop_api.txt
echo     Disease:    logs\disease.txt
echo     Sensor UI:  logs\sensor_dashboard.txt
echo     Disease UI: logs\disease_dashboard.txt
echo ===========================================================
echo.
echo [OPTION] View a log file:
echo   [1] Gateway      [2] Auth        [3] Sensor
echo   [4] Crop API     [5] Disease     [6] Sensor Dashboard
echo   [7] Disease Dashboard            [0] Back to Menu
set "LOGCHOICE="
set /p LOGCHOICE="> Select log to view (or press 0 to return): "

if "%LOGCHOICE%"=="1" (
    if exist "%ROOT%logs\gateway.txt" (
        more "%ROOT%logs\gateway.txt"
    ) else (
        echo [ERROR] Log file not found. Has gateway been started yet?
        timeout /t 2 /nobreak >nul
    )
) else if "%LOGCHOICE%"=="2" (
    if exist "%ROOT%logs\auth.txt" (
        more "%ROOT%logs\auth.txt"
    ) else (
        echo [ERROR] Log file not found. Has auth been started yet?
        timeout /t 2 /nobreak >nul
    )
) else if "%LOGCHOICE%"=="3" (
    if exist "%ROOT%logs\sensor.txt" (
        more "%ROOT%logs\sensor.txt"
    ) else (
        echo [ERROR] Log file not found. Has sensor been started yet?
        timeout /t 2 /nobreak >nul
    )
) else if "%LOGCHOICE%"=="4" (
    if exist "%ROOT%logs\crop_api.txt" (
        more "%ROOT%logs\crop_api.txt"
    ) else (
        echo [ERROR] Log file not found. Has crop API been started yet?
        timeout /t 2 /nobreak >nul
    )
) else if "%LOGCHOICE%"=="5" (
    if exist "%ROOT%logs\disease.txt" (
        more "%ROOT%logs\disease.txt"
    ) else (
        echo [ERROR] Log file not found. Has disease been started yet?
        timeout /t 2 /nobreak >nul
    )
) else if "%LOGCHOICE%"=="6" (
    if exist "%ROOT%logs\sensor_dashboard.txt" (
        more "%ROOT%logs\sensor_dashboard.txt"
    ) else (
        echo [ERROR] Log file not found. Has sensor dashboard been started yet?
        timeout /t 2 /nobreak >nul
    )
) else if "%LOGCHOICE%"=="7" (
    if exist "%ROOT%logs\disease_dashboard.txt" (
        more "%ROOT%logs\disease_dashboard.txt"
    ) else (
        echo [ERROR] Log file not found. Has disease dashboard been started yet?
        timeout /t 2 /nobreak >nul
    )
)

exit /b


REM =========================================================================
REM [9] INITIALIZE DATABASES
REM =========================================================================

:init_databases
cls
echo ===========================================================
echo          INITIALIZE PostgreSQL DATABASES
echo ===========================================================
echo.
echo   This will initialize all PostgreSQL databases (Auth and AgroSensor).
echo.
set /p "CONFIRM=Proceed with database initialization? (yes/no): "

if /i "%CONFIRM%"=="yes" (
    echo.
    echo Initializing databases...
    if exist "%ROOT%init_databases.bat" (
        call "%ROOT%init_databases.bat"
    ) else (
        echo [ERROR] init_databases.bat not found in project root.
    )
    echo.
    echo Database initialization complete.
    echo.
    echo Importing crop recommendation dataset...
    if exist "%ROOT%Crop_Recommendation_Engine\db_import.py" (
        cd /d "%ROOT%Crop_Recommendation_Engine"
        python db_import.py
    ) else (
        echo [ERROR] db_import.py not found.
    )
) else (
    echo Cancelled.
)

echo.
pause >nul
exit /b


REM =========================================================================
REM END OF SCRIPT
REM =========================================================================

endlocal
exit /b 0
