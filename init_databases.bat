@echo off
title AgroModules - PostgreSQL Database Setup
color 0E
echo ===========================================================
echo   AgroModules - PostgreSQL Database Initialization
echo   Creates: agrodb (Auth) + agrosensor (AgroSensor)
echo ===========================================================
echo.

cd /d "%~dp0"

REM --- Auto-detect psql ---
set PSQL_CMD=
where psql >nul 2>&1
if not errorlevel 1 (
    set PSQL_CMD=psql
    echo [OK] psql found in PATH.
    goto :found_psql
)

echo psql not in PATH. Searching common locations...
echo.

REM Search PostgreSQL installation directories
for %%V in (17 16 15 14 13) do (
    if exist "C:\Program Files\PostgreSQL\%%V\bin\psql.exe" (
        set "PSQL_CMD=C:\Program Files\PostgreSQL\%%V\bin\psql.exe"
        echo [OK] Found psql at: !PSQL_CMD!
        goto :found_psql
    )
)

REM Try x86 path
for %%V in (17 16 15 14 13) do (
    if exist "C:\Program Files (x86)\PostgreSQL\%%V\bin\psql.exe" (
        set "PSQL_CMD=C:\Program Files (x86)\PostgreSQL\%%V\bin\psql.exe"
        echo [OK] Found psql at: !PSQL_CMD!
        goto :found_psql
    )
)

REM Try pgAdmin bundled psql
for /d %%D in ("C:\Program Files\pgAdmin 4\*") do (
    if exist "%%D\runtime\psql.exe" (
        set "PSQL_CMD=%%D\runtime\psql.exe"
        echo [OK] Found psql in pgAdmin: !PSQL_CMD!
        goto :found_psql
    )
)

echo [ERROR] psql not found anywhere.
echo.
echo   Please ensure PostgreSQL is installed and provide the path:
echo   Common locations:
echo     C:\Program Files\PostgreSQL\16\bin\
echo     C:\Program Files\PostgreSQL\15\bin\
echo.
echo   Or add PostgreSQL bin to your PATH and run again.
pause
exit /b 1

:found_psql
setlocal enabledelayedexpansion
set PGPASSWORD=proShadow
echo.
echo Using: %PSQL_CMD%
echo.

REM === Create agrodb (Auth Service) ===
echo -------------------------------------------
echo [1/2] Creating database: agrodb (Auth)
echo -------------------------------------------

"%PSQL_CMD%" -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'agrodb'" 2>nul | findstr /C:"1" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   Database 'agrodb' already exists. Skipping.
) else (
    echo   Creating 'agrodb'...
    "%PSQL_CMD%" -U postgres -c "CREATE DATABASE agrodb;"
    if %ERRORLEVEL% EQU 0 (
        echo   [OK] Database 'agrodb' created successfully.
    ) else (
        echo   [ERROR] Failed to create 'agrodb'. Check PostgreSQL credentials.
    )
)
echo.

REM === Create agrosensor (AgroSensor Service) ===
echo -------------------------------------------
echo [2/2] Creating database: agrosensor (AgroSensor)
echo -------------------------------------------

"%PSQL_CMD%" -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'agrosensor'" 2>nul | findstr /C:"1" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   Database 'agrosensor' already exists. Skipping.
) else (
    echo   Creating 'agrosensor'...
    "%PSQL_CMD%" -U postgres -c "CREATE DATABASE agrosensor;"
    if %ERRORLEVEL% EQU 0 (
        echo   [OK] Database 'agrosensor' created successfully.
    ) else (
        echo   [ERROR] Failed to create 'agrosensor'. Check PostgreSQL credentials.
    )
)
echo.

REM === Verify ===
echo -------------------------------------------
echo Verification - Listing databases:
echo -------------------------------------------
"%PSQL_CMD%" -U postgres -l 2>nul | findstr /I "agrodb agrosensor"
echo.

echo ===========================================================
echo   [OK] Database initialization complete.
echo   Tables are auto-created when each service starts.
echo ===========================================================
echo.
