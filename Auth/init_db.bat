@echo off
setlocal enabledelayedexpansion
echo ===========================================
echo   AgroModules - PostgreSQL DB Setup (Auth)
echo ===========================================
echo.

REM --- Auto-detect psql ---
set PSQL_CMD=
where psql >nul 2>&1
if not errorlevel 1 (
    set PSQL_CMD=psql
    echo [OK] psql found in PATH.
    goto :found_psql
)

echo psql not in PATH. Searching common locations...

for %%V in (17 16 15 14 13) do (
    if exist "C:\Program Files\PostgreSQL\%%V\bin\psql.exe" (
        set "PSQL_CMD=C:\Program Files\PostgreSQL\%%V\bin\psql.exe"
        echo [OK] Found psql at: !PSQL_CMD!
        goto :found_psql
    )
)

for %%V in (17 16 15 14 13) do (
    if exist "C:\Program Files (x86)\PostgreSQL\%%V\bin\psql.exe" (
        set "PSQL_CMD=C:\Program Files (x86)\PostgreSQL\%%V\bin\psql.exe"
        echo [OK] Found psql at: !PSQL_CMD!
        goto :found_psql
    )
)

echo [ERROR] psql not found in PATH or common locations.
echo   Add PostgreSQL bin folder to your PATH, e.g.:
echo   C:\Program Files\PostgreSQL\16\bin
echo.
echo   Or run init_databases.bat from the project root.
echo.
pause
exit /b 1

:found_psql
echo.
echo Creating database 'agrodb' if it does not exist...
echo.

"%PSQL_CMD%" -U postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'agrodb'" 2>nul | findstr /C:"1" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   Database 'agrodb' already exists. Skipping creation.
) else (
    "%PSQL_CMD%" -U postgres -c "CREATE DATABASE agrodb;"
    if %ERRORLEVEL% EQU 0 (
        echo   [OK] Database 'agrodb' created successfully.
    ) else (
        echo   [ERROR] Failed to create database. Check PostgreSQL credentials.
    )
)

echo.
echo Done. Tables will be auto-created when the Auth service starts.
echo.
pause
