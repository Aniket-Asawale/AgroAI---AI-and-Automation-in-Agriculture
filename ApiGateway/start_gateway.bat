@echo off
echo Starting AgroModules API Gateway on port 8080...
cd /d "%~dp0"
uvicorn main:app --reload --port 8080

