@echo off
title LocateMe - FastAPI Backend Server
echo ===================================================
echo   Starting LocateMe FastAPI Backend Server...
echo ===================================================
echo API Docs will be available at: http://127.0.0.1:8000/docs
cd /d "%~dp0"
call .venv\Scripts\activate.bat
uvicorn api.main:app --reload --port 8000
pause
