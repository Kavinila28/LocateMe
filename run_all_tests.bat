@echo off
title LocateMe - PyTest Verification Suite
echo ===================================================
echo   Running LocateMe Complete Automated Test Suite...
echo ===================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
pytest tests/ -v
pause
