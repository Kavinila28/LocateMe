@echo off
title LocateMe - Streamlit Operator Dashboard
echo ===================================================
echo   Starting LocateMe Streamlit Operator Dashboard...
echo ===================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app/dashboard.py
pause
