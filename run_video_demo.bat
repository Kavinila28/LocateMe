@echo off
title LocateMe - CCTV Video Screening Demo
echo ===================================================
echo   Running LocateMe Video Screening CLI Demo...
echo ===================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python process_video.py --video data/test_videos/sample_cctv_feed.mp4
pause
