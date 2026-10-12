@echo off
title Flight Price Dashboard + AI Copilot
cd /d "%~dp0"
echo =======================================================
echo   Khoi dong Flight Price Dashboard va AI Copilot...
echo =======================================================
python serve.py
if errorlevel 1 (
    echo.
    echo Khong tim thay Python hoac bi loi. Dang thu mo truc tiep index.html...
    start index.html
    pause
)
