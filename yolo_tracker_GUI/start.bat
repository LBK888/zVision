@echo off
title Multi-Camera Timelapse Analyzer Launcher
echo Starting Multi-Camera Timelapse Analyzer...
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with error code %ERRORLEVEL%.
    pause
) else (
    echo.
    echo Application closed.
    pause
)
