@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "UVC Lab" ".venv\Scripts\pythonw.exe" -m uvc_lab
) else (
  python -m uvc_lab
)
