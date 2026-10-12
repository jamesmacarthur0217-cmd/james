@echo off
rem VisionLabeler: stop sending this PC's webcam and remove everything setup-camera.bat added.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-camera.ps1" -Uninstall
echo.
pause
