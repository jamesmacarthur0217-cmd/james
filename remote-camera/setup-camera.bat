@echo off
rem VisionLabeler: send the webcam of this PC to the VisionLabeler PC through Tailscale (see README.txt).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-camera.ps1" %*
echo.
pause
