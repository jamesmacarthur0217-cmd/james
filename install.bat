@echo off
rem Install VisionLabeler on this PC, or check what is missing (see install.ps1 for the options):
rem   install.bat               install, asking before each step
rem   install.bat -CheckOnly    only report; changes nothing
rem   install.bat -Offline D:\vl-offline
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
pause
