@echo off
rem Stop VisionLabeler from starting automatically at login.
set "LINK=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\VisionLabeler.lnk"
if exist "%LINK%" (
  del "%LINK%"
  echo VisionLabeler will no longer start automatically.
) else (
  echo VisionLabeler was not set to start automatically.
)
pause
