@echo off
setlocal
cd /d "%~dp0"
title Stopping VisionLabeler
where docker >nul 2>&1 || set "PATH=%PATH%;%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin;%ProgramFiles%\Docker\Docker\resources\bin"
echo ============================================
echo   Stopping VisionLabeler
echo ============================================

echo [1/3] Stopping the app (people in view are saved and their labeling finishes first)...
set "APP_PID="
if exist "datapp.pid" set /p APP_PID=<"datapp.pid"
curl -sf -X POST http://127.0.0.1:8000/api/shutdown >nul 2>&1
if errorlevel 1 (
  echo       The app was not running.
  goto app_done
)
rem Wait for the app process itself to exit (not just the web page), up to about 2.5 minutes,
rem so it finishes saving before the database is stopped.
for /l %%i in (1,1,150) do (
  if defined APP_PID (
    tasklist /fi "PID eq %APP_PID%" 2>nul | find "%APP_PID%" >nul || goto app_stopped
  ) else (
    curl -s -o nul http://127.0.0.1:8000/api/status >nul 2>&1 || goto app_stopped
  )
  timeout /t 1 /nobreak >nul
)
echo       The app is still finishing after 2.5 minutes; the database is left running. Check the app window.
goto end
:app_stopped
echo       App stopped.
:app_done

echo [2/3] Stopping the database...
docker compose stop >nul 2>&1
echo       Database stopped (your data is kept).

echo [3/3] Freeing GPU memory used by the vision model...
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" stop qwen3-vl:4b-instruct >nul 2>&1
echo       Done.
echo.
echo VisionLabeler is fully stopped. It is now safe to shut down the PC.
:end
timeout /t 8
