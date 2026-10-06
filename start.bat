@echo off
setlocal
cd /d "%~dp0"
title VisionLabeler
echo ============================================
echo   Starting VisionLabeler
echo ============================================

rem Docker's command-line tool, in case this window's PATH does not have it yet.
where docker >nul 2>&1 || set "PATH=%PATH%;%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin;%ProgramFiles%\Docker\Docker\resources\bin"

rem 1. Docker Desktop (holds the database). It starts with Windows but needs about a minute after boot.
docker info >nul 2>&1
if not errorlevel 1 goto docker_ready
echo [1/3] Waiting for Docker Desktop (can take a minute after the PC starts)...
tasklist /fi "imagename eq Docker Desktop.exe" | find /i "Docker Desktop.exe" >nul
if errorlevel 1 (
  if exist "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe" start "" "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe"
  if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
)
for /l %%i in (1,1,90) do (
  docker info >nul 2>&1 && goto docker_ready
  timeout /t 2 /nobreak >nul
)
echo.
echo Docker did not become ready within 3 minutes.
echo Open Docker Desktop, wait until it says "Engine running", then run start.bat again.
pause
exit /b 1
:docker_ready
echo [1/3] Docker is ready

rem 2. The database container.
docker compose up -d --wait >nul 2>&1
if errorlevel 1 (
  echo Could not start the database container. Check Docker Desktop for errors.
  pause
  exit /b 1
)
echo [2/3] Database is ready

rem 3. Ollama (runs the vision model). It starts with Windows; start it if it is not running.
curl -sf http://localhost:11434/api/version >nul 2>&1
if not errorlevel 1 goto ollama_ready
echo [3/3] Starting Ollama...
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe" start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
for /l %%i in (1,1,30) do (
  curl -sf http://localhost:11434/api/version >nul 2>&1 && goto ollama_ready
  timeout /t 1 /nobreak >nul
)
echo [3/3] Ollama is not responding: the app will run with colors only and say so in the log.
goto run
:ollama_ready
echo [3/3] Ollama is ready

:run
echo.
echo Starting the app. The browser opens in about 15 seconds: http://127.0.0.1:8000
echo To stop: the Stop button in the web page, stop.bat, or Ctrl+C in this window.
echo.
if not "%VL_NO_BROWSER%"=="1" start "" cmd /c "timeout /t 15 >nul & start http://127.0.0.1:8000"
rem The supervisor restarts the app if it crashes; a normal stop ends it.
.venv\Scripts\python.exe -m labeler.supervisor %*
echo.
echo VisionLabeler has stopped.
pause
