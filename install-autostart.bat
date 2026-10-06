@echo off
rem Start VisionLabeler automatically when you log in to Windows (adds a shortcut to your Startup folder).
rem Undo with remove-autostart.bat.
cd /d "%~dp0"
powershell -NoProfile -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Startup') + '\VisionLabeler.lnk'); $s.TargetPath = '%~dp0start.bat'; $s.WorkingDirectory = '%~dp0'; $s.WindowStyle = 7; $s.Description = 'Start VisionLabeler'; $s.Save()"
if errorlevel 1 (
  echo Could not create the startup shortcut.
) else (
  echo VisionLabeler will now start automatically when you log in.
  echo It starts minimized; the browser opens by itself. To undo, run remove-autostart.bat.
)
pause
