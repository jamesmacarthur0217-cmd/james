<#
Install VisionLabeler on a Windows PC, or check what is missing. Safe to run again: finished steps are skipped.

  install.bat                        install, asking before each big step (double-click it)
  install.bat -CheckOnly             only report what is installed or missing; changes nothing
  install.bat -Offline D:\vl-offline use a package made by tools\pack_offline.ps1 (no internet needed)
  install.bat -Yes                   do not ask before each step (the admin password is still asked)

What it does: checks Windows and the NVIDIA GPU; installs Python 3.13, Docker Desktop and Ollama with winget
if missing; makes the Python environment (.venv) with CUDA PyTorch and requirements.txt; writes .env with a
random database password; starts the database; downloads the models; offers a login account for other devices
and the start-at-login shortcut; finally checks the running app (/api/health).
#>
param([switch]$CheckOnly, [string]$Offline = "", [switch]$Yes)

# Native programs write progress to stderr; with "Stop" Windows PowerShell 5.1 would treat that as a failure.
$ErrorActionPreference = "Continue"
$Root = $PSScriptRoot
Set-Location $Root
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$TorchIndex = "https://download.pytorch.org/whl/cu128"
$Image = "pgvector/pgvector:pg17"
$script:Missing = @()
$script:Stopped = $false

function Say($text, $color = "Gray") { Write-Host $text -ForegroundColor $color }
function Ok($name, $detail) {
  Write-Host ("  [OK]      {0}: {1}" -f $name, $detail) -ForegroundColor Green
  $script:Missing = @($script:Missing | Where-Object { $_ -ne $name })   # fixed during this run
}
function Info($name, $detail) { Write-Host ("  [INFO]    {0}: {1}" -f $name, $detail) -ForegroundColor Gray }
function Miss($name, $detail) {
  Write-Host ("  [MISSING] {0}: {1}" -f $name, $detail) -ForegroundColor Yellow
  $script:Missing += $name
}
function Fail($text) { Write-Host "  $text" -ForegroundColor Red }

function Ask($question) {
  if ($CheckOnly) { return $false }
  if ($Yes) { return $true }
  $a = Read-Host "  $question [y/n]"
  return ($a -match '^(y|yes)$')
}

function Step($title) { Say ""; Say "== $title ==" "Cyan" }

function Run($exe, [string[]]$arguments) {
  # Runs a program in this window and says whether it worked. Out-Host: its output is shown, and is not
  # mixed into this function's answer.
  & $exe @arguments | Out-Host
  return ($LASTEXITCODE -eq 0)
}

function Has-Winget { return [bool](Get-Command winget -ErrorAction SilentlyContinue) }

function Winget-Install($id, $what) {
  if ($Offline) {
    Fail "$what is missing and this is an offline install: install it from its setup file, then run install.bat again."
    return $false
  }
  if (-not (Has-Winget)) {
    Fail "winget is not available: install $what yourself, then run install.bat again."
    return $false
  }
  if (-not (Ask "Install $what now with winget?")) { return $false }
  return (Run "winget" @("install", "--id", $id, "-e", "--accept-package-agreements", "--accept-source-agreements"))
}

function Find-Python {
  if (Get-Command py -ErrorAction SilentlyContinue) {
    $p = & py -3.13 -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $p) { return $p.Trim() }
  }
  foreach ($c in @("$env:LOCALAPPDATA\Programs\Python\Python313\python.exe", "$env:ProgramFiles\Python313\python.exe")) {
    if (Test-Path $c) { return $c }
  }
  return $null
}

function Find-Docker {
  $cmd = Get-Command docker -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  foreach ($c in @("$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe",
                   "$env:ProgramFiles\Docker\Docker\resources\bin\docker.exe")) {
    if (Test-Path $c) { return $c }
  }
  return $null
}

function Find-DockerDesktop {
  foreach ($c in @("$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe",
                   "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe")) {
    if (Test-Path $c) { return $c }
  }
  return $null
}

function Find-Ollama {
  $cmd = Get-Command ollama -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $c = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
  if (Test-Path $c) { return $c }
  return $null
}

function Ollama-Running {
  $v = & curl.exe -s -m 3 http://127.0.0.1:11434/api/version 2>$null
  return ($LASTEXITCODE -eq 0 -and $v -match "version")
}

function Docker-Running($docker) {
  & $docker info *> $null
  return ($LASTEXITCODE -eq 0)
}

function Venv-Report {
  # One line: torch version and whether CUDA works, or an error.
  if (-not (Test-Path $VenvPy)) { return $null }
  $out = & $VenvPy -c "import torch, fastapi, ultralytics, transformers, psycopg, sqlalchemy, cv2; print(torch.__version__, torch.cuda.is_available())" 2>$null
  if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
  return ($out | Select-Object -Last 1).Trim()
}

if ($CheckOnly) { Say "VisionLabeler: checking this PC (nothing is changed)" "Cyan" }
else { Say "VisionLabeler: installing in $Root" "Cyan" }
if ($Offline) {
  $Offline = (Resolve-Path $Offline -ErrorAction SilentlyContinue).Path
  if (-not $Offline -or -not (Test-Path (Join-Path $Offline "models"))) {
    Fail "The offline package folder was not found (it is made by tools\pack_offline.ps1)."
    exit 1
  }
  Say "Using the offline package in $Offline"
}

# ---------------------------------------------------------------------------------------------------
Step "1. Windows and the graphics card"
$os = Get-CimInstance Win32_OperatingSystem
if ([Environment]::Is64BitOperatingSystem -and $os.Caption -match "Windows (10|11)") { Ok "Windows" $os.Caption }
else { Miss "Windows" "$($os.Caption): Windows 10 or 11, 64-bit, is needed" }
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
  $gpu = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null | Select-Object -First 1
  Ok "NVIDIA GPU" "$gpu"
} else {
  Miss "NVIDIA GPU" "no NVIDIA driver found (nvidia-smi). The app needs an NVIDIA GPU with 8 GB or more; install its driver from nvidia.com"
}

# ---------------------------------------------------------------------------------------------------
Step "2. Python environment (.venv)"
$venv = Venv-Report
if ($venv) {
  if ($venv -match "True$") { Ok "Python environment" ".venv with torch $($venv -replace ' True$', '') (CUDA works)" }
  else { Miss "Python environment" ".venv has torch $($venv -replace ' False$', '') but CUDA does not work (CPU PyTorch, or a GPU driver problem)" }
} else {
  $python = Find-Python
  if ($python) { Ok "Python 3.13" $python } else { Miss "Python 3.13" "not installed" }
  if (Test-Path $VenvPy) { Miss "Python environment" ".venv exists but packages are missing or broken" }
  else { Miss "Python environment" "no .venv yet" }
  if (-not $CheckOnly) {
    if (-not $python) {
      if (Winget-Install "Python.Python.3.13" "Python 3.13") { $python = Find-Python }
      if ($python) { Ok "Python 3.13" $python }
    }
    if ($python -and (Ask "Make the Python environment and install the packages (about 4 GB download online)?")) {
      $ok = $true
      if (-not (Test-Path $VenvPy)) { $ok = Run $python @("-m", "venv", ".venv") }
      if ($ok -and $Offline) {
        $wheels = Join-Path $Offline "wheels"
        $ok = Run $VenvPy @("-m", "pip", "install", "--no-index", "--find-links", $wheels, "-r", (Join-Path $wheels "requirements-frozen.txt"))
      } elseif ($ok) {
        $ok = (Run $VenvPy @("-m", "pip", "install", "--upgrade", "pip")) -and
              (Run $VenvPy @("-m", "pip", "install", "torch", "torchvision", "--index-url", $TorchIndex)) -and
              (Run $VenvPy @("-m", "pip", "install", "-r", "requirements.txt"))
      }
      $venv = Venv-Report
      if ($ok -and $venv) { Ok "Python environment" "installed (torch $venv)" } else { Fail "Installing the packages failed; see the messages above." }
    }
  }
}

# ---------------------------------------------------------------------------------------------------
Step "3. Settings file with the database password (.env)"
$envFile = Join-Path $Root ".env"
$envText = ""
if (Test-Path $envFile) { $envText = Get-Content $envFile -Raw }
if ($envText -match "(?m)^POSTGRES_PASSWORD=.+" -and $envText -match "(?m)^DATABASE_URL=.+") {
  Ok ".env" "has the database password and address"
} else {
  Miss ".env" "missing or incomplete"
  if (-not $CheckOnly -and -not (Test-Path $envFile) -and (Ask "Write .env with a new random database password?")) {
    $chars = [char[]]"abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    $bytes = New-Object byte[] 32
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $pw = -join ($bytes | ForEach-Object { $chars[$_ % $chars.Length] })
    $text = "# Local secrets for VisionLabeler. Do not commit or share.`r`nPOSTGRES_PASSWORD=$pw`r`n" +
            "DATABASE_URL=postgresql+psycopg://visionlabeler:$pw@127.0.0.1:5432/visionlabeler`r`n"
    [IO.File]::WriteAllText($envFile, $text, (New-Object Text.UTF8Encoding $false))
    Ok ".env" "written (the password is only in this file)"
  } elseif (-not $CheckOnly -and (Test-Path $envFile)) {
    Fail ".env exists but lacks POSTGRES_PASSWORD or DATABASE_URL; it is not overwritten (an existing database uses it). Fix it by hand."
  }
}

# ---------------------------------------------------------------------------------------------------
Step "4. Ollama (runs the vision model)"
$ollama = Find-Ollama
if ($ollama) {
  if (Ollama-Running) { Ok "Ollama" "installed and running" } else { Info "Ollama" "installed, not running now (it starts with Windows; start.bat starts it too)" }
} else {
  Miss "Ollama" "not installed"
  if (-not $CheckOnly -and (Winget-Install "Ollama.Ollama" "Ollama")) {
    $ollama = Find-Ollama
    if ($ollama) { Ok "Ollama" "installed" }
  }
}

# ---------------------------------------------------------------------------------------------------
Step "5. Docker Desktop and the database"
$docker = Find-Docker
$dbReady = $false
if (-not $docker) {
  Miss "Docker Desktop" "not installed"
  if (-not $CheckOnly -and (Winget-Install "Docker.DockerDesktop" "Docker Desktop")) {
    Say "  Docker Desktop is installed. Windows may ask to restart. Then open Docker Desktop once, accept its terms," "Yellow"
    Say "  wait until it says 'Engine running', and run install.bat again to finish." "Yellow"
    $script:Stopped = $true
  }
} else {
  Ok "Docker Desktop" $docker
  $running = Docker-Running $docker
  if (-not $running -and -not $CheckOnly) {
    $desktop = Find-DockerDesktop
    if ($desktop -and (Ask "Docker is not running. Start Docker Desktop and wait for it (up to 3 minutes)?")) {
      Start-Process $desktop
      for ($i = 0; $i -lt 90 -and -not $running; $i++) { Start-Sleep -Seconds 2; $running = Docker-Running $docker }
    }
  }
  if (-not $running) {
    if ($CheckOnly) { Info "Docker engine" "not running now (start.bat starts it), so the database was not checked" }
    else { Miss "Docker engine" "not running: open Docker Desktop, wait for 'Engine running', run install.bat again" }
  } else {
    Ok "Docker engine" "running"
    & $docker image inspect $Image *> $null
    $hasImage = ($LASTEXITCODE -eq 0)
    if ($hasImage) { Ok "Database image" $Image } else { Miss "Database image" "$Image not downloaded yet" }
    $state = & $docker inspect -f "{{.State.Running}}" visionlabeler-db 2>$null
    if ($state -eq "true") { Ok "Database" "container visionlabeler-db is running"; $dbReady = $true }
    elseif ($CheckOnly) { Info "Database" "container not running now (start.bat starts it)" }
    if (-not $CheckOnly -and -not $dbReady -and (Test-Path $envFile)) {
      if (-not $hasImage -and $Offline) {
        $tar = Join-Path $Offline "docker\pgvector-pg17.tar"
        if (Test-Path $tar) { [void](Run $docker @("load", "-i", $tar)) } else { Fail "No $tar in the offline package." }
      }
      if (Ask "Start the database (docker compose up)?") {
        $dbReady = Run $docker @("compose", "up", "-d", "--wait")
        if ($dbReady) { Ok "Database image" $Image; Ok "Database" "started" }
        else { Fail "The database did not start; see the messages above." }
      }
    }
  }
}

# ---------------------------------------------------------------------------------------------------
Step "6. Models (detector, appearance and age models, vision model)"
if (-not (Venv-Report)) {
  Info "Models" "checked after the Python environment is installed"
} else {
  & $VenvPy tools\models.py check
  $modelsOk = ($LASTEXITCODE -eq 0)
  if ($modelsOk) { Ok "Models" "all on this PC" }
  else {
    Miss "Models" "some are missing (above)"
    if (-not $CheckOnly -and $Offline) {
      if (Ask "Copy the models from the offline package?") { [void](Run $VenvPy @("tools\models.py", "unpack", $Offline)) }
    } elseif (-not $CheckOnly -and (Ask "Download the missing models (about 5 GB)?")) {
      [void](Run $VenvPy @("tools\models.py", "download"))
      $vlm = (& $VenvPy -c "import sys; sys.path.insert(0, 'tools'); import models; print(models.names()['ollama'])").Trim()
      if ($ollama) {
        if (-not (Ollama-Running)) {
          $app = "$env:LOCALAPPDATA\Programs\Ollama\ollama app.exe"
          if (Test-Path $app) { Start-Process $app; for ($i = 0; $i -lt 30 -and -not (Ollama-Running); $i++) { Start-Sleep 1 } }
        }
        [void](Run $ollama @("pull", $vlm))
      } else { Fail "Ollama is not installed, so $vlm could not be downloaded." }
    }
    if (-not $CheckOnly) { & $VenvPy tools\models.py check | Out-Null; if ($LASTEXITCODE -eq 0) { Ok "Models" "all on this PC now" } }
  }
}

# ---------------------------------------------------------------------------------------------------
Step "6b. FFmpeg (records video while people are in view)"
if (-not (Venv-Report)) {
  Info "FFmpeg" "checked after the Python environment is installed"
} else {
  & $VenvPy tools\get_ffmpeg.py check | Out-Null
  if ($LASTEXITCODE -eq 0) { Ok "FFmpeg" "ready (data\ffmpeg)" }
  else {
    Miss "FFmpeg" "not set up: nothing is recorded until it is"
    if (-not $CheckOnly -and $Offline -and (Test-Path (Join-Path $Offline "ffmpeg\ffmpeg.exe"))) {
      if (Ask "Copy FFmpeg from the offline package?") { [void](Run $VenvPy @("tools\get_ffmpeg.py", "unpack", $Offline)) }
    } elseif (-not $CheckOnly -and (Ask "Download FFmpeg (LGPL build from github.com/BtbN/FFmpeg-Builds, about 170 MB)?")) {
      [void](Run $VenvPy @("tools\get_ffmpeg.py", "download"))
    }
    if (-not $CheckOnly) { & $VenvPy tools\get_ffmpeg.py check | Out-Null; if ($LASTEXITCODE -eq 0) { Ok "FFmpeg" "ready now" } }
  }
}

# ---------------------------------------------------------------------------------------------------
Step "6c. MediaMTX (only for cameras sent from a phone or another PC)"
if (-not (Venv-Report)) {
  Info "MediaMTX" "checked after the Python environment is installed"
} else {
  & $VenvPy tools\get_mediamtx.py check | Out-Null
  if ($LASTEXITCODE -eq 0) { Ok "MediaMTX" "ready (data\mediamtx)" }
  else {
    Info "MediaMTX" "not set up (only needed for cameras sent from a phone or another PC)"
    if (-not $CheckOnly -and $Offline -and (Test-Path (Join-Path $Offline "mediamtx\mediamtx.exe"))) {
      if (Ask "Copy MediaMTX from the offline package?") { [void](Run $VenvPy @("tools\get_mediamtx.py", "unpack", $Offline)) }
    } elseif (-not $CheckOnly -and (Ask "Download MediaMTX (MIT license, from github.com/bluenviron/mediamtx, about 29 MB)?")) {
      [void](Run $VenvPy @("tools\get_mediamtx.py", "download"))
    }
  }
}

# ---------------------------------------------------------------------------------------------------
Step "7. Login account for other devices (optional)"
if (-not $dbReady -or -not (Venv-Report)) {
  Info "Accounts" "checked when the database is running"
} else {
  # Counts accounts without changing anything (no tables are created here).
  $count = & $VenvPy -c "from labeler.config import load_config; from sqlalchemy import create_engine, text; e = create_engine(load_config()['database']['url']); c = e.connect(); print(c.execute(text('select count(*) from users')).scalar())" 2>$null
  if ($LASTEXITCODE -eq 0 -and [int]$count -gt 0) { Ok "Accounts" "$count login account(s)" }
  else {
    Info "Accounts" "none yet: this PC never needs one; phones and other PCs do (to log in)"
    if (-not $CheckOnly -and (Ask "Create an admin login account now?")) {
      $name = Read-Host "  Account name (Enter = admin)"
      if (-not $name) { $name = "admin" }
      [void](Run $VenvPy @("tools\users.py", "add", $name, "--role", "admin"))
    }
  }
}

# ---------------------------------------------------------------------------------------------------
Step "8. Start when you log in to Windows (optional)"
$shortcut = Join-Path ([Environment]::GetFolderPath("Startup")) "VisionLabeler.lnk"
if (Test-Path $shortcut) { Ok "Start at login" "on (remove-autostart.bat turns it off)" }
else {
  Info "Start at login" "off"
  if (-not $CheckOnly -and (Ask "Start VisionLabeler automatically when you log in?")) {
    $s = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcut)
    $s.TargetPath = Join-Path $Root "start.bat"; $s.WorkingDirectory = $Root; $s.WindowStyle = 7
    $s.Description = "Start VisionLabeler"; $s.Save()
    Ok "Start at login" "on"
  }
}

# ---------------------------------------------------------------------------------------------------
Step "9. Health check of the running app"
$health = & curl.exe -s -m 15 http://127.0.0.1:8000/api/health 2>$null
if (-not $health -and -not $CheckOnly -and $script:Missing.Count -eq 0 -and -not $script:Stopped -and
    (Ask "Start VisionLabeler now (start.bat) and check it?")) {
  Start-Process (Join-Path $Root "start.bat")
  Say "  Waiting for the app (up to 3 minutes)..."
  for ($i = 0; $i -lt 90 -and -not $health; $i++) {
    Start-Sleep -Seconds 2
    $health = & curl.exe -s -m 15 http://127.0.0.1:8000/api/health 2>$null
  }
}
if ($health) {
  try { $h = $health | ConvertFrom-Json } catch { $h = $null }
  if ($h -and $h.ok) { Ok "App" "running, everything works" }
  elseif ($h) { Miss "App" "running, but: $($h.problems -join '; ')" }
  else { Info "App" "answers, but not with a health check (an older version?)" }
  if ($h) { foreach ($w in $h.warnings) { Info "App" "worth a look: $w" } }
} else {
  Info "App" "not running now (start it with start.bat)"
}

# ---------------------------------------------------------------------------------------------------
Say ""
if ($script:Missing.Count -eq 0 -and -not $script:Stopped) {
  if ($CheckOnly) { Say "Everything needed is installed." "Green" }
  else { Say "VisionLabeler is installed. Start it with start.bat; it opens http://127.0.0.1:8000" "Green" }
  exit 0
}
if ($CheckOnly) { Say ("Missing or not working: {0}. Run install.bat to install them." -f ($script:Missing -join ", ")) "Yellow" }
else { Say ("Not finished: {0}. Fix what the messages above say, then run install.bat again." -f ($script:Missing -join ", ")) "Yellow" }
exit 1
