<#
Build an offline package: everything install.ps1 downloads, in one folder, for a PC without internet.

  powershell -ExecutionPolicy Bypass -File tools\pack_offline.ps1 -Out D:\vl-offline
  options: -SkipWheels  -SkipDocker  -SkipModels   (to build or refresh only some parts)

Run it on a PC where VisionLabeler works (this one): the Python packages are the exact versions installed in
.venv (pip freeze), downloaded once more as files (about 4 GB, needs internet); the models (detector, SigLIP 2,
age model, the Ollama vision model) and the database image are copied from this PC (Docker must be running for
the image). About 10 GB in total.

On the new PC: copy the VisionLabeler folder and this package, install Python 3.13 (64-bit), Docker Desktop and
Ollama from their setup files (not included), then run:  install.bat -Offline D:\vl-offline
The new PC must have the same Python (3.13, 64-bit Windows): the packages are made for it.
#>
param([Parameter(Mandatory = $true)][string]$Out, [switch]$SkipWheels, [switch]$SkipDocker, [switch]$SkipModels)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
$Image = "pgvector/pgvector:pg17"
$failed = @()

function Say($text, $color = "Gray") { Write-Host $text -ForegroundColor $color }

if (-not (Test-Path $VenvPy)) { Say "No .venv here: run this on a PC where VisionLabeler is installed." "Red"; exit 1 }
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path
Say "Building the offline package in $Out" "Cyan"

if (-not $SkipWheels) {
  Say ""; Say "== Python packages (exact versions from .venv) ==" "Cyan"
  $wheels = Join-Path $Out "wheels"
  New-Item -ItemType Directory -Force $wheels | Out-Null
  $frozen = Join-Path $wheels "requirements-frozen.txt"
  # Packages installed from a local folder or URL ("name @ file:...") cannot be downloaded again.
  $lines = & $VenvPy -m pip freeze --exclude-editable | Where-Object { $_ -and $_ -notmatch " @ " }
  [IO.File]::WriteAllLines($frozen, [string[]]$lines, (New-Object Text.UTF8Encoding $false))
  Say ("  {0} packages listed in requirements-frozen.txt" -f $lines.Count)
  # torch +cu128 builds are on the PyTorch index; everything else on PyPI.
  & $VenvPy -m pip download -r $frozen -d $wheels --only-binary=:all: `
      --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple | Out-Host
  if ($LASTEXITCODE -ne 0) { $failed += "Python packages" }
}

if (-not $SkipDocker) {
  Say ""; Say "== Database image ==" "Cyan"
  $docker = (Get-Command docker -ErrorAction SilentlyContinue).Source
  if (-not $docker) { $docker = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe" }
  & $docker image inspect $Image *> $null
  if ($LASTEXITCODE -ne 0) {
    Say "  $Image is not on this PC, or Docker is not running (start Docker Desktop and try again)." "Red"
    $failed += "database image"
  } else {
    New-Item -ItemType Directory -Force (Join-Path $Out "docker") | Out-Null
    & $docker save -o (Join-Path $Out "docker\pgvector-pg17.tar") $Image | Out-Host
    if ($LASTEXITCODE -ne 0) { $failed += "database image" } else { Say "  saved docker\pgvector-pg17.tar" }
  }
}

if (-not $SkipModels) {
  Say ""; Say "== Models ==" "Cyan"
  & $VenvPy tools\models.py pack $Out | Out-Host
  if ($LASTEXITCODE -ne 0) { $failed += "models" }
}

$readme = @"
VisionLabeler offline package (made $(Get-Date -Format "yyyy-MM-dd") on $env:COMPUTERNAME).

On the new PC:
1. Install Python 3.13 (64-bit), Docker Desktop and Ollama from their setup files.
2. Copy the VisionLabeler folder (the program) to the new PC, e.g. C:\VisionLabeler.
3. Run:  install.bat -Offline "<this folder>"

wheels\   Python packages (pip install --no-index), exact versions of the PC that made this package
models\   detector, Hugging Face models, the Ollama vision model
docker\   the database image (docker load)
"@
[IO.File]::WriteAllText((Join-Path $Out "README.txt"), $readme, (New-Object Text.UTF8Encoding $false))

$gb = (Get-ChildItem $Out -Recurse -File | Measure-Object Length -Sum).Sum / 1e9
Say ""
if ($failed.Count) { Say ("Finished with problems: {0}. Package so far: {1:N1} GB in {2}" -f ($failed -join ", "), $gb, $Out) "Yellow"; exit 1 }
Say ("Offline package ready: {0:N1} GB in {1}" -f $gb, $Out) "Green"
exit 0
