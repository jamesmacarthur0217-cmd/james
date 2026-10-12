<#
VisionLabeler: a camera at another site. Run on the OTHER PC (the one whose webcam should be watched), not on the
VisionLabeler PC. It sends this PC's webcam to the VisionLabeler PC through Tailscale:

  webcam -> FFmpeg (H.264) -> MediaMTX (RTSP on port 8554, password, Tailscale only) -> VisionLabeler PC

  setup-camera.bat              set up (asks before each download and before changing the firewall)
  remove-camera.bat             remove it all again (the same as setup-camera.bat -Uninstall)

What it does: checks Tailscale; downloads MediaMTX (28.6 MB, MIT license) and FFmpeg (170.5 MB, LGPL), each
checked against its published checksum; finds the webcam and tests it; picks an H.264 encoder that works here;
writes the settings with a new random password; lets only Tailscale devices reach port 8554 in the Windows firewall
(asks for administrator once); starts it now and with Windows (no window); prints the camera address to enter on
the VisionLabeler PC (Settings > Cameras > Add camera). Everything is kept in %LOCALAPPDATA%\VisionLabelerCamera.

Options for testing: -InstallDir <folder> -MediaMtxZip <zip> -FfmpegDir <folder with ffmpeg.exe>
-TestVideo <file instead of the webcam> -Camera <webcam name> -NoFirewall -NoAutostart -Yes
This file must stay plain ASCII: Windows PowerShell 5.1 reads scripts as ANSI.
#>
param([string]$InstallDir = "$env:LOCALAPPDATA\VisionLabelerCamera", [string]$MediaMtxZip = "", [string]$FfmpegDir = "",
      [string]$TestVideo = "", [string]$Camera = "", [switch]$NoFirewall, [switch]$NoAutostart, [switch]$Uninstall,
      [switch]$Yes)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is many times slower with the progress bar

$MediaMtxRelease = "https://github.com/bluenviron/mediamtx/releases/download/v1.21.2/"
$MediaMtxZipName = "mediamtx_v1.21.2_windows_amd64.zip"
$FfmpegRelease = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
$FfmpegZipName = "ffmpeg-n8.1-latest-win64-lgpl-8.1.zip"
$Port = 8554
$RuleName = "VisionLabeler camera (Tailscale)"
$StartupLink = Join-Path ([Environment]::GetFolderPath("Startup")) "VisionLabeler camera.lnk"

function Say($text, $color = "Gray") { Write-Host $text -ForegroundColor $color }
function Step($title) { Say ""; Say "== $title ==" "Cyan" }
function Ask($question) {
  if ($Yes) { return $true }
  $answer = Read-Host "$question [y/n]"
  return $answer -match "^(y|yes)$"
}
function Stop-Here($text) { Say ""; Say $text "Red"; Say ""; exit 1 }

function Run-Elevated($command) {
  # One command as administrator (Windows asks); -EncodedCommand avoids any quoting problems.
  $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
  $p = Start-Process powershell -Verb RunAs -Wait -PassThru -ArgumentList "-NoProfile -EncodedCommand $encoded"
  return ($p.ExitCode -eq 0)
}

function Stop-Camera {
  Get-Process mediamtx -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$InstallDir\*" } | Stop-Process -Force
  Get-Process ffmpeg -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$InstallDir\*" } | Stop-Process -Force
}

function Download-Checked($release, $name, $out) {
  $sums = (Invoke-WebRequest -UseBasicParsing ($release + "checksums.sha256")).Content
  if ($sums -is [byte[]]) { $sums = [Text.Encoding]::ASCII.GetString($sums) }
  $want = $null
  foreach ($line in ($sums -split "`n")) {
    $parts = $line.Trim() -split "\s+"
    if ($parts.Count -eq 2 -and $parts[1].TrimStart("*") -eq $name) { $want = $parts[0].ToLower() }
  }
  if (-not $want) { Stop-Here "$name is not in the release's checksums: the release changed. Ask for an updated script." }
  Say "  downloading $name ..."
  Invoke-WebRequest -UseBasicParsing ($release + $name) -OutFile $out
  $got = (Get-FileHash $out -Algorithm SHA256).Hash.ToLower()
  if ($got -ne $want) { Remove-Item $out -Force; Stop-Here "The download of $name does not match its checksum: not used." }
  Say "  checksum OK"
}

function Extract-One($zip, $entryEnd, $out) {
  # One file from a zip (no paths from the archive are used).
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  $z = [IO.Compression.ZipFile]::OpenRead($zip)
  try {
    $entry = $z.Entries | Where-Object { $_.FullName.EndsWith($entryEnd) } | Select-Object -First 1
    if (-not $entry) { Stop-Here "$entryEnd is not in $zip" }
    [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $out, $true)
  } finally { $z.Dispose() }
}

function Ffmpeg-Ok([string[]]$arguments) {
  # $true when FFmpeg runs these arguments without an error (its output is not shown). Start-Process in
  # PowerShell 5.1 does not quote arguments with spaces (a webcam called "Integrated Camera"), so they are quoted here.
  $line = ($arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join " "
  $p = Start-Process -FilePath $Ffmpeg -ArgumentList $line -NoNewWindow -Wait -PassThru `
         -RedirectStandardError (Join-Path $env:TEMP "vl-camera-ffmpeg.txt") -RedirectStandardOutput (Join-Path $env:TEMP "vl-camera-out.txt")
  return ($p.ExitCode -eq 0)
}

function Quote-Arg($text) { return '"' + $text + '"' }

# ---------------------------------------------------------------------------------------------------
if ($Uninstall) {
  Step "Removing the VisionLabeler camera"
  Stop-Camera
  if (Test-Path $StartupLink) { Remove-Item $StartupLink -Force; Say "  no longer starts with Windows" }
  if (Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue) {
    Say "  removing the firewall rule (Windows asks for administrator)"
    [void](Run-Elevated "Remove-NetFirewallRule -DisplayName '$RuleName'")
  }
  if (Test-Path $InstallDir) { Remove-Item $InstallDir -Recurse -Force; Say "  removed $InstallDir" }
  Say "Done: this PC no longer sends its camera." "Green"
  exit 0
}

Say "VisionLabeler camera: sends this PC's webcam to the VisionLabeler PC through Tailscale." "Cyan"

# ---------------------------------------------------------------------------------------------------
Step "1. Tailscale"
$Tailscale = "$env:ProgramFiles\Tailscale\tailscale.exe"
if (-not (Test-Path $Tailscale)) { Stop-Here "Tailscale is not installed. Install it from https://tailscale.com/download, sign in, then run this again." }
$TsIp = ((& $Tailscale ip -4) 2>$null | Select-Object -First 1)
if (-not $TsIp -or -not $TsIp.StartsWith("100.")) { Stop-Here "Tailscale is not connected. Open Tailscale, make sure it says Connected (on the VisionLabeler owner's network), then run this again." }
Say "  connected; this PC is $TsIp" "Green"

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$Ffmpeg = Join-Path $InstallDir "ffmpeg.exe"
$Mediamtx = Join-Path $InstallDir "mediamtx.exe"
Stop-Camera   # a running copy keeps its files open

# ---------------------------------------------------------------------------------------------------
Step "2. MediaMTX (the stream server, MIT license)"
if (Test-Path $Mediamtx) { Say "  already here" "Green" }
else {
  $zip = $MediaMtxZip
  if (-not $zip) {
    if (-not (Ask "Download MediaMTX v1.21.2 from github.com/bluenviron/mediamtx (28.6 MB)?")) { Stop-Here "MediaMTX is needed. Nothing was changed." }
    $zip = Join-Path $env:TEMP $MediaMtxZipName
    Download-Checked $MediaMtxRelease $MediaMtxZipName $zip
  }
  Extract-One $zip "mediamtx.exe" $Mediamtx
  Extract-One $zip "LICENSE" (Join-Path $InstallDir "LICENSE-mediamtx.txt")
  if (-not $MediaMtxZip) { Remove-Item $zip -Force }
  Say "  ready" "Green"
}

# ---------------------------------------------------------------------------------------------------
Step "3. FFmpeg (reads the webcam and makes H.264, LGPL)"
if ($FfmpegDir) {
  Copy-Item (Join-Path $FfmpegDir "ffmpeg.exe") $Ffmpeg -Force
  if (Test-Path (Join-Path $FfmpegDir "LICENSE.txt")) { Copy-Item (Join-Path $FfmpegDir "LICENSE.txt") (Join-Path $InstallDir "LICENSE-ffmpeg.txt") -Force }
}
if (Test-Path $Ffmpeg) { Say "  ready" "Green" }
else {
  if (-not (Ask "Download FFmpeg 8.1 (LGPL build) from github.com/BtbN/FFmpeg-Builds (170.5 MB)?")) { Stop-Here "FFmpeg is needed. Nothing else was changed." }
  $zip = Join-Path $env:TEMP $FfmpegZipName
  Download-Checked $FfmpegRelease $FfmpegZipName $zip
  Extract-One $zip "bin/ffmpeg.exe" $Ffmpeg
  Extract-One $zip "LICENSE.txt" (Join-Path $InstallDir "LICENSE-ffmpeg.txt")
  Remove-Item $zip -Force
  Say "  ready" "Green"
}

# ---------------------------------------------------------------------------------------------------
Step "4. H.264 encoder"
$Encoder = $null
foreach ($enc in @("h264_nvenc", "h264_mf", "libopenh264")) {
  if (Ffmpeg-Ok @("-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=256x144:rate=10:duration=1", "-c:v", $enc, "-f", "null", "-")) {
    $Encoder = $enc; break
  }
}
if (-not $Encoder) { Stop-Here "No H.264 encoder works on this PC." }
Say "  $Encoder" "Green"
$EncoderArgs = @{ "h264_nvenc" = "-preset p4 -tune ll -rc cbr"; "h264_mf" = "-rate_control cbr"; "libopenh264" = "" }[$Encoder]

# ---------------------------------------------------------------------------------------------------
Step "5. Webcam"
if ($TestVideo) {
  $Source = "-re -stream_loop -1 -i " + (Quote-Arg (Resolve-Path $TestVideo).Path)
  Say "  test video instead of the webcam: $TestVideo" "Yellow"
} else {
  [void](Ffmpeg-Ok @("-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"))
  $lines = Get-Content (Join-Path $env:TEMP "vl-camera-ffmpeg.txt")
  $cams = @($lines | Where-Object { $_ -match '"(.+)" \(video\)' } | ForEach-Object { [regex]::Match($_, '"(.+)" \(video\)').Groups[1].Value })
  if ($Camera) { $name = $Camera }
  elseif ($cams.Count -eq 0) { Stop-Here "No webcam found. Plug it in (and close apps that use it), then run this again." }
  elseif ($cams.Count -eq 1) { $name = $cams[0] }
  else {
    for ($i = 0; $i -lt $cams.Count; $i++) { Say ("  {0}. {1}" -f ($i + 1), $cams[$i]) }
    $pick = Read-Host "Which webcam (number)?"
    $name = $cams[[int]$pick - 1]
  }
  Say "  $name"
  if ($name -match '[%^&|<>"]') { Stop-Here "The webcam's name has a character this script cannot pass on: $name" }
  # 1280x720 at 15 frames per second if the webcam offers it, else its own default.
  $sized = @("-f", "dshow", "-rtbufsize", "100M", "-video_size", "1280x720", "-framerate", "15", "-i", "video=$name")
  $plain = @("-f", "dshow", "-rtbufsize", "100M", "-i", "video=$name")
  $test = @("-t", "3", "-f", "null", "-")
  Say "  testing it for 3 seconds (close Teams, Zoom or the camera app if it fails) ..."
  if (Ffmpeg-Ok (@("-hide_banner", "-loglevel", "error") + $sized + $test)) {
    $Source = "-f dshow -rtbufsize 100M -video_size 1280x720 -framerate 15 -i " + (Quote-Arg "video=$name")
    Say "  works at 1280x720, 15 fps" "Green"
  } elseif (Ffmpeg-Ok (@("-hide_banner", "-loglevel", "error") + $plain + $test)) {
    $Source = "-f dshow -rtbufsize 100M -i " + (Quote-Arg "video=$name")
    Say "  works (its own picture size)" "Green"
  }
  else { Stop-Here ("The webcam could not be read: " + ((Get-Content (Join-Path $env:TEMP "vl-camera-ffmpeg.txt") | Select-Object -Last 3) -join " ")) }
}

# ---------------------------------------------------------------------------------------------------
Step "6. Settings"
$AddressFile = Join-Path $InstallDir "camera-address.txt"
$Password = $null
if (Test-Path $AddressFile) {
  $m = [regex]::Match((Get-Content $AddressFile -Raw), "rtsp://visionlabeler:([A-Za-z0-9]+)@")
  if ($m.Success) { $Password = $m.Groups[1].Value; Say "  keeping the password made before" }
}
if (-not $Password) {
  $chars = [char[]]"ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789"
  $bytes = New-Object byte[] 24
  [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
  $Password = -join ($bytes | ForEach-Object { $chars[$_ % $chars.Length] })
}
$Hash = [Convert]::ToBase64String([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($Password)))

$StreamCmd = Join-Path $InstallDir "stream.cmd"
$cmd = "@echo off`r`n" +
  "rem Started by MediaMTX: the webcam as H.264 to MediaMTX on this PC (setup-camera.ps1 writes this file).`r`n" +
  (Quote-Arg $Ffmpeg) + " -hide_banner -loglevel error $Source -an -vf " + (Quote-Arg "scale='min(1280,iw)':-2,fps=15") +
  " -c:v $Encoder $EncoderArgs -b:v 1500k -maxrate 2000k -bufsize 2000k -g 30 -pix_fmt yuv420p" +
  " -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:$Port/cam`r`n"
[IO.File]::WriteAllText($StreamCmd, $cmd, (New-Object Text.ASCIIEncoding))

$yml = @"
# Written by setup-camera.ps1 (VisionLabeler). Run setup-camera.bat again to change it.
logLevel: warn
logDestinations: [file]
logFile: mediamtx.log
authMethod: internal
authInternalUsers:
  - user: any
    pass:
    ips: ["127.0.0.1", "::1"]
    permissions:
      - action: publish
        path: cam
  - user: visionlabeler
    pass: "sha256:$Hash"
    ips: ["100.64.0.0/10", "127.0.0.1", "::1"]
    permissions:
      - action: read
        path: cam
api: false
metrics: false
pprof: false
playback: false
rtsp: true
rtspTransports: [tcp]
rtspEncryption: "no"
rtspAddress: :$Port
rtmp: false
hls: false
webrtc: false
srt: false
paths:
  cam:
    source: publisher
    runOnInit: '"$StreamCmd"'
    runOnInitRestart: yes
"@
[IO.File]::WriteAllText((Join-Path $InstallDir "mediamtx.yml"), $yml, (New-Object Text.UTF8Encoding $false))
Say "  written (only Tailscale devices with the password may watch)" "Green"

# ---------------------------------------------------------------------------------------------------
Step "7. Windows firewall (Tailscale devices only, port $Port)"
if ($NoFirewall) { Say "  skipped (-NoFirewall)" "Yellow" }
elseif (Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue) { Say "  rule already there" "Green" }
elseif (Ask "Add the firewall rule now? Windows asks for administrator once") {
  $ok = Run-Elevated "New-NetFirewallRule -DisplayName '$RuleName' -Direction Inbound -Protocol TCP -LocalPort $Port -RemoteAddress 100.64.0.0/10 -Action Allow -Profile Any | Out-Null"
  if ($ok -and (Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue)) { Say "  added" "Green" }
  else { Say "  not added: the VisionLabeler PC cannot reach this camera until it is (run this again)" "Red" }
} else { Say "  not added: the VisionLabeler PC cannot reach this camera until it is" "Red" }

# ---------------------------------------------------------------------------------------------------
Step "8. Start (now and with Windows)"
$Vbs = Join-Path $InstallDir "start-hidden.vbs"
$vbsText = "Set s = CreateObject(""WScript.Shell"")`r`ns.CurrentDirectory = ""$InstallDir""`r`ns.Run """"""$Mediamtx"""""", 0, False`r`n"
[IO.File]::WriteAllText($Vbs, $vbsText, (New-Object Text.ASCIIEncoding))
if ($NoAutostart) { Say "  not started with Windows (-NoAutostart)" "Yellow" }
else {
  $shell = New-Object -ComObject WScript.Shell
  $link = $shell.CreateShortcut($StartupLink)
  $link.TargetPath = "$env:WINDIR\System32\wscript.exe"
  $link.Arguments = Quote-Arg $Vbs
  $link.WorkingDirectory = $InstallDir
  $link.Save()
  Say "  starts with Windows (no window)" "Green"
}
Start-Process "$env:WINDIR\System32\wscript.exe" -ArgumentList (Quote-Arg $Vbs) -WorkingDirectory $InstallDir
Say "  started; checking the stream ..."
$ok = $false
for ($i = 0; $i -lt 10 -and -not $ok; $i++) {
  Start-Sleep 2
  $ok = Ffmpeg-Ok @("-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp", "-i", "rtsp://visionlabeler:$Password@127.0.0.1:$Port/cam", "-t", "2", "-f", "null", "-")
}
if (-not $ok) { Stop-Here ("The stream did not start. See " + (Join-Path $InstallDir "mediamtx.log")) }
Say "  the stream works" "Green"

# ---------------------------------------------------------------------------------------------------
$Address = "rtsp://visionlabeler:$Password@${TsIp}:$Port/cam"
[IO.File]::WriteAllText($AddressFile, "Camera address for the VisionLabeler PC (keep it private):`r`n$Address`r`n", (New-Object Text.ASCIIEncoding))
try { Set-Clipboard -Value $Address } catch { }
Step "Done"
Say "On the VisionLabeler PC: Settings > Cameras > Add camera, and enter as its source:" "Green"
Say ""
Say "  $Address" "White"
Say ""
Say "(Also copied to the clipboard and saved in $AddressFile.)"
Say "Send it privately (it holds the password). This PC must stay on, awake and connected to Tailscale."
Say "To remove everything again: remove-camera.bat"
