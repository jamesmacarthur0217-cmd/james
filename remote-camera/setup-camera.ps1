<#
VisionLabeler: a camera at another site. Run on the OTHER PC (the one whose webcam should be watched), not on the
VisionLabeler PC. It SENDS this PC's webcam to the VisionLabeler PC through Tailscale:

  webcam -> FFmpeg (H.264) --(outgoing connection, RTSP over TCP)--> VisionLabeler PC (its receiver, port 8554)

Sending is an outgoing connection, so this PC needs no open port, no firewall change and no administrator: it also
works on PCs whose firewall blocks everything coming in (company PCs: ask whether this is allowed there).

  setup-camera.bat     set up: asks for the send address shown on the VisionLabeler PC (Settings > Cameras),
                       downloads FFmpeg (170.5 MB, LGPL, checked against its published checksum), finds and tests
                       the webcam, picks an H.264 encoder, sends a test picture, then sends from every login
                       (no window; it reconnects by itself)
  remove-camera.bat    remove it all again

Options for testing: -Address <send address> -InstallDir <folder> -FfmpegDir <folder with ffmpeg.exe>
-TestVideo <file instead of the webcam> -Camera <webcam name> -NoAutostart -Yes
This file must stay plain ASCII: Windows PowerShell 5.1 reads scripts as ANSI.
#>
param([string]$Address = "", [string]$InstallDir = "$env:LOCALAPPDATA\VisionLabelerCamera", [string]$FfmpegDir = "",
      [string]$TestVideo = "", [string]$Camera = "", [switch]$NoAutostart, [switch]$Uninstall, [switch]$Yes)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is many times slower with the progress bar

$FfmpegRelease = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
$FfmpegZipName = "ffmpeg-n8.1-latest-win64-lgpl-8.1.zip"
$OldRuleName = "VisionLabeler camera (Tailscale)"   # the firewall rule of the earlier version (it served instead of sending)
$StartupLink = Join-Path ([Environment]::GetFolderPath("Startup")) "VisionLabeler camera.lnk"
$AddressPattern = '^rtsp://([A-Za-z0-9_-]+):([A-Za-z0-9]+)@([0-9.]+):([0-9]+)/([A-Za-z0-9_-]+)$'

function Say($text, $color = "Gray") { Write-Host $text -ForegroundColor $color }
function Step($title) { Say ""; Say "== $title ==" "Cyan" }
function Ask($question) {
  if ($Yes) { return $true }
  $answer = Read-Host "$question [y/n]"
  return $answer -match "^(y|yes)$"
}
function Stop-Here($text) { Say ""; Say $text "Red"; Say ""; exit 1 }
function Quote-Arg($text) { return '"' + $text + '"' }

function Stop-Camera {
  # The sending loop (cmd running send.cmd), FFmpeg, and MediaMTX of the earlier version, all from this folder.
  Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*$InstallDir*send.cmd*" } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Get-Process ffmpeg, mediamtx -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$InstallDir\*" } | Stop-Process -Force
}

function Remove-OldRule {
  if (Get-NetFirewallRule -DisplayName $OldRuleName -ErrorAction SilentlyContinue) {
    if (Ask "Remove the firewall rule the earlier version added (no longer needed; Windows asks for administrator)?") {
      $cmd = "Remove-NetFirewallRule -DisplayName '$OldRuleName'"
      $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($cmd))
      try { Start-Process powershell -Verb RunAs -Wait -ArgumentList "-NoProfile -EncodedCommand $encoded" } catch { Say "  not removed" "Yellow" }
    }
  }
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

function Ffmpeg-Run([string[]]$arguments) {
  # FFmpeg's exit code (its messages go to a file). Start-Process in PowerShell 5.1 does not quote arguments with
  # spaces (a webcam called "Integrated Camera"), so they are quoted here.
  $line = ($arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join " "
  $p = Start-Process -FilePath $Ffmpeg -ArgumentList $line -NoNewWindow -Wait -PassThru `
         -RedirectStandardError (Join-Path $env:TEMP "vl-camera-ffmpeg.txt") -RedirectStandardOutput (Join-Path $env:TEMP "vl-camera-out.txt")
  return $p.ExitCode
}
function Ffmpeg-Ok([string[]]$arguments) { return ((Ffmpeg-Run $arguments) -eq 0) }
function Ffmpeg-Said { return ((Get-Content (Join-Path $env:TEMP "vl-camera-ffmpeg.txt") -ErrorAction SilentlyContinue | Select-Object -Last 3) -join " ") }

# ---------------------------------------------------------------------------------------------------
if ($Uninstall) {
  Step "Removing the VisionLabeler camera"
  Stop-Camera
  if (Test-Path $StartupLink) { Remove-Item $StartupLink -Force; Say "  no longer starts with Windows" }
  Remove-OldRule
  if (Test-Path $InstallDir) { Remove-Item $InstallDir -Recurse -Force; Say "  removed $InstallDir" }
  Say "Done: this PC no longer sends its camera." "Green"
  exit 0
}

Say "VisionLabeler camera: sends this PC's webcam to the VisionLabeler PC." "Cyan"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$Ffmpeg = Join-Path $InstallDir "ffmpeg.exe"
$AddressFile = Join-Path $InstallDir "send-address.txt"
Stop-Camera   # a running copy keeps its files open

# ---------------------------------------------------------------------------------------------------
Step "1. Where to send"
if (-not $Address -and (Test-Path $AddressFile)) {
  $Address = (Get-Content $AddressFile | Where-Object { $_ -match "^rtsp://" } | Select-Object -First 1)
  if ($Address) { Say "  the address entered before" }
}
if (-not $Address) {
  Say "  On the VisionLabeler PC: Settings > Cameras > add a camera 'Sent from another device', Save, and copy"
  Say "  its address for a PC (it starts with rtsp://)."
  $Address = (Read-Host "Paste the address here").Trim()
}
$m = [regex]::Match($Address, $AddressPattern)
if (-not $m.Success) { Stop-Here "That does not look like a send address (rtsp://name:password@100.x.x.x:8554/name). Copy it again from the Settings page." }
$TargetIp = $m.Groups[3].Value
Say ("  camera '" + $m.Groups[5].Value + "' on " + $TargetIp + ":" + $m.Groups[4].Value) "Green"

# ---------------------------------------------------------------------------------------------------
Step "2. Tailscale"
if ($TargetIp.StartsWith("100.")) {
  $Tailscale = "$env:ProgramFiles\Tailscale\tailscale.exe"
  if (-not (Test-Path $Tailscale)) { Stop-Here "Tailscale is not installed. Install it from https://tailscale.com/download, sign in, then run this again." }
  $TsIp = ((& $Tailscale ip -4) 2>$null | Select-Object -First 1)
  if (-not $TsIp -or -not $TsIp.StartsWith("100.")) { Stop-Here "Tailscale is not connected. Open Tailscale, make sure it says Connected (on the VisionLabeler owner's network), then run this again." }
  Say "  connected; this PC is $TsIp" "Green"
} else { Say "  not needed (the VisionLabeler PC is on this network)" }

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
foreach ($old in @("mediamtx.exe", "mediamtx.yml", "mediamtx.log", "stream.cmd", "camera-address.txt", "LICENSE-mediamtx.txt", "server.key", "server.crt", "auto.key", "auto.crt")) {
  Remove-Item (Join-Path $InstallDir $old) -Force -ErrorAction SilentlyContinue   # the earlier version (it served)
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
$EncoderArgs = @{ "h264_nvenc" = @("-preset", "p4", "-tune", "ll", "-rc", "cbr"); "h264_mf" = @("-rate_control", "cbr"); "libopenh264" = @() }[$Encoder]

# ---------------------------------------------------------------------------------------------------
Step "5. Webcam"
if ($TestVideo) {
  $SourceArgs = @("-re", "-stream_loop", "-1", "-i", (Resolve-Path $TestVideo).Path)
  Say "  test video instead of the webcam: $TestVideo" "Yellow"
} else {
  [void](Ffmpeg-Run @("-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"))
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
  Say "  testing it for 3 seconds (close Teams, Zoom or the camera app if it fails) ..."
  if (Ffmpeg-Ok (@("-hide_banner", "-loglevel", "error") + $sized + @("-t", "3", "-f", "null", "-"))) { $SourceArgs = $sized; Say "  works at 1280x720, 15 fps" "Green" }
  elseif (Ffmpeg-Ok (@("-hide_banner", "-loglevel", "error") + $plain + @("-t", "3", "-f", "null", "-"))) { $SourceArgs = $plain; Say "  works (its own picture size)" "Green" }
  else { Stop-Here ("The webcam could not be read: " + (Ffmpeg-Said)) }
}
$EncodeArgs = @("-an", "-vf", "scale='min(1280,iw)':-2,fps=15", "-c:v", $Encoder) + $EncoderArgs +
              @("-b:v", "1500k", "-maxrate", "2000k", "-bufsize", "2000k", "-g", "30", "-pix_fmt", "yuv420p",
                "-timeout", "10000000", "-f", "rtsp", "-rtsp_transport", "tcp", $Address)
# -timeout (10 s, in microseconds): without it FFmpeg waits forever when the VisionLabeler PC stops answering
# halfway (it restarted, or the connection broke), and the loop below never gets to start it again.

# ---------------------------------------------------------------------------------------------------
Step "6. A test picture to the VisionLabeler PC"
Say "  sending for 5 seconds ..."
if (-not (Ffmpeg-Ok (@("-hide_banner", "-loglevel", "error") + $SourceArgs + @("-t", "5") + $EncodeArgs))) {
  $said = Ffmpeg-Said
  if ($said -match "401") { Stop-Here "The VisionLabeler PC refused the password: copy the address again from its Settings page (or press New password there)." }
  if ($said -match "404|not found|not configured") { Stop-Here "The VisionLabeler PC does not know this camera: Save it on its Settings page and press Restart now there." }
  Stop-Here ("The VisionLabeler PC did not answer (is VisionLabeler running there, and its firewall rule for port " + $m.Groups[4].Value + " added?). FFmpeg said: " + $said)
}
Say "  received by the VisionLabeler PC" "Green"

# ---------------------------------------------------------------------------------------------------
Step "7. Send (now and at every login)"
$quoted = ($SourceArgs + $EncodeArgs | ForEach-Object { if ($_ -match '[\s'',]') { '"' + $_ + '"' } else { $_ } }) -join " "
$SendCmd = Join-Path $InstallDir "send.cmd"
$cmd = "@echo off`r`n" +
  "rem Sends the webcam to the VisionLabeler PC; starts again 5 seconds after any stop (setup-camera.ps1 writes this file).`r`n" +
  "rem (ping waits the 5 seconds: timeout.exe needs a visible console window.)`r`n" +
  ":again`r`n" +
  (Quote-Arg $Ffmpeg) + " -hide_banner -loglevel error $quoted`r`n" +
  "ping -n 6 127.0.0.1 >nul`r`n" +
  "goto again`r`n"
[IO.File]::WriteAllText($SendCmd, $cmd, (New-Object Text.ASCIIEncoding))
[IO.File]::WriteAllText($AddressFile, "Send address (keep it private):`r`n$Address`r`n", (New-Object Text.ASCIIEncoding))
$Vbs = Join-Path $InstallDir "start-hidden.vbs"
$vbsText = "Set s = CreateObject(""WScript.Shell"")`r`ns.CurrentDirectory = ""$InstallDir""`r`ns.Run """"""$SendCmd"""""", 0, False`r`n"
[IO.File]::WriteAllText($Vbs, $vbsText, (New-Object Text.ASCIIEncoding))
if ($NoAutostart) { Say "  not started with Windows (-NoAutostart)" "Yellow" }
else {
  $shell = New-Object -ComObject WScript.Shell
  $link = $shell.CreateShortcut($StartupLink)
  $link.TargetPath = "$env:WINDIR\System32\wscript.exe"
  $link.Arguments = Quote-Arg $Vbs
  $link.WorkingDirectory = $InstallDir
  $link.Save()
  Say "  sends at every login (no window)" "Green"
}
Start-Process "$env:WINDIR\System32\wscript.exe" -ArgumentList (Quote-Arg $Vbs) -WorkingDirectory $InstallDir
Say "  sending now" "Green"
Remove-OldRule

Step "Done"
Say "This PC now sends its webcam to the VisionLabeler PC. It must stay on, awake and connected to Tailscale." "Green"
Say "It uses about 1.5 Mbit/s of upload. To stop and remove everything: remove-camera.bat"
