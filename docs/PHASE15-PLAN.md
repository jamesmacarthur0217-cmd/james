# Phase 15 plan: an extra camera that works anywhere (approved 2026-10-12)

## Context

Phase 14 made another PC's webcam a camera by having **this PC pull** the stream from the other PC
(`remote-camera/`, MediaMTX on the other PC, port 8554). On Ti Xuan's PC it was blocked: a company **ESET
firewall** replaces the Windows firewall and drops incoming connections. Outgoing connections from that PC work
(Ti Xuan already opens this PC's pages over Tailscale). The user chose: "an extra camera that works anywhere".

So turn the direction around: **the other device sends (pushes) its picture to this PC**, which runs the receiver.
The sending device then needs no incoming port, no firewall change and no server: only FFmpeg on a PC, or a free
streaming app on a **phone**. The same receiver gives the user a second camera without buying one (an old phone).

```
other PC:  webcam -> FFmpeg --(outgoing, RTSP over TCP, Tailscale)--> this PC: MediaMTX :8554 -> VisionLabeler
phone:     camera -> streaming app --(outgoing, RTMP, Tailscale)----> this PC: MediaMTX :1935 -> VisionLabeler
```

Note for the user: running streaming software on a **company PC** may be against company rules; the plan works
the same with a personal PC or a phone at that site.

## Step 1: the receiver on this PC (`labeler/receiver.py`)

- A camera can have `source: push` (Settings page: "Sent from another device"). For such cameras run.py starts
  **MediaMTX** here (`data/mediamtx/mediamtx.exe`, v1.21.2, MIT; downloaded with the user's OK, checksum checked,
  via a new `tools/get_mediamtx.py` sharing `expected_sum`/`sha256`/zip helpers with `tools/get_ffmpeg.py`).
- `receiver.write_config(cfg)` writes `data/mediamtx/mediamtx.yml` from the cameras: one path per pushed camera;
  **publish** needs that camera's own random password (stored hashed `sha256:` in the settings, shown once on the
  Settings page) and a Tailscale address (`100.64.0.0/10`); **read** only from 127.0.0.1 with an internal password
  made at each start; RTSP over TCP on 8554 and RTMP on 1935 (phone apps speak RTMP); HLS, WebRTC, SRT, API off.
- `FrameSource` maps `push` to `rtsp://<internal user>:<pw>@127.0.0.1:8554/<camera>` (labeler/sources.py; the
  existing reconnect loop already waits for a sender that is not there yet; `mask_password` keeps logs clean).
- Lifecycle in run.py, like `_start_recorders`: start MediaMTX hidden with the app, stop it at shutdown (before the
  15 s camera-loop wait), restart it if it dies (watchdog thread), free a port left by a crashed earlier copy;
  test configs (`tools/make_test_config.py`) get their own folder and ports.
- Health page: "receiver" line (MediaMTX found/running, which cameras are sending now: MediaMTX's API on
  127.0.0.1 only, or from the camera health already there) and the firewall check in `labeler/remote.py` extended
  to ports 8554/1935 (same netsh parsing: `firewall_rules`, `_covers_tailscale`).
- This PC's firewall: one more Tailscale-only rule for TCP 8554 and 1935 (the user runs one administrator line,
  as for 8443); `install.ps1` step and offline package include MediaMTX.

## Step 2: senders

- **PC:** rework `remote-camera/` to send instead of serve: `setup-camera.bat` asks for the **send address**
  (copied from this PC's Settings page, e.g. `rtsp://tixuan:<pw>@100.116.21.17:8554/tixuan`), downloads only
  FFmpeg (170.5 MB, checksum), finds and tests the webcam (as now), picks the encoder (as now), writes a `send.cmd`
  that runs FFmpeg in a loop (reconnects after a drop or when this PC restarts), starts it hidden at login.
  No MediaMTX, no firewall rule, no administrator. `remove-camera.bat` as now. The pull version is removed
  (push works wherever pull did; one way to support).
- **Phone:** no code; Settings page shows the RTMP address for that camera and `docs/PHONE-CAMERA.md` explains
  (Tailscale app on the phone, a free RTMP streaming app such as Larix Broadcaster, 1280x720, ~1.5 Mbit/s;
  keep the app open and the phone charging; check the app's exact steps when writing).

## Step 3: Settings page and docs

- Settings → Cameras: source type "Sent from another device"; on save a name-based path and a random password;
  the page shows the two send addresses (PC: RTSP, phone: RTMP) with copy buttons, and "New password".
  Validation as for other cameras; audited (password never in the audit log).
- README "A camera at another site" rewritten for sending; data-protection note; HANDOFF/ROADMAP.

## Critical files

`labeler/receiver.py` (new), `run.py` (start/stop/watchdog), `labeler/sources.py` (push source), `labeler/settings.py`
+ `static/settings.html` (camera type, addresses), `labeler/healthcheck.py` + `labeler/remote.py` (lines/ports),
`labeler/config.py` + `config.yaml` (`receiver:` defaults: ports, dir), `tools/get_mediamtx.py` (new),
`remote-camera/setup-camera.ps1`, `install.ps1`, `tools/pack_offline.ps1`, `tools/make_test_config.py`.
Reuse: `tools/get_ffmpeg.py` helpers, `labeler/recorder.pick_encoder`-style encoder test, `config.mask_password`,
`remote.firewall_rules`, the Phase 14 MediaMTX settings that were tested (auth, TCP-only, protocols off).

## Verification

- Tests: config writing (paths, hashed passwords, Tailscale-only publish, local-only read, protocols off), push
  source mapping, lifecycle with a fake MediaMTX (start, restart after a crash, stop), settings validation and
  that passwords never reach the audit log or logs, firewall check for the new ports, ASCII check of the script.
- Real runs on this PC (test config, port 8001, its own MediaMTX ports): FFmpeg sending a MOT16 video to this PC's
  Tailscale address → people found and recorded; RTMP sending (as a phone app would) → same; wrong/no password and
  a non-Tailscale address refused; sender stops and comes back → camera offline then back by itself; MediaMTX
  killed → restarted by the watchdog; the Settings flow in the browser pane.
- With the user: a phone (theirs) as a test camera; Ti Xuan's PC only if the user decides it is allowed there.
