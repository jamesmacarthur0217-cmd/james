# Phase 13 plan: recording (approved choices 2026-10-12)

## Context

The system analyzes live video but stores only pictures, a short video of each person and alert clips. The user
wants recordings they can watch back. Their choices (2026-10-12):

- **When**: only while people are in view, from 10 s before the first person appears to 10 s after the last one
  leaves (a person returning within those 10 s continues the same recording).
- **Format**: H.264 in MP4, encoded by FFmpeg on the graphics card (NVENC); plays in any browser and on phones.
- **Keep**: 3 days and at most 100 GB, oldest deleted first.
- **Who**: admins and viewers; viewers see blurred faces (as in the live view); every viewing and download is in
  the Audit log.

Measured before planning (2026-10-12, MOT16 street footage, OpenCV's own encoders, the only ones on this PC now):
WebM/VP8 77-305 GB per camera per day and 1-4 CPU cores; MP4/mp4v 23-135 GB per day and 4-25% of a core. The
earlier guess of 5-20 GB per day was wrong for these encoders; H.264 should be 3-5 times smaller (to be measured,
step 1), and a quiet room at home much smaller than a busy street.

## Step 1: FFmpeg and measurements

- Download (with the user's OK): `ffmpeg-n8.1-latest-win64-lgpl-8.1.zip`, 170.5 MB, from
  https://github.com/BtbN/FFmpeg-Builds/releases (LGPL build: H.264 through NVENC or Windows Media Foundation;
  no GPL encoders). Unpack into `tools/ffmpeg/` or `data/ffmpeg/` (not in git); config `recording.ffmpeg`.
- Measure on MOT16 clips and on the user's webcam (lens cover open): `h264_nvenc` at a few quality levels, size
  per hour, GPU/CPU use, effect on the pipeline's FPS; fallback `h264_mf` when there is no NVIDIA card. Pick the
  default size, frame rate (about 10 fps) and quality from the numbers.

**Step 1 results (2026-10-12)**: FFmpeg n8.1 LGPL downloaded with the user's OK (checksum matched the release's
checksums.sha256); `ffmpeg.exe`, `ffprobe.exe` and `LICENSE.txt` in `data/ffmpeg/`. Encoders: `h264_nvenc`
(NVIDIA), `h264_mf` (Windows), `libopenh264`, also AMD/Intel ones. Measured on MOT16-04 and -09 (fixed cameras,
busy streets) at 10 fps, recording all the time, raw frames piped in, fragmented MP4 out:

| Size | Encoder, quality | GB per camera per day | FFmpeg CPU |
|---|---|---|---|
| 1280×720 | NVENC cq 23 / **28** / 33 | 17-22 / **9-13** / 5-7 | 2-6% of a core |
| 1280×720 | Media Foundation, quality 60 | 10-24 | 13-15% |
| 1280×720 | OpenH264, 2 Mbit/s | 21 | 5-7% |
| 1920×1080 | NVENC cq 28 | 20-26 | 8-9% |

Defaults: 1280×720 (smaller cameras keep their size), 10 fps, NVENC cq 28 (looks the same as cq 23 at normal size;
cq 33 slightly soft); fallback Media Foundation, then OpenH264. The webcam measurement (quiet room, recording only
while people are in view) comes with the first real run in step 2 (the lens cover was closed so far).

## Step 2: the recorder (`labeler/recorder.py`)

- One per live camera, fed from `Pipeline.run` with every frame read (before idle mode skips any), sampled at
  `recording.fps`; frames go to an FFmpeg process over a pipe (raw frames in, H.264 MP4 out) on its own thread,
  so a slow disk never slows the pipeline.
- Pre-buffer: the last 10 s kept as JPEG in memory (as `labeler/clips.py` does), written first when a person
  appears. Start: a person tracked (`Pipeline.tracks`); stop: 10 s after the last one. Files of at most 10
  minutes: `data/recordings/<camera>/<YYYY-MM-DD>/<HHMMSS>.mp4`, fragmented MP4 so a crash leaves a playable file.
- `storage.blur_faces` blurs faces before writing. For viewers, the head boxes of each recorded frame (known from
  the pipeline's keypoints) go into a small file next to the video, used to blur at playback.
- Table `recordings` (camera, start, end, file, size, people seen); a startup check closes files left open.
- Retention every hour and while writing: older than `recording.keep_days` (3) or over `recording.max_gb` (100)
  → oldest deleted; below `storage.min_free_gb` recording pauses and the Health page says so.

**Step 2 results (2026-10-12)**: `labeler/recorder.py`, table `recordings`, `recording:` in config.yaml (on by
default: the user's choice), hooked into `Pipeline.run` (every frame read, before idle mode skips any), started by
run.py when FFmpeg and an encoder are found; `repair()` at start, `prune()` with the hourly retention. Test copies
record into their own folder (`tools/make_test_config.py`), the restore drill not at all. Measured on a test copy
with MOT16-11 (busy mall, 1080p, no VLM) as the camera:
- size: about 9 MB a minute while recording (13.6 GB/day if it never stopped);
- camera loop: 18.5 frames/s without recording, 17.9-18.4 with, after moving shrinking, blurring and JPEG-making to
  the writer thread (before that: 16.3-16.7); about 10% of one CPU core more in all, FFmpeg 4%;
- a normal stop: frames in the video, the frame list and the database all equal (295, 304). Found and fixed on the
  way: run.py waits 15 s for a camera loop, which wrote person videos first, so recordings were left open; the
  recorder now closes first, and its frame list is written line by line;
- killed with the whole process tree (like a power cut): the video lost its last ~3 s (265 of 299 frames); the
  repair at the next start trusts the video (ffprobe) and records 266.

## Step 3: watching

- **Recordings** page: pick a camera and a day; a bar shows the recorded stretches (and when people were seen);
  click to play from that moment. From a person, an alert or the timeline: "▶ Recording" jumps to that moment.
- Admins: the MP4 file itself (seeking through HTTP range requests). Viewers: the app decodes, blurs faces with the
  stored head boxes and streams it like the live view (no direct file access).
- Admins: download a time range (cut without re-encoding) and delete a time range. Everything audited.

**Step 3 results (2026-10-12)**: `labeler/recordings_web.py` + `static/recordings.html` (menu link on every page,
viewers too). Day bar (recorded stretches, people's arrivals, a crosshair with the time; click to play from that
moment), list with Play / Download / Delete (admins), deep links `/recordings?camera=…&at=…` from each person on
the Live page (and the timeline) and each alert. Admins: the MP4 itself (Range requests; the finished file has an
`mfra` index, so seeking works); a recording still being written plays as a stream. Viewers: only the stream, faces
blurred from the stored head boxes; the file, downloads and deletes answer 403. The camera's name and the time are
written on every recorded frame (exact in the player, downloads and streams, however unevenly the camera ran). One
audit entry per playback (not per seek). Checked in the browser pane at desktop width.

## Step 4: settings, health, docs

- Settings page section *Recording*: on/off, when (people / always), keep days, max GB. Health page: recorder
  running per camera, FFmpeg found and which encoder, disk used by recordings, oldest recording.
- `install.ps1` / offline package include FFmpeg; README (feature, measured sizes), data-protection summary
  (recordings, 3 days), HANDOFF/ROADMAP. Deleting a person does not cut them out of recordings (said in the
  README; the time-range delete is the tool for that).

**Step 4 results (2026-10-12)**: Settings page *Recording* (on/off, when, keep days, max GB; validated, audited);
Health page *recording* line (off / FFmpeg missing / no encoder / paused: problem / frames dropped / files, GB,
oldest, encoder); `tools/get_ffmpeg.py` (check, download with the release's checksum, pack, unpack), install.ps1
step 6b and the offline package include FFmpeg; README *Recording*, the Web UI table, licenses (FFmpeg LGPL),
`docs/data-protection.md` template rows (the user's filled copy in `data/privacy` is theirs to update). The code
default is off (`DEFAULTS`), the project's config.yaml turns it on (the user's choice). Still to do with the user:
a long-run check with recording on (needs the PC awake) and the webcam measurement (lens cover).

## Verification

- Tests: start/stop and pre-buffer logic, file splitting, retention by age and size, the free-disk pause, viewer
  access only through the blurred stream, audit entries, settings validation (fake FFmpeg in tests; CI has none).
- Real runs on port 8001 (test config): MOT16 videos as a camera, then the webcam; a person video and alert clip
  still work; kill the app mid-recording and play the file; a 24-hour long-run check (`tools/soak.py`) with
  recording on, watching memory, disk and FPS; playback and seeking in the browser pane, as admin and as viewer.
- Measured numbers in the README: GB per day per camera at home and on a busy street, GPU/CPU use, FPS before/after.
