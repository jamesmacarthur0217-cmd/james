# Phases 36-40 plan: first-run check, faster playback, PC watch, unusual quiet, weekly report (approved 2026-10-15; Phase 36 started 2026-10-17)

## Context

On 2026-10-15 Phases 16, 27, 28 (steps 2-4) and 30-35 were finished and pushed (`b15c94f`, 795 tests, GitHub green).
Nothing buildable was left: Phase 26 needs the user's own app running the new version (it has not run Phases 13-35
yet), Phase 28 step 1 waits until after 2026-10-19 (Ubuntu 26 runner), Phase 29 needs more cameras (the user has one
webcam). The user chose this round on 2026-10-15, in this order: **36 first-run check, 37 faster playback, 38 PC watch,
39 unusual quiet, 40 weekly report**. Offered and not chosen (don't offer again unless asked): simple view, full-system
long run, privacy hours, snooze alerts, activity calendar, camera moved/blocked, left-behind objects; earlier: phone
alerts, picture backups (only with a second disk), family vs strangers, licenses, sounds, privacy masks, Malay/Chinese
pages, blurred export.

## Phase 36 (A): First-run check — "what's new" and a guided check

The user's app jumps from before Phase 13 to Phase 35 at its next restart; this makes that first run (and Phase 26)
easy and safe.
1. **What's new**: the app remembers the version it last ran (`data/last_version.txt`, git `HEAD`; written at start)
   and, when it changed, the Live page shows a banner "New in this version" linking to `/whats-new`: the README
   sections added since then, in plain words (a short list kept in `labeler/whatsnew.py`, one entry per phase with its
   commit, filtered by `git merge-base --is-ancestor`). Viewers see only what applies to them.
   **Done 2026-10-17**, a little differently: the list is in the order phases were finished, so the state file
   (`data/last_version.json`: newest entry ran, and the one before the last update) needs no git (works for offline
   installs too); a restart of the same version keeps the banner (✕ or opening the page hides it per browser). The
   user's app has no state file yet, so the first start judges by the tables/columns the database had to add.
2. **First-run check page** (`/first-run`, admins): a checklist that reads existing APIs: Health all green
   (`/api/health`), database updated (new tables present), Check the picture (`/api/cameras/<name>/picture`),
   recording works (a file from today: `/api/recordings`), Home/Away state, the built-in rules exist (Fall, No movement,
   Help sign), and a **"try the help sign"** button: for 30 s the camera reports a confirmed sign to the page instead of
   making an alert (a `GestureWatch` test flag on the Pipeline; no alert, no clip). Each item ✓ / ✗ with what to do.
   **Done 2026-10-18** (`labeler/firstrun.py`, `static/first-run.html`). Found on the real run: a hand already up when
   the button is pressed was never counted (its sign was given before), so starting the test restarts the count.
3. Tests, a real run (a test copy started on an old `last_version`), docs.

## Phase 37 (L): Faster playback on the Recordings page

1. Admins (the MP4 file): `video.playbackRate` 1x/2x/4x/8x, buttons −10 s / +10 s, keys (space, ←/→, 1/2/4/8);
   also for the time-lapse and search moments (the moment's end check uses currentTime, so it still works).
2. Viewers (the blurred MJPEG stream, `recordings_web.recording_stream`): a `speed` parameter (send every n-th frame,
   same pacing), and skip = a new stream from `t ± 10`.
3. Tests (stream with speed sends fewer frames, same duration), browser check, docs.

## Phase 38 (K): PC watch

1. `labeler/pcwatch.py`, checked every minute / hour (cached): GPU temperature and power (`nvidia-smi --query-gpu=
   temperature.gpu,power.draw,memory.used`), disk health (`Get-PhysicalDisk | Select HealthStatus,OperationalStatus`,
   PowerShell, read-only, hourly), unexpected stops: a heartbeat file (`data/heartbeat.json`, written every minute) —
   found at start without a clean-exit mark = "the app or PC stopped unexpectedly around <time>" (stop.bat and Restart
   now write the mark). Thresholds: GPU ≥ 85 °C, disk not Healthy.
2. Health page lines ("PC": temperature, disk; "unexpected stops (7 days)"), a pop-up like camera problems for a new
   problem, stored like camera events (`camera_events` with camera "PC", so Alerts/Summary list them).
3. Tests with fake command outputs; real run (heartbeat left by killing a test copy, then starting it); docs.

## Phase 39 (B): Unusual quiet — learn the usual day, alert on an odd quiet

Builds on Phase 31 (`labeler/quiet.py`) and the history used by `labeler/today.py` (same weekday, last 4 weeks).
1. `labeler/routine.py`: per kiosk camera, per weekday kind (weekday / weekend) and hour, the share of days on which
   someone was seen in that hour (tracks of the last 4-8 weeks; needs ≥ 14 days of history, else "still learning").
2. Built-in rule "Unusual quiet" (condition `unusual = yes`, only when Home): fires when nobody has been seen since
   the start of an hour in which people were seen on at least 80% of usual days, by 1 h after it (e.g. usually up by
   08:00, nobody by 09:00). One alert per day per camera; the message says what is usual. Pop-up like no movement.
3. Measuring: synthetic histories (a regular household, an irregular one) for false/late alerts; the user's own data
   once their app has run 2+ weeks (part of Phase 26). Health line "routine: learning (5 of 14 days)". Docs.

## Phase 40 (H): Weekly report

1. `labeler/summary.py` period "week" (the last 7 days, or a chosen week): per day counts (people, vehicles, animals,
   alerts by kind, camera problems, PC problems), recording hours and storage, checks (Phase 24 results), quiet/help/
   fall alerts listed.
2. Summary page "This week" tab with a printable layout (`@media print`: one page, no menus; "Print / Save as PDF"
   button using the browser's print). Viewers: without clips and camera problems, as the Summary.
3. Tests, browser check (print preview width), docs.

## Way of working (unchanged)

Ask before each phase and each step; one commit per step; full suite green; push and check `gh run list`; real run on
a test copy (port 8001, `tools/make_test_config.py`), never the user's app (port 8000) or data; schema changes checked
on a scratch PostgreSQL database; measure before claiming a gain; edit with the Write/Edit tools or Python scripts
written with Write (pitfalls 1, 22, 25); short, simple answers for the user.

## Verification

- Step 0: control-character scan of the changed docs; GitHub run green; the new session's start message points to
  `docs/PHASE36-40-PLAN.md`.
- Each phase: its tests, full suite (≥ 795), real run on a test copy, browser check (admin, viewer path, phone width).
