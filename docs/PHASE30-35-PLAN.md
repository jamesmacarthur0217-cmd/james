# Phases 30-35 plan: picture check, no movement, help sign, photo search in recordings, time-lapse, update button (approved 2026-10-15, not started)

## Context

On 2026-10-15 Phases 27 (scene question areas and limits), 28 (maintenance; step 1, the GitHub runner check, waits
until after 2026-10-19) and 16 (search recordings by words) were finished and pushed (738 tests, GitHub green).
The user has **one camera** (the USB webcam), so Phase 29 (more cameras) waits. Phase 26 (a real-camera week) needs
their own app running with the webcam (it has not run Phases 13-16 and 27-28 yet: a restart, `stop.bat` then
`start.bat`, updates the PostgreSQL database by itself).

The user asked for new ideas that work with one webcam and chose all six, in this order, with these choices
(2026-10-15). Not to be offered again unless the user asks: phone alerts, backups of pictures/videos (only with a
second disk), family vs strangers, commercial licenses (Phase 1), sounds, privacy masks, Malay/Chinese pages and a
blurred export (offered 2026-10-15; the user asked for other ideas instead).

Way of working (as before): ask before each phase and each step; one commit per step; full suite green; `gh run list`
green after pushing; a real run on a test copy (port 8001, `tools/make_test_config.py`); schema changes checked on a
scratch PostgreSQL database; measure before claiming a gain (tune on one half, report on the other); README / HANDOFF /
ROADMAP updated; ask before any download.

## Phase 30: Camera picture check

The webcam's lens cover was closed for a whole 24-hour test (2026-10-11) and only "blank picture" caught it. A check
of the picture itself says what to fix, in plain words.

1. `labeler/picture_check.py`: from frames the camera already has (no extra model): covered / blank (exists in
   `labeler/health.py`, reused), **blurry** (variance of the Laplacian, compared with the camera's own usual value),
   **too dark** / **too bright** (brightness histogram), **light from behind** (a bright background with dark people:
   their boxes against the frame), **faces too small or seen from above** (kiosk cameras: face sizes and the head
   angle from the body points the detector already gives; no face, no age). Measured on COCO and MOT16 frames made
   worse on purpose (blur, darkness, glare; one half to set the limits, the other to report) and on the user's webcam
   when their app runs.
2. Health page line "camera picture" and a **Check the picture** button on the Settings page's camera row: a still
   with the findings and what to do ("open the lens cover", "clean the lens", "turn on a light", "move the camera
   lower"); a banner on the Live page only for problems that last.
3. Docs; README numbers.

## Phase 31: "No movement" alert

For someone living alone (an elderly parent): an alert when nobody has been seen for a long time while someone should
be in. User's choice: **4 hours, from 08:00 to 22:00, only when the house is set to Home** (labeler/arming.py);
changeable on the Settings page (hours, from-until, per camera; off by default for security cameras).

1. A built-in rule "No movement" (like "Fall", `alerts.FALLEN`): the camera's last person seen (Pipeline / db) and the
   idle mode (`idle:`) already know "nobody in view"; a check once a minute; one alert per quiet stretch; it ends
   when someone is seen again (the alert says how long it was quiet). Camera problems (offline, covered) give their own
   alert instead, not "no movement".
2. Pop-up on every page (like falls, without the beep unless chosen), Alerts list, Summary page, Health line; the
   Settings page section.
3. Tests with a clock (quiet 3 h 59 min: none; 4 h: one; Away: none; outside the hours: none, the time counts from
   08:00); a real run with an empty video; docs.

## Phase 32: Help sign

A silent call for help: **one hand held above the head for 3 seconds** (user's choice) gives an alert. Pose points
are already there (`labeler/falls.py` uses them).

1. `labeler/gestures.py`: a wrist above the top of the head (nose/eyes/ears, with a margin by body size), elbow up,
   held 3 s (`gestures.hold_s`), the person facing roughly the camera; one alert per person per sign; areas can turn
   it off (like `lying_ok`).
2. Measured: false alarms on MOT16 walking videos and UR Fall everyday activities (want 0), found on photos and short
   clips of a hand raised (public photos checked by eye, plus the user's own test in front of the webcam when they
   want); tune on one half, report on the other.
3. Built-in rule "Help sign", red pop-up with a beep on every page (like falls), clip; Settings (on/off, seconds);
   docs.

## Phase 33: Search recordings by photo

Like Phase 16, with a picture: "where is this person in the recordings?". `labeler/photo.py` (PhotoSearch.vector)
gives the vector, `db.nearest` the people, `labeler/moments.py` the moments.

1. `POST /api/recordings/search/photo` (the picture as the request body, as `/api/search/photo`; not stored; viewers
   too; audited) -> the same answer as the words search, with the similarity per person; a limit on similarity so
   people who only look a little alike are not listed (measured on MOT16 with true identities, as Phase 8's photo
   search: far cameras are weak, say so).
2. Recordings page: "or search with a photo" (choose a file, or drop it); the same results list and Play / Play all.
3. Docs.

## Phase 34: Time-lapse of the day

A short video (about 1 minute) of everything one camera recorded in a day, to skim it quickly.

1. `labeler/timelapse.py`: from the day's recording files (labeler/recorder.py) with FFmpeg (already installed,
   NVENC): every n-th frame so the day fits the chosen length; made on request (a progress line), kept until the
   day's recordings are deleted; the time stays written on every frame. Measured: how long making it takes per
   recorded hour, the file size.
2. Recordings page: **Time-lapse** for the shown day and camera; clicking in it opens the real recording at that
   moment. Who may watch: admins; viewers only with faces blurred (needs decoding every frame with the head boxes:
   measure the cost, then ask the user).
3. Docs.

## Phase 35: Update button

Updating today needs `git pull` in a terminal. A button on the Settings page does it safely.

1. `labeler/updater.py`: `git fetch` from the user's GitHub (`origin`), shows what is new (the commit titles), and on
   **Update now** (admins, asks first, audited): only a fast-forward `git pull --ff-only` (refuses if files were
   changed by hand or the history differs), then `pip install -r requirements.txt` only when it changed, then the
   database backup and a restart (exit 4, the supervisor restarts at once, as "Restart now"). If anything fails it
   stops and says what, and the old version keeps running. Health line "updates: 3 new changes" (checked once a day).
2. Tests with a scratch git repository (new commits, local changes, a failed pull); a real update on a test copy of
   the repository; docs and the START-STOP-GUIDE.

## Verification (every phase)

- Its own tests, full suite green, GitHub green.
- A real run on a test copy; pictures checked with the measuring tool on public data (one half to set limits, the
  other to report) and, when the user's app runs, on their webcam.
- Pages checked in the browser pane (admin; viewer path; phone width).
