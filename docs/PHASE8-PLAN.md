# Phase 8 plan: counting check, a video of each person, maintenance, four extras (approved 2026-10-10)

## Context

Phases 0 and 2-7 are done (381 tests, code backed up to the user's private GitHub; CI green). The user chose
(2026-10-10): **check people counting**, **a short video of each person**, **maintenance**, plus four extras:
**today at a glance**, **search by photo**, **storage use**, **original video link** for imported people. Not
chosen: telling family from strangers. The second site waits for the user.

Why counting: in Phase 7, re-linking by appearance alone joined 24 of 34 tracks on an overhead video; plausible
by eye, never checked. Wrong joins make counts, Reports and "where was this person" wrong on far cameras.

Order: smallest and safest first. Each step: tests, full suite green, real run on port 8001 with
`tools/make_test_config.py`, one commit, pushed to GitHub (push works from the shell now), README /
START-STOP-GUIDE / HANDOFF / ROADMAP updated. Ask before any download or package install.

## Step 1: maintenance

- **Midnight test**: `tests/test_alerts_reports.py` makes people with `datetime.now()` and asks for "today"
  separately (line ~115): across midnight they differ. Confirm by running it with the clock just before midnight
  (monkeypatched `datetime`), fix it, and check the other tests that build "today" the same way.
- **Security updates**: with the user's OK, install `pip-audit` into `.venv` (a dev tool), list packages with
  known vulnerabilities, update those, then full tests + a real run (models still load, FPS unchanged). Offer
  GitHub's free Dependabot security alerts (`.github/dependabot.yml`, weekly, pip only): it opens update
  pull requests in the private repository, so ask first.
- **Certificate expiry**: Health line "HTTPS certificate valid until …", a warning 30 days before it ends
  (`labeler/tls.py` already logs at 14 days; certificates last 825 days), with the fix (delete it, restart).
- **Skipped installer tests**: ask again (about 4 GB download): offline wheels into a fresh `.venv`, tests there.

## Step 2: check people counting (re-linking), with exact markings

- `vtest.avi` is PETS 2009 S2L1, view 1 (768x576, 795 frames). Published ground truth gives each person's box in
  every frame. With the user's OK, download that marking file (small XML/CSV; name, source and size first).
- New `tools/relink_check.py`: import a video headless (reuse `labeler/importer.py` and the measurement script
  from Phase 7), log every re-link decision (similarity, box overlap, gap, accepted or not), match each track to
  the true person by box overlap, and report: true people, people counted, wrong joins (two different people
  made one), missed joins (one person split), plus a contact sheet of every wrong join.
- Measure at `long_gap_min_similarity` 0.90 / 0.93 / 0.95 / 0.98 and with a movement check (how far the box
  moved during the gap vs. a walking speed). Choose what gives the fewest wrong joins without many more splits,
  per mode if they differ (`tracking.relink` for security can get its own values in config, as `min_box_area`
  does). Also measure the timeline threshold (0.8) the same way, since it uses the same appearance similarity.
- Change defaults only if the numbers say so; write the table in README "One person, one record".

## Step 3: a short video of each person, and storage use

- Extend `labeler/clips.py` (`ClipRecorder`, which keeps the live JPEGs for alert clips): a person's clip starts
  1 s before they appear and ends 1 s after they leave, capped at `person_videos.max_s` (20 s), at 5 fps and
  640 px wide (smaller than alert clips). One clip per visit (`clips/person_<id>_<n>.webm`), written by the existing
  writer thread; dropped for people too short to keep. New column `tracks.video_paths` (`db.NEW_COLUMNS`);
  `delete_tracks` and retention remove them (as pictures). Not for imported footage (see step 4).
- Privacy as alert clips: admin-only (viewers get no person videos); made from blurred frames with
  `storage.blur_faces`. Config `person_videos: {enabled, max_s, fps, max_width}` + Settings page on/off.
- People list / Review / timeline: a "video" button on a person, playing it inline (`/clips/...`, audited).
- **Storage use** (`labeler/healthcheck.py`): disk used by pictures, person videos, alert clips, backups, logs;
  average per person over the last 7 days and "days of space left" at that rate before `storage.min_free_gb`.
- Measure: clip size per person and CPU time on the live webcam test copy; numbers in README.

## Step 4: original video link (imported people)

- Imported people store their file and moment (`tracks.source_file`, `tracks.source_offset_s`): the people
  list shows "seen in door_cam.mp4 at 0:31"; admins get a "play from 0:31" link served from the original file
  (only files that an import used, checked against the import history in the database; audited). No copy.

## Step 5: search by photo

- Live page Ask box gets "search by photo": the admin or viewer picks a picture; the server finds people in it
  with the detector (largest person, or the whole picture if none), makes its appearance vector with
  `Embedder.images` (`labeler/embed.py`) and ranks people with `db.nearest`, like look-alikes. The photo is not
  stored (only its use is in the Audit log). Limit 10 MB, JPEG/PNG; same viewer rules as search.

## Step 6: today at a glance

- A strip at the top of the Live page: people today (and the usual number for this weekday, last 4 weeks),
  busiest hour, alerts today, camera problems today, newest backup. One endpoint `/api/today` built from
  `labeler/reports.py` and the existing tables; refreshed every minute. Viewers see people counts only.

## Critical files

`labeler/clips.py`, `labeler/pipeline.py` (start/stop person clips), `labeler/db.py`, `labeler/importer.py`,
`labeler/healthcheck.py`, `labeler/tls.py`, `labeler/web.py`, `labeler/embed.py`, `labeler/reports.py`,
`labeler/settings.py`, `labeler/static/index.html` (+ settings, review), `config.yaml` / `labeler/config.py`,
new `tools/relink_check.py`, `tests/test_*.py`.

Reuse: `ClipRecorder` buffer and writer, `privacy.remove_picture`, `db.nearest` / `get_tracks`, `Embedder.images`,
`reports.summarize`, `healthcheck._check`, the Phase 7 measurement script pattern, `tools/make_test_config.py`.

## Verification

- `pytest -q` green after each step; CI green on GitHub after each push.
- Step 2: a table of true vs. counted people and wrong/missed joins per setting, from the ground truth.
- Steps 3-6: real runs on port 8001 (scratch config; webcam or demo photo, the test video for imports): person
  videos play and are deleted with the person; storage numbers add up; photo search finds the right person in the
  test video from one of its own frames; the glance strip matches Reports for today.
- No leftover `python.exe run.py`; scratch files deleted; HANDOFF/ROADMAP updated; one commit per step.
