# VisionLabeler: next improvements roadmap

## Status (2026-10-14)

**Phases 0 and 2-11 are done** (each item below is marked with what was done, measured or rejected;
details in docs/HANDOFF.md and README.md). **Phase 1 is deferred**: the user uses the system at home, not
commercially yet. Phase 10 (long-run check, idle mode, restore drill) and Phase 11 (keyboard answer key, MOT16
import, first kept retraining) are done (2026-10-12). Phase 12 step 1 (a re-identification model) was measured
on MOT16 and not adopted (2026-10-12); the rest of Phase 12 waits for real extra cameras. Phase 13 (recording
while people are in view), Phase 14 (a camera at another site, through Tailscale) and Phase 15 (that camera sent
from a phone or PC, so it works through company firewalls) are done (2026-10-12). Phase 17 (mobility aids and fall
alerts) is done (2026-10-13). Phase 18 (vehicles) is done. Phases 19 (Home / Away), 20 (loitering) and 21 (animals) are done (2026-10-14). 662 tests pass.

## Context (as planned 2026-10-07)

VisionLabeler (`C:\VisionLabeler`) is a working local system: live cameras → YOLO11-pose + ByteTrack →
color estimator, fast classifiers, age model, Qwen3-VL 4B (Ollama) → Postgres + pgvector → web UI (Live,
Review, Reports, Alerts, Answer key, Ask box, delete). At planning time: 75 tests, 88% on a 39-person answer
key, age model 60% exact / 96% ±1 group on FairFace, ~2 s per person on an RTX 4070 SUPER.

Answers to scoping questions then: **used for both kiosk and security cameras**, and **commercial** use (since
changed: personal home use for now). Each phase below is sized for one or two sessions and is independently
shippable.

## Working notes for the next session (learned the hard way)

- Start every session with `git log --oneline | head` and `.venv\Scripts\python.exe -m pytest -q`.
- **Edit with the Edit/Write tools, or a Python script written to a file.** Bash heredocs in this
  environment collapse `\\` → `\`, which turned `\n` into real newlines and `\b` into backspaces
  (broke a page script and `stop.bat`). `tests/test_pages.py` now catches JS syntax errors.
- Ollama must be addressed as `http://127.0.0.1:11434` (`localhost` adds ~2 s per request on this PC).
- Test the app on `--port 8001 --source samples\bus.jpg` with a scratch config from
  `tools/make_test_config.py` (the real config opens the app to the network); stop it with `POST /api/shutdown`; afterwards
  confirm no leftover `python.exe run.py` processes (`Get-CimInstance Win32_Process`), since leftovers
  load the GPU and skew benchmarks. The app now refuses a second copy on the same port.
- Database: Docker container `visionlabeler-db` (Postgres 17 + pgvector), URL in `.env`.
- The user's DB holds 27 people (26 in their answer key); don't delete anything without asking.

## Phase 0: Carry-over (small, do first)

1. **Done 2026-10-08** (answer key 67% → 75%; see HANDOFF.md). **Retrain the fast classifiers** with the
   improved gender prompt and decade age ranges:
   `tools/teacher_label.py` (~1 h, re-label after moving the old `teacher.jsonl` aside), then
   `tools/train_attributes.py ... --no-balance`. Their old age classifier is currently ignored by the
   vocabulary guard in `labeler/student.py`.
2. Ask the user to label 50–100 of their own people on the **Answer key** page; record the per-camera
   score as the baseline for later phases (`GET /api/answer-key/score`).

## Phase 1: Commercial license readiness

**Deferred (user, 2026-10-08): not for commercial use yet.** Skip this phase until the user says the system
will be sold or used commercially; then ask the decision below again.

Blockers found: **Ultralytics YOLO is AGPL-3.0** (detection, pose and its ByteTrack implementation), and the
fast classifiers were trained on **COCO photos with mixed Flickr licenses**. Fine: Qwen3-VL, SigLIP 2, age model
(Apache-2.0), FairFace (CC BY 4.0, needs attribution), Postgres, FastAPI.

1. **Decision point for the user** (ask at the start): buy an Ultralytics Enterprise license (no code change),
   or replace the detector stack. Recommended replacement if not buying:
   - detection: RF-DETR (Apache-2.0, `rfdetr` package);
   - pose keypoints: RTMPose via `rtmlib` (ONNX, Apache-2.0), same COCO-17 keypoint layout the code uses;
   - tracking: `supervision` ByteTrack (MIT), keeping the tuned values from `labeler/trackers/people.yaml`.
2. Introduce `labeler/detect.py` with one interface, `PeopleTracker.update(frame) -> [(track_id, box, conf,
   kpts)]`, and move the Ultralytics call out of `Pipeline.process` (`labeler/pipeline.py`) behind it. Both
   backends live behind a `detector.backend` config option, so they can be compared and switched back.
3. Re-run the tracking checks with the new backend: `tools/` clip tests (gap 2 s / 10 s / 40 s → 1/1/2
   records), FPS per camera, keypoint quality (torso color boxes, face crops for the age model).
4. Retrain the fast classifiers on the customer's own reviewed camera data instead of COCO: new
   `tools/dataset_from_db.py` that exports finished people with crops (reuse `export_dataset.py` logic).
5. Add `LICENSES.md`: every model and dataset, license, attribution text (FairFace).

## Phase 2: Privacy and security for business use

1. **Done 2026-10-08.** **Accounts and roles** replacing the single Basic-auth password in `labeler/web.py`: `admin` (delete,
   export, rules, settings) and `viewer` (live view and search only). Hashed passwords in a new `users`
   table; session cookie; login page.
2. **Done 2026-10-08.** **Audit log** table: who viewed, searched, exported, deleted, changed rules, with time and IP. Admin page.
3. **Done 2026-10-08** (self-signed, built in; no Caddy: a local proxy would bypass the login). **HTTPS** for network access (self-signed certificate generated on setup, or Caddy in docker-compose).
4. **Done 2026-10-08.** **Face blurring option** for viewers and for saved crops (`storage.blur_faces`), using the face box from
   `labeler/geometry.py:face_box`.
5. **Done 2026-10-08** (plus hourly cleanup while running). Per-camera retention (`cameras[].retention_days`), and a documented erasure procedure (search the person,
   then Delete selected).
6. **Done 2026-10-08** (`docs/camera-notice.md`, `docs/data-protection.md`). Templates in `docs/`: camera notice sign text, short data-protection summary (what is stored, how long, who
   can see it). Recommend BitLocker on the PC for encryption at rest.

## Phase 3: Shared camera features (kiosk and security)

1. **Done 2026-10-08** (stored in the database, not config). **Zones and counting lines** per camera (config plus a small drawing tool on the Live page): count entries
   and exits, time spent in each zone; store as `zone_events`. Alerts can use "entered zone X".
2. **Done 2026-10-08.** **Dwell time and engagement**: time in view (already `started_at` / `ended_at`), share of time facing the
   camera (from `geometry.frontality`). Added to Reports.
3. **Done 2026-10-08.** **Camera health**: alerts for camera offline, blank picture, or FPS drop (status already tracked in
   `Pipeline.status()`); shown on the Live page and in Alerts.
4. **Done 2026-10-08** (WebM/VP8). **Event clips**: keep a ~10 s rolling buffer per camera and save a short clip around each alert; playable
   on the Alerts page. Counts toward retention.

## Phase 4: Mode-specific features

Kiosk:
- **Done 2026-10-08.** Demographics report by hour (age group × gender), dwell histogram, busiest periods (extends
  `labeler/reports.py`).
- **Tried 2026-10-08, rejected** (worse on COCO; see HANDOFF). Higher-resolution upper-body crop for clothing; top type is the weakest attribute (73%), so also test
  vocabulary refinements (coat vs jacket, polo, tank top) against the answer key.

Security:
- **Done 2026-10-08** (per-camera imgsz; tiling measured and rejected). **Small, distant people**: per-camera `imgsz` and optional tiled detection (slicing) for 4 MP+ cameras;
  measure the recall gain on footage with far-away people.
- **Done 2026-10-08.** **Night / infrared**: an IR mode that skips color attributes (low-color detection already exists in
  `labeler/colors.py:is_low_color`) and labels shape and carried items only.
- **Done 2026-10-08.** **Person timeline across cameras**: "where was this person": look-alike search across cameras, ordered by
  time, with a time-plausibility filter. A dedicated re-identification model only after a license check.
- **Measured 2026-10-08** (README "How many cameras?"; batching / GPU decoding not done). **Capacity**: GPU video decoding and batched detection across cameras; document how many cameras the
  4070 SUPER handles.

## Phase 5: Accuracy loop

- **Done 2026-10-09.** Scheduled retraining of the fast classifiers from reviewed labels (weekly), keeping the previous model if
  the answer-key score drops.
- **Done 2026-10-09** (spot checks). Review queue picks varied people, not just uncertain ones, so corrections cover more situations.
- **Done 2026-10-09.** Per-camera accuracy history from the answer key, shown on the Answer key page.

## Phase 6: Operations (done 2026-10-10)

**Approved plan: [PHASE6-PLAN.md](PHASE6-PLAN.md)** (user's choices: console window kept, no service; Settings
page for common settings; health check; installer; private GitHub with tests). Earlier outline:

1. **Done 2026-10-09** (log files; health check with a Health page and a monitoring token). Log files with rotation (`data/logs`, for run.py and the supervisor), plus a `/api/health` endpoint
   (app, database, Ollama, each camera; also usable from a monitoring tool).
2. **Not done, by the user's choice** (console window kept; see PHASE6-PLAN.md). Run as a Windows service (e.g. NSSM wrapping `python -m labeler.supervisor` — a download, ask first — or a
   Task Scheduler "at startup" task without a window) instead of a console window. It must keep start.bat's
   waits (Docker, Ollama) and stop.bat's clean stop (shutdown API, final saves, DB backup).
3. **Done 2026-10-09** (`data/settings.yaml`, Restart now). A settings page (admin, audited) for cameras, retention, HTTPS, blurring, retraining (zones and rules
   already have their own pages), replacing hand-edited `config.yaml`. Keep config.yaml as commented defaults
   and store changes separately (a file or the database) so comments are never lost; validate every value;
   say when a restart is needed.
4. **Done 2026-10-09** (`install.bat`, `tools/pack_offline.ps1`). An installer script for a new PC: Python venv, Docker, Ollama and model downloads, first admin account,
   with offline model packaging for sites without internet.
5. **Done 2026-10-10** (pushed to the user's private repository; `.github/workflows/tests.yml`). Optional: push to a
   private GitHub repo with CI running the test suite (ask first).

Then: switch to planning mode and propose the next round (open items: HANDOFF section 8).

## Phase 7: Backups, second site, recorded videos (done 2026-10-10)

**Approved plan: [PHASE7-PLAN.md](PHASE7-PLAN.md)** (user's choices 2026-10-10; no phone alerts).

1. **Done 2026-10-10** (real backup + restore check: 27 of 27 people, 2 s). Daily database backup while the app runs, with
   an optional second copy and a weekly restore check.
2. **Checks and guide done 2026-10-10** (connecting the other PC: the user, later). Second site over
   Tailscale: self-checks on the Health page and a plain guide; the user connects the other PC.
3. **Done 2026-10-10** (measured on vtest.avi; README). Import recorded videos on a new Import page: fast (no waiting, every k-th frame) and with real recording
   times (file name, MP4 metadata, or typed), kept until deleted by default.

## Phase 8: Counting check, person videos, maintenance, extras (done 2026-10-10)

**Approved plan: [PHASE8-PLAN.md](PHASE8-PLAN.md)** (user's choices 2026-10-10; not chosen: family vs. strangers).

1. **Done 2026-10-10** (database image test waits for Docker). Maintenance: the midnight test, security updates,
   certificate expiry on the Health page, skipped installer tests.
2. **Done 2026-10-10** (security 0.98: 20 of 24 joins were wrong at 0.90). Check people counting (re-linking)
   against the published ground truth of the test video; change only if measured.
3. **Done 2026-10-10** (median 1.3 MB per person at 480 px). A short video of each person (admin-only), and
   storage use on the Health page.
4. **Done 2026-10-10** (excerpt cut on first view). Original video link for imported people. 5. **Done 2026-10-10** (a hint on far cameras: 6/18 first). Search by photo. 6. **Done 2026-10-10.** Today at a glance on the Live page.

## Phase 10: Long-run check, idle mode, restore drill (done 2026-10-12)

**Approved plan: [PHASE10-PLAN.md](PHASE10-PLAN.md)** (the user chose a webcam test copy for the soak and no second
long run).

1. **Done 2026-10-11** (`tools/soak.py`). 24 hours: no leak, no errors; the lens cover was closed, so no people were
   labeled. The tool now reads the app's own GPU memory (the whole GPU's number rose with other programs).
2. **Done 2026-10-12** (on by default). Idle mode: 12/12 people found, 0.1-0.4 s later; GPU 16.9 → 11.9 W while empty.
3. **Done 2026-10-12** (`tools/restore_drill.py`). 27/27 people and all pages back in 22 s. Found: the backups hold
   the database only, not pictures or person videos.

## Phase 11: Better labels on the user's webcam (done 2026-10-12)

1. **Done** (labels kept hidden, the user's choice). Answer key by keyboard; before about 34 s per person.
2. **Done with a public dataset** (the user had no suitable videos): MOT16, CC BY-NC-SA 3.0, 127 people imported.
3. **Done**: first kept retraining, 61.8% → 62.1% on 26 answer-key people (no measurable change).

## Phase 12: Re-identification and more cameras (step 1 measured 2026-10-12; the rest waits for cameras)

1. **Measured, not adopted**: OSNet-AIN re-identification model vs SigLIP 2 on MOT16 (look-alikes in the
   pipeline 24/74 vs 34/74). Kept as a measuring option (`tools/relink_check.py --reid`) for outdoor footage.
2. More cameras (batching) and 3. per-camera tuning: when the user adds cameras.

## Phase 13: Recording (done 2026-10-12, except the long-run check with the user)

**Plan: [PHASE13-PLAN.md](PHASE13-PLAN.md)** (user's choices: record while people are in view with 10 s before and
after; H.264 through FFmpeg on the graphics card; keep 3 days / 100 GB; admins and viewers, viewers blurred).
1. **Done**: FFmpeg 8.1 LGPL (downloaded with OK); NVENC 9-13 GB per camera per day on a busy street, 2-6% of a core.
2. **Done**: `labeler/recorder.py`; camera loop speed unchanged within noise; crash-safe files.
3. **Done**: Recordings page; admins the file, viewers a blurred stream; links from people and alerts.
4. **Done**: Settings, Health, installer and offline package, README, data-protection template.
Waiting for the user: a long-run check with recording on, and the webcam measurement (lens cover open).

## Phase 14: A camera at another site (done 2026-10-12)

The user's choices: a setup script for the other PC, kiosk mode. `remote-camera/` (setup-camera.bat,
remove-camera.bat, README.txt): MediaMTX + FFmpeg, RTSP over TCP on 8554 with a random password, readable only from
Tailscale addresses, firewall rule, autostart. Measured with a video file: 1.4 Mbit/s, read at ~20 frames/s, refused
without the password or from the home network, recovers by itself after the stream stops. Waiting for Ti Xuan to run
it on their PC. Ran there 2026-10-12: blocked by the company ESET firewall; left aside by the user.

## Phase 15: An extra camera that works anywhere (done 2026-10-12, except a test with a real phone or PC)

**Approved plan: [PHASE15-PLAN.md](PHASE15-PLAN.md)**: the other device sends its picture to this PC (MediaMTX
here receives), so it needs no incoming port: works through company firewalls, and with a phone as a camera.
Done: `labeler/receiver.py`, the sending `remote-camera/` script, the Settings page box. Waiting for the user: this
PC's Tailscale connected again, the firewall rule for 8554/1935, then a phone or PC as a real test.

## Phase 16: Search recordings by words (approved 2026-10-12, not started)

**Approved plan: [PHASE16-PLAN.md](PHASE16-PLAN.md)**: type a description, get video moments from all cameras to
play (built on the Ask box, the recordings table and the Recordings page). Start after the Phase 15 phone test.

## Phase 17: Mobility aids and fall alerts (done 2026-10-13)

**Plan: [PHASE17-PLAN.md](PHASE17-PLAN.md)** (user's choices: mobility aids on by default with a switch; fall alerts as
a pop-up with sound on every page; measured on public fall videos). The Phase 15 phone test and Phase 16 were skipped
for now by the user (2026-10-13).
1. **Done 2026-10-13**: mobility-aid label (wheelchair 211 of 218 on held-out Open Images photos, 0 false alarms in
   1,532 COCO people, +0.13 s per person).
2. **Done 2026-10-13** (with step 4's measuring): fall detection from body points; side camera 13/15 falls and 1/20
   false alarms on the second half of UR Fall (15/15, 0/20 on the first), 0/7 on walking videos; ceiling cameras not
   supported.
3. **Done 2026-10-13**: the built-in "Fall" rule, a red pop-up with a beep on every page until OK, clips that show
   the fall; a person can be alerted for a second fall.
5. **Done 2026-10-13**: Settings (fall alerts on/off, seconds, per camera) and docs. **Phase 17 is done.**

## Phase 18: Vehicles (done 2026-10-13)

**Plan: [PHASE18-PLAN.md](PHASE18-PLAN.md)** (user's choices: cars, motorbikes, bicycles, buses and lorries; no number
plates; type, color and make/model).
1. **Done 2026-10-13**: detect, track and store vehicles; parked vehicles one record (MOT16-04); COCO cars 79% found.
2. **Done 2026-10-13**: type, color, make and model by the vision model (Stanford Cars: make 88%, model 57%, type 93%).
3. **Done 2026-10-13**: Vehicles list on the Live page, Ask box questions, Reports, counting lines.
4. **Done 2026-10-13**: alert rules on vehicles (type, color, zone, time of day), pop-ups and clips.
5. **Done 2026-10-13**: Settings, Health, installer/offline package, docs. **Phase 18 is done.**

## Phases 19-22: Home/Away, loitering, animals, scene questions (approved 2026-10-13; Phases 19-21 done 2026-10-14)

**Plan: [PHASE19-22-PLAN.md](PHASE19-22-PLAN.md)**: the user chose all four new ideas as future phases, in this order:
19 Home/Away arming (rules "only when Away"), 20 loitering alert (stayed in an area too long), 21 animals (with the
vehicle detector), 22 scene questions (the vision model answers yes/no questions about a camera every few minutes).
Ask the user before each phase and each step.

Phase 19 started 2026-10-14 (user's choices: admins **and viewers** may switch; a weekly schedule).
1. **Done 2026-10-14**: Home/Away state (`labeler/arming.py`, table `arming`), weekly schedule, `/api/arming`.
2. **Done 2026-10-14**: rule condition `armed = away / home` (people and vehicles; the Fall rule stays always on).
3. **Done 2026-10-14**: Home/Away button on every page (asks first), Live tile, Health line, Alerts page condition,
   Settings schedule box, README "Home / Away".
4. **Done 2026-10-14**: the new table checked on PostgreSQL (scratch database), docs. **Phase 19 is done.**
   Next: Phase 20 (loitering alert); ask the user first.

Phase 20 started 2026-10-14.
1. **Done 2026-10-14**: condition `stayed = <area>:<seconds>` for people and vehicles, checked once a second
   (`Pipeline._check_stays`, `AlertEngine.stay_limits`); real run: 4 people and a bus alerted 21 s into a 20 s rule.
2. **Done 2026-10-14**: Alerts page "stayed in area (loitering)" (area x 30 s ... 1 h); with hours and Away: none at
   Home, all 4 people within 3 s of switching to Away; README "Loitering".
3. **Done 2026-10-14**: the live picture shows the time inside areas that have a loitering rule ("street 1:03" above a
   person's label, "#2 bus parked street 1:02" on a vehicle).
4. **Done 2026-10-14**: `tools/loiter_check.py` on MOT16-02/04/09 (people at least half visible): stayed 5 s found
   51/93, 6 false of 75 alerts; 15 s found 13/47 (tracks break). Tried, not kept: carrying the time over a re-link
   (same found, 2 more false). **Phase 20 is done.** Next: Phase 21 (animals); ask the user first.

Phase 21 started 2026-10-14 (user's choices: dogs, cats and birds; security cameras only, like vehicles).
1. **Done 2026-10-14**: one detector pass for vehicles and animals; `labeler/animals.py` (AnimalStore; the vehicle
   watcher with animal names, "resting"), tables `animals`, `animal_zone_events` (checked on PostgreSQL), retention,
   orange boxes on the live picture. Seen on a COCO photo: the husky (0.40-0.45) is below the tracker's 0.5 start
   threshold: to be measured in step 4.
2. **Done 2026-10-14**: "Animals seen" on the Live page (kind, color, day, delete; viewers may look), Ask box words
   (dog, puppy, cat, kitten, bird, crow, pigeon, pet, animal ...; `agent.animal_query`), Reports "Animals per hour".
   Plan split with the user: 3 alert rules, 4 measuring, 5 Settings/Health/docs.
3. **Done 2026-10-14**: alert rules `animal = any/dog/cat/bird` with zone, hours, Away and stayed
   (`AlertEngine.check_animal`, `alerts.rule_kind`); `alert_events.animal_id` (NEW_COLUMNS, checked on PostgreSQL);
   pop-ups, clips, Alerts list; deleting an animal deletes its alerts.
4. **Done 2026-10-14**: `tools/animal_check.py`: COCO dogs 75%, cats 83%, birds 68% (medium/large, at 0.5; 0.4 not
   worth it); false alarms 0 on MOT16, 3 on UR Fall (a person crawling = "dog"). Tried, not kept: ignoring animal boxes
   that are a found person (lost real animals). README "Animals".
5. **Done 2026-10-14**: Settings (Animals section and column), Health "animals" line, docs. **Phase 21 is done.**
   Next: Phase 22 (scene questions); ask the user first.

Phase 22 started 2026-10-15 (user's choices: every 5 minutes by default; alert after two answers in a row).
1. **Done 2026-10-15**: tables `scene_questions`, `scene_answers` (checked on PostgreSQL; answers kept like the audit
   log), admin API `/api/scene`, "Scene questions" box on the Alerts page.

## Critical files

- `labeler/pipeline.py` (detector call, labeling, zones/health hooks), `labeler/web.py` (auth, new pages),
  `labeler/db.py` (users, audit, zone_events tables), `labeler/reports.py`, `labeler/alerts.py`,
  `config.yaml` / `labeler/config.py` (new options with defaults), `run.py`, `labeler/static/*.html`.
- New: `labeler/detect.py`, `tools/dataset_from_db.py`, `LICENSES.md`, `docs/`.

## Verification (every phase)

- `pytest -q` stays green; add tests for each new table, endpoint and rule (pattern: `tests/test_*.py`
  with the `db` / `camera` fixtures in `tests/conftest.py` and FastAPI `TestClient`).
- Run the app on port 8001 with the demo photo and the webcam; check pages in the browser pane and stop
  via `/api/shutdown`.
- Re-score with `tools/evaluate.py` and the Answer key page after any model or prompt change; re-run
  `tools/bench_age.py` / `tools/bench_vlm.py` when speed or age could be affected.
- Phase 1 specifically: the tracking gap tests and FPS must match or beat the YOLO baseline before
  switching the default backend.
- Commit each phase separately with a descriptive message.
