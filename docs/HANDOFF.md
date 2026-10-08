# VisionLabeler: session handoff

Read this first in a new session. It says what the system is, how it is built, how to run and test it,
what to watch out for, and what to do next. The detailed plan for the next work is in
[ROADMAP.md](ROADMAP.md); user-facing docs are [README.md](../README.md) and
[START-STOP-GUIDE.md](../START-STOP-GUIDE.md).

## 1. What it is

A local AI camera system (Windows 11 PC, RTX 4070 SUPER 12 GB). When a person appears on a camera it
detects and tracks them, analyzes them live, and records them with labels such as "man, age 30-39, red
t-shirt, blue jeans, carrying backpack". Everything runs on this PC; no cloud.

The user (non-native English speaker, communicates simply) intends it for **both kiosk and security
cameras** and **commercial use**. They have tested the current version and are happy with it.

## 2. Current state (2026-10-07)

- Git: local repo, branch `main`, clean. Latest commits:
  `5912ff8` roadmap · `43d5238` docs · `fe031af` accuracy · `c323fb0` questions + fixes ·
  `b903945` reports/alerts · `99a6f26` cameras/reliability/login · `430bfd1` delete + tests · `cd13ef6` baseline.
- Tests: 157 pass (`.venv\Scripts\python.exe -m pytest -q`, ~25 s).
- 2026-10-08, Phase 4 capacity measured (Phase 4 complete): 2 cameras at full 15 fps, 4 at ~10 fps each, 6-8 at
  5-7 fps with a growing labeling queue (README "How many cameras?"). GPU 65-79% busy, so the limit is probably
  the single Python process; batching or one process per camera not done yet.
- 2026-10-08, Phase 4 far-away people: measured on COCO val2017 (true boxes from Ultralytics'
  coco2017labels.zip, downloaded with the user's OK to data/datasets/coco/labels/val2017). Tiling rejected (13x
  false alarms); per-camera `imgsz` added, guided by the "found by height" table in README (Detection).
- 2026-10-08, Phase 4 timeline done: `labeler/timeline.py` (look-alikes in time order, overlapping sightings
  removed), `/api/tracks/{id}/timeline`, Timeline button on Live. Threshold 0.8 not measured across cameras.
- 2026-10-08, Phase 4 night mode done: mostly black-and-white frames (`Pipeline._is_night`) -> colors stored as
  unknown from every source; night answers are blanked before joining the VLM history (no carry-over at dawn).
- 2026-10-08, Phase 4 kiosk step 2 (clothing): an extra upper-body close-up for the VLM was measured and
  REJECTED (own camera no gain and slower; COCO 89.9% -> 86.1%, bottoms worse). Kept instead: the kiosk prompt
  now names the pictures sent (`vlm.IMAGES_KIOSK`): own camera 85.9% -> 88.9%, COCO (without age) 88.3% ->
  89.5%, but COCO top type 70% -> 63% (2 people). Vocabulary refinements (coat, polo) not tried: the answer
  keys don't have those values, so they can't be measured yet.
- 2026-10-08, Phase 4 kiosk step 1 done: Reports by hour (heatmap hour x gender/age group, time-in-view
  histogram, typical day over several days; `reports.hours_of_day`, `reports.dwell_histogram`). Charts share
  `drawBars` in reports.html; heatmap blues skip the middle steps so cell numbers keep >= 5.1:1 contrast.
- 2026-10-08, Phase 3 step 4 done (Phase 3 complete): alert clips (`labeler/clips.py`; files
  `<crop_dir>/clips/alert_<id>.webm`, named after the alert, so no DB column; route `/clips/{name}`, admin).
  OpenCV here cannot write H.264 (no OpenH264 DLL); VP8 WebM works and plays in browsers.
- 2026-10-08, Phase 3 step 3 done: camera health (`labeler/health.py`, table `camera_events`, thread in
  run.py, `camera_problems` in /api/status, banner + pop-up on Live, history on Alerts). A missing video *file*
  still stops the app (setup error, restarted by start.bat); an unreachable camera is reported as offline.
- 2026-10-08, Phase 3 step 2 done: time in view and facing the camera per person (`tracks.in_view_s`,
  `tracks.facing_s`, `geometry.facing_camera`), in Reports and the CSV. Facing threshold set from 6 demo people.
- 2026-10-08, Phase 3 step 1 done: zones and counting lines (`labeler/zones.py`, tables `zones` and
  `zone_events`, API `/api/zones`, drawing tool on the Live page, counts in Reports, alert condition
  `["zone", name]` in `labeler/alerts.py`). Pages are now sent with Cache-Control: no-cache.
- 2026-10-08, remote access for a second site: the user's config now has `web.host: 0.0.0.0` and
  `web.https: true`; plan is Tailscale on both PCs (README "From another site"). Without an account the app
  falls back to this PC only (`web.check_access`). BitLocker on C: turned on by the user (was encrypting).
- 2026-10-08, Phase 2 step 6 done (Phase 2 complete): `docs/camera-notice.md`, `docs/data-protection.md`
  (templates; keep them in line with the code when storage or access changes). run.py now loads `.env` before
  any import, so `HF_HUB_OFFLINE=1` there works.
- 2026-10-08, Phase 2 step 5 done: per-camera `retention_days`; retention now also runs every hour
  (`run.py:_retention_loop`), not only at startup. Erasure procedure in README ("Erasing one person's data").
- 2026-10-08, Phase 2 step 4 done: face blurring (`labeler/privacy.py`, `geometry.head_box`): viewers get
  blurred live video/thumbnails/pictures (`web.blur_faces_for_viewers`, default on; blurred copies in
  `data/crops/blurred/`, served by the `/crops/{name}` route); `storage.blur_faces` blurs saved pictures on disk.
- 2026-10-08, Phase 2 steps 1–3 done: login accounts with admin/viewer roles replace the shared password;
  audit log (table `audit_log`, admin page /audit, `audit()` calls in `labeler/web.py` endpoints);
  HTTPS (`web.https`, off by default): HTTPS on 8443 for the network + plain HTTP on 127.0.0.1 only
  (`run.py:_web_servers`, `labeler/tls.py`). New package: `cryptography` (installed, in requirements.txt).
  The user's config has `web.host: 127.0.0.1` and no accounts, so nothing changes for them until they open
  the app to the network.
- 2026-10-08, Phase 0 step 1 done: teacher labels redone with the current prompt (1,534 people, 57 min;
  old labels kept as `data/datasets/coco_people/teacher.old-prompt.jsonl`), fast classifiers retrained:
  answer key 67% → 75%, gender 56% → 97%. They still cover every attribute for 0% of people, so the VLM
  labels everyone. While the age model is loaded they give no age (it would outrank the better age model).
  Step 2 (the user labels 50–100 own people on the Answer key page) is in progress: 26 people on cam1,
  mean 75% (2026-10-08; too few to trust yet). Weakest: upper_type 45%, eyewear 57%, upper_color 68%.
  (It first showed 65%: age was scored 0 because the page asks age_group and kiosk stores age_range; fixed.)
- 2026-10-08, kiosk prompt: +2 sentences (t-shirt vs shirt, thin/clear glasses). Own webcam 80% → 86%,
  COCO key 88% → 85% (more "unknown" for legs only). See README "Accuracy". Fast classifiers retrained on
  labels from this prompt (51 min; previous labels kept as `teacher.before-tee-prompt.jsonl`).
- The fast classifiers transfer poorly from COCO to the user's webcam, so a VLM answer now beats a student
  answer in `db.resolve`. Real fix: Phase 1 step 4 (train them on the user's own reviewed people) once there
  are enough; until then they mainly give early live labels.
- Phase 1 (licenses) deferred by the user: not for commercial use yet. Don't start it until they say so.
- Config: one camera (`cameras: [{name: cam1, source: 0}]`, the USB webcam), kiosk mode.
- Database: holds ~55 test people from development (demo photo and webcam tests). The user can remove them
  with Delete all; don't delete them without asking.
- Measured: 85% mean accuracy on a hand-made 39-person answer key (88% before the 2026-10-08 prompt change); age model 60% exact / 96% within one
  group on 300 FairFace faces; ~2 s per person for full labels; ~30 fps detection.

## 3. Architecture

```
camera(s) ─ labeler/sources.py (MSMF webcam, files, RTSP; latest-frame reader thread)
   └─ labeler/pipeline.py  one Pipeline per camera (thread)
        YOLO11s-pose + ByteTrack (labeler/trackers/people.yaml)
        TrackState per person: best frames, colors, pose, live thumb, labels
        re-linking of returning people (SigLIP similarity + position + time)
        ignore rules: kiosk min box 3%, ignore_zones; deleted people stay ignored
        per person: colors (labeler/colors.py), embedding (labeler/embed.py, SigLIP 2),
                    fast classifiers (labeler/student.py), age model (labeler/age.py),
                    VLM job (labeler/vlm.py → Ollama qwen3-vl:4b-instruct), re-analysis every 15 s,
                    alert check (labeler/alerts.py)
   └─ labeler/db.py  SQLAlchemy models + queries (Postgres+pgvector; SQLite works for tests)
        tables: cameras, tracks, attributes, embeddings, reviews, alert_rules, alert_events, answer_key,
                users, login_sessions, audit_log
   labeler/accounts.py  password hashing (scrypt), roles (admin/viewer allowlist), login lock-out;
                        tools/users.py manages accounts. This PC (127.0.0.1) is always admin, except requests
                        carrying forwarding headers (web.is_this_pc): a proxy/tunnel on this PC must not be admin.
   └─ labeler/web.py FastAPI app + labeler/static/*.html (no build step)
        pages: / (Live), /review, /reports, /alerts, /answer-key, /audit, /login
        API: /stream, /api/live, /api/status, /api/tracks, /api/tracks/delete, /api/search, /api/ask,
             /api/review, /api/reports/*, /api/alerts/*, /api/answer-key/*, /api/shutdown
run.py            builds everything; single-copy check; watchdog; PID file data/app.pid; exit 3 = crash
labeler/supervisor.py  restarts run.py after crashes (5 in 10 min → gives up); used by start.bat
labeler/agent.py  Ask box: question → validated search plan (local model) → DB filter + embedding rank
labeler/reports.py summaries and CSV;  labeler/schema.py the fixed attribute vocabulary
start.bat / stop.bat  daily start/stop (wait for Docker+Ollama; stop waits for app exit, backs up DB)
tools/            bench_vlm.py, bench_age.py, evaluate.py (answer key), teacher_label.py,
                  train_attributes.py, extract_people.py, backup_db.py
```

Label sources stored per attribute: `vlm`, `color_estimator`, `student`, `age_model`, and `human` (review).
Disagreeing confident sources or low confidence → review queue. A human answer settles all sources.

Key design decisions (keep unless there is a measured reason):
- Fixed vocabulary, JSON-schema output from the VLM; carried items asked as yes/no per item.
- Age from the age model, not the VLM (measured better); age_range in decades (0-9 … 70+).
- VLM at `127.0.0.1`, `vlm.workers: 1` (2 at once measured no faster once the localhost delay was fixed).
- Rejected after testing: several people per VLM request (mixes people up), Qwen3-VL 8B (no better, slower),
  box-motion and VLM "is this a real person" filters (didn't separate screens from people).

## 4. Environment

| Thing | Where / how |
|---|---|
| Python | `C:\VisionLabeler\.venv` (Python 3.13, torch 2.11 cu128) |
| Database | Docker Desktop container `visionlabeler-db` (pgvector/pgvector:pg17), 127.0.0.1:5432, password + URL in `.env` (private, not in git) |
| Docker CLI | may be missing from PATH in tool shells: `C:\Users\JAMES\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe` |
| Ollama | `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`, starts with Windows; models `qwen3-vl:4b-instruct` (used), `qwen3-vl:8b-instruct` and `qwen3-vl:4b` (unused, can be removed) |
| Models on disk | `yolo11s-pose.pt` (project root), SigLIP 2 + age model in the Hugging Face cache, `models/attributes.joblib` |
| Datasets | `data/datasets/coco_people` (1,534 crops, teacher labels, embeddings, `answer_key.jsonl`), `data/datasets/fairface/validation.parquet` |
| Node.js | installed; used by `tests/test_pages.py` to syntax-check page scripts |

## 5. Run and test

```bash
.venv\Scripts\python.exe -m pytest -q
```

```bash
.venv\Scripts\python.exe run.py --source samples\bus.jpg --port 8001
```

Then open http://127.0.0.1:8001 (the browser pane works), and stop with
`curl -X POST http://127.0.0.1:8001/api/shutdown`. Headless check:
`run.py --source samples\zidane.jpg --no-web --max-frames 40`. Two-camera test: a temporary config with a
`cameras:` list passed via `--config`.

## 6. Pitfalls (each cost real time)

1. **Bash heredocs collapse `\\` into `\`** before Python sees them. `\n` in a string became a real newline
   (broke a page's JavaScript) and `\b` became a backspace (broke `stop.bat`). Make edits with the
   Edit/Write tools, or write a Python script to a file and run it. Never `rm` with an unguarded variable.
2. **Leftover test copies**: after testing, confirm no `python.exe run.py` processes remain
   (`Get-CimInstance Win32_Process -Filter "Name='python.exe'"`). Leftovers once made Ollama 10x slower and
   skewed benchmarks. The app now refuses a second copy on the same port, and stops even with a Live page open.
3. **`localhost` adds ~2 s** to every HTTP request on this PC; use `127.0.0.1` for Ollama and tests.
4. **Webcam**: Media Foundation (`cv2.CAP_MSMF`) with `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0`.
   DirectShow returns black frames from this camera; the first frames can be blank while it warms up.
5. **The user's own window**: if a `start.bat` copy is running on port 8000, don't kill it without asking;
   test on port 8001.
6. **Privacy**: when checking the webcam, prefer the API/logs over screenshots; delete any test recordings.
7. **Batch files** must keep CRLF line endings (`.gitattributes` enforces this).
8. **Commit messages with double quotes** fail from PowerShell 5.1 (it splits them into extra arguments):
   write the message to a file and use `git commit -F <file>`.
9. **Test runs** of the app use a scratch config (database, pictures and certificate outside `data/`) with
   `web.host: 127.0.0.1` and `web.https: false`: the real config opens the app to the network with HTTPS.
10. **Schema changes**: new tables appear by themselves (`create_all`); new columns on existing tables go in
   `db.NEW_COLUMNS` (added at startup).

## 7. Conventions

- Match the existing style: plain-language docstrings and comments that say *why*; no build tooling for the
  web pages; config options get defaults in `labeler/config.py` and a commented line in `config.yaml`.
- Every change: tests added or updated, full suite green, a real run on port 8001 for anything that touches
  the pipeline or pages, then one commit per feature ending with the Co-Authored-By line.
- Measure before claiming a speed or accuracy gain (`tools/bench_*.py`, `tools/evaluate.py`); report
  numbers honestly, including corrections of earlier claims.
- Explain results to the user simply; they prefer short, concrete answers.

## 8. Open items and next steps

Follow [ROADMAP.md](ROADMAP.md). In short:

0. ~~Retrain the fast classifiers~~ (done 2026-10-08). Still open: the user labels 50–100 of their own people
   on the Answer key page; record `GET /api/answer-key/score` (its `cameras` part) as the per-camera baseline.
1. **Commercial licenses** (deferred: not commercial yet): Ultralytics YOLO is AGPL-3.0. Ask the user: buy an Ultralytics license, or replace
   with RF-DETR + RTMPose (rtmlib) + supervision ByteTrack behind a new `labeler/detect.py` interface. Retrain
   the classifiers on the user's own data instead of COCO. Add LICENSES.md.
2. Privacy/security: admin/viewer accounts, audit log, HTTPS, face blurring, per-camera retention.
3. Zones and counting, dwell time, camera-health alerts, alert video clips.
4. Kiosk demographics and clothing detail; security tiling for distant people, IR mode, cross-camera timeline.
5. Scheduled retraining from reviewed labels. 6. Windows service, logs, settings page, installer.

## 9. Message to start the next session

> I'm continuing the VisionLabeler project in C:\VisionLabeler. Please read docs/HANDOFF.md and
> docs/ROADMAP.md first, run the tests to confirm everything is green, then start with Phase 0 of the
> roadmap. Ask me before Phase 1's license decision.
