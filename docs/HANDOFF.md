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
- Tests: 75 pass (`.venv\Scripts\python.exe -m pytest -q`, ~12 s).
- Config: one camera (`cameras: [{name: cam1, source: 0}]`, the USB webcam), kiosk mode.
- Database: holds ~55 test people from development (demo photo and webcam tests). The user can remove them
  with Delete all; don't delete them without asking.
- Measured: 88% mean accuracy on a hand-made 39-person answer key; age model 60% exact / 96% within one
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
        tables: cameras, tracks, attributes, embeddings, reviews, alert_rules, alert_events, answer_key
   └─ labeler/web.py FastAPI app + labeler/static/*.html (no build step)
        pages: / (Live), /review, /reports, /alerts, /answer-key
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

0. Retrain the fast classifiers (teacher labels were made with the old prompt and age ranges; their age part
   is ignored for now). Ask the user to label 50–100 of their own people on the Answer key page.
1. **Commercial licenses**: Ultralytics YOLO is AGPL-3.0. Ask the user: buy an Ultralytics license, or replace
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
