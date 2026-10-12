# VisionLabeler: session handoff

Read this first in a new session. It says what the system is, how it is built, how to run and test it, what to
watch out for, and what to do next. The plan is in [ROADMAP.md](ROADMAP.md); user-facing docs are
[README.md](../README.md) (every feature, with measured numbers) and [START-STOP-GUIDE.md](../START-STOP-GUIDE.md).

## 1. What it is, and for whom

A local AI camera system on one Windows 11 PC (RTX 4070 SUPER 12 GB). When a person appears on a camera it
detects and tracks them, labels them live ("man, age 30-39, red t-shirt, blue jeans, carrying backpack"), and
records them: searchable, counted in zones, reported, alerted on. Everything runs on this PC; no cloud.

The user (non-native English speaker; prefers **short, simple, concrete answers**; in **Malaysia**) uses it **at
home, for personal use** (home security and testing the kiosk features). It is **not commercial yet**, so
Roadmap Phase 1 (licenses) is deferred until they say otherwise. Under Malaysia's PDPA, purely personal
household use is generally outside the Act; their filled-in camera sign and data-protection summary are in
`data/privacy/` (git-ignored, contains their email); the blank templates are in `docs/`.

## 2. Current state (2026-10-12)

- Git: branch `main`, pushed to the user's **private** GitHub repository (`origin`, 2026-10-10; the first push
  was signed in by the user in the browser, Git Credential Manager keeps it). Push after each commit. Roadmap **Phases 0, 2-11 done**, Phase 12 step 1 measured (not adopted), Phase 13 (recording) done; Phase 1 deferred. Phase 11: `5341d56` keyboard answer key. Phase 10:
  `2429790`/`bc23363` soak tool · `0449255` idle mode · `23273b1` restore drill. Phase 9 (small
  fixes): `8675c46` list redraw · `81dbcb2` backup.dir · `482bdd4` CI v7 · `0296435` relink_check help. Phase 8
  commits, newest first: `3794de3` today at a glance · `e3a6179` photo search · `ea97937` original excerpts ·
  `4893b9e` person videos · `bae4034` counting check · `0a44601` maintenance · `f38ea1e` plan. Phase 7
  commits: `e1c1f24` video import · `2c39ed7` second-site checks · `bf55621`/`0dacfe2` daily
  backup · `fab8340` plan. Phase 6 commits: `2b6d3f2` GitHub workflow · `39c7dd0` installer · `a63017b` Settings page · `4fb0c3b`
  health check · `f32543d` log files · `be09f44` stop.bat fix. Earlier: `958d80c` spot checks · `a18e8a8` weekly
  retraining · `e54c242` capacity · `be9c236` far-away people · `953b756` timeline · `14cc7b3` kiosk prompt ·
  `21dc165` night mode · `ab90e2e` reports by hour · `722c4b5` alert clips · `21e9d2d` camera health ·
  `1b51cb6` time in view · zones (`bbcacf6`, `1b43675`, `66a0d83`) · `a777656` remote access ·
  privacy/security (`853b25d` … `f81ebd2`).
- Tests: **504 pass** (`.venv\Scripts\python.exe -m pytest -q`, ~45 s).
- The user's webcam had its **lens cover closed** for the whole 24-hour check (2026-10-11); they were told to open
  it. A blank picture shows as a camera problem on the Live and Health pages.
- The user's config: one USB webcam (`cam1`, kiosk mode); `web.host: 0.0.0.0` and `web.https: true` (ready for a
  second site over Tailscale, not set up yet: see README "From another site"); one viewer account `TIXUAN`.
- The user's database: 27 webcam people, 26 of them in the answer key (their own labels; mean ~75% before the
  2026-10-08 prompt change, ~89% for the VLM alone after it on those people), plus **127 MOT16 people** imported
  2026-10-12 as the recorded camera `mot16` (dated March 2016; CC BY-NC-SA 3.0, non-commercial). With them the
  weekly retraining ran and was kept for the first time (`data/models/attributes.joblib`, 61.8% → 62.1%).
- BitLocker is on for drive C: (the user turned it on 2026-10-08).
- **Docker was not running** at the end of the session (the database is in Docker): `start.bat` starts it;
  tools like `tools/users.py` need it running first (they hang otherwise).
- Proposal card in the user's session list: "Import recorded videos (fast, with real times)" (not started).

## 3. Architecture

```
camera(s) ─ labeler/sources.py (MSMF webcam, files/images at their frame rate, RTSP; latest-frame thread)
   └─ labeler/pipeline.py  one Pipeline per camera (thread)
        YOLO11s-pose + ByteTrack (labeler/trackers/people.yaml); per-camera detector size (cameras[].imgsz)
        TrackState per person: best frames, colors, pose, thumbs (+ blurred), zone state, time in view/facing
        re-linking of returning people (SigLIP similarity + position + time)
        per person: colors (colors.py), embedding (embed.py, SigLIP 2), fast classifiers (student.py),
                    age model (age.py), VLM job (vlm.py → Ollama qwen3-vl:4b-instruct), re-analysis every 15 s,
                    alerts (alerts.py; conditions on labels and "zone"), alert clips (clips.py, WebM/VP8)
        night mode: mostly black-and-white frames → no colors from any source (_is_night, night_ids)
        zones.py: areas (enter/exit, dwell) and counting lines (in/out); drawn on the live picture
   └─ labeler/db.py  SQLAlchemy models + queries (Postgres+pgvector; SQLite for tests)
        tables: cameras, tracks (+in_view_s, facing_s), attributes, embeddings, reviews, alert_rules,
                alert_events, answer_key, users, login_sessions, audit_log, zones, zone_events,
                camera_events, model_history, accuracy_history
        new columns on existing tables: db.NEW_COLUMNS (added at startup); new tables: create_all
   └─ labeler/web.py FastAPI app + labeler/static/*.html (no build step; session.js shared by all pages)
        pages: / (Live: video, zones tool, people list, Ask, Look-alikes, Timeline), /review, /reports,
               /alerts (rules, alerts + clips, camera problems), /answer-key (labeling, score, accuracy
               history, retraining), /audit, /settings, /health, /login
        login middleware: this PC (127.0.0.1 without forwarding headers) = admin; others log in;
        viewer = allowlist in accounts.py; pages/scripts sent with Cache-Control: no-cache
labeler/accounts.py  scrypt passwords, roles, login lock-out; tools/users.py manages accounts (+ audit)
labeler/privacy.py   face blurring (viewers; optional on disk), picture/clip removal
labeler/tls.py       self-signed HTTPS certificate (web.https: HTTPS on 8443 + plain HTTP on 127.0.0.1 only)
labeler/health.py    camera problems (offline / blank / slow) → camera_events, Live banner, Alerts history
labeler/timeline.py  where a person was seen: look-alikes across cameras in time order, overlaps removed
labeler/reports.py   summaries, hours of day, dwell histogram, engagement, CSV
labeler/training.py  fast-classifier training (shared by tools/train_attributes.py and retrain.py)
labeler/retrain.py   weekly retraining on own cameras with an answer-key gate; daily accuracy snapshot
labeler/agent.py     Ask box: question → validated search plan (local model) → DB filter + embedding rank
run.py               builds everything; .env loaded first; retention (startup + hourly), health, retrain
                     threads; one server or HTTPS pair (_web_servers); watchdog; exit 3 = crash
labeler/supervisor.py restarts run.py after crashes (5 in 10 min → gives up); used by start.bat
labeler/logs.py      console + data/logs/visionlabeler.log (10 MB × 5) + supervisor.log; RecentErrors
labeler/healthcheck.py /api/health + /health: db, Ollama, cameras, disk, backup, retrain, log errors;
                     200/503; ?token= (web.health_token) = status only; config.mask_password for URLs
labeler/backups.py   daily backup thread (tools/backup_db.py; --copy-to, weekly --verify into a test DB)
labeler/clips.py     alert clips + person videos (start/finish/drop_person; keys from Pipeline._clip_keys;
                     480 px, 5 fps, 20 s; tracks.video_paths; web.videos() = admins, written files only)
labeler/importer.py  Import page: start times (typed/name/mvhd/mtime), RecordedVideo (sources.py),
                     VideoClock (pipeline.clock), one job at a time, VLM throttle, no alerts/clips;
                     cameras.retention_days (0 = keep) used by _apply_retention; run.py routes VLM answers;
                     tracks.source_file/source_offset_s, make_excerpt -> clips/excerpt_<id>.webm (admins)
labeler/today.py     Today at a glance (/api/today): same weekday, last 4 weeks, by the same time
labeler/photo.py     search by photo: own YOLO copy (lazy), person in the middle else largest, Embedder,
                     db.nearest; raw request body (no python-multipart); POST /api/search/photo (viewers too)
labeler/remote.py    Health checks for other devices / Tailscale / firewall (netsh, read-only, cached 5 min)
labeler/settings.py  Settings page: data/settings.yaml merged over config.yaml in load_config (only the
                     project's config.yaml, or a config naming settings_file:); validated; masked passwords;
                     POST /api/restart → exit 4 → supervisor restarts at once (VL_SUPERVISED=1 from it)
start.bat / stop.bat daily start/stop (wait for Docker + Ollama; stop waits for app exit, backs up DB)
install.bat/.ps1     new PC: winget Python/Docker/Ollama, .venv, .env, DB, models, admin, autostart,
                     health; -CheckOnly, -Offline <pkg>; PowerShell 5.1 + ASCII only (tests check)
tools/relink_check.py counting check against MOTChallenge gt.txt (true identities); per-join right/wrong
tools/models.py      check / download / pack / unpack models (HF cache, Ollama manifest+blobs, YOLO)
tools/pack_offline.ps1 offline package: pip freeze wheels (cu128 index), models, docker save
tools/               evaluate.py (COCO answer key), bench_vlm.py, bench_age.py, teacher_label.py,
                     train_attributes.py, extract_people.py, backup_db.py, users.py, make_test_config.py
```

Key decisions (keep unless there is a measured reason; numbers in README):
- Fixed vocabulary, JSON-schema VLM output; carried items as yes/no per item. Kiosk prompt names the pictures
  ("Images: 1 = …; 2 = …") and defines t-shirt vs shirt and thin glasses.
- Age from the age model; the fast classifiers give no age while it is loaded; a VLM answer beats theirs.
- `vlm.workers: 1`, Ollama at `127.0.0.1`. Rejected after measuring: several people per VLM request, Qwen3-VL 8B,
  extra upper-body close-up, 2x2 tiled detection, H.264 clips (no encoder here).

## 4. Environment

| Thing | Where / how |
|---|---|
| Python | `C:\VisionLabeler\.venv` (Python 3.13, torch 2.11 cu128); `cryptography` added for HTTPS |
| Database | Docker Desktop container `visionlabeler-db` (pgvector/pgvector:pg17), 127.0.0.1:5432, URL in `.env` |
| Docker CLI | may be missing from PATH: `C:\Users\JAMES\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe`; start: Docker Desktop, then `docker compose up -d --wait` in the project (what start.bat does) |
| Ollama | `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`, starts with Windows; `qwen3-vl:4b-instruct` used |
| Models | `yolo11s-pose.pt`; SigLIP 2 + age model in the Hugging Face cache; `models/attributes.joblib` (shipped fast classifiers, in git); `data/models/attributes.joblib` (retrained, preferred when present) |
| Datasets | `data/datasets/coco_people` (1,534 crops, teacher labels, embeddings, `answer_key.jsonl`); `data/datasets/coco/val2017` (5,000 images) + `labels/val2017` (YOLO boxes, from Ultralytics' coco2017labels.zip, downloaded with the user's OK); `data/datasets/fairface` |
| User files | `data/privacy/` (filled sign + summary), `data/tls/` (certificate, made on first HTTPS start), `data/crops/` (+ `blurred/`, `clips/`), `data/backups/` |
| Node.js | used by `tests/test_pages.py` to syntax-check page scripts and `static/*.js` |

## 5. Run and test

```bash
.venv\Scripts\python.exe -m pytest -q
```

Try the app **without touching the user's data** (scratch database, pictures, certificate, retrained models):

```bash
.venv\Scripts\python.exe tools\make_test_config.py <scratch folder>
```

```bash
.venv\Scripts\python.exe run.py --config <scratch folder>\test_config.yaml --source samples\bus.jpg --port 8001
```

Open http://127.0.0.1:8001 (the browser pane works; this PC is admin), stop with
`curl -X POST http://127.0.0.1:8001/api/shutdown`, then check no `python.exe run.py` is left. Headless:
`--no-web --max-frames 100`. Options: `--two-cameras`, `--https`, `--fast-health`. For page-only checks with
made-up data, serve `create_app(cfg, db, None)` with uvicorn on 8001 from a script.

## 6. Pitfalls (each cost real time)

1. **Bash heredocs collapse `\\` into `\`** (broke a page script and `stop.bat`). Edit with the Edit/Write
   tools, or write a Python script to a file. Never `rm` with an unguarded variable.
2. **Leftover test copies**: confirm no `python.exe run.py` remains (`Get-CimInstance Win32_Process -Filter
   "Name='python.exe'"`). An app shutting down saves people first: wait for the processes to go.
3. **`localhost` adds ~2 s** per HTTP request on this PC; use `127.0.0.1`.
4. **Webcam**: Media Foundation with `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0`; first frames can be blank.
5. **The user's own copy** on port 8000: don't stop it without asking; test on 8001.
6. **Privacy**: prefer the API/logs over screenshots of the webcam; delete test recordings. Never write the
   user's email or `data/privacy/` content into tracked files.
7. **Batch files** keep CRLF (`.gitattributes`).
8. **Commit messages with double quotes** break in PowerShell 5.1: write the message to a file, `git commit -F`.
9. **The real config opens the app to the network with HTTPS**: always test with `tools/make_test_config.py`.
10. **Schema**: new columns go in `db.NEW_COLUMNS`; new tables appear by themselves. Checked on PostgreSQL.
11. **Docker after a reboot** takes ~1-2 min; anything touching the database hangs until then.
12. **Browser pane**: it resizes while you work, so draw charts at the element's width and redraw on resize
   (ResizeObserver); screenshots sometimes time out: measure with JavaScript instead.
13. **Cached pages**: fixed (no-cache), but browsers that cached older pages need Ctrl+F5 once.
14. **`Remove-Item` with a variable path** may be blocked by the tool's safety check: use plain paths.
15. **Windows' 260-character path limit**: Hugging Face cache paths are long; copies inside a long folder fail
   with WinError 3. `tools/models.py` copies with `\\?\` paths and skips `.no_exist`.
16. **PowerShell functions return all uncaptured output**: pipe native programs to `Out-Host` inside a function
   that returns a value (`Run` in install.ps1). Scripts must stay ASCII (5.1 reads them as ANSI; a test checks).
17. **Tests must never write the user's files**: the `supervisor` fixture and `settings_file:` in test configs
   keep logs and settings in tmp folders; `load_config` uses `data/settings.yaml` only for the real config.yaml.
18. **"now" twice in a test** can fall on both sides of midnight: make people at noon of a chosen day and ask
   for that day (fixed 2026-10-10 in test_alerts_reports and test_zones; it failed once at 00:00).

## 7. Conventions

- Match the existing style: plain-language docstrings and comments that say *why*; no build tooling for pages;
  config options get a default in `labeler/config.py` and a commented line in `config.yaml`.
- Every change: tests, full suite green, a real run on port 8001 (scratch config) for anything touching the
  pipeline or pages, one commit per feature ending with the Co-Authored-By line; README/HANDOFF/ROADMAP updated.
- Measure before claiming a gain; report numbers honestly, including rejected ideas and corrections.
- Charts follow the dataviz skill (validated palette, one axis, hover, table view, both color modes).
- Ask before downloads, installing packages, or anything outward-facing; the user decides.

## 8. Phase 6, 7 and 8 results, and what is next

**Phase 6 (operations) is done** (plan: [PHASE6-PLAN.md](PHASE6-PLAN.md)): log files, health check, Settings page
with Restart now, installer with offline package, GitHub test workflow. Not done, by the user's choice: a Windows
service (console window kept). GitHub: pushed 2026-10-10; the first Actions run is checked by the user (the
browser pane is not signed in to GitHub, and the repository is private).
Untested installer parts (user said skip): the offline package's Python wheels (~4 GB download) with a fresh
`.venv`, `docker save` of the database image, the installer's database/account checks with Docker running.

**Phase 7 is done** (2026-10-10, plan: [PHASE7-PLAN.md](PHASE7-PLAN.md)):
- Daily backup while running, second copy (`backup.copy_to`), weekly restore check; checked on the real database
  (27 of 27 people restored, 2 s).
- Second site: Health page checks every step, guide `docs/SECOND-SITE.md`. **Still to do with the user**
  ("later"): Tailscale on the other PC, the firewall rule (admin PowerShell, the user runs it), first login.
  This PC: Tailscale running, no other device yet, no firewall rule yet.
- Import page for recorded videos (README "Importing recorded videos", measured on OpenCV's vtest.avi).

**Phase 8 is done** (2026-10-10, plan: [PHASE8-PLAN.md](PHASE8-PLAN.md)):
- Maintenance: the midnight test found and fixed; certificate end date on Health; pip-audit (setuptools 81.0;
  PyTorch 2.11 kept, its advisory is about torch.jit.script); Dependabot security PRs (the user turns on alerts in
  the repository settings); the offline installer tested end to end except the database image.
- Counting check against true identities (`tools/relink_check.py`): security cameras now re-link by appearance
  only at 0.98 (at 0.90, 20 of 24 joins were wrong); kiosk stays 0.90.
- A video of each person (480 px, median 1.3 MB) and the storage line on Health; original-video excerpts for
  imported people; search by photo (a hint on far cameras: 6 of 18 first); today at a glance on the Live page.

**Phase 9 (small fixes) is done** (2026-10-10, [PHASE9-PLAN.md](PHASE9-PLAN.md)): the people list redraws only
when it changed; `backup.dir` (test copies keep their own backups); CI actions v7; relink_check help;
the offline package's database image and the installer's database checks tested.

**Phase 10 is done** (2026-10-12, [PHASE10-PLAN.md](PHASE10-PLAN.md); numbers in README "Speed" and
"Reliability"):
- `tools/soak.py`: 24 h on a webcam test copy, no leak or error (no people: lens cover closed; the user chose no
  second run). Watch **private** memory (Windows trims RSS) and the app's own GPU memory (Windows "GPU Process
  Memory" counter; nvidia-smi shows N/A per process here, and the whole GPU's number includes other programs).
- Idle mode (`idle:`; Pipeline.skip_while_idle / _wake_on_detection): measured with a synthetic video (an empty COCO
  photo with COCO person crops walking across, 12 arrivals; the script was in the session scratchpad, not kept).
  Lesson: a video file plays every frame, so time it by frame number, not the clock (a busy PC makes it lag).
- `tools/restore_drill.py`: whole way back in 22 s. **Backups hold the database only**: pictures and person
  videos (`data/crops`) are not backed up anywhere.

**Phase 11 is done** (2026-10-12): Answer key by keyboard (labels stay hidden, the user's choice; before about 34 s
per person, the page shows the new pace: not measured with the user yet); MOT16 downloaded (with OK) and imported
by the user on the Import page (my API call into the real app was refused by the permission check: the user starts
writes to their real app themselves); first kept retraining, no measurable change on 26 answer-key people.
Lesson: the auto-mode check blocks writes to the user's real app (port 8000): give the user the steps instead.

**Phase 12 step 1 was measured** (2026-10-12, the user asked to go ahead before adding cameras): OSNet-AIN
(`labeler/reid.py`, weights in `data/models/reid`, downloaded with OK) was not better than SigLIP 2 inside the
pipeline on 7 MOT16 clips (look-alikes 24/74 vs 34/74; README "One person, one record"); not adopted, kept as
`tools/relink_check.py --reid`. MOT16 train clips as videos with walking-people-only markings:
`data/datasets/mot16/train_videos`, `data/datasets/mot16/gt` (use these with relink_check).

**Phase 13 (recording) is done** (2026-10-12, [PHASE13-PLAN.md](PHASE13-PLAN.md) has the numbers):
`labeler/recorder.py` (fed from `Pipeline.run` with every frame read; FFmpeg in `data/ffmpeg`, NVENC),
`labeler/recordings_web.py` + Recordings page, table `recordings`, `tools/get_ffmpeg.py`. Lessons: run.py waits only
15 s for a camera loop at shutdown, so anything slow in its `finally` (person videos) must come after quick closes;
the shared `.player` class in style.css is the full-screen video pop-up (don't reuse the name); CSS `display`
overrides the `hidden` attribute unless a `[hidden]` rule wins. The user's own app records from its next restart.

**Phase 14 (a camera at another site) is done** (2026-10-12, README "A camera at another site"): the user's
second PC (Ti Xuan's, `XY-TIXUAN`, 100.106.249.52) joined their Tailscale network as a member and logged in to the
app (their browser's antivirus HTTPS scanning first showed a different certificate; fixed on their side).
`remote-camera/setup-camera.ps1` (MediaMTX v1.21.2 + FFmpeg, RTSP over TCP with a password, Tailscale-only firewall
rule, autostart) tested here with a video file; network cameras are read over TCP and their passwords masked in
logs. Ti Xuan still has to run it (the user sends them the folder); then the user adds the printed address on the
Settings page (kiosk mode). **Result (2026-10-12)**: Ti Xuan ran it (stream fine on their PC, Windows firewall rule
in place, Tailscale shields down), but their PC is a company PC with **ESET Firewall**, which replaces the Windows
firewall and blocks port 8554 from outside; the user chose to leave the remote camera aside and was told to have
Ti Xuan run remove-camera.bat. Viewing from that PC still works. Lessons: `$Input` is reserved in PowerShell; Start-Process in 5.1 does not quote
arguments with spaces; Go programs (MediaMTX) refuse relative paths to commands, so runOnInit uses the full path.

**Phase 15 (a camera sent from a phone or another PC) is done** (2026-10-12, [PHASE15-PLAN.md](PHASE15-PLAN.md)):
the device pushes to this PC; `labeler/receiver.py` runs MediaMTX (`data/mediamtx`) with the app (cameras with
`source: push`, per-camera send passwords stored hashed, Tailscale-only, read from 127.0.0.1 with a per-start
password via `Pipeline.source_override`, watchdog restart); Settings page box with the send addresses (password
shown once); `remote-camera/` now sends (FFmpeg loop, no admin). Lessons: MediaMTX 1.21 has MoQ on a fixed port
(8892) by default: turn it off or a second copy fails; FFmpeg's RTSP output needs `-timeout` or it hangs forever
when the server dies halfway; `timeout.exe` fails without a visible console (use ping); a `<select>` on the
Settings page must be read as text (saving was broken by the Phase 13 "When" drop-down until this phase).
**Waiting for the user:** this PC's Tailscale was stuck "starting" at the end; the firewall rule for ports
8554/1935; a real phone or second PC as the test.

**Next**: with the user: a long-run check with recording on and the webcam measurement (lens cover). The rest of
Phase 12 (more cameras, per-camera tuning, re-measuring re-identification and the short-gap re-link rule on
outdoor footage) when the user adds cameras.

Open items (also in that plan):
- Pictures and person videos are not in the backups (restore drill, 2026-10-12). The user chose to leave it so
  (2026-10-12); offer again only if they set up a second disk (`backup.copy_to`).
- Second site over Tailscale: waits for the user (docs/SECOND-SITE.md; the Health page checks each step).
- Appearance on far or overhead cameras is weak (re-linking, look-alikes, timeline, photo search). A
  re-identification model (OSNet-AIN) did not help on MOT16 overall but did on a far view: re-measure on the
  user's outdoor footage. On a crowded moving-camera clip the short-gap re-link rule made 7 wrong joins of 8.
- Measure re-linking and photo search on close-up (kiosk) footage with true identities when there is some.
- Phase 1 licenses (only when the use becomes commercial; ask the user then).
- More cameras than ~4: batching or one process per camera (README "How many cameras?").
- The user keeps labeling on the Answer key page (aim 50-100 webcam people; the retraining check is noise with
  26). Measure their seconds per person with the keyboard page (before: about 34 s).
- Before any commercial use: remove the `mot16` camera's people and retrain (MOT16 is non-commercial).
- Vocabulary refinements (coat, polo, tank top) need answer-key values to be measurable.

## 9. Message to start the next session

> I'm continuing the VisionLabeler project in C:\VisionLabeler. Please read docs/HANDOFF.md and
> docs/ROADMAP.md first, run the tests to confirm everything is green, then continue with docs/PHASE10-PLAN.md:
> ask me before each phase starts, and do it step by step.
