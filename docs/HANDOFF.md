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

## 2. Current state (2026-10-14)

- Git: branch `main`, pushed to the user's **private** GitHub repository (`origin`, 2026-10-10; the first push
  was signed in by the user in the browser, Git Credential Manager keeps it). Push after each commit. Roadmap **Phases 0, 2-11 done**, Phase 12 step 1 measured (not adopted), Phases 13-15 and 17-25 done (25: `21d51cf` · `3f20e28` · `483f048`; 24: `f4f1a1d` · `be64dd8` · `cbaea65`; 23: `02c7cf9` · `2989afc`; 22: `722118c` · `cba3db4` · `14f8c19` · `938f2c3` · Settings; 21: `98be1cd` · `5bc87d8` · `7215d4c` · `8f694b4` · Settings; 20: `98f436a` rule · `89af3de` page · `3141852` live picture · tool; 19: `46daf57` Home/Away state · `1843a06` rules · `b41530b` pages; 17: `81da79e` mobility aids · `aba64cb` fall detection · `023bb6e` fall alerts · Settings; 18: vehicles `2790822` · `91d90e4` · `8374fad` · `0d87bc0` · Settings); Phase 16 approved, not started; Phase 1 deferred. Phase 11: `5341d56` keyboard answer key. Phase 10:
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
- Tests: **767 pass** (`.venv\Scripts\python.exe -m pytest -q`, ~90 s).
- The user's webcam had its **lens cover closed** for the whole 24-hour check (2026-10-11); they were told to open
  it. A blank picture shows as a camera problem on the Live and Health pages.
- The user's config: one USB webcam (`cam1`, kiosk mode); `web.host: 0.0.0.0` and `web.https: true` (ready for a
  second site over Tailscale, not set up yet: see README "From another site"); one viewer account `TIXUAN`.
- The user's database: 27 webcam people, 26 of them in the answer key (their own labels; mean ~75% before the
  2026-10-08 prompt change, ~89% for the VLM alone after it on those people), plus **127 MOT16 people** imported
  2026-10-12 as the recorded camera `mot16` (dated March 2016; CC BY-NC-SA 3.0, non-commercial). With them the
  weekly retraining ran and was kept for the first time (`data/models/attributes.joblib`, 61.8% → 62.1%).
- BitLocker is on for drive C: (the user turned it on 2026-10-08).
- Docker and the database container were running at the end of 2026-10-13 (healthy); VisionLabeler itself was not
  running. `start.bat` starts Docker if needed; tools like `tools/users.py` need it running first (they hang otherwise).
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
        loitering: _check_stays once a second: people/vehicles inside areas -> alerts.check(stayed={area: s})
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
labeler/vehicles.py  vehicles: second detector (yolo11s.pt) + tracker, VehicleWatch (records, parked, zones, pictures
                     vehicle_<id>.jpg + blurred copy), VehicleStore (db), apply_description (vision model's answer)
labeler/animals.py   animals (Phase 21): dogs, cats, birds from the vehicle detector's pass (Pipeline._update_vehicles,
                     _object_classes); VehicleWatch(names=kinds_map(..), still_word="resting") + AnimalStore; tables
                     animals, animal_zone_events; cameras[].animals (security by default); no idle wake, no recording;
                     /api/animals (+ /delete), Live "Animals seen", agent.animal_query, Reports "animals";
                     rules "animal = dog" (AlertEngine.check_animal, Pipeline._check_animal_alerts), alert_events.animal_id
labeler/falls.py     fall detection from body points (FallWatch.update per person, tick per frame); Pipeline._on_fall
                     -> AlertEngine.check(fallen=True) (rule "Fall", alerts.FALLEN); pop-ups for every page in
                     static/session.js (falls: red, beep until OK); zones.lying_ok; tools/fall_check.py measures it
labeler/arming.py    Home / Away (Phase 19): table `arming` (each change; newest = now), weekly Away schedule
                     (`arming.schedule`, "mon-fri 08:00-18:00"), a button press holds until the schedule's next
                     change; GET/POST /api/arming (viewers too, audited); one Arming made in run.py, shared by
                     AlertEngine(db, arming) and create_app (rule condition "armed = away/home"); button in
                     static/session.js (window.VL.arming, "vl-arming" event, VL.switchArming), Live tile, Health
                     line (healthcheck.check_arming), Settings field away_schedule
labeler/scene.py     scene questions (Phase 22; one area of the picture: Phase 27 area_picture): SceneRunner (run.py thread "scene"), job ids -(2e9 + question id) routed
                     in run.py route_result; VLMClient.label(frame, None, ("scene", question)), PROMPT_SCENE, 1024 px;
                     VLMWorker.idle; Pipeline.scene_frame (clean frame + heads every 2 s); tables scene_questions/answers
labeler/checks.py    "was this right?" (Phase 24): candidates per kind (vehicle, animal, alert, scene answer; 30 days,
                     newest first), next_item (the kind with fewest checks first), table checks (verdict yes/no/
                     unsure, correction), /check page + /api/check/next, POST /api/check (admins)
labeler/summary.py   what happened in a period (Phase 25): away_period from the arming history, night 18-08, today;
                     build(): people, vehicles, animals, alerts, scene answers that changed, camera problems, one
                     timeline; /api/summary (viewers: no clips, no camera problems)
labeler/settings.py  Settings page: data/settings.yaml merged over config.yaml in load_config (only the
                     project's config.yaml, or a config naming settings_file:); validated; masked passwords;
                     POST /api/restart → exit 4 → supervisor restarts at once (VL_SUPERVISED=1 from it)
start.bat / stop.bat daily start/stop (wait for Docker + Ollama; stop waits for app exit, backs up DB)
install.bat/.ps1     new PC: winget Python/Docker/Ollama, .venv, .env, DB, models, admin, autostart,
                     health; -CheckOnly, -Offline <pkg>; PowerShell 5.1 + ASCII only (tests check)
tools/scene_check.py scene questions on UR Fall (lying) and MOT16 (car) frames with known answers
tools/animal_check.py animals on COCO (per confidence) and false alarms on MOT16 / UR Fall (color half only!)
tools/loiter_check.py loitering alerts against MOT16 true markings (found, false, late; --min-visibility)
tools/relink_check.py counting check against MOTChallenge gt.txt (true identities); per-join right/wrong
tools/models.py      check / download / pack / unpack models (HF cache, Ollama manifest+blobs, YOLO)
tools/pack_offline.ps1 offline package: pip freeze wheels (cu130 index), models, docker save
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
| Python | `C:\VisionLabeler\.venv` (Python 3.13, torch 2.14.1 cu130 since 2026-10-15); `cryptography` added for HTTPS; `httpx2` for tests |
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
19. **The measuring tools missed bugs a real run on a test copy found** (Phase 17-18): two alerts for one fall, no alert
   after a video loops, a zone ignored out of sight, an alert before the vehicle's picture. Always do the real run.
20. **Ollama reuses its work on a picture**: asking twice in a row about the same picture makes the second request
   ~25% faster; time whole passes, not alternating requests (tools/mobility_check.py speed).
21. **Tune on one half, report on the other**, and say so when the rules changed after seeing the second half.
22. **Heredocs again** (pitfall 1) turned `\v` in a README path into a control character (2026-10-13): edits with
   backslashes go in a script file written with the Write tool; a scan for control characters before committing helps.
23. **Tests must not need git-ignored files** (models like `yolo11s.pt`, datasets): GitHub has none of them. One test
   did, and the GitHub run failed for 23 pushes (2026-10-13/14) unnoticed. Check `gh run list --limit 3` after pushing
   (gh is signed in on this PC); to test like GitHub, rename the file for one run.
24. **Python's sqlite3 commits ALTER/CREATE/DROP by itself**, outside `engine.begin()`: a table rebuild that failed
   halfway left a test database with only `alert_events_old` (2026-10-14, before the fix was committed). For
   several DDL statements that must be all-or-nothing, use the raw connection with `isolation_level = None` and an
   explicit BEGIN / COMMIT / ROLLBACK (`Database._rebuild_sqlite`), and test a failure halfway.

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
**State at the end of the session (2026-10-12, checked):** this PC's Tailscale connected again (Ti Xuan's PC offline);
the firewall rule "VisionLabeler cameras (Tailscale)" for 8554/1935 is in place; VisionLabeler was not running.
**Still to do with the user (they have not done it yet):** (1) Settings → "Add camera sent from a phone or PC" →
Save → copy the two send addresses (password shown once) → Restart now; (2) send from a device: their phone
(Tailscale app + Larix Broadcaster, the phone/RTMP address) is the easiest test; another PC gets the NEW
`remote-camera` folder (`remote-camera.zip` in the project folder was made again from it on 2026-10-15; git-ignored); Ti Xuan's company PC only if their company allows it. Then check the Health page line
"cameras from other devices" and that people are found and recorded. Note: the user's real app has never run the
Phase 13-15 code yet (recording, receiver) until they restart it.

**Phase 17 (mobility aids and fall alerts) is done** (2026-10-13, [PHASE17-PLAN.md](PHASE17-PLAN.md); numbers in README
"Mobility aids" and "Fall alerts"). The user skipped the Phase 15 phone test and Phase 16 ("skip both"; don't push them
again unless asked). Commits: `81da79e` mobility aids · `aba64cb` fall detection · `023bb6e` fall alerts · `3b7ffd5`
Settings.
- Mobility aids: label `mobility_aid`, asked as one choice with look-alike answers stored as "none" (`schema.DECOYS`);
  `labels.mobility_aids` (Settings) turns it off; use `schema.mode_fields(mode, cfg["labels"])`, not `MODE_FIELDS`.
  Measured with `tools/mobility_check.py` (COCO; Open Images photos in `data/datasets/openimages_aids`, truth by eye).
- Falls: `labeler/falls.py` (FallWatch.update per person, tick per frame; side cameras only), alert rule "Fall"
  (`alerts.FALLEN`, made at start), pop-ups for every page in `static/session.js` (falls red with a beep until OK:
  the user heard it), zones `lying_ok`. Measured with `tools/fall_check.py` on UR Fall (`data/datasets/urfd`).

**Phase 18 (vehicles) is done** (2026-10-13, [PHASE18-PLAN.md](PHASE18-PLAN.md); README "Vehicles"). User's choices: cars,
motorbikes, bicycles, buses, lorries; no number plates; make/model by the vision model. Commits: `2790822` detect/track
· `91d90e4` describe · `8374fad` list/Ask/Reports/lines · `0d87bc0` alerts · `9a03ef4` Settings/Health.
- `labeler/vehicles.py` (VehicleWatch + VehicleStore), `yolo11s.pt` (git-ignored), second tracker
  `trackers/vehicles.yaml`, every 2nd frame, security cameras by default (`cameras[].vehicles`). Parked vehicles stay
  one record (still 60 s; a vehicle still 5 s is remembered 2 min when lost). Tables `vehicles`, `vehicle_zone_events`.
- The vision model describes each vehicle once (`vlm.PROMPT_VEHICLE`; jobs with negative ids routed by
  `Pipeline.on_vlm_result`). Stanford Cars (unseen 300): make 88%, model 57%, type 93%.
- Vehicle alert rules: a rule with a `vehicle` condition (+ `vehicle_color`, `zone`, `hours`; `hours` also for people);
  `alert_events.vehicle_id`, `track_id` nullable (`db.NULLABLE_NOW`).
- Measured with `tools/vehicle_check.py` (coco, speed, count on MOT16, makes on `data/datasets/stanford_cars`).

**The user's own app has not run Phases 13-25 yet**: at its next restart it gets them and changes the PostgreSQL
database by itself (zones.lying_ok; the alert "one per rule and person" constraint dropped; alert_events.vehicle_id and
track_id nullable; new tables vehicles, vehicle_zone_events; Phases 19-22: new tables arming, animals,
animal_zone_events, scene_questions, scene_answers, checks; alert_events.animal_id, .scene_id, .picture; alert_events.rule_id
made optional). Each change was checked on a scratch database on the same
server (scripts in the session scratchpad, not kept: create a database, recreate the old layout, start `Database(url)`,
check, drop it). Ask the user to restart (stop.bat, start.bat) and look at the Health page.

Waiting for the user: a camera that sees a room from the side (falls) and one that sees the gate or street in security
mode (vehicles), then real numbers (makes like Perodua and Proton are not measured); marking beds/sofas "Lying is
normal here". Phase 16 ([PHASE16-PLAN.md](PHASE16-PLAN.md), search recordings by words) stays approved, not started.

**Phase 19 (Home / Away) is done** (2026-10-14, [PHASE19-22-PLAN.md](PHASE19-22-PLAN.md); README "Home / Away"). User's
choices: admins **and viewers** may switch; a weekly schedule; a tap asks first. `labeler/arming.py` (table `arming`,
column `changed_by`, not the reserved word `by`), rule condition `armed = away/home`, button in `static/session.js`, Live
tile, Health line, Settings box `away_schedule` (-> `arming.schedule`, applies after a restart like every setting).
The new table was checked on a scratch PostgreSQL database (script in the session scratchpad, not kept). The user's own
app gets it at its next restart (new table only, no change to old ones). Docker Desktop was started for that check
(2026-10-14) and left running. The Settings page's "unsaved changes" mark that reacted only to the cameras
table was fixed in Phase 23 step 1.

**Phase 20 (loitering) is done** (2026-10-14; README "Loitering"): condition `stayed = <area>:<seconds>` (people and
vehicles; `alerts.parse_stayed`, `AlertEngine.stay_limits`), `Pipeline._check_stays` once a second, the Alerts page
picker (area x 30 s ... 1 h), the time on the live picture (`Pipeline._stay_text`, `VehicleWatch.shown(t, extra)`).
Measured with `tools/loiter_check.py` (MOT16 02/04/09, true markings, people at least half visible): 5 s found 51/93,
15 s 13/47; misses are people never detected (far, half hidden) and broken tracks (the time starts again). Tried and
not kept: carrying the area time over a re-link (no gain, 2 more false alarms; most broken tracks are not re-linked on
security cameras). Not measured: minutes-long stays, the user's own footage.

**Phase 21 (animals) is done** (2026-10-14; README "Animals"). User's choices: dogs, cats and birds; security cameras only.
`labeler/animals.py`: the vehicle detector's pass also asks for the animals' COCO classes (`Pipeline._object_classes`);
the vehicle watcher follows them with their own names ("resting" = parked); tables `animals`, `animal_zone_events`;
`alert_events.animal_id`; Live "Animals seen", Ask box words, Reports, alert rules `animal = dog` (+ zone, hours, armed,
stayed), Settings section/column, Health line. Measured with `tools/animal_check.py`: COCO dogs 75%, cats 83%, birds 68%
at 0.5 (0.4 not worth it); 0 false on MOT16, 3 on UR Fall (a person crawling). Pitfall: UR Fall videos are a depth map
(left) next to the color picture (right): use the right half. Tried, not kept: ignoring animal boxes that are a found
person. All new tables/columns checked on scratch PostgreSQL databases. Not measured: the user's camera, night pictures.

**Phase 22 (scene questions) started 2026-10-14**: steps 1-3 done (step 2: the runner asks and stores answers; step 3:
real alerts with their own picture and a clip; an old SQLite database cannot store them, PostgreSQL is updated at
start); step 1 (tables `scene_questions`/`scene_answers`, `/api/scene`
(admins, audited), Alerts page box). Docker was off again later on 2026-10-14 (PC restarted): a database check hung
until it was started (pitfall 11).

**Phase 22 (scene questions) is done** (2026-10-14; README "Scene questions"). User's choices: every 5 minutes by default,
an alert after two answers in a row. `labeler/scene.py` (SceneRunner), tables scene_questions/scene_answers, scene
alerts (alert_events.scene_id/.picture, rule_id optional; an old SQLite database cannot store them), Alerts page box,
Settings section, Health line with an overdue warning. Measured with `tools/scene_check.py`: lying on the floor 64/64
and 513/523 (wrong: kneeling), a car 116/116 and 135/135, 0.6 s per answer. Not measured: smoke, fire, gates, lights,
night, the user's cameras.

**The Phases 19-22 plan is finished.**

**Phase 23 (small fixes) is done** (2026-10-14): `02c7cf9` Settings unsaved mark; step 2 not needed (console text is
fine, the plan's assumption was wrong); old SQLite alert tables rebuilt at start (`Database._rebuild_sqlite`).

**Phase 24 ("was this right?") is done** (2026-10-14; README "Checking on your own cameras"): `f4f1a1d` Check page ·
`be64dd8` results · `cbaea65` often-wrong notes · docs. Waiting for the user: real checks on their cameras (after
they restart their app). Step 3: checks.flags, /api/check/flags, notes (Check page, table checks; results: checks.results/summary_line, Health
"checked by you", Reports "checks").

**Phase 25 ("while you were away") is done** (2026-10-14; README "What happened (Summary)"): `21d51cf` summary ·
`3f20e28` page · `483f048` welcome-back note · docs (step 3: session.js welcomeBack) (summary data and API; the Summary page `static/summary.html`, `?period=`).

**Phase 27 (2026-10-14/15)** (the user skipped Phase 16 for now and picked 27 first). Step 1 done: a scene question
can be about one area drawn on the Live page (`scene_questions.zone_id`; `labeler/scene.py` `area_picture`, `AREA_WAY`;
`/api/scene` `zone_id`, 0 = whole picture again; deleting the area sends its questions back to the whole picture).
Measured with `tools/scene_area_check.py`: cut-out 94.4%, red outline 90.2% (answers about the whole picture anyway),
whole 66.5%. Step 2 done: limits "only when Away" (`only_away`) and `hours` per question (`scene.asked_now`,
`limits_since`; SceneRunner gets the Arming from run.py; `healthcheck.check_scene(..., arming)`). Step 3 done
(2026-10-15): Alerts page box "Look at", "Ask" (hours), "only when Away", also in each row. **Phase 27 is done.**
Next: ask the user which of 16 (search recordings), 26 (real-camera week: their app must be restarted first), 28
(maintenance; GitHub runner check after 2026-10-19), 29 (more cameras) comes next.

**Phase 28 started 2026-10-15** (user's choice): step 2 done, `httpx2` 2.13.1 in `.venv` (with OK) and in the GitHub
test install, so Starlette's test client no longer warns. Step 3 done: PyTorch 2.11 cu128 -> **2.14.1 cu130**
(torchvision 0.29.1; the cu128 index has nothing newer than 2.11), setuptools 81 -> 84, pip-audit clean; install.ps1 and
pack_offline.ps1 use the cu130 index and the installer requires NVIDIA driver 580+. Compared on the same inputs with a
scratchpad script (not kept): identical boxes/points/vectors/age, fall_check and relink_check unchanged. Roll back if
ever needed: `pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128`.
Step 4 done: `remote-camera.zip` made again from the current (sending) folder, user's choice; git-ignored. Left:
step 1 (check the GitHub runs after 2026-10-19, when `ubuntu-latest` becomes Ubuntu 26; fix if anything breaks).

**Phase 16 started 2026-10-15** (the user has no extra camera; chose 16): step 1 done, `labeler/moments.py` (finds ->
moments; `PAD_S` 3, `MAX_JOIN_S` 60: a busy camera chained everyone into one 2-minute moment before that limit),
`POST /api/recordings/search` in `labeler/recordings_web.py` (`add_routes(..., agent, embedder)`; agent.run, else
db.nearest by text), viewers allowed (accounts.VIEWER), stream `?d=`. Parts carry offset_s/end_offset_s (real seconds:
the viewers' stream) and video_s/video_end_s (frame index / fps: the admins' player and the cut download). Step 2 done:
the Recordings page's "Find in recordings" box (`static/recordings.html`: drawFound, playMoment, play(..., until) stops
the file at video_end_s or waits d + 1.5 s for the stream, then the next part / Play all queue). Checked in the browser
(admin; the viewer path by switching the page's flag; phone width). **Phase 16 is done.**
Next: **[PHASE30-35-PLAN.md](PHASE30-35-PLAN.md)** (approved 2026-10-15): 30 camera picture check, 31 "no movement"
alert (4 h, 08:00-22:00, only when Home), 32 help sign (one hand above the head 3 s), 33 search recordings by photo,
34 time-lapse of the day, 35 Update button. Also open: Phase 26 (real-camera week: their app must run with the webcam,
lens cover open), Phase 28 step 1 after 2026-10-19; Phase 29 waits for extra cameras (the user has none, 2026-10-15).
Not chosen 2026-10-15 (don't offer again unless asked): sounds, privacy masks, Malay/Chinese pages, blurred export.

**Phase 30 started 2026-10-15**: step 1 done, `labeler/picture_check.py` (`stats`, `findings`, `check_frame`, `faces`;
`LIMITS` from `tools/picture_check_eval.py --tune`; the blur measure is Laplacian variance / picture variance: random
noise is no test picture for it, as blurring noise removes its contrast too, so tests use `samples/bus.jpg`). Step 2
done: `PictureWatch` (Pipeline.picture, checked in Pipeline.run next to scene_frame; `scene_people` = boxes in
scene_frame), `healthcheck.check_picture`, `GET /api/cameras/{name}/picture` (admins), Live banner `#pic-problem`
(index.html showPictureProblems from /api/status cameras[].picture), Settings row button `.check-pic`. **Phase 30 is
done.**

**Phase 31 started 2026-10-15**: step 1 done, `labeler/quiet.py` (QuietWatch; run.py thread "quiet", `quiet:` config),
`alerts.QUIET` / `QUIET_RULE` / `ensure_quiet_rule` / `quiet_rules`, `Pipeline.person_seen_wall`, `db.record_alert`
with no track (repeat), `db.update_alert_message`, list_alerts lists rule-only alerts (`"quiet": true`), retention
deletes alerts about no one per camera. A rule without a camera watches kiosk cameras only. Step 2 done (pop-up
`toast quiet`, `window.VL.alertWho` shared by session.js and alerts.html, picker, Health `check_quiet(app.state.quiet)`,
Settings `quiet_enabled`). Step 3 done: real run (empty scene 11 min then MOT16-09 people, a 10-minute rule): alert at
10 min, "seen again at 15:29, after 11 min". **Phase 31 is done.**

**Phase 32 started 2026-10-15**: step 1 done, `labeler/gestures.py` (hands_up per frame, GestureWatch for the hold;
DEFAULTS margin 0.15 of the torso, hold 3 s, gap 0.5 s, rearm 2 s, both_ok, max_tilt 45: `upright`). Step 2 done:
`tools/gesture_check.py videos` (reuses tools/fall_check.py's cached detections, leaves out its frozen tail) and
`photos --start 1500` (mosaics in a folder to judge by eye; the first 1,500 photos were used to add the upright check).
README "Help sign". Next: step 3 (connect to Pipeline like FallWatch, built-in rule "Help sign", red pop-up with a
beep, Settings; real run with a still photo of a raised hand as the camera).

**Earlier plan: Phase 16** search recordings by words ([PHASE16-PLAN.md](PHASE16-PLAN.md); approved, skipped for now), **then
[PHASE26-29-PLAN.md](PHASE26-29-PLAN.md)** (approved 2026-10-14): 26 real-camera week with the user, 27 scene question
areas and limits, 28 maintenance (after 2026-10-19: Ubuntu 26 runner), 29 more cameras. The user picks the order.: 23 small fixes, 24 "was this right?"
on the user's cameras, 25 "while you were away" summary, then 16 search recordings by words. Not chosen: phone alerts
(don't offer again unless asked). Ask before each phase and each step. Before Phase 24 the user restarts their app. Its opening questions are
already answered (2026-10-14): every 5 minutes by default, alert after two "yes" in a row. Ask the user before starting
the phase, then before each step. State at the end of 2026-10-14: Docker Desktop was started for the PostgreSQL checks
and left running; the user's own app has not run Phases 13-21 yet (ask them to restart it: stop.bat, start.bat).
Later, with the user: a long-run check with recording on, the webcam measurement (lens cover), the rest of Phase 12
when cameras are added.

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
- Before any commercial use: remove the `mot16` camera's people and retrain (MOT16 is non-commercial). The
  measuring sets UR Fall (CC BY-NC-SA 4.0), Open Images (CC BY 2.0) and Stanford Cars (research use) are only used by
  the tools in `tools/`, never by the app; they can be deleted (`data/datasets/urfd`, `openimages_aids`, `stanford_cars`).
- Vocabulary refinements (coat, polo, tank top) need answer-key values to be measurable.

## 9. Message to start the next session

> I'm continuing the VisionLabeler project in C:\VisionLabeler. Please read docs/HANDOFF.md and
> docs/ROADMAP.md first, run the tests to confirm everything is green, then continue with docs/PHASE30-35-PLAN.md
> (and the GitHub runner check of Phase 28 after 2026-10-19); ask me before each phase and each step starts, and do it
> step by step.
