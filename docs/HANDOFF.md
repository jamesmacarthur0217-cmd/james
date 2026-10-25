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

## 2. Current state (2026-10-25)

**Start here (2026-10-25, end of the session; last commit `0171fd4` + this handoff; 1016 tests + 1 PostgreSQL test that
is skipped without `VL_TEST_POSTGRES_URL`; GitHub green; working tree clean; no test copy running).**

**Plan in progress: [PHASE50-55-PLAN.md](PHASE50-55-PLAN.md)** (approved 2026-10-25, the user's choices, in this
order): 50 vision model capacity, 51 Settings for the safety checks, 52 a safety overview page, 53 night-time
measuring, 54 more behaviour signs, 55 our own pest model; and Phase 26 (real-camera week) after the user restarts
their app. Same way of working: **ask before each phase and each step; discuss any architecture first**; short, simple
answers.
- **Phase 50 steps 1-2 done** (design agreed first, plan `fd41660`): step 1 `e89cfac` (`tools/incident_bench.py --busy`:
  a busy camera in real time from MOT16-09's true marks, checks through the real confirm queue); the vision model gives
  **about 42 answers a minute** on this PC (a person's label = 2); **2 requests at once: no gain** (the user's Ollama
  setting left alone; the old "1.9x" note in VLMWorker corrected). Step 2 `0171fd4`: checks use the model's **idle
  time** (`ConfirmQueue._labels_waiting`: the per-minute limit only while labels wait); **passers-by under 2 s get
  colours only** (`vlm.label_after_s`, `Pipeline._due_for_label`); **one picture per person while other labels wait**
  (`vlm.one_when_busy`, `Pipeline.pictures_per_person`; the user's choice: two pictures differ in carried items 37%,
  colours 22-23%, age/gender 0%). Result: 21 people a minute: nothing dropped (before 15 of 27), labels 5 s; a crowd
  (87 a minute) is still over capacity (README "Speed").
- **Next: Phase 50 step 3** (the vision model's load on the Health page: share of time busy, waits, drops in the
  "safety checks" line); ask the user before starting it. Then Phase 51 (discuss its design first).
- Waiting for the user: restart their own app (stop.bat, start.bat; then the First-run check; Phase 26); the MIVIA fire
  videos. Open (offer, don't change unasked): the floor check's third look (not chosen 2026-10-25).

**Finished plan: [PHASE44-49-PLAN.md](PHASE44-49-PLAN.md)**: safety incidents on **security cameras** (the 11 kinds
the user listed + "suspicious behaviour", judged by what people do, never by looks). Each phase's design was agreed
with the user first (their rule: **discuss any architecture before building it**; ask before each phase and step).
- **Phase 44 done** (2026-10-20: `55a60c3` area roles, `2e6bca2` confirm queue, `3132c2e` safety alerts): quick checks
  raise a *suspect*, the vision model confirms from up to 3 pictures of the last seconds, only "yes" alerts
  (`labeler/incidents.py`); a role per drawn area (`zones.role`); 12 built-in rules (one per kind); critical kinds pop
  up red with a beep.
- **Phase 45 done** (`3e28566` tool, `0f22525` falls/lying/slumped, `3c95afa` children, `85da519` crowding):
  `labeler/safety.py`. Falls on safety cameras are checked by the vision model too (user's choice) with a **fail-safe**
  (only a clear "no" stops the alert); children by label or small body (user's choice); crowding: "at most N people"
  per area (`zones.max_people`, Live page shows "now N"), counted once a second by a copy of the plain detector at
  1280 px (the pose detector finds 0-40% in crowds); the vision model decides if it is dangerous (user's choice).
  Vision-model queue order: **critical, warning, people's labels, slow round** (warnings were refused behind labels).
- **Phase 46 done** (`059c8f8` model, `544f7db` check, `ad84a2a` measured): our own fire/smoke detector trained on
  D-Fire (CC0; `tools/train_fire.py`, `data/models/fire/fire_smoke.pt`, test mAP50 0.77), once a second
  (`labeler/fire.py`); D-Fire's 100 videos: 38 of 50 fires found (median 3 s), 5 of 50 tricky false. **MIVIA videos:
  the user requests them; measure with `tools/incident_check.py` when they arrive.** Indoor fires not measured.
- **Phase 47 step 1 done** (`9878bc1`, `1d5ba2f`): dangerous objects held at a wrist (`labeler/weapons.py`): knives,
  scissors, bats from the plain detector; **our own gun detector** (`tools/train_guns.py`, Open Images, CC BY 2.0;
  `data/models/guns/guns.pt`; 76% of guns found vs YOLOE's 10%); YOLOE prompt-free (27 MB) only for machetes/swords;
  **any real gun in hand alerts** (user's choice; the vision model only checks it is a real gun). 130 of 217 held
  guns alerted, 0 false on 389 everyday photos. Knives held as a threat: no photos to measure.
- **Phase 47 step 2 done** (2026-10-21; design agreed with the user first, plan `2ea1e80`): floor hazards and blocked
  exits (`labeler/floor.py`): walkway / exit areas against each safety camera's empty picture (saved in
  `<crop_dir>/empty/<camera>.png`); the vision model sees the empty picture and now; two "yes" a minute apart alert, a
  "no" puts the change into the empty picture. Measured with made-up videos (`tools/make_floor_videos.py`,
  `data/datasets/floor_made`, 1.1 GB, git-ignored): objects 5 of 6, spills 1 of 3, 0 false alerts (README "Floor
  hazards and blocked exits"). Seen while measuring: on the moving MOT20 cameras the fall check (Phase 45) raised
  23-30 fall suspects per 4-minute video; not looked into (safety cameras are normally still).
- **Phase 47 step 3 done; Phase 47 is done** (2026-10-21; design agreed first, plan `3803356`): unsafe equipment by
  10-minute rounds over every area with a role (`labeler/equipment.py`; two "yes" a minute apart alert; README
  "Unsafe equipment; pests left out"): a chair on its side 6 of 9, a cable 4 of 9, COCO rooms 2 of 150 false. **Pests
  left out** (the user's choice after measuring): YOLOE found 3% of 251 Open Images rodent photos
  (`data/datasets/openimages_pests`, 74 MB, downloaded with OK) and takes computer mice for mice;
  `tools/pest_check.py` kept for a better model (our own, as for guns, if the user asks).
- **Phase 48 done** (2026-10-23; design agreed first, plan `e81a2db`): fights from body points (SafetyWatch.fights:
  two people close, a fast wrist touching or reaching the other; speed 2.5 body heights a second chosen on half the
  clips + MOT16), one "yes" alerts (critical), one fight check per camera a minute; collisions through the fall check
  ("after touching person #12"). Surveillance Camera Fight Dataset (`data/datasets/survfight`, 87 MB, downloaded with
  OK): 39 of 75 fights alerted, 0 false alerts on the everyday videos (README "Fights and collisions").
- **Phase 49 done; the Phases 44-49 plan is finished** (2026-10-25; design agreed first, plan `bb4860d`): suspicious
  behaviour, judged only by what people do (`labeler/behaviour.py`): climbing at an area with the new role "fence or
  wall" (the vision model confirms; critical at night 22-06 or when Away), a restricted area and pacing back and forth
  (alert at once at night or when Away: `ConfirmQueue.direct`, the vision model's words added after), the camera
  covered or turned (when a person was seen in the 30 s before). Made-up cases found, 0 false alerts on still cameras
  in 86 everyday videos (README "Suspicious behaviour").
- Then the user chose the Phases 50-55 plan (above).

**Measuring tools** (all on recorded videos / photos, the real camera code, the vision model):
`tools/incident_check.py` (videos + truth CSV `video,kind,start_s,end_s,half`; `--labels`, `--areas`, `--hold-s`,
`--fps`, `--match` for tuning halves), `tools/crowd_check.py`, `tools/incident_bench.py` (vision-model cost),
`tools/weapon_check.py` (coco / people / guns / openimages / photos), `tools/train_fire.py`, `tools/train_guns.py`,
`tools/make_floor_videos.py` (made-up floor videos from still scenes), `tools/equipment_check.py`,
`tools/pest_check.py`; `tools/make_behaviour_videos.py` (Phase 49); `tools/incident_check.py --make-truth survfight` (fights; tune `--match '[02468]\.mp4$'`).
Numbers and limits are in README sections "Safety incidents" ... "Dangerous objects".

**Data downloaded this time (all with the user's OK, git-ignored, measuring/training only):** MOT20 (`data/datasets/
mot20`: zip 4.7 GB + 3.2 GB unpacked + videos; the zip may be deleted), D-Fire (`data/datasets/dfire`: zip 3.0 GB +
unpacked + 100 sample videos 0.44 GB), Open Images guns (`data/datasets/openimages_weapons` 127 MB test,
`data/datasets/oi_guns` 1.03 GB training), YOLOE (`data/models/yoloe`, 27 MB), Open Images rodents
(`data/datasets/openimages_pests`, 74 MB), the fight clips (`data/datasets/survfight`, 87 MB, a git clone). Made here:
`data/datasets/floor_made` (1.1 GB, `tools/make_floor_videos.py`). The user may want old zips deleted
(ask; never delete without OK).

**Models only on this PC** (not in git; `tools/models.py check / pack` lists them as optional): `data/models/fire/
fire_smoke.pt`, `data/models/guns/guns.pt`, `data/models/yoloe/yoloe-11s-seg-pf.pt`. Without them those checks are off
and the log says so.

**Still waiting for the user:** restart their own app (stop.bat, start.bat) so it runs the new version (the
safe-update test passed on their database 2026-10-19), then the First-run check, then Phase 26 (real-camera week);
the MIVIA fire videos. Their own camera is one kiosk webcam: none of the safety checks run on it (security cameras
only); real numbers need a security camera.

**State at the end:** Docker Desktop running; no test copy running; Ollama has qwen3-vl:4b-instruct (used) and two
unused models (the user decides).

**Latest (2026-10-19, end of the session; last commit `6d7fe99` + this handoff, 884 tests + 1 PostgreSQL test that is
skipped without `VL_TEST_POSTGRES_URL`, GitHub green).** Done since 2026-10-17, newest last: Phase 36 What's new +
First-run check (`67b69b3`, incl. the PostgreSQL start-up hang fix), 37 faster playback (`c8e1fe3`), 38 PC watch
(`32e6b8c`), 39 unusual quiet (`114d8f5`), 40 the week, printable (`83f1ba4`), 28 step 1 Ubuntu 26 runner checked once
(`04b9f46`), 41 safe-update test (`6546920`), 42 recordings kept around alerts + the clock on recordings fixed
(`c5aa0e8`, `6d7fe99`). **Next: Phase 43** (the usual day as a chart), the last of
[PHASE41-43-PLAN.md](PHASE41-43-PLAN.md); its steps are written there in detail.

- **Waiting for the user:** restart their own app (stop.bat, start.bat). The safe-update test passed on their real
  database (2026-10-19, read only: 1.1 s, 154 people, 9 tables and 5 columns added, nothing lost); then the First-run
  check page (link in the Live page's "New in this version" line). Then Phase 26 (real-camera week, lens cover open).
  Before any fine-tuning of the vision model: label 200-300 webcam people on the Answer key page (26 now).
- **Decided this session:** age fine-tuning is a separate project in `C:\AgeModel` (another session did it: MiVOLO v2
  and a Qwen3-VL age variant, share zips in `C:\AgeModel\share`); never wire it into VisionLabeler unless the user asks.
  MaskFlow (a SAM 3 labelling platform, github.com/ZhangEmo28/maskflow): not now; worth a look only if YOLO must learn
  something new or work on a hard camera. Not chosen for Phases 41-46: PC history chart, weekly report saved by
  itself, a week in one minute. A long full-system run (people in view + recording on) was mentioned as never done.
- **State:** Docker Desktop was started by this session and left running; no test copy running. Ollama also holds
  `qwen3-vl:8b-instruct` and `qwen3-vl:4b` (unused by the app, ~9 GB; the user decides whether to delete them).

**2026-10-18: a start-up hang fixed** (`labeler/db.py` `_drop_old_unique` / `_add_missing_columns`, pitfall 30): the user's
own app would have hung for ever at its next start on PostgreSQL. Their database (read only, 2026-10-18) still has
the old `alert_events (rule_id, track_id)` unique constraint, no `zones.lying_ok`, and a `recordings` table with 2
recordings (2026-10-12/13): so their app ran Phase 13-15 code then, not 17+. Checked on a throwaway database built
by the pre-Phase-13 code: opens in 1.2 s, 9 tables and 5 columns added, constraint gone. `tests/test_postgres.py` (skipped
without `VL_TEST_POSTGRES_URL`) failed on the old code. Docker Desktop was started for it and left running.

**Phase 37 is done (2026-10-18)**: Recordings page 1x-8x, -10/+10 s, keys; the video (`endTimer` checks a moment's end
every 50 ms) and the stream (`recording_stream(..., speed)`, paced by the clock; page `streamAt`, `restartStream`,
`pauseStream` keeps the frame via a canvas). Next: Phase 38 (PC watch).

**Phase 38 started 2026-10-18**: step 1 done (`labeler/pcwatch.py`, run.py thread "pcwatch"); step 2 done (camera
events with camera "PC": kinds hot / disk / stopped, `PCWatch.sync`, `db.add_camera_event`; Health `check_pc`,
`check_stops`; /api/status camera_problems includes `pcwatch.current()`). Step 3 (tests with made-up outputs, a
real run with a copy ended by force) was done within steps 1-2. **Phase 38 is done (2026-10-18).**

**Phase 39 started 2026-10-18**: step 1 done (`labeler/routine.py`: `learn`, `Routine.usual_hours`, `away_spans`;
`db.track_spans`; config `routine`); step 2 done (`labeler/unusual.py` UnusualWatch, run.py thread "unusual",
`alerts.UNUSUAL` / `UNUSUAL_RULE` / `ensure_unusual_rule`, list_alerts `"unusual"`, session.js pop-up, Alerts page
picker). Step 3 done: measured with `tools/unusual_check.py`, the per-hour rule was replaced by whole quiet stretches
(`routine.quiet_share` 0.97, `routine.min_quiet_h` 3, `Routine.stretch`); day summaries `routine_days`
(`routine.store_days`, called first in run.py `_apply_retention`) so 8 weeks survive the 30-day retention; Health
`check_routine`. **Phase 39 is done (2026-10-18).** The new table was not checked on PostgreSQL (Docker was off;
new tables come from create_all, which has never failed; check at the next PostgreSQL start).

**Phase 40 started 2026-10-19**: step 1 done (`summary.week_period`, `summary.week`, `alert_kind`; /api/summary
`period=week` (+ `week=<day>`)); step 2 done (Summary page "This week": `drawWeek`, `shiftWeek`, weekbar,
`@media print`); step 3 (tests, browser check, docs) within it. **Phase 40 is done; the Phases 36-40 plan is
finished (2026-10-19).** **Phase 28 step 1 done (2026-10-19, `bcfa09a`)**: the tests ran once on ubuntu-26.04
next to ubuntu-latest (still 24.04): the same 856 passed, 6 skipped; the extra job was removed again (the user asked
for a one-time check). **Phase 28 is done.** Next: the user picks from the ideas for Phases 41-46 (offered
2026-10-19: safe-update test on a copy of their database, keep recordings around alerts longer, the usual day as a
chart, PC history chart, weekly report saved by itself, a week in one minute). **The user chose 1, 2, 3:
[PHASE41-43-PLAN.md](PHASE41-43-PLAN.md)** (proposed 2026-10-19, waiting for approval). Phase 26 once their app
runs the new version.

**Phase 41 started 2026-10-19**: step 1 done (`tools/upgrade_check.py`; checked on a pre-Phase-13 database on
PostgreSQL, and it caught the old hang with the pre-fix code). That run also created `routine_days` on PostgreSQL:
the Phase 39 table is checked there now. Docker was started for it and left running. Step 2 done (run.py `_open_database`: messages, every 30 s while
slow; `updater.run_upgrade_check` / `check_words` / `last_check`, Updater `check_db`: a failed test puts the
running version back; Health `check_upgrade`; /api/update `safe_test`; Settings `#upd-safe`; config
`updates.check_report`). Step 3 done: the check on the user's real database (read only, with their OK): 1.1 s,
154 people, 9 tables, 5 columns, nothing lost, What's new 22 (from Phase 17). **Phase 41 is done (2026-10-19).**
The user can now restart their app (stop.bat, start.bat) and open the First-run check.

**Phase 42 started 2026-10-19**: step 1 done (kept parts around alerts, `recordings.kept` checked on PostgreSQL;
also the clock on recordings fixed: `recorder.stamp` outline drawn with the same thickness, pitfall 32); step 2
done (`/api/recordings` items `kept`, day bar `--kept` = categorical slot 3, validated with the dataviz validator;
`db.recording_stats` kept_files/kept_bytes; Health recording line; `summary.week` days `kept_min`; Settings
`recording_alert_*`); step 3's tests and real run were within steps 1-2. **Phase 42 is done (2026-10-19).** Next:
Phase 43 (the usual day as a chart).

**Phase 36 is done (2026-10-18).** Step 1 done ("What's new": `labeler/whatsnew.py`, Live banner, `/whats-new`; 808
tests); step 2 done (First-run check `/first-run`, `labeler/firstrun.py`, Try the help sign; 819 tests). A still photo
with a raised hand for help-sign runs: `data/datasets/coco/val2017/000000032081.jpg` (tennis serve). **Every new phase adds one entry to `whatsnew.ENTRIES`** (with a `marker` when it adds a table or column).

**Earlier (2026-10-15, last commit `b15c94f`, 795 tests, GitHub green):** done this day, newest last: Phase 16 search
recordings by words (`9758ea5` `591615a`), 27 scene question areas and limits (`56fecb7` `fcbc306` `806e0ae`), 28
steps 2-4 (`cae1a17` httpx2, `5304118` PyTorch 2.14.1 cu130 + setuptools 84, `ae73616` zip), 30 camera picture check
(`94df715` `03ec5b3`), 31 no movement (`1b51a57` `ab51c07` `f5349fd`), 32 help sign (`c197716` `764dee7` `55f7758`),
33 search recordings by photo (`0de9d76` `8a0f48d`), 34 time-lapse (`ed76cd9` `c2bb71c`), 35 Update button (`b15c94f`).
**The Phases 30-35 plan is finished; next: [PHASE36-40-PLAN.md](PHASE36-40-PLAN.md)** (approved 2026-10-15). The user's
own app has still **not run Phases 13-35**: at its next start (stop.bat, start.bat) it updates PostgreSQL by itself
(new since Phase 25: scene_questions zone_id / only_away / hours; earlier changes were checked on scratch databases).
The user has one webcam (no Phase 29); Docker was running at the end of the day; no test copy left running.

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
- Tests: **795 pass** (`.venv\Scripts\python.exe -m pytest -q`, ~90 s).
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
labeler/moments.py   search recordings (Phase 16/33): finds -> moments (3 s pad, joined < 60 s, parts per file);
                     /api/recordings/search (+ /photo, min_similarity 0.85); Recordings page Find box, Play all
labeler/picture_check.py camera picture check (Phase 30): covered/dark/bright/blurry/backlight/faces; PictureWatch
                     per camera (80% of 10 min); Health line, Live banner, Settings "Check the picture"
labeler/quiet.py     no movement (Phase 31): QuietWatch thread, built-in rule "No movement" (quiet=240, 08-22, Home)
labeler/gestures.py  help sign (Phase 32): hands_up (upright body) + GestureWatch (3 s); built-in rule "Help sign"
labeler/timelapse.py the day's time-lapse (Phase 34): Maker, frame map, blurred copy for viewers, "older" for today
labeler/updater.py   Update button (Phase 35): daily git fetch, ff-only pull, pip when requirements changed, restart
labeler/whatsnew.py  What's new (Phase 36): ENTRIES in the order phases were finished (add one per new phase);
                     data/last_version.json {last, before}; no file -> guess from Database.created_tables /
                     added_columns (Entry.marker); Live banner (index.html whatsNew, localStorage whatsNewSeen), /whats-new
labeler/routine.py   the usual day (Phase 39): per kiosk camera, weekday/weekend x hour, share of counted days with
                     someone seen (8 weeks; days with nobody and Away hours left out; learning below 14 days)
labeler/safety.py    quick checks for people (Phase 45): SafetyWatch per safety camera: lying still 20 s, slumped 30 s,
                     not moved 5 min (slow round), children (alone 2 min / cooking area / climbing), crowding (over an
                     area's limit 30 s, running together); Pipeline._raise (crop around the person), _on_fall via the
                     queue with a fail-safe, _look_objects (plain detector 1280 px once a second: counts + knives etc.)
labeler/fire.py      fire and smoke (Phase 46): FireWatch, our D-Fire model once a second, 3 of 5 looks, cooking areas
labeler/weapons.py   dangerous objects (Phase 47): WeaponWatch (at a wrist on 2 of 3 looks), our gun model, YOLOE for
                     machetes/swords, GUN_QUESTION (any real gun alerts)
labeler/floor.py     floor hazards and blocked exits (Phase 47 step 2): FloorWatch per safety camera (empty picture =
                     median of 15 looks without people, saved <crop_dir>/empty/<camera>.png; a change held 30 s on a
                     walkway / 2 min in an exit; the lights = learned again); Pipeline._look_at_floor sends the empty
                     picture + now (Suspect.cut tighter, alert_if_yes False on the first look, True on the second)
labeler/equipment.py unsafe equipment (Phase 47 step 3): EquipmentRounds, every area with a role to the vision model
                     every 10 min (Pipeline._round_equipment, level round; two "yes" a minute apart; "reported")
labeler/behaviour.py suspicious behaviour (Phase 49): BehaviourWatch (climbing at a "fence" area -> confirm; restricted
                     area and pacing -> ConfirmQueue.direct at night/Away; camera() covered/turned from the middle
                     value of 5 looks against 10-15 s before) -> Pipeline._look_for_behaviour
labeler/safety.py    + fights (Phase 48): SafetyWatch.fights (pairs close together, a fast wrist touching / reaching the
                     other) -> Pipeline._raise_fight (one check per camera a minute); touched() -> a fall's note
                     ("after touching person #12", Pipeline._touched_before_fall, AlertEngine.check(note=))
labeler/incidents.py safety incidents (Phase 44): FrameMemory (Pipeline.memory, 1 picture/s for 10 s on safety cameras),
                     CATEGORIES (12 kinds: level, words, question), ConfirmQueue (run.py thread "incidents", job ids
                     -(1e9 + n), critical first, confirm_per_min, max_wait_s, cooldown_s), IncidentAlerts (a "yes" ->
                     alert_events.category/level/confirm, picture, clip); vlm.VLMWorker is one priority queue (CRITICAL,
                     WARNING, PEOPLE, ROUND since Phase 45; slot first, then the job); zones.role (zones.ROLES);
                     GET /api/safety
labeler/safety.py    quick checks for people (Phase 45): SafetyWatch per safety camera (lying still, slumped, not moved,
                     children: alone / cooking area / climbing, crowding: over an area's limit / running together);
                     Pipeline._raise, _on_fall via the queue with a fail-safe, _count_people (plain detector, 1280 px,
                     once a second, every area: "people_now" on /api/zones); tools/incident_check.py, crowd_check.py
labeler/unusual.py   (+ Phase 43: UnusualWatch.today, Routine.chart -> GET /api/routine, Health chart #usual)
labeler/unusual.py   unusual quiet (Phase 39): built-in rule "Unusual quiet" (unusual = yes, armed = home); fires when
                     today's quiet stretch (>= 3 whole hours) had someone on >= 97% of history days; once a day
labeler/pcwatch.py   PC watch (Phase 38): nvidia-smi each minute, Get-PhysicalDisk hourly, heartbeat file (pcwatch.
                     heartbeat_file) + mark_ended at run.py's very end; previous_stop at start (boot time via
                     GetTickCount64: PC restarted vs app ended); app.state.pcwatch (state, problems)
labeler/firstrun.py  First-run check (Phase 36): server_checks (database, built-in rules, recording today, Home/Away);
                     static/first-run.html adds /api/health and each camera's picture check; "Try the help sign" =
                     Pipeline.start_help_test (30 s: _on_help reports to help_test, no alert; GestureWatch.rearm in the
                     camera thread), /api/first-run/help-test
start.bat / stop.bat daily start/stop (wait for Docker + Ollama; stop waits for app exit, backs up DB)
install.bat/.ps1     new PC: winget Python/Docker/Ollama, .venv, .env, DB, models, admin, autostart,
                     health; -CheckOnly, -Offline <pkg>; PowerShell 5.1 + ASCII only (tests check)
labeler/recorder.py  + Phase 42: keep_around_alerts (before prune deletes a file: alert_stretches, frames copied with
                     OpenCV + FFmpegWriter, <HHMMSS>-alert.mp4, recordings.kept = "alert", alert_keep_days);
                     db.recordings_to_prune(..., kept_days) deletes normal ones first when over max_gb
tools/upgrade_check.py safe-update test (Phase 41): pg_dump | pg_restore inside the container into
                     visionlabeler_upgrade, Database(url) in a child with a time limit, rows/rules before and after,
                     pg_stat_activity on a hang; data/upgrade_check.json
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
25. **Bash heredocs also eat a backslash at a line end** (a Python `\` line continuation was joined into one line,
   2026-10-15): write edit scripts with the Write tool and run them with Python.
26. **The browser pane's tabs can disappear between turns**: open the page again with `navigate` before using refs.
27. **Windows asyncio "WinError 10054"** (a browser dropping a video connection) was logged as ERROR and counted by the
   Health page; it is filtered now (`logs.DropClosedConnections`).
28. **Test stand-ins of `AlertEngine.check`** must accept `helped=` (the help sign) as well as `fallen=` and `stayed=`.
29. **The time-lapse reads files with OpenCV**, not FFmpeg's concat demuxer, which dropped half the frames when a
   camera's size changed during the day.
30. **PostgreSQL waits for itself**: reading a table's layout with `inspect(engine)` while a transaction has just
   changed that table uses a second connection, which waits for ever (no deadlock error: two clients). Use
   `inspect(conn)` inside the transaction. The scratch checks of Phases 17-22 missed it; the real check is a
   database built by the OLD code (`git archive <old commit> labeler | tar -x -C <scratch>/old`, run Database
   there), then today's code. `_no_endless_wait` makes such a wait fail after 60 s instead.
31. **GitHub runs on UTC**, 8 hours behind Malaysia: a test that takes a recent day or week as "over" by the real
   clock passes here after midnight and fails there (Phase 40 step 1, fixed in `94ab78d`). Put test data in a week
   that is past on any clock, or pass `now`.
32. **Text drawn twice must have the same thickness** in OpenCV: letters are spaced by thickness, so a thick dark
   outline under thin white text drifts apart along the line (the recordings' clock looked doubled for weeks). Also:
   in Python edit scripts, code that contains "\n" goes into raw strings (r'''...'''), or it becomes a real line
   break in the file (Phase 42).
33. **OneDrive refuses scripts** (403 / sign-in needed, 2026-10-20): D-Fire was downloaded through the built-in
   browser pane (click Download; the file lands in the user's Downloads folder, may keep its final name while still
   growing: wait until its size stops changing), then moved into data/datasets. Check the size on the page first.
34. **Licences of "free" models**: YOLOE with your own words needs Apple's MobileCLIP, whose weights are **research
   only** (LICENSE_MODELS: no product use). The prompt-free YOLOE (fixed 4,585 words) needs no text model. Read the
   weights' licence, not only the code's.
35. **Ultralytics training downloads yolo11n.pt** to test half precision (check_amp): set
   `ultralytics.engine.trainer.check_amp = lambda model: True` (train_fire.py / train_guns.py) to avoid an unasked
   download.
36. **Ollama's context is 4096 tokens**: 4 pictures at 768 px are refused (400). A bigger num_ctx per request would
   reload the model between requests. Keep incident checks at 3 pictures (incidents.MAX_FRAMES).
37. **A shared vision-model queue starves low priorities** on a busy camera: warnings were refused behind people's
   labels until they were moved ahead; a check that waits must never block a critical one (ConfirmQueue lets a
   critical go while a lower one is out).
38. **Measure with the camera's own frame rate**: the fall check caught 9 of 15 falls at 10 pictures a second and 13 at
   30 (tools/incident_check.py --fps). And the person (pose) detector misses most people in crowds: count crowds with
   the plain detector at 1280 px.
39. **Check that a "still" test video really is still** (Phase 47 step 2): all four MOT20 cameras pan and zoom, and
   most of D-Fire's sample videos come from turning cameras that visit preset views ("Preset 5" on screen). Compare a
   few frames across the video (median gray difference, `cv2.phaseCorrelate`) before using one as a fixed view.
   Still here: MOT16-02, -04, -09; D-Fire FP1-5, 7-10, 15, 19-22 (FP9/FP10 have paths, the rest hills).
40. **A sign raised per person floods on a busy camera** (Phase 48-49): a fight raised by each pair of one fight, and a
   restricted area entered by people one after another, gave 2 and 29 alerts until they were limited to one a minute
   per camera (and area and sign). Every new per-person sign needs such a limit; a looping test clip (new track
   numbers each loop) shows the flood quickly.
41. **PowerShell 5.1's `Set-Content -Encoding utf8` writes a byte-order mark** (it happened to a plan doc,
   2026-10-25): edit files with the Edit/Write tools or Python, not Set-Content.
42. **The vision model is the shared limit** (Phase 50): about 42 answers a minute here; every new check or label
   costs from the same budget. Measure a change with `tools/incident_bench.py --busy` (people from MOT16's true marks,
   `--share` for the load) before claiming it fits; a looping short clip in a test copy floods it with "new" people
   (each loop new track numbers), heavier than a real camera.

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
**Step 1 done 2026-10-19**: checked beforehand on ubuntu-26.04, nothing broke (see section 2).

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
README "Help sign". Step 3 done: `Pipeline.gestures` (GestureWatch per camera, `cameras[].help_sign`), `_on_help`,
`alerts.HELP` / `HELP_RULE` / `ensure_help_rule`, `check(..., helped=True)` (repeat like falls), list_alerts `"help"`,
checks.alert_what "help" / "quiet", session.js `toast fall help` (beeps like a fall), Settings `help_enabled` /
`help_hold_s`. Test stand-ins of the alert engine need the `helped` argument. **Phase 32 is done.**

**Phase 33 started 2026-10-15**: step 1 done, `POST /api/recordings/search/photo` in `labeler/recordings_web.py`
(`add_routes(..., photo)`; shares `app.state.photo`), `moments(..., closest_first=True)` and items' `similarity`,
config `photo_search` (min_similarity 0.85, limit 20), `tools/photo_match_check.py`. Step 2 done: Recordings page
"or with a photo" (`#photo-file`; chips show "% alike"). `logs.DropClosedConnections` on the asyncio logger: Windows
WinError 10054 when a browser drops a video connection was logged as ERROR (also in the user's own log, 7 times) and
counted by the Health page. **Phase 33 is done.**

**Phase 34 started 2026-10-15** (viewers: blurred copy, the user's choice): step 1 done, `labeler/timelapse.py`.
OpenCV reads the day's files one by one (grab for skipped frames) into `recorder.FFmpegWriter` at 25 fps; FFmpeg's
concat demuxer was tried first and lost half the frames when the camera size changed during the day. Step 2 done:
`recordings_web` time-lapse routes (`app.state.timelapse` Maker; camera names checked against known cameras),
`timelapse.current` marks a map "older" when only the open recording grew (Make again, `again: true`), page
`checkLapse/makeLapse/pollLapse/playLapse`, `#lapse-open` -> playAt. **Phase 34 is done.** Next: Phase 35 (Update
button), the last of the plan.

**Phase 35 is done 2026-10-15** (`labeler/updater.py`, `app.state.updater` made in run.py with the database backup;
web `/api/update`; Settings "Updates" box; Health `check_updates`). The status parsing bug found by the tests: git's
" M file" lines must not be stripped (it cut the first letter of the name). **The Phases 30-35 plan is finished.**
Next: ask the user (Phase 26 real-camera week once their app runs the new version; Phase 28 step 1 after 2026-10-19;
Phase 29 when cameras are added; or new ideas).

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

**Next (2026-10-15): [PHASE36-40-PLAN.md](PHASE36-40-PLAN.md)**, the user's choices in this order: 36 first-run check
("what's new" + a guided check page), 37 faster playback, 38 PC watch, 39 unusual quiet, 40 weekly report. Also: Phase
26 once the user's app runs the new version; Phase 28 step 1 (GitHub runs on Ubuntu 26) after 2026-10-19. Offered on
2026-10-15 and not chosen (don't offer again unless asked): simple view, full-system long run, privacy hours, snooze
alerts, activity calendar, camera moved/blocked, left-behind objects (and earlier: phone alerts, picture backups,
family vs strangers, licenses, sounds, privacy masks, Malay/Chinese pages, blurred export).

## 9. Message to start the next session

> I'm continuing the VisionLabeler project in C:\VisionLabeler. Please read docs/HANDOFF.md and docs/ROADMAP.md
> first, run the tests to confirm everything is green, then continue with docs/PHASE50-55-PLAN.md (Phase 50 step 3:
> the vision model's load on the Health page). Ask me before each phase and each step starts, discuss any
> architecture with me first, and do it step by step.
