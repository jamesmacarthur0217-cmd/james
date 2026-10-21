# Phases 44-49 plan: safety incidents and suspicious behaviour on security cameras (architecture agreed with the user 2026-10-20; Phase 44 done)

## Context

On 2026-10-20 the Phases 41-43 plan was finished. The user listed 11 kinds of safety incident to detect (people needing
help, falls and collisions, fire and smoke, floor hazards, aggression, dangerous objects, child safety, blocked
access, unsafe equipment, animals and pests, unusual crowding) and added a 12th: **dangerous or suspicious behaviour**.
The place is not decided (a kiosk or something else); the user chose **security cameras** (a wide view of a room,
entrance or yard: most of these need the whole scene, not one person up close). Use stays personal / testing (Phase 1
licenses stay deferred; ask again before any business use: YOLO, YOLOE / YOLO-World are AGPL-3.0, and most of the test
sets below are non-commercial).

Today each safety feature works on its own: falls and the help sign from body points (`labeler/falls.py`,
`labeler/gestures.py`), no movement and unusual quiet, animals (dogs, cats, birds), loitering (`stayed`), areas and
counting lines (`labeler/zones.py`), scene questions to the vision model every few minutes (`labeler/scene.py`), the
camera picture check (covered, dark, blurry). Twelve kinds on one GPU need a shared layer: the vision model takes about
0.6 s per answer (more with several pictures) and also labels every person, so it cannot be asked twelve questions about
every area every second.

## The design (three shared parts, built in Phase 44)

**Agreed with the user 2026-10-20** (they asked to discuss any architecture before it is built): (1) fast checks,
then the vision model confirms; (2) incidents stored in today's `alert_events` with new columns, not a new table;
(3) a role on each drawn area, not separate safety zones; (4) extra models (fire/smoke, pests, guns) decided per phase,
license checked, download with their OK. Any further architecture choice (new tables, new threads, new models,
changes to the vision model queue) is discussed with the user before it is built.

1. **Area roles.** Each drawn area gets a role: walkway, exit, cooking, food prep, dining, play, restricted, bed/sofa
   (today's "lying is normal here"), or none. The same thing is normal in one area and an alarm in another (a knife or a
   flame in the cooking area; a child in the play area; a person lying on the bed). Roles are a list in the config, so
   another place can name its own.
2. **Spot fast, confirm slowly.**
   - Fast, every frame or every second, no vision model: YOLO body points and objects, small extra detectors where
     needed (fire/smoke, pests), and simple rules (lying, slumped, not moving, speed, people per area, a knife near a
     wrist, a patch of floor that changed against the area's empty picture). They only raise a *suspect*.
   - Slow: the vision model looks at the suspect (3-4 frames from the last seconds, the area cut out, as in Phase 27)
     with that category's question and answers yes / no / unsure plus a few words on what it sees. An alert comes only
     on yes (twice in a row for things that stay, like a spill or a blocked exit).
   - One queue with priorities: critical suspects first (fire, weapon, fight, person down, child in danger), then
     warnings (moved ahead of people's labels in Phase 45, the user's choice), then people's labels, then slow rounds
     over each area for things with no fast sign (cables, a broken chair). A limit per minute keeps people's labels going; a suspect that waited too long is dropped and counted.
3. **One incident record.** Category, level (*critical*: fire, weapon, fight, person down, child in danger, climbing
   in at night; *warning*: spill, blocked exit, crowding, pests, equipment, suspicious behaviour), area, picture and
   clip, the vision model's answer and words. Stored in `alert_events` (new columns), shown with today's pop-ups (critical:
   red with a beep, like falls), Alerts page, Summary, week report; "was this right?" (Phase 24, Check page) per
   category, so every category has a measured false-alarm rate on the user's own cameras.

**Behaviour, never appearance.** Suspicious-behaviour and every other rule look at what a person *does* (where, how
long, how they move, what they hold), never at age, gender, clothing, skin or a face covering (masks and helmets are
normal in Malaysia). No face recognition, no automatic action: an alert shows the clip and the reason; a person decides.

## Phase 44: Safety base (area roles, suspects, confirm queue, incidents)

1. **Area roles**: `zones.role` (in `db.NEW_COLUMNS`, checked on a scratch PostgreSQL database built by the old code,
   pitfall 30); `lying_ok` becomes role "bed" (kept in sync for old rules); the Live page's area tool gets a role
   picker; `safety.roles` in the config; `cameras[].safety` (on by default for security cameras). Tests.
   **Done 2026-10-20**: roles fixed in `zones.ROLES` (later checks depend on their meaning); the config option is
   `zones.roles` (which ones the page offers; all by default); `zones.role` + `db.set_zone_role`, PATCH
   `/api/zones/{id}` (admins, audited), GET `/api/zones/roles`; the Live page picks a role when drawing and changes it in
   the list. An old area with lying_ok reads as "bed". PostgreSQL: a database built by the old code opened in 0.2 s with
   `zones.role` added and the old bed area read as "bed". Browser: drawn, saved as bed, changed to walkway (audited),
   phone width fits (a zone row wraps). `cameras[].safety` moves to step 2, where something uses it. 899 tests.
2. **Suspects and the confirm queue**: `labeler/incidents.py`: `Suspect` (category, camera, area, people, frames from a
   short ring kept by the Pipeline, reason), `ConfirmQueue` (priorities, per-minute limit, a cooldown per category and
   area, drop when too old), `vlm.PROMPT_INCIDENT` per category with several pictures (JSON: answer, what), job numbers
   in their own range (like scene questions, routed in run.py). Measure the vision model's time with 1, 3 and 4 frames
   on this PC and how many confirms per minute leave people's labels on time.
   **Done 2026-10-20** (the user chose: one priority queue, pictures from memory, a per-minute limit):
   `vlm.VLMWorker` is one priority queue (CRITICAL 0, PEOPLE 10, WARNING 20, ROUND 30; `max_queue` counts PEOPLE and
   below, so critical checks are never refused; the dispatcher takes a free slot first, then the most urgent job: found
   by a test, it used to hold the next job while waiting, so one label could still jump ahead); `labeler/incidents.py`
   (FrameMemory, CATEGORIES with the first questions, Suspect, ConfirmQueue: critical first, one at a time,
   `confirm_per_min` 6, `max_wait_s` 30, `cooldown_s` 60, recent answers and counts); `vlm.incident_prompt` /
   `clean_incident` (yes / no / unsure + a few words); Pipeline.memory on safety cameras (`cameras[].safety`, security
   by default); run.py routes job numbers -(1e9 + n). Measured with `tools/incident_bench.py` (MOT16-09, qwen3-vl 4B,
   one at a time): a check takes 0.6 s with 1 picture, 1.4 s with 3; **4 pictures are refused** (4263 tokens over
   Ollama's 4096 context; a bigger context per request would reload the model), so at most 3 (`MAX_FRAMES`). A very
   busy camera (a new person every 4 s, 2 pictures each): people's labels waited 3.6 s without checks, 3.9 s with 3
   critical checks a minute, 7.7 s (90%: 12.5 s, a small backlog) with 6; a critical check waited ~3.7 s (the label
   being made finishes first). 6 stays the ceiling (checks come only when a fast check sees something). Real run on a
   test copy with a security camera (MOT16-09): "safety checks on: cam1", labels as before, no errors. Nothing raises
   suspects yet (Phase 45+), so the whole way from a suspect to an answer inside the app is tested with stand-ins and
   the tool, not yet live. 909 tests.
3. **Incidents on the pages**: `alert_events.category` / `level` / `confirm` (+ PostgreSQL check); built-in rules per
   category (condition `incident = <category>`, with zone, hours, Home / Away like other rules); pop-ups by level;
   Alerts page filter; Check page kind "incident"; Health line "safety checks" (suspects and confirms per hour, the
   queue's wait, dropped ones); Settings box (each category on/off). `tools/incident_check.py`: runs videos with true
   time spans per category and reports found / false / late (the base for every later phase's measuring).
   **Done 2026-10-20** (the user chose: a built-in rule per kind, alert only on yes, critical red with a beep):
   `alert_events.category` / `level` / `confirm` (checked on a scratch PostgreSQL database built by the step 2 code:
   0.2 s, 3 columns added, the old alert kept); `alerts.ensure_incident_rules` (12 rules, made only when a safety camera
   and the vision model are there, so a kiosk-only home gets none), `AlertEngine.check_incident` (camera, area, hours,
   Home / Away), `incidents.IncidentAlerts` (newest picture with the area outlined + blurred copy, clip);
   `db.record_incident_alert`, list_alerts `category` / `level` / `confirm` / `words`; session.js critical = red with a
   beep, warning = plain; Alerts page "Show" filter, `critical` mark, rule picker "safety incident"; GET `/api/safety`;
   Check page kind per safety kind ("fire or smoke alerts"); summary `incident` listed in the week; Health
   `check_safety`; Settings `safety_enabled`, `safety_confirm_per_min`, camera column "Safety". Real run: two stand-in
   suspects through the real vision model into a running test copy: UR Fall lying -> "yes" (1.8 s), red pop-up, alert,
   Check page item; MOT16 street asked about fire -> "no", no alert. **Moved to Phase 45 step 1**: the measuring tool
   `tools/incident_check.py` (nothing raises suspects before Phase 45, so it would measure nothing yet).
   **Phase 44 is done** (916 tests).

## Phase 45: People (help, falls, children, crowding)

**Design agreed with the user 2026-10-20**: (1) the quick checks run in each camera every frame (a small safety watch
per Pipeline fed with the body points and boxes it already has, like FallWatch), not in a separate thread; (2) **falls
are checked by the vision model too** (the user's choice, against the recommendation): a fall FallWatch confirms becomes
a "fall" suspect; "yes" fires today's "Fall" rule (person, clip, FALL pop-up); a clear "no" stops it; **fail-safe**: no
vision model, a failed check or one dropped for waiting too long still alerts, so a fall is never lost to Ollama; a
camera without picture memory (kiosk, safety off) alerts at once as today. The help sign stays instant; (3) a child is
the age group "child" from the labels **or a small body** next to adults nearby (the user's choice: sooner, more false
alarms; the vision model confirms); (4) crowding: a limit per area ("at most N people", set when drawing; none = not
checked).

0. (from Phase 44) `tools/incident_check.py`: videos with true time spans per kind -> found / false alarms per hour /
   late, through the real quick checks and the vision model.
   **Done 2026-10-20**: the app's own Pipeline on each video's clock (VideoClock, scratch database, security camera,
   safety on), the confirm queue on the video's clock too, the video waiting while the vision model answers (quality,
   not speed); truth CSV `video,kind,start_s,end_s,half`; `--make-truth urfd` writes
   `data/datasets/urfd/incidents_truth.csv` (30 falls: "fall" over the falling frames, "person_down" from lying to the
   end + 60 s; 40 everyday clips as negatives; the color half); `--hold-s`, `--no-vlm` (quick checks alone), `--json`.
   Reports spans, raised (suspects), found (confirmed), false per hour at both stages, late, answers. Checked: 3 videos
   run (0 suspects: no quick check yet); a stand-in suspect through the real vision model on fall-01 counted as found
   (8.3 s after lying began) and a stand-in fire as a false suspect, not a false alert. Also: a suspect's time is now
   the camera's own (the newest picture's), right for videos. 921 tests.

1. **Person down or slumped**: today's lying from body points works for side views; add *slumped* (sitting, head below
   the shoulders, still for 20 s) and *down after a fall* on security views; overturned wheelchair (mobility aid
   "wheelchair" + lying). Confirm: "Does this person need help (collapsed, slumped, in distress)?". Distress has no fast
   sign: a slow round over people who have not moved for long. Measured with UR Fall (have it) and normal sitting
   (MOT16, COCO).
   **Done 2026-10-20**: `labeler/safety.py` SafetyWatch (lying still 20 s, slumped 30 s, not moved 5 min = slow round
   with a distress question), `zones.role_at`; Pipeline `_raise` (a crop around the person), `_on_fall` through the
   queue with the fail-safe (Suspect.on_done; only a clear "no" stops it; cooldown, no pictures, failed, dropped: alert);
   `incidents.FALL_QUESTION` (what is visible: "need help?" called a real fall "exercising"); a fall marks lying still
   as handled. Measured with `tools/incident_check.py` (README "People: falls, lying still, slumped"): report half
   14 of 15 falls with an alert (before 13), 3 false alerts on 20 everyday clips (before 1); MOT16 none. Not done:
   overturned wheelchair (lying still covers the person). 931 tests.
2. **Child safety**: age group child (age model / vision model) and rules: a child with no adult within a distance
   (scaled by body height) for 60 s; a child in a cooking, food-prep or restricted area; climbing (hips above the
   child's own standing height, or on a table box). Confirm before alerting.
   **Done 2026-10-20**: SafetyWatch `_children` (label, else small upright body next to an upright adult at the same
   depth, 80% of 3 s), alone (`child_alone_s` 120, 0 = off), cooking / food prep / restricted area (`child_area_s` 3),
   climbing (feet up 25% of the height, the body not smaller, 2 s); Pipeline `_age_of`. `tools/incident_check.py
   --labels --areas`. Measured only false alarms (no video with children here): 77 videos, no child suspect; the size
   rule took 17 of ~500 people for a child at some moment, 3 of them real children (README "Children"). Also fixed: a
   vision model worker closed while jobs were left raised an error in its thread (now: empty answers). 935 tests.
3. **Crowding**: people per area against the area's limit (a setting per area; density from the area's size in the
   picture), held 30 s; *pushing*: many people in an area speeding up the same way, bodies overlapping; "someone trapped
   against a barrier" only through the vision model. Measured on MOT16 and, with the user's OK to download, MOT20 (crowds,
   non-commercial).
   **Done 2026-10-20** (MOT20 downloaded with the user's OK, 5.0 GB, train part unpacked, videos made):
   `zones.max_people` (Live page "at most N people", "now N"; PATCH `/api/zones/{id}` role and/or limit; checked on a
   scratch PostgreSQL database built by the old code), SafetyWatch `crowd` (over the limit `crowd_s` 30, running
   together), the vision model decides (the user's choice). Found and changed with the user: the pose detector finds
   0-40% of people in a crowd, so a copy of the plain detector counts every area once a second at 1280 px (14 ms;
   75-90% found, 25% in the densest MOT20-05); warnings were refused behind a busy camera's labels, so the queue is
   now critical, warning, people, round, and a critical check never waits for a lower one. `tools/crowd_check.py`.
   **Phase 45 is done.**

## Phase 46: Fire and smoke

**Design agreed with the user 2026-10-20**: (1) **our own model**: D-Fire (21,527 photos with fire and smoke boxes,
GAIA; its GitHub license file: "made available to the public domain under the Creative Commons Zero v1.0 Universal
license"; mirrors say CC BY 4.0, the original says CC0) downloaded (size checked first) and a small YOLO11 trained
here, measured on D-Fire's own test part (trained with Ultralytics: AGPL-3.0 like the other YOLO models, Phase 1);
not chosen: ready-made models (AGPL, training data unclear; Pyronear is for wildfires), the vision model alone;
(2) once a second on safety cameras (like the crowd count), not every frame; (3) fire and smoke inside a "cooking"
area are ignored unless they reach past it; (4) test videos: D-Fire's sample surveillance videos, the videos here for
false alarms (MOT16, MOT20, UR Fall) and COCO kitchen photos, **and MIVIA** (31 videos, 14 with fire, 17 tricky
without: red objects, clouds, smoke; no terms stated; the user requests it at
mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/ and it is used when it arrives).

1. **A small fire/smoke detector**: find one (a YOLO trained on D-Fire or similar; license checked before any
   download, which needs the user's OK) or train YOLO11n on D-Fire here; every second on safety cameras at 640 px.
   Flames inside a cooking area are normal unless they grow past it; smoke counts everywhere outside it.
   **Done 2026-10-20**: D-Fire downloaded with the user's OK through the browser pane (OneDrive refuses scripts:
   3.04 GB, 21,527 pictures + labels, no unsafe paths; moved from Downloads to data/datasets/dfire/download).
   `tools/train_fire.py` (prepare: D-Fire's train/test, a tenth of train kept apart to choose the best round; train:
   from yolo11s.pt already here, 640 px, 40 rounds in 1 h 11 min, Ultralytics' half-precision check skipped as it would
   download yolo11n.pt; test once). Test part (4,302 pictures, 4 with broken labels skipped): mAP50 0.767 (smoke 0.827,
   P 0.81 R 0.77; fire 0.706, P 0.71 R 0.64), 9.5 ms a picture. Model: data/models/fire/fire_smoke.pt (git-ignored;
   tools/models.py packs it in step 2).
   **Step 2 (the check in the app) done 2026-10-20**: `labeler/fire.py` FireWatch (once a second, 3 of the last 5 looks,
   cooking areas ignored unless reached past, again after 60 s of nothing), Pipeline `_look_for_fire` (own model,
   640 px, conf 0.35), a critical "fire" suspect with a crop and pictures over 8 s (`suspect(span_s=)`); config
   `safety.fire_*`; `tools/models.py` lists and packs the trained model (optional). Real run: a D-Fire fire picture
   raised and confirmed ("large fire with smoke"), a clear one nothing. 954 tests.
2. **Confirm and measure**: the vision model on 3 frames over 10 s (smoke spreads, flames flicker). False alarms on
   things that look like fire: stoves, candles, sunsets, red lights, steam, a TV (COCO kitchens, MOT16, UR Fall);
   found / late on fire videos. Compare with a plain scene question ("Is there fire or smoke?") every 5 minutes.
3. Pages, Settings, README; critical level with a beep.
   **Steps 2-3 measured 2026-10-20** (D-Fire's 100 sample videos, 0.44 GB, downloaded with the user's OK through the
   browser; truth `data/datasets/dfire/videos_truth.csv`): the app found 38 of 50 fire videos (median 3 s), 5 false of
   50 tricky no-fire videos; the vision model alone on one picture each: 36 and 5 (README "Fire and smoke"). Everyday
   videos: nothing. Pages and the beep came from Phase 44 (critical level). **Phase 46 is done**; to add when the user
   has them: the MIVIA videos (requested by the user).

## Phase 47: Objects and places (weapons, blocked exits, floor hazards, equipment, pests)

**Design agreed with the user 2026-10-21**: (1) dangerous objects: knives, scissors and bats from the plain detector's
once-a-second pass (the crowd count's), held at a person's wrist outside a cooking / food-prep area, **and an
open-vocabulary model, YOLOE (Ultralytics 8.4 has it; weights to download with the user's OK after checking size and
license; AGPL), for guns** ("gun, pistol, rifle, machete"), kept only if it measures well; (2) floor hazards, blocked
exits and left objects by comparing walkway / exit areas with their empty picture; (3) unsafe equipment by the vision
model's slow round over each area with a role every 10 minutes; (4) pests: the same YOLOE asked for "rat, mouse,
cockroach" in food areas, kept only if good when measured (else left out, said so). One YOLOE serves (1) and (4).
Measuring guns and pests needs pictures of them (e.g. Open Images classes, CC BY 2.0): a download asked for when the
step comes.

1. **Dangerous objects**: the object pass already runs on security cameras (`Pipeline._object_classes`); add COCO knife,
   scissors and baseball bat; *held* = the object's box at a wrist for 1 s. Cooking and food-prep areas: ignored; dining,
   walkway, exit: a suspect. Confirm: "Is the person holding a knife or weapon in a threatening way, or using it for
   food or work?". Guns are not in COCO: only an open-vocabulary detector (step 3) or the vision model can see them;
   measured separately and said honestly.
   **Done 2026-10-21** (the user chose YOLOE prompt-free, 27 MB, after MobileCLIP's research-only license came up):
   `labeler/weapons.py` (WeaponWatch, at_wrist), Pipeline `_look_objects` (counts + knives / scissors / bats in one
   pass), `_look_yoloe`, `_check_weapons`; config `safety.weapon_*`; `tools/models.py` lists YOLOE as optional;
   `tools/weapon_check.py`. False alarms: 0 of 87 everyday knives / scissors / bats at a wrist confirmed (COCO),
   YOLOE 1 of 500 ordinary photos, everyday videos nothing. Then (the user's choices, 2026-10-21): Open Images gun
   photos downloaded for testing (439, 127 MB); YOLOE found only 10% of the guns, so **our own gun detector** was
   trained on Open Images' training photos (2,236 + 1,000 without guns, 1.03 GB; `tools/get_gun_training.py`,
   `tools/train_guns.py`): 76% of the guns found; YOLOE kept for machetes and swords only; **any real gun in hand
   alerts** (GUN_QUESTION). Whole check: 130 of 217 held guns alerted, 0 of 89 everyday knives / bats, 0 of 300
   ordinary people photos (README "Dangerous objects"). **Step 1 done.**
2. **The empty picture of an area**: for walkway and exit areas, a picture of the area with nobody in it (a median of
   recent empty frames, refreshed daily and after lighting changes). A patch that changed and stays changed while
   nobody stands there becomes a suspect: *blocked exit* (exit area, 2 min), *floor hazard* (walkway, 30 s: spill,
   glass, dropped food), *left object* (a bag; also used in Phase 49). Confirm: "What is on the floor / in the exit:
   spill, broken glass, bag, furniture, nothing?".
   **Design agreed with the user 2026-10-21**: (1) one empty picture per safety camera, **saved as a file** in data/
   (kept across restarts), made from many small frames with the parts where people stand left out, refreshed daily
   and after a lighting change (the whole picture changed: made again, nothing raised); (2) compared once a second in
   the camera's own loop at a small size, only inside walkway and exit areas; no new table, thread, model or download;
   (3) the vision model sees **the empty picture and now** (picture 1 = the area empty, 2 = now: what changed?);
   (4) an alert after **two "yes" about a minute apart**; a "no" (a chair moved) puts the change into the empty picture
   so it is not asked again; (5) a left bag is a floor hazard (walkway) or blocked exit (exit) for now; Phase 49 adds
   who left it; (6) measured with **made-up tests**: objects (bags, chairs, boxes, dark "spill" patches) pasted onto
   the floor of still-camera videos here (MOT16, UR Fall), false alarms on the unchanged videos; no download.
   **Done 2026-10-21**: `labeler/floor.py` FloorWatch (empty picture from the median of 15 looks 2 s apart, people
   left out: an average smeared missed passers-by in; a spot next to a change does not follow either; the lights /
   a moved camera = learned again), Pipeline `_look_at_floor` (tracked people + the plain detector's boxes),
   `Suspect.alert_if_yes` (a first look notes "yes", the second alerts) and `Suspect.cut` (a tighter cut-out),
   config `safety.floor_*`, Health line "floor and exits", What's new entry p47. The question asks to compare the two
   pictures (tune half: 11 of 13 changes instead of 9). `tools/make_floor_videos.py`: UR Fall clips are too short and
   the MOT20 cameras and most of D-Fire's move, so MOT16-02/-04/-09 and D-Fire's still guard booth (FP9, FP10).
   Report half: objects 5 of 6 alerted (exits 3 of 3), spills 1 of 3 (dark ones look like shadows), 0 false alerts,
   1.5 min on a walkway, 3 min in an exit. The real run on a test copy found the same suitcase asked about again
   (spots hidden by a passer-by began to differ later): an alert or a "no" now covers the whole connected change;
   the report half was run again after it (same numbers). README "Floor hazards and blocked exits". **Step 2 done.**
3. **Equipment and pests**: equipment through slow rounds of the vision model per area (an overturned table or chair,
   cables across the walkway, something hanging loose) and a table box that changed shape against the empty picture.
   Pests: no rodents in COCO; try YOLOE / YOLO-World (open-vocabulary, Ultralytics) with "rat, mouse, cockroach" in food
   areas, often at night (black-and-white pictures). If it is poor when measured, leave pests out and say so.
   **Design agreed with the user 2026-10-21**: (1) equipment: every 10 minutes each safety camera asks the vision
   model about **every area with a role**, from the newest picture with the area cut out, at the lowest priority
   (round, after people's labels); an alert after **two "yes" a minute apart** (as for floors); (2) pests: YOLOE
   prompt-free (already here, its fixed word list has rat, mouse, rodent, roach; gecko and lizard, the harmless cicak,
   never count; insects too small for a room-wide camera) in the same once-a-second YOLOE pass as machetes and swords,
   **only inside cooking, food prep and dining areas**; seen on 2 of the last 3 looks, the vision model checks the last
   seconds and **one "yes" alerts** (a mouse runs away); (3) measured with **Open Images photos of real mice / rats**
   (size and licence checked, downloaded only with the user's OK), false alarms on COCO computer mice and kitchens
   (here) and the everyday videos; equipment: false alarms on everyday videos and made-up cases (e.g. an overturned
   chair pasted in); (4) no new table, thread or model.
   **Done 2026-10-21**: `labeler/equipment.py` EquipmentRounds (first round 1 min after the start, areas 20 s apart;
   "reported" after an alert until a "no"; an unanswered or dropped round again in a minute), Pipeline
   `_round_equipment` (level round; the second look as a warning so it is not left behind), config
   `safety.equipment_*`, Health line "unsafe equipment". `tools/equipment_check.py` (report half): a chair on its side
   6 of 9 "yes", a cable across 4 of 9, ordinary scenes and offices 0 of 29, COCO rooms 2 of 150. Real run on a test
   copy: the alert at the second look. **Pests left out** (the user's choice after measuring): YOLOE found 3% of 251
   Open Images rodent photos (downloaded with OK, 74 MB) and took computer mice for mice (34 of 46);
   `tools/pest_check.py` kept for a better model. **Step 3 done; Phase 47 is done.**

## Phase 48: Aggression and collisions

1. **Fast signs from body points between two people close together**: wrist speed against body size, an arm reaching the
   other's head or body, one falling right after contact, an object leaving a hand at speed (throwing). Collisions: a
   person falling right after touching a moving large object (trolley through the open-vocabulary detector).
2. **Confirm on a short clip** (4 frames over 2 s): "Are these people fighting or attacking someone?". Measured on
   RWF-2000 / Hockey Fight (non-commercial, downloads with OK), and false alarms on hugs, handshakes, dancing, children
   playing, sports.
3. Pages, levels, README.

## Phase 49: Suspicious behaviour (behaviour only)

1. **Behaviour signs** (rules, each with its own setting): came back to the same area 3 times in 30 minutes (re-linked
   person); pacing back and forth across an area; stopping at several doors or vehicles in a row; a hand at a parked
   vehicle's door or window for 5 s; climbing a fence or wall (hips rising along an area's edge); in a restricted
   area at night or when Away (today's zone + hours + Away, now with the role); leaving a bag and walking out of view
   (Phase 47 step 2); following someone closely through an exit line (two crossings within 2 s); running off right after
   another alert; covering or turning the camera (the picture check's covered / a new "moved" finding).
2. **Confirm with a clip and a reason**: the vision model is asked what the person is *doing* (the prompt names the
   behaviour and forbids judging by looks); the alert says the behaviour ("came back 3 times in 20 min; looked into 2
   cars"). Levels: climbing in at night = critical, the rest warning.
3. **Measure false alarms on normal footage** (MOT16: shoppers and walkers; target zero to a few per hour), the real
   cases we can make on a test copy (an empty COCO scene with people pasted to walk a path), README with the
   behaviour-only rule, Settings.

## Way of working (unchanged)

Ask before each phase and each step; one commit per step; full suite green; push and check `gh run list`; real run on
a test copy (port 8001, `tools/make_test_config.py`), never the user's app (port 8000) or data; schema changes checked
on a scratch PostgreSQL database built by the old code; downloads (models, datasets) only with the user's OK and after
checking their license; measure before claiming a gain, tune on one half and report on the other (pitfall 21); edit
with the Write/Edit tools or Python scripts written with Write (pitfalls 1, 22, 25); tests never depend on the real
date (pitfall 31); short, simple answers for the user. Each phase adds its entry to `labeler/whatsnew.py`.

## Open points for the user

- The user has one webcam (a desk view, kiosk mode): every phase is measured on recorded videos and test copies; real
  numbers need a security camera on the real place (Phase 26 / 29).
- The GPU budget is measured in Phase 44 step 2; with many security cameras the confirm limit may have to be lower.
- Business use later: Phase 1 (licenses) first.

## Verification

- Each phase: its tests, the full suite (>= 892), `tools/incident_check.py` numbers per category (found, false per hour,
  late), a real run on a test copy, browser check where pages change.
- Phase 44: the queue keeps people's labels on time with the confirm limit reached (measured).
