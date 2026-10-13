# Phases 19-22 plan: Home/Away, loitering, animals, scene questions (approved 2026-10-13; Phases 19-20 done 2026-10-14)

## Context

Phases 17 (mobility aids, falls) and 18 (vehicles) are done; the user asked for the next ideas and chose all four
new ones as future phases: **Home/Away arming, loitering alert, animals, scene questions**. Order: Home/Away first,
because every later alert type can then say "only when Away"; loitering is small; animals reuse the vehicle detector;
scene questions are the biggest. One phase at a time, the user is asked before each phase and each step; one commit
per step, tests, full suite green, a real run on a test copy (port 8001, `tools/make_test_config.py`), README/HANDOFF/
ROADMAP. Saved as `docs/PHASE19-22-PLAN.md` once approved; the next session starts from it (HANDOFF section 9).

Reused everywhere: alert rules as condition lists (`labeler/alerts.py`: `matches`, `describe`, `AlertEngine.check` /
`check_vehicle`, `is_vehicle_rule`, `HOURS`), rule validation in `labeler/web.py` (`add_rule`), the Alerts page
condition picker (`static/alerts.html`: `fields`, `NICE`, `describe`), pop-ups on every page (`static/session.js`),
alert clips (`Pipeline._check_alerts`, `ClipRecorder.request`), schema changes (`db.NEW_COLUMNS`, `NULLABLE_NOW`,
new tables via create_all; PostgreSQL checked on a scratch database first), Settings (`labeler/settings.py` SIMPLE),
Health lines (`labeler/healthcheck.py`).

## Phase 19: Home / Away arming

Ask at the start: may viewers (family) switch it, or admins only; a weekly schedule yes/no.
**Answers (2026-10-14): admins and viewers may switch; yes, a weekly schedule.**
1. **Done 2026-10-14.** State: table `arming` (mode home/away, since, by whom, why: button or schedule); `labeler/arming.py`
   (`current()`, `set(mode, by)`, schedule from Settings: e.g. weekdays 08:00-18:00 Away; a button press holds until
   the next schedule change). API GET/POST `/api/arming` (audited).
2. **Done 2026-10-14.** Rules: condition `armed = away / home` (attrs["armed"] in `AlertEngine.check`, `check_vehicle`); rules without it
   work as now. The built-in "Fall" rule stays always on.
3. **Done 2026-10-14** (a tap asks first, the user's choice). Pages: a Home/Away switch in the header of every page (session.js, big enough for a phone), shown on the Live
   page's "Today" tiles and the Health page; Alerts page "only when Away/Home".
4. **Done 2026-10-14** (PostgreSQL checked on a scratch database). Tests (schedule edges, override until the next
   change, rule matching), real run, docs. **Phase 19 is done** (commits `46daf57`, `1843a06`, `b41530b`, docs).

## Phase 20: Loitering alert (staying too long)

1. **Done 2026-10-14.** Condition `stayed = <zone>:<seconds>` (e.g. `gate:120`): the person (or vehicle) has been inside that area at least
   that long. The time inside is already kept (`zones.PersonZones.inside` = zone -> entered at); the pipeline checks
   people inside areas once a second (not every frame) with `AlertEngine.check(..., attrs with stayed seconds)`;
   once per rule and person as now. Vehicles the same in `VehicleWatch` (a car standing at the gate 10 minutes).
2. **Done 2026-10-14** (with the Alerts page picker). Combines with `hours` and `armed` (e.g. *stayed at gate 2 min, 22:00-06:00, only when Away*).
3. **Done 2026-10-14.** Live picture: seconds inside an area next to the person (only for areas with such a rule).
4. **Done 2026-10-14** (`tools/loiter_check.py`; README "Loitering"; the user's footage: later). Measure: MOT16 (people waiting at a crossing) and the user's own footage when there is some; tests; docs.

## Phase 21: Animals

Ask at the start: which animals (COCO has bird, cat, dog, horse, sheep, cow; monkeys and monitor lizards are not in it:
said honestly). **Answers (2026-10-14): dogs, cats and birds; security cameras only.**
1. **Done 2026-10-14.** One detector call for vehicles and animals (the classes added to the vehicle detector's list: no extra cost);
   generalize `VehicleWatch` (kinds map + store) and add table `animals` (+ zone events) with an AnimalStore; animals
   resting stay one record like parked vehicles; on all cameras or security only (asked).
2. **Done 2026-10-14** (list, Ask box, Reports; the alert rules became step 2b with the user). Live page "Animals seen", Ask box words (dog, cat, bird...), Reports counts, alert rules `animal = dog`
   (with zone, hours, armed).
2b. **Done 2026-10-14**: the alert rules `animal = dog` (with zone, hours, armed, stayed).
3. Measure: COCO val2017 animal detection (marked boxes already here), false alarms on MOT16 and the UR Fall
   activities (indoors, no animals).
4. Settings/Health, docs.

## Phase 22: Scene questions

Ask at the start: default interval; alert after one "yes" or two in a row.
1. Table `scene_questions` (camera, question, every_s default 300, alert on yes/no, enabled, last answer/time/picture)
   and `scene_answers` (history, kept like the audit log). Admins write them on the Alerts page.
2. Runner: for each due question, the camera's latest clean frame (the pipeline keeps an unannotated copy every
   few seconds) goes to the vision model only when its queue is empty (people and vehicles first); JSON answer
   yes/no/unsure; an alert when the answer turns to "yes" (two in a row by default, against single wrong answers).
   Job ids in their own range (vehicles use negative ids), routed back like vehicles.
3. Alerts: `alert_events` with neither a person nor a vehicle (allowed since Phase 18) plus a picture
   (`alert_events.picture`, NEW_COLUMNS); pop-up, Alerts list, clip as now; "▶ Recording".
4. Measure reliability with questions whose answers are known: "Is someone lying on the floor?" on UR Fall frames
   (frame marks), "Is there a car?" on MOT16-04/13 frames (marked cars); speed per question. No smoke/fire footage
   here: said honestly, and the user tests it with their own camera.
5. Settings (on/off, default interval), Health line (questions, last answers, overdue), docs.

## Critical files

`labeler/alerts.py`, `labeler/web.py`, `labeler/db.py`, `labeler/pipeline.py`, `labeler/vehicles.py`, `labeler/zones.py`,
`labeler/vlm.py`, `run.py`, `labeler/settings.py`, `labeler/healthcheck.py`, `static/session.js`, `static/alerts.html`,
`static/index.html`, `static/settings.html`; new: `labeler/arming.py`, `labeler/scene.py`, tests per phase
(`tests/test_arming.py`, `test_loitering.py`, `test_animals.py`, `test_scene.py`), measuring tools in `tools/`.

## Verification (every phase)

- Unit tests for each new rule condition, table and endpoint (patterns: tests/test_vehicles.py, tests/test_falls.py);
  full suite green (592 now).
- PostgreSQL schema changes checked on a scratch database on the user's server before committing (as in Phase 17-18).
- A real run on a test copy in the browser pane: the switch/condition/list works, an alert pops up with a picture
  and clip; no leftover `python.exe run.py`.
- Numbers measured honestly in README (what was found, what was not tested).
