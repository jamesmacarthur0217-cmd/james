# Phases 44-49 plan: safety incidents and suspicious behaviour on security cameras (architecture agreed with the user 2026-10-20; no phase started)

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
     people's labels, then warnings, then slow rounds over each area for things with no fast sign (cables, a broken
     chair). A limit per minute keeps people's labels going; a suspect that waited too long is dropped and counted.
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
3. **Incidents on the pages**: `alert_events.category` / `level` / `confirm` (+ PostgreSQL check); built-in rules per
   category (condition `incident = <category>`, with zone, hours, Home / Away like other rules); pop-ups by level;
   Alerts page filter; Check page kind "incident"; Health line "safety checks" (suspects and confirms per hour, the
   queue's wait, dropped ones); Settings box (each category on/off). `tools/incident_check.py`: runs videos with true
   time spans per category and reports found / false / late (the base for every later phase's measuring).

## Phase 45: People (help, falls, children, crowding)

1. **Person down or slumped**: today's lying from body points works for side views; add *slumped* (sitting, head below
   the shoulders, still for 20 s) and *down after a fall* on security views; overturned wheelchair (mobility aid
   "wheelchair" + lying). Confirm: "Does this person need help (collapsed, slumped, in distress)?". Distress has no fast
   sign: a slow round over people who have not moved for long. Measured with UR Fall (have it) and normal sitting
   (MOT16, COCO).
2. **Child safety**: age group child (age model / vision model) and rules: a child with no adult within a distance
   (scaled by body height) for 60 s; a child in a cooking, food-prep or restricted area; climbing (hips above the
   child's own standing height, or on a table box). Confirm before alerting.
3. **Crowding**: people per area against the area's limit (a setting per area; density from the area's size in the
   picture), held 30 s; *pushing*: many people in an area speeding up the same way, bodies overlapping; "someone trapped
   against a barrier" only through the vision model. Measured on MOT16 and, with the user's OK to download, MOT20 (crowds,
   non-commercial).

## Phase 46: Fire and smoke

1. **A small fire/smoke detector**: find one (a YOLO trained on D-Fire or similar; license checked before any
   download, which needs the user's OK) or train YOLO11n on D-Fire here; every second on safety cameras at 640 px.
   Flames inside a cooking area are normal unless they grow past it; smoke counts everywhere outside it.
2. **Confirm and measure**: the vision model on 3 frames over 10 s (smoke spreads, flames flicker). False alarms on
   things that look like fire: stoves, candles, sunsets, red lights, steam, a TV (COCO kitchens, MOT16, UR Fall);
   found / late on fire videos. Compare with a plain scene question ("Is there fire or smoke?") every 5 minutes.
3. Pages, Settings, README; critical level with a beep.

## Phase 47: Objects and places (weapons, blocked exits, floor hazards, equipment, pests)

1. **Dangerous objects**: the object pass already runs on security cameras (`Pipeline._object_classes`); add COCO knife,
   scissors and baseball bat; *held* = the object's box at a wrist for 1 s. Cooking and food-prep areas: ignored; dining,
   walkway, exit: a suspect. Confirm: "Is the person holding a knife or weapon in a threatening way, or using it for
   food or work?". Guns are not in COCO: only an open-vocabulary detector (step 3) or the vision model can see them;
   measured separately and said honestly.
2. **The empty picture of an area**: for walkway and exit areas, a picture of the area with nobody in it (a median of
   recent empty frames, refreshed daily and after lighting changes). A patch that changed and stays changed while
   nobody stands there becomes a suspect: *blocked exit* (exit area, 2 min), *floor hazard* (walkway, 30 s: spill,
   glass, dropped food), *left object* (a bag; also used in Phase 49). Confirm: "What is on the floor / in the exit:
   spill, broken glass, bag, furniture, nothing?".
3. **Equipment and pests**: equipment through slow rounds of the vision model per area (an overturned table or chair,
   cables across the walkway, something hanging loose) and a table box that changed shape against the empty picture.
   Pests: no rodents in COCO; try YOLOE / YOLO-World (open-vocabulary, Ultralytics) with "rat, mouse, cockroach" in food
   areas, often at night (black-and-white pictures). If it is poor when measured, leave pests out and say so.

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
