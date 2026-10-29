# Phases 50-55 plan: capacity, settings, an overview, night, more signs, pests (approved 2026-10-25)

## Context

The Phases 44-49 plan (safety incidents and suspicious behaviour on security cameras) is finished. On 2026-10-25 the
user chose seven ideas for what comes next: six new phases below, and Phase 26 (a real-camera week), which waits for
the user to restart their own app. Not chosen this time: the floor check's third look (a "yes" then "no" still puts the
object into the empty picture).

Way of working as before: ask before each phase and each step; discuss any architecture first; one commit per step;
full suite green; push and check GitHub; a real run on a test copy (port 8001, `tools/make_test_config.py`), never the
user's app or data; downloads only with the user's OK after checking size and licence; tune on one half, report on the
other (pitfall 21); every new per-person sign gets a once-a-minute limit (pitfall 40); each phase adds its entry to
`labeler/whatsnew.py`.

## Phase 50: Vision model capacity (first: the later phases add checks)

On busy cameras safety checks were dropped after waiting 30 s (Phase 49's real run: the question for a direct alert,
the equipment round; pitfall 37). Before adding more checks, measure and make room.

**Design agreed with the user 2026-10-25** (the plan was approved in this order the same day): (1) safety checks use
the vision model's **idle time**: the per-minute limit (`confirm_per_min`) holds only while people's labels are
waiting; when nothing waits, checks go at once (today: at most 6 a minute, one at a time, always); (2) on a busy
camera **people passing in under 2 s get colours only** (no vision-model label); people who stay are labelled as now;
not chosen: the words question last, longer waits for rounds, one picture for warnings; (3) **measure two requests at
once** (Ollama answering 2 in parallel on the GPU): changing the user's Ollama setting only with their OK and only if
it gives more answers a minute; (4) measured with a busy camera made from the videos here (a crowded clip in a loop,
the real vision model); no download. Re-labels already wait until the queue is empty (unchanged).
1. **Measure**: `tools/incident_bench.py` grows a "busy camera" run: people's labels, re-labels every 15 s, safety
   checks of each level at a set rate; report how long each kind waits and how many are dropped, per minute of video.
   **Done 2026-10-25** (`incident_bench.py --busy`: people from MOT16-09's true marks looped, checks through the real
   confirm queue at 1 critical, 6 warnings, 2 rounds and 1 words question a minute, 3 min each, the real model):
   the vision model gives **about 42 answers a minute** (1.4 s each; a person's label takes 2). Busy (87 people a
   minute): labels waited 86 s, 168 of 261 refused (the queue full), warnings 14 of 18 dropped, rounds 4 of 6, the
   critical ones on time (2.4 s). Moderate (21 a minute): the model already **busy 100%**, labels 9 s, warnings 10 of
   18 dropped (the fixed 6 a minute), rounds 3 of 6, critical 2.1 s. **Two at once: no gain** (44-45 answers a minute
   instead of 41-43; critical checks waited 4 s instead of 2.4): as measured before (README "Speed"), the GPU is the
   limit; the user's Ollama setting is left alone. So "use idle time" helps only when there is idle time, which a
   camera with 20 people a minute does not leave.
2. **Make room** (the choice is discussed with the user first; options): re-label less often while safety checks wait;
   skip re-labels of people already labelled well; one picture instead of three for warnings; a longer wait for warnings
   than for rounds. Keep whatever measures better, say what was tried.
   **Done 2026-10-25**: the confirm queue's per-minute limit holds only while people's labels wait
   (`ConfirmQueue._labels_waiting`); first labels only after `vlm.label_after_s` 2 s in view (`Pipeline._due_for_label`;
   16% of MOT16-09's people pass quicker); and (the user's choice after the numbers, with the "try one picture" they
   asked for) **one picture per person while other labels wait** (`vlm.one_when_busy`, `Pipeline.pictures_per_person`):
   two pictures of the same person differ in carried items 37%, colours 22-23%, types 10-13%, age and gender 0% (60
   MOT16 people; label accuracy itself cannot be measured here: only the best crop of a person is kept), so one
   picture only when busy. Result (README "Speed"): 21 people a minute: nothing dropped (before 15 of 27), labels 5 s;
   87 a minute: still over capacity (labels refused 92 of 261 instead of 168; checks below critical dropped).
   Real run (MOT16-02 looping): no label refused, no error; 2 equipment rounds dropped, the next one answered.
3. Health page: the vision model's load (share of time busy, waits) in the "safety checks" line.
   **Done 2026-10-25** (the user's choice: on the **"vision model (Ollama)" line**, which always shows; the safety
   line shows only with a security camera, and the user's own camera is a kiosk webcam): `VLMWorker.load()` keeps the
   last 10 minutes in memory (`LOAD_WINDOW_S`: busy share, answers a minute, people's label waits, people refused);
   `ConfirmQueue.dropped_within`; `healthcheck.load_words`. Refused or dropped: a warning, never a problem. Real run
   (MOT16-02 looping, security): "busy 93%, 25 answers a minute, people's labels waited 40 s, 9 people refused ... more
   work than it can do" (the looping clip floods it, pitfall 42). **Phase 50 is done.**

## Phase 51: Settings for the safety checks

Phases 45-49 added many options with no Settings fields (only `safety_enabled` and `safety_confirm_per_min` are there).

**Design agreed with the user 2026-10-25** (all three recommended): (1) **one on/off switch per kind** (10: falls,
people down, children, crowding, fire, dangerous objects, floor and exits, equipment, fights, behaviour; behaviour is
one switch for its 4 signs). Fire, objects, floor, equipment and behaviour have config options already, falls has its
Settings switch; people down, children, crowding and fights get new options (`safety.down_enabled`,
`children_enabled`, `crowd_enabled`, `fights_enabled`, default on). (2) **Main times only**: night hours, lying still
20 s, slumped 30 s, child alone 2 min, crowded for 30 s, floor 30 s, exit 2 min, equipment round 10 min, restricted 3 s,
pacing turns 4. Measured thresholds (fight speed, fire / gun confidence and the like) stay in config.yaml only, so the
README's measured numbers stay true. (3) **After a restart**, like every other setting (data/settings.yaml, the
Restart now button); no new architecture.
1. Settings page box "Safety checks": each kind on/off (falls, people down, children, crowding, fire, dangerous objects,
   floor and exits, equipment, fights, behaviour), night hours, the main times (lying still, floor 30 s / exit 2 min,
   equipment every 10 min, restricted 3 s, pacing turns), validated like the other fields; per camera "Safety" stays.
2. Tests, a browser check (phone width), README.
   **Steps 1-2 done 2026-10-25** in one commit: `safety.down_enabled`, `children_enabled`, `crowd_enabled`,
   `fights_enabled` (SafetyWatch `down_on` ...; people are still followed, so a fall's "after touching" works);
   19 fields in `settings.SIMPLE`, night hours checked and written as 23-05; the page's own "Safety checks" box (the
   two old safety fields moved into it; the per-minute hint now says it holds only while labels wait). Real run: saved
   floor and fights off and night 23-5 on the page (phone width, no sideways scroll; "night" refused with a message);
   after a restart the floor check was gone and the values in use. **Phase 51 is done.**

## Phase 52: Safety overview page

**Design agreed with the user 2026-10-25** (all three recommended): (1) the latest checks come from **memory, as now**
(`ConfirmQueue.recent`, the last 50 since the start, as the Health line); no new table; alerts are in the database
anyway; (2) **admins only**, links from the Alerts and Health pages; (3) the floor check's empty picture as a **small
picture with its date** ("learned 14:02, learned again 2 times"; FloorWatch gets a `learned_at`), click for full size.
Steps: 1. `/api/safety` (admins): per safety camera which checks run and why one is off (switched off on the Settings
page, its model missing), areas with roles and what each role means for the checks, the empty picture's state (+ the
picture itself), the latest checks, alerts per kind this week with the Check page's "right N of M"; tests. 2. The page
`/safety`, the links, a browser check (phone width), README.
**Step 1 done 2026-10-25**: `labeler/safety_view.py` (`overview`, `KINDS`, `role_meaning` with the running times),
added to the existing `/api/safety` as `"overview"` (the Alerts page's keys kept); `/api/safety/empty/<camera>`
(JPEG, admins); `FloorWatch.learned_at`; `ConfirmQueue.latest()`. Alerts per kind: the last 7 days. Real run
(MOT16-02, three areas drawn): all ten kinds on, each area's meaning, the empty picture learned (the walking people
left out), no errors.
**Step 2 done 2026-10-25; Phase 52 is done**: `labeler/static/safety.html` (`/safety`, admins), links from the Alerts
page and the Health page's lines "safety checks", "floor and exits", "unsafe equipment"; a red note when part of
the walkway or exit areas was never seen empty (`ready`). Browser checks at phone width (375 px, no sideways scroll)
on a test copy: MOT16-02 looping (areas, the empty picture) and bus.jpg (an answered equipment round in the table;
the note at 57% seen empty). Found: `function top()` broke the page (pitfall 43, now a test); and a round waiting
in the vision model's queue blocks the warnings (pitfall 44, not fixed: a queue change, discuss first).
1. A page (`/safety`, admins; a link from the Alerts page and Health): per safety camera which checks run, which areas
   have which role (and what that role means for the checks), the empty picture (floor check) and when it was learned,
   the latest checks with the vision model's answers, and alerts per kind this week with the Check page's "right N of
   M" where there are checks.
2. Viewers: not shown (camera pictures and alerts are admin things), unless the user wants otherwise.
3. Browser check, README.

## Fix before Phase 53: a waiting round blocks warnings (HANDOFF pitfall 44)

Found in Phase 52's real run: the confirm queue sends one non-critical check at a time; a round (the equipment look,
"not moved for 5 min") goes into the vision model's queue last, behind people's labels; on a flooded camera it waited
8 minutes and every warning behind it was dropped after 30 s (8 dropped). **Fix agreed with the user 2026-10-25 (both
recommended parts)**: (A) a round waiting in the vision model's queue no longer blocks warnings (warnings still one at
a time; rounds one at a time among themselves; critical as before); (B) a round not started within `max_wait_s` (30 s)
is taken back out of the vision model's queue and counted as dropped, so no answer comes about old pictures.
1. Measure first: `tools/incident_bench.py --busy` (MOT16-09, a crowd and a moderate share), warnings and rounds
   dropped, as in Phase 50.
   **Done 2026-10-25** (3 min each, the real model; the bench now also counts checks "still out at the end", taken
   when the time ends): **a crowd** (87 people a minute): one round answered only after **219 s** (stuck behind
   labels), rounds 4 of 6 dropped and 2 still out, **warnings 14 of 18 dropped** and 3 still out, words 2 of 3
   dropped; critical 3 of 3 on time (2.2 s); labels waited 46 s, 119 refused. **Moderate** (18 a minute, share 0.25):
   nothing dropped, rounds waited 12 s, warnings 4 s, critical 2.8 s. So the blocking shows only over capacity.
2. Build A and B (VLMWorker can take back a job that has not started; ConfirmQueue), tests.
   **Done 2026-10-25**: `VLMWorker.take_back(job)` (a job not started is skipped when reached, never answered;
   `pending` leaves it out); `ConfirmQueue._may_send` (one at a time per lane: critical waits only for critical, a
   warning for critical or warning, a round for a round; the per-minute limit below critical as before) and
   `_take_back_rounds` (a round not started within `max_wait_s` after it was sent: taken back, dropped, its on_done
   hears None; one already started is left to finish). 1038 tests.
3. Measure again the same way; a real run on a test copy (MOT16-02 looping, as when it was found); README, pitfall 44.
   **Done 2026-10-25; the fix is done.** Crowd: warnings answered 9 of 18 (before 1), dropped 6 (before 14), the
   words question 3 of 3 (before 1); rounds 0 answered, 5 dropped (2 taken back), none answered late (before one
   after 219 s); critical 2.9 s. 18 a minute: nothing dropped, as before. The warnings still dropped on a crowd are
   held by the per-minute limit while labels wait. Real run (MOT16-02 looping): 14 rounds taken back after 30 s in
   8 min (only rounds came up in that clip), no errors. README "Speed".

## Phase 53: Night-time measuring

None of the safety checks were measured at night (dark or infrared, black-and-white pictures).

**Design agreed with the user 2026-10-26** (all three recommended): (1) **two kinds of night**, each measuring video
made in both: *IR* (black and white, flat contrast, grain, slightly soft, brighter in the middle like the camera's own
IR lamp) and *dark colour* (about a quarter of the brightness, heavy grain: no IR, street lights); (2) **the plan's
four checks plus fire**: falls (UR Fall), fights (the fight clips), floor and exits (`floor_made`), behaviour
(`behaviour_made`, with "camera covered" false alarms on the everyday videos: a dark picture has little contrast), fire
and smoke (D-Fire's sample videos); (3) **real night footage looked for after the made-up run** (size and licence
shown, nothing downloaded without the user's OK). A change found needed (e.g. other thresholds at night) is discussed
with the user before it is built.
Steps: 1. `tools/make_night_videos.py` (IR and dark versions into `data/datasets/night_made`, git-ignored; previews
to check they look right). 2. Measure each check by day (the README's numbers, the same tools and halves) against IR
and dark. 3. The numbers to the user; any change discussed, built, measured again; then the search for real footage;
README with day and night numbers.
**Step 1 done 2026-10-26** (`d4d8052`, `6c9e285`): `tools/make_night_videos.py`; 506 videos in each kind (UR Fall,
the fight clips with and without fights, D-Fire's samples, floor_made, behaviour_made, the everyday videos),
`data/datasets/night_made/ir` 4.7 GB and `dark` 3.8 GB, the same frame counts; truth and areas files beside them.
H.264 on the graphics card through the app's FFmpeg, as a camera sends: OpenCV's mp4v made 0.83 GB of one
2-minute floor video with grain (about 50 GB for all), so the first run was stopped and its files deleted.
Previews in `night_made/preview` (day | ir | dark).
**Step 2 done 2026-10-26** (`tools/night_check.py`: each set by day, IR and dark with the README's day settings,
every kind reported on every set; today's code by day too, so the comparison is fair; by day the README's numbers
came back):

| set | by day | IR | dark colour | why (looked into) |
|---|---|---|---|---|
| falls (UR Fall report half) | 13 of 15 (+1 lying still) | **1** (+1) | **3** (+2) | people are found (94-98% of
frames, as by day) but while on the floor their body points are too unsure to read "lying" (by day most frames,
at night 1-10): seen down in 7 / 6 of 15 instead of 14 |
| fights (odd clips) | 37 of 75, 6 false | 27, 8 false | 29, 4 false | less sure body points |
| fire (D-Fire videos) | 39 of 50, 5 false | **3**, 1 false | 27, 3 false | IR: our detector (trained on colour photos)
raises 3 of 50 (by day 43): smoke in black and white looks like fog |
| floor / exit (report videos) | 3 of 6 / 3 of 3 | **1 / 0** | **0 / 0** | changes fainter in a flat or dark picture
(spills and exits often not raised at all), and the vision model says "no" more on gray pictures |
| behaviour (made-up) | 4 of 4 | 4 of 4 | 4 of 4, **1 false** | the dark "lights dimmed" video: "camera covered"
(a dark picture dimmed has too little contrast) |
| everyday videos (36 min) | camera "turned" 6 (true: moving cameras) | 5 | 4 | no false "covered" at night; one
fall alert in IR and in dark (day 0) |

Found while measuring: at the default 10 pictures a second (every other frame of a 15-frame video) pacing was
never found, by today's code and by Phase 49's (checked in a temporary copy of `256e9b8`); at the videos' own 15
it is 4 of 4 again (pitfall 38: measure at the camera's own rate). The runner now uses 15 for that set.
**Step 3 agreed with the user 2026-10-26** (all four recommended fixes; fights left as they are, 37 -> 27-29): each
rule changed **for day and night alike, kept only if the day numbers stay the same or better** (the user's choice;
no night switch). Each tuned on the tuning half and reported on the other (pitfall 21), measured by day, IR and
dark with `tools/night_check.py`; ask before each:
- 3a **falls**: when the body points are unsure, the box shape counts too (clearly wider than tall = lying).
  **Done 2026-10-26, nothing kept** (the user's choice): the box shape is used already; the real cause is the
  tracker (a fallen person found at 0.2-0.4 at night, a new number needs 0.5; numbers come and go with gaps).
  Tried: weak detections no number took continuing the person (IR tuning half 2 of 15 either way: undone); a
  lenient tracker for the fall check only, estimated (IR 2 -> 4 of 15, day 15 and dark 11 both ways, no false
  alarms): not built. Falls looked at again with real night video (3e). README "Safety checks at night".
- 3b **camera covered**: only when the picture becomes almost uniform compared with just before, not when a dark
  picture gets darker (lights off at night).
  **Done 2026-10-26**: covered = flat **and the edges gone** (edge match with before under `moved_corr` 0.35;
  covered 0.0, dimmed 0.87-0.92 by day, IR and dark). Measured again: covered and turned 4 of 4 each condition,
  the dark dimmed false alarm gone, everyday unchanged. One dimmed video: chosen and checked on the same videos.
- 3c **floor and exits**: a change compared with the picture's own contrast instead of a fixed amount; the vision
  model told the pictures may be from a night camera.
  **Done 2026-10-27**: `FloorWatch.threshold` = floor_diff x the empty picture's contrast / `floor_contrast_ref` 50,
  between `floor_diff_min` 10 and floor_diff 30; `floor.NIGHT_NOTE` in the questions. Tuning videos: day 6/6, IR
  4/6, dark 4/6. Report videos: day 6 of 9 (unchanged), IR 5 (was 1), dark 3 (was 0), no false alerts.
- 3d **fire in IR**: our fire model trained again with black-and-white copies of its D-Fire photos (here already,
  no download; 2-4 h on the graphics card); the old model kept as a backup, the new one used only if not worse by
  day.
  **Done 2026-10-29, not used** (the user's rule): `tools/train_fire.py bw` / `train --bw` / `test --night`;
  `fire_smoke_bw.pt` test photos mAP50 day 0.781 / IR 0.713 / dark 0.694 (old 0.767 / 0.138 / 0.444), but the 100
  videos: day 27 fires / 1 false (old 39 / 5), IR 13 / 4 (old 3 / 1), dark 23 / 3 (old 27 / 3). Worse by day: the
  old model stays; the new file kept unused. An IR-only use of it was not decided (the user stopped the question).
- 3e the search for real night footage (size and licence shown, nothing downloaded without the user's OK); README
  with day and night numbers.
1. Made-up night videos from the ones here (darker, noisier, black-and-white like infrared) for falls (UR Fall), fights
   (the fight clips), floors and behaviour; measure found / false alarms as by day, with the same tools.
2. Optionally real night footage: a public set with night or infrared video (size and licence checked, asked first).
3. Change only what measures better (e.g. thresholds by night), README with day and night numbers.

## Phase 54: More behaviour signs

The four of Phase 49's list not built yet (the design per sign discussed with the user first):
1. Came back again and again: the same person (re-linked) entering the same area 3 times in 30 minutes (a pattern:
   at night or when Away).
2. A bag left by someone who walked away: the floor check's new object (Phase 47) and the person last next to it, who
   then left the view.
3. At parked cars: a hand at a parked car's door or window for 5 s, or stopping at 2 or more parked cars in a row
   (vehicles are tracked on security cameras).
4. Following through a door: two people crossing an exit line within 2 s the same way.
Measured like Phase 49: made-up cases on test copies, false alarms on the everyday videos.

## Phase 55: Our own pest model

YOLOE found 3% of rodents (Phase 47 step 3). Like the gun model (Phase 47 step 1):
1. Training photos: Open Images "Mouse" (rodents) training photos plus photos without (computer mice, kitchens) as
   negatives (download size and licence checked, asked first); `tools/train_pests.py`.
2. Train a small YOLO11 here, measure on the 251 validation/test photos already here (`tools/pest_check.py`), close
   and far; keep it only if it measures well; then the check in food areas as designed in Phase 47 (2 of 3 looks, one
   "yes" alerts).
3. Cockroaches: no boxes in Open Images; said honestly.

## Phase 26 (waiting): a real-camera week

After the user restarts their own app (stop.bat, start.bat; the safe-update test passed on their database), a week of
"was this right?" checks on their real camera, as planned in PHASE26-29-PLAN.md. Their camera is a kiosk webcam: the
safety checks run on security cameras only, so this week measures the people labels unless a security camera is added.

## Verification

Each phase: its tests, the full suite (>= 1011), a real run on a test copy, browser checks where pages change, the
numbers in the README.
