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
2. **Make room** (the choice is discussed with the user first; options): re-label less often while safety checks wait;
   skip re-labels of people already labelled well; one picture instead of three for warnings; a longer wait for warnings
   than for rounds. Keep whatever measures better, say what was tried.
3. Health page: the vision model's load (share of time busy, waits) in the "safety checks" line.

## Phase 51: Settings for the safety checks

Phases 45-49 added many options with no Settings fields (only `safety_enabled` and `safety_confirm_per_min` are there).
1. Settings page box "Safety checks": each kind on/off (falls, people down, children, crowding, fire, dangerous objects,
   floor and exits, equipment, fights, behaviour), night hours, the main times (lying still, floor 30 s / exit 2 min,
   equipment every 10 min, restricted 3 s, pacing turns), validated like the other fields; per camera "Safety" stays.
2. Tests, a browser check (phone width), README.

## Phase 52: Safety overview page

1. A page (`/safety`, admins; a link from the Alerts page and Health): per safety camera which checks run, which areas
   have which role (and what that role means for the checks), the empty picture (floor check) and when it was learned,
   the latest checks with the vision model's answers, and alerts per kind this week with the Check page's "right N of
   M" where there are checks.
2. Viewers: not shown (camera pictures and alerts are admin things), unless the user wants otherwise.
3. Browser check, README.

## Phase 53: Night-time measuring

None of the safety checks were measured at night (dark or infrared, black-and-white pictures).
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
