# Phase 17 plan: mobility aids and fall alerts (approved 2026-10-13)

## Context

The user skipped the Phase 15 phone test and Phase 16 for now (2026-10-13) and chose two new features for home use:
**see visible mobility aids** ("check disability") and **a fall alert**. Their choices (2026-10-13):

- Mobility aids = what a camera can actually see: **wheelchair, walking stick, crutches, walker**. Hidden
  disabilities are not guessed. **On by default**, a switch on the Settings page turns it off (it is
  health-related data: the data-protection template says so).
- Fall alerts reach the user as a **pop-up with a sound on any open VisionLabeler page** (this PC or a phone
  browser through Tailscale) plus the Alerts page with a clip. Nothing new to install, nothing leaves the network.
- Fall alerts are measured on **public fall videos**: the UR Fall Detection Dataset (University of Rzeszow;
  30 falls + 40 everyday activities such as sitting down, lying on a bed, picking something up; RGB 640×480;
  **CC BY-NC-SA 4.0**, non-commercial, used for measuring only, like MOT16). Downloaded only with the user's OK
  (size said first).

Built on what exists: the fixed vocabulary with yes/no list fields (`labeler/schema.py`, like `carrying`), the
VLM prompt (`labeler/vlm.py`), body points for every person on every frame (`TrackState.kpts`, YOLO11-pose), alert
rules with computed conditions (`labeler/alerts.py`, like `zone`), alert clips (`labeler/clips.py`), zones
(`labeler/zones.py`) and the Live page's alert pop-up (`static/index.html`).

Not a medical device: a fall out of view, or on a camera that does not see the floor (the user's desk webcam
probably does not), is not seen. README says so.

## Step 1: mobility aids label

- `schema.LIST_FIELDS["mobility_aid"] = [wheelchair, walking stick, crutches, walker]`, both modes, one yes/no per
  item (small models invent items when asked to fill a list). Prompt lines: an office chair is not a wheelchair; an
  umbrella is not a walking stick; a shopping trolley or pram is not a walker; "yes" only when the person is clearly
  using it.
- `labels.mobility_aids: true` (config default + commented line) and a Settings switch; off → not asked, not stored.
- VLM only: too rare for the fast classifiers to learn; check that the student guard and retraining ignore the new
  field. Ask box words (wheelchair, cane, walking stick, crutch, walker, rollator, zimmer frame); alert rules get it
  by themselves (field list); a chip in the live label; Review page edits it like `carrying`.
- Measure before keeping it:
  1. false alarms: the VLM on the 1,534 COCO people → how many get any aid "yes"; I check those pictures by eye;
  2. caught: COCO people found by SigLIP text search ("a person in a wheelchair", "… with a walking stick",
     "… on crutches", "… with a walker"), checked by eye → a small list with true answers (expect 10-40; said
     honestly if too few);
  3. cost: VLM seconds per person (`tools/bench_vlm.py`) and the other fields' scores unchanged
     (`tools/evaluate.py` COCO answer key; the user's 26 webcam people on the Answer key page).

**Step 1 results (2026-10-13)**: done (README "Mobility aids"). Asked yes/no per aid, the model called ski and hiking
poles a walking stick (8 of 8 "aids" in the first 200 COCO people wrong); now **one choice with look-alike answers**
(`schema.DECOYS`: poles, sports equipment, an ordinary chair, a trolley/pram, pushing or standing next to a
wheelchair), stored as "none". COCO had only 2 real aids, so 166 **Open Images** photos with wheelchairs and crutches
were downloaded (OK given; `tools/get_aid_photos.py`) and checked by eye; the prompt was tuned on the validation half
only. Final: COCO 2 of 2 caught, 0 false alarms in 1,532; Open Images test half wheelchair 211 of 218, walking stick
3 of 3, 3 false alarms in 556 (2 people kneeling next to a wheelchair user); +0.13 s per person; answer key without
age 89.5% → 90.4%. Crutches and walkers too rare to measure. Lesson: asking Ollama twice in a row about the same
picture makes the second request look ~25% faster (it reuses its work on the picture): time whole passes.

## Step 2: fall detection (`labeler/falls.py`)

- Per person, each frame, from body points: **lying** when the shoulders-to-hips line is closer to horizontal than
  vertical and the head is near hip height (box shape as the fallback when points are weak).
- **A fall** = upright → lying within about 2 s (the head drops a large part of the body height), then lying for
  `fall.after_s` (default 10 s, Settings). Lying down slowly (going to bed) is not a fall.
- Zones: an area can be marked **"lying is normal here"** (bed, sofa): no fall alerts inside (zones table, new
  column through `db.NEW_COLUMNS`; checkbox in the Live page zones tool).
- Per camera `fall_alert` (default on; off for cameras that never see the floor), Settings page.
- Live picture: the person's box turns red with "FALL?" while it lasts.

**Step 2 results (2026-10-13)**: done (README "Fall alerts"), together with the measuring of step 4 (the rules
needed the videos). URFD downloaded with OK (100 MP4, 116 MB: the color picture is the right half, 320x240; plus
the per-frame lying marks). `labeler/falls.py` (FallWatch: update per person, tick per frame), pipeline hook
(`_on_fall`, red box "FALL?"/"FALL"), zones `lying_ok` ("Lying is normal here" on the Live page), `falls:` config,
`cameras[].fall_alert`. Side camera: first half 15/15 falls, 0/20 false alarms; second half 13/15, 1/20 (the rules
were changed after seeing it: README says so); MOT16 walking 0/7; ceiling camera 0/30 (not supported). Found on the
way: the tracker renumbers a falling person (sometimes in the same frame, or reusing an old number), loses people
lying half cut off at the bottom edge, and can give one fallen person two boxes; a person walking out of the bottom
of a high street view looked like "fell out of sight" until only body points decide at the edge; kneeling under a
bed looked like lying until the box had to be wider than tall. Lesson: a real app run found 3 of these that the
measuring tool did not (two alerts for one fall, no alert in a video's second loop, the zone ignored out of sight).

## Step 3: fall alerts everywhere

- AlertEngine: computed condition `fallen = yes` (like `zone`); a built-in **"Fall"** rule made at first start
  (enabled; can be limited to a camera or turned off on the Alerts page). The alert clip is saved as now (10 s
  before), so the fall itself is in it; the recording (Phase 13) has it too.
- The alert pop-up moves from the Live page into `static/session.js`, so **every page** shows new alerts (also on a
  phone). Fall alerts play a repeating sound and stay until **OK** is clicked; other alerts as now (20 s, no sound).
  Browsers play sound only after one click on the page: the page says "click once to turn on sound".
- Viewers (e.g. family) also get fall pop-ups, picture blurred as everywhere; other alerts stay admin-only.

**Step 3 results (2026-10-13)**: done. `alerts.FALLEN` condition ("fell"), `AlertEngine.ensure_fall_rule` (run.py),
`check(..., fallen=True)` for confirmed falls (repeat allowed: `db.DROPPED_UNIQUE` drops the old
"one alert per rule and person" constraint on PostgreSQL at start, checked on a scratch database; an old SQLite file
skips the repeat with a warning), events carry `fall`; clips reach back 15 s for falls (`ClipRecorder.keep_at_least`,
`request(before_s=)`); the pop-ups moved from the Live page into `session.js` (every page; fall: red, stays until
OK, two-tone beep every 2 s, a hint while the browser blocks sound). Checked in the browser pane on the Health page
of a test copy: the pop-up with picture and hint, a click turns sound on (the hint goes; the beep itself was not
heard), OK closes, the next fall pops up again; the clip (19 s) shows standing, falling and lying. Found: the
ClipRecorder was used before it was made in Pipeline.__init__ (a test caught it); db.py had no logger.

## Step 4: measure on fall videos (`tools/fall_check.py`)

- After the user's OK: URFD camera-0 RGB of the 30 falls and 40 activities (PNG sequences → MP4 with the FFmpeg in
  `data/ffmpeg`), in `data/datasets/urfd`. Camera 1 (seen from above) of the 30 falls as a second, untuned view.
- `tools/fall_check.py`: detector + `falls.py` on each video (no database; fast): falls caught and how many seconds
  after the fall; false alarms on the 40 activities and on the MOT16 walking videos (expect 0). Thresholds set on
  camera 0, then reported unchanged on camera 1. Aim: ≥ 25 of 30 caught, ≤ 2 false alarms in 40.
- Real run on a test copy (port 8001, scratch config): a URFD fall video as the camera → pop-up with sound on the
  Live page and another page, the Alerts page with the clip, the Recordings page.

## Step 5: Settings, docs

Settings page (mobility aids on/off; fall alert per camera; seconds), README sections ("Mobility aids", "Fall
alerts", with the measured numbers and the limits), data-protection template line, HANDOFF/ROADMAP. One commit per
step, tests, full suite green.

## Critical files

`labeler/schema.py`, `labeler/vlm.py`, `labeler/agent.py`, `labeler/student.py` / `labeler/training.py` (ignore
the new field), `labeler/falls.py` (new), `labeler/pipeline.py` (posture per frame, overlay), `labeler/alerts.py`
(`fallen` condition, built-in rule), `labeler/zones.py` + `labeler/db.py` (zone flag), `labeler/static/session.js`
+ `index.html` + `alerts.html` + `settings.html`, `labeler/settings.py`, `labeler/config.py` + `config.yaml`,
`tools/fall_check.py` (new), README, `docs/data-protection.md`.

## Verification

- Tests: schema/clean for the new field (off → absent), prompt contents, agent words; `falls.py` on made-up
  keypoint sequences (sudden fall → alert after N s; slow lying down → none; lying inside a "normal" area → none;
  getting up before N s → none); the `fallen` condition and the built-in rule (once per person); viewers get fall
  alerts only; Settings validation; zone flag stored on PostgreSQL and SQLite.
- Measurements above (COCO for aids, URFD + MOT16 for falls), reported honestly, including what did not work.
- Real run on port 8001 in the browser pane; no leftover `python.exe run.py`.
