# VisionLabeler

Local camera system that detects people, tracks them, and labels each person with a fixed set of attributes
(apparent age, gender, clothing type and color, headwear, eyewear, carried items, mobility aids). Uncertain labels go to a
human review queue; reviewed labels become training data. Everything runs on this PC: no cloud calls.

**Daily use: see [START-STOP-GUIDE.md](START-STOP-GUIDE.md).** `start.bat` after turning on the PC,
`stop.bat` before shutting it down.

## How it works

```
Camera(s): webcam / video / RTSP
  → YOLO11-pose + ByteTrack        detect and track every person (GPU, every frame, ~30 fps)
  → best-frame selection           keep the sharpest, largest, most front-facing frames per person
  → color estimator                clothing colors from torso/leg regions found by pose keypoints
  → fast classifiers ("student")   instant partial labels from the SigLIP embedding
  → age model                      apparent age from face crops (dima806/fairface_age_image_detection)
  → Qwen3-VL 4B (via Ollama)       the remaining attributes, re-checked every 15 s while in view
  → SigLIP 2 embedding             one appearance vector per person, for search and re-linking
  → reconcile                      low confidence, or two sources disagreeing → review queue
  → alert rules                    notify when a matching person appears
  → PostgreSQL + pgvector          people, labels (with source and confidence), embeddings, reviews, alerts
  → web UI                         live view, search and questions, review, reports, alerts, answer key
```

Two modes, global (`mode:`) or per camera:

| | kiosk | security |
|---|---|---|
| For | people 0.5–1.5 m away, facing the camera | people far away, often seen from above |
| Age | `age_range` in decades (0-9, 10-19, … 70+) | `age_group` (child / teen / adult / senior) |
| Face attributes | eyewear, face mask | none |
| Minimum person size | 3% of the frame | none |

## Installing on a new PC

Copy the `VisionLabeler` folder to the new PC (Windows 10/11, NVIDIA GPU with 8 GB or more) and double-click
**`install.bat`**. It asks before each big step and is safe to run again (finished steps are skipped):

1. checks Windows and the GPU driver (`nvidia-smi`);
2. installs Python 3.13, Ollama and Docker Desktop with `winget` if they are missing (after Docker Desktop is
   installed, Windows may need a restart and Docker Desktop must be opened once; then run `install.bat` again);
3. makes `.venv` with CUDA PyTorch (CUDA 13.0: NVIDIA driver 580 or newer) and `requirements.txt` (about 4 GB);
4. writes `.env` with a new random database password (an existing `.env` is never overwritten);
5. starts the database (`docker compose up`);
6. downloads the models: detector, SigLIP 2, age model (`tools\models.py download`) and the vision model
   (`ollama pull`), about 5 GB;
7. offers a login account for other devices (this PC never needs one) and the start-at-login shortcut;
8. starts the app and shows the health check.

`install.bat -CheckOnly` only reports what is installed or missing and changes nothing (exit code 1 when
something is missing). `tools\models.py check` lists the models alone.

**Without internet**: on a PC where VisionLabeler works, build an offline package (about 10 GB):

```bash
powershell -ExecutionPolicy Bypass -File tools\pack_offline.ps1 -Out D:\vl-offline
```

It holds the exact Python package versions of this PC (`pip freeze`, downloaded once more as files), the
models copied from this PC (detector, SigLIP 2, age model, the Ollama model) and the database image
(`docker save`; Docker must be running). On the new PC install Python 3.13 (64-bit), Docker Desktop and Ollama
from their setup files (not in the package), then run `install.bat -Offline D:\vl-offline`. Checked on this PC:
the models part was packed (5.2 GB in 5 s) and unpacked into empty model folders, the vision model's file
matched its checksum, and both Hugging Face models loaded with networking turned off; the whole package (8.2 GB)
was built, a new `.venv` installed from it with pip offline (`--no-index`, 5 minutes), CUDA worked in it, and all
tests passed there (2026-10-10); the database image part was saved (0.2 GB) and the tarball holds
`pgvector/pgvector:pg17`, and `install.bat -CheckOnly` with Docker running found the image, the running
database and the login account. Not tried: `docker load` on a PC that has no image yet.

## Running it

Docker Desktop must be running (it holds the database). Double-click `start.bat`: it waits for Docker and
Ollama, starts the database, then the app, then opens http://127.0.0.1:8000. From a terminal:

```bash
.venv\Scripts\python.exe -m labeler.supervisor
```

That runs the app and restarts it after a crash. `run.py` runs it once:

```bash
.venv\Scripts\python.exe run.py --source "C:\videos\lobby.mp4"
```

Options: `--source` (one camera with this source), `--mode`, `--no-vlm` (colors only), `--no-web`,
`--max-frames N`, `--port N`, `--list-cameras`, `--config other.yaml`.

### Cameras

`cameras:` in `config.yaml` lists every camera (or add them on the **Settings** page); each runs its own
detection loop, and the vision model, embeddings and database are shared. The web page shows a tab per camera.

```yaml
cameras:
  - name: cam1
    source: 0
  - name: entrance
    source: rtsp://user:password@192.168.1.50:554/stream1
    mode: security
```

Webcams are read through Media Foundation (DirectShow gave black frames on this PC). Use
`run.py --list-cameras` to find the right number. "NO PICTURE FROM CAMERA" on the video means a lens cover
or another app (Teams, Discord) holding the camera.

## Web UI

| Page | What it does |
|---|---|
| **Live** | Moving video with skeletons, person number, time in view, status and labels; full screen; **In view now** cards updated several times a second; pop-up alerts |
| **People seen** (on Live) | Everyone recorded. **Ask** box for questions or descriptions; filter by attribute; look-alikes; tick people and **Delete selected** or **Delete all…** |
| **Review queue** | Disagreements between models first, then least confident; your answer settles every source |
| **Reports** | People per hour/day, attribute breakdowns, date range and camera filters, **Download CSV** |
| **Alerts** | Rules like "red top + backpack", "anyone" or "stayed at the gate 2 min" (per camera, time of day, only when Away); each fires once per person |
| **Summary** | What happened while you were away, last night, today or any period: counts, people, one timeline |
| **Recordings** | Video of each camera while people are in view: a bar per day, play from any moment (viewers: faces blurred) |
| **Answer key** | Label recorded people yourself, blind, and see how accurate the system is on this camera |
| **Check** (admin) | "Was this right?" about vehicles, animals, alerts and scene answers on your cameras: real accuracy here |
| **Import** (admin) | Add recorded videos with their real times (see Importing recorded videos) |
| **Settings** (admin) | Cameras, how long data is kept, backups, face blurring, HTTPS, weekly retraining, the Home / Away schedule, vehicles, animals, scene questions; **Restart now** |
| **Health** | Whether the database, Ollama, cameras, disk, backups and storage are fine (see Reliability) |

### Settings page

The common settings can be changed on the **Settings** page instead of editing `config.yaml`: cameras (add,
remove, source, mode, own retention, detector size, fall alerts), how long people and the audit log are kept, face
blurring (for viewers, and in saved pictures), mobility-aid labels, fall alerts (on/off, seconds down: 3-300), vehicles
(on/off, per camera, which kinds, parked after 10-3600 s), HTTPS,
the weekly retraining, idle mode, and recording. Every value is checked before it is
saved (a webcam number 0-9, a file that exists, or a camera address with a host; detector size 640/960/1280;
0-3650 days), and every change is in the Audit log. Only the changed values are saved, in `data\settings.yaml`
(config.yaml keeps its comments and stays the list of defaults; delete the file to go back to it).

Changes take effect after a restart: **Restart now** stops the app cleanly (people in view are saved) and the
supervisor started by `start.bat` starts it again at once; this is not counted as a crash. A copy started
without `start.bat` says to restart by hand. A camera's name cannot be changed (zones, rules and recorded people
use it); a removed camera's people stay until their retention ends. Camera passwords are shown as `•••`;
leaving the mask keeps the stored password, and the Audit log never contains it. Zones are drawn on the Live
page, alert rules are on the Alerts page, and accounts are managed with `tools\users.py`.

### Today at a glance

The top of the Live page shows today so far: people today against the same weekday in the last 4 weeks by the
same time of day (only weeks with people count: a day the app was off says nothing), the busiest hour, alerts
and camera problems today (admins; click for the Alerts page), and the time since the last database backup
(admins; a warning after 2 days). Updated every minute; it matches Reports for today (checked 2026-10-10).

### Asking questions

The Ask box understands questions like "who wore a red top after 3pm?", "women with a backpack yesterday",
"a 30 year old man on the entrance camera" or plain descriptions like "man in a beige fur coat". The local
model turns the question into a search plan (time, camera, conditions, appearance), shown above the results
so misunderstandings are visible. A condition is only used if the question mentions it, and a camera only if
it is named: the model otherwise invents filters. When few people match, the closest look-alikes whose labels
differ are added and marked, because labels can be wrong. Interpreting a question takes under a second.

### Search by photo

**Search by photo** (next to Ask) takes a JPEG or PNG from this PC or a phone (up to 10 MB) and lists the recorded
people who look most like the person in it, on all cameras and imported videos, closest first. The detector
finds the people in the picture; the one in the middle is searched (else the largest; with nobody found, the
whole picture). The photo is not stored; the search is in the Audit log. Viewers may use it, like text search.

Measured on the overhead test video (people about 90 pixels tall): for 18 people, a picture cut from a frame of
each found that person first 6 times (8 among the first 3); even a person's own saved picture ranked them 11th
of 40. On far or overhead cameras this is a hint, as with look-alikes; close-up photos and kiosk cameras should
do better but were not measured. Preferring the person in the middle changed nothing on this video (chosen
because people centre what they photograph). The first search loads a detector copy (about 1 s).

### Deleting

Tick people in **People seen** (or **Select all shown**) and press **Delete selected**, or use
**Delete all…** (asks you to type DELETE ALL). Their pictures, labels, review history, embeddings and alerts
are removed. A deleted person still in view is shown as a gray "deleted" box and not recorded again.

## Checking on your own cameras

The numbers in this README come from public videos and photos. The **Check** page (admins) measures how often the
system is right **on your own cameras**: it shows one thing at a time from the last 30 days, with its picture (an
alert with its clip), and you answer with a key:

- **Y** right, **N** wrong, **U** not sure, **S** skip. After "wrong" for a vehicle or an animal, **1-9** says what it
  really was ("cat", "not an animal"), or **Enter** goes on without.
- What it asks: vehicles ("Is this a white car?"), animals ("Is this a dog?"), alerts of every kind (falls, loitering,
  scene questions, vehicle, animal and person rules: "Was this alert right?") and each scene question's latest answer.
  The kinds take turns (the one with the fewest checks first), newest first; checking again replaces the old answer.

**Results**: the page's table shows right N of M per kind and per camera ("Not sure" counts neither way), with the
common mistakes ("dog → cat: 3"); under 10 answers it says "few checks yet". The same numbers are a Health page line
("checked by you") and a Reports panel (checks made on the chosen days). Once something has at least 10 answers and
under 70% right, it gets a note: an alert rule or a scene question on the Alerts page ("often wrong: 4 of 12 right"),
a vehicle or animal kind on a camera on the Check page, with what to try (other words for a question; turning animals
or vehicles off for a camera that never sees real ones). Nothing changes by itself.

A few minutes a day for a week gives the first real numbers for your cameras. Checks are kept when the thing checked is
deleted later (they are the measurement).

## Accuracy (measured)

**Answer key**: 39 COCO people labeled by hand (`data/datasets/coco_people/answer_key.jsonl`, unsure
attributes left out), scored with `tools/evaluate.py`:

| | Qwen3-VL 4B (used) | 4B without face crop | Qwen3-VL 8B | fast classifiers |
|---|---|---|---|---|
| Mean accuracy | **86%** | 79%* | 70%* | 75% |
| Gender | 97% | 34%* | 44%* | 97% |
| Top color / type | 88% / 63% | 88% / 67% | 69% / 60% | 53% / 73% |
| Bottoms type / color | 91% / 96% | 96% / 100% | 74% / 70% | 83% / 65% |
| Headwear / eyewear / carrying | 100% / 91% / 90% | same | 92% / 87% / 87% | 92% / 87% / 77% |
| Time per person | 2.2 s | 1.8 s | 3.0 s | ~0 |

\* before the gender prompt change. Overall = right answers out of all labels ("unknown" counts as wrong).
The gender prompt change took gender from 44% to 97%. Retraining the fast classifiers on labels made with
the new prompt took them from 67% to 75% (gender 56% → 97%).

**Kiosk prompt, 2026-10-08**: two sentences added (t-shirt has no collar, shirt has one; thin or clear
glasses count as glasses). On 24 people from the user's own webcam (Answer key page, vision model only): mean
80% → 86%, top type 64% → 91%, glasses 62% → 77%, nothing worse. On the COCO key above: 88% → 85%, because
the model now answers "unknown" more often for legs (4 cases; no more wrong values) and its age (not used;
the age model gives age) changed. Longer rewordings were tested and rejected: they turned jackets into
shirts and sunglasses into glasses. The security prompt is unchanged. The fast classifiers were then
retrained on labels from this prompt: COCO key unchanged at 75%; on the user's webcam (from stored
embeddings) top type 27% → 82%, glasses 38% → 46%.

**Kiosk prompt, naming the pictures (2026-10-08)**: "Images: 1 = the whole person; 2 = a close-up of their
face" instead of "Image 2, if present, is a close-up of their face". Own webcam (24 people): 85.9% → 88.9%
(gender 69% → 81%, glasses 77% → 85%, the rest unchanged). COCO key: 85% → 86% (without age, which the app
does not take from the vision model: 88.3% → 89.5%; bottoms type 83% → 91%, color 91% → 96%, top color
84% → 88%, but top type 70% → 63%, 2 people). Small samples; kept because both keys improved overall.
**Rejected**: sending an extra close-up of the upper body for the top's type and color. Own webcam: no change
in top type or color, slower (3.1 s vs 2.1 s per person); COCO: 89.9% → 86.1% (top type +1 person, bottoms
type 91% → 78% and color 96% → 78%).

The fast classifiers learned from COCO photos and transfer poorly to the webcam (top color 5%, carrying 25%
there; their confident glasses answers right only 5 of 11). So once the vision model has answered an
attribute, its answer is used over theirs, even at lower confidence (`resolve` in `labeler/db.py`).

**Age** on 300 FairFace faces (human-labeled age groups), `tools/bench_age.py`:

| | exact age group | within one group | bias | time per face |
|---|---|---|---|---|
| **age model (used)** | **60%** | **96%** | −0.12 | 0.01 s |
| Qwen3-VL 4B | 51% | 90% | +0.01 | 0.67 s |
| Qwen3-VL 8B | 50% | 90% | +0.34 (too old) | 0.99 s |

The age model was trained on FairFace-style photos, which flatters it on that test. The Answer key page
measures everything on your own camera; 50-100 labeled people give a useful score. The score for one camera:
`/api/answer-key/score?camera=cam1` (without `?camera=`, `cameras` lists each camera's mean score).
Labeling is by keyboard (or mouse): each attribute is a line of choices with keys **1-9, a-d**; a key picks in
the outlined line and moves to the next, **n** = not sure, **↑ ↓** move, **Enter** saves and shows the next person
at once (it is fetched ahead), **s** skips; carrying takes several items, **0** = nothing. The system's labels stay
hidden: shown labels would be accepted too easily and make the score look better than it is. The page shows how
many people were labeled this visit and the typical seconds per person (before the keys: about 34 s, the middle
of 24 saves on 2026-10-08).

## Speed

Measured on this PC (RTX 4070 SUPER): a person is fully labeled about **1.8–2.1 s** after labeling starts;
detection and colors run every frame for everyone. Things that turned out to matter:

- Ollama at `127.0.0.1`, not `localhost`: on Windows `localhost` added ~2 s to every request (4.4 s → 1.8 s).
- One request at a time (`vlm.workers: 1`): with that delay gone, 2 at once is no faster.
- Several people in one request was rejected: answers mixed people up (gender matched only 54%).
- The 8B model is slower and was not more accurate.

**Idle mode** (`idle:` in config.yaml, Settings page *Power*; on by default): when nobody has been in view for
60 s (`idle.after_s`), the detector looks only 3 times a second (`idle.fps`), and the live picture updates as
often; the Live page shows "(idle)". The first clear detection of a person who would be recorded brings it back
to full speed at once, without waiting for the tracker to confirm them. The camera health check does not call an
idle camera slow. Measured 2026-10-12 on a 7.4-minute test video (an empty scene; 12 people, from a 1.5 s
crossing to standing 5 s, after 25-40 s empty gaps; idle after 10 s for the test): **12 of 12 people found** with
idle mode on and off, the same records (14); found 0.1-0.4 s later than without it (worst 1.1 s, for a slow
walker entering at the edge, against 0.6 s); **GPU power while nobody was in view 16.9 W → 11.9 W** (GPU use
45% → 14%; the GPU drew 13.8 W with the app not running, so the app's own share is nearly gone); 45% fewer
frames processed over the whole video.

## One person, one record

- **Tracker** (`labeler/trackers/people.yaml`): weak detections only keep a known person tracked; a lost
  person is remembered ~3 s.
- **Re-linking** (`tracking.relink`): someone who reappears within 5 s near the same spot (similarity
  ≥ 0.75), or within 30 s looking very similar, continues the same record. "Very similar" (appearance alone)
  is ≥ 0.90 for kiosk cameras and ≥ 0.98 for security cameras (`long_gap_min_similarity`, per mode). Tested on
  webcam footage: 2 s and 10 s gaps → 1 record (before: 2); 40 s → 2 records, as intended.

**Measured against true identities** (2026-10-10, `tools/relink_check.py`): the test video (OpenCV's vtest.avi
is the PETS 2009 S2L1 recording: people walking, seen from above, 768×576) with MOTChallenge's per-frame markings
of its 19 people; detector size 1280, 10 frames per second. A join is wrong when the returning piece is a
different true person.

| Appearance-only bar | Records (19 people) | Records mixing 2+ people | Joins | Right | Wrong |
|---|---|---|---|---|---|
| 0.90 (the old value for all cameras) | 10 | 7 | 24 | 4 | 20 |
| 0.93 | 15 | 7 | 19 | 7 | 12 |
| 0.95 | 20 | 5 | 14 | 8 | 6 |
| **0.98 (now for security)** | 33 | 2 | 1 | 1 | 0 |
| off | 34 | 2 | 0 | – | – |

Small people seen from above look alike: right joins scored 0.92-0.97, wrong ones 0.90-0.97, so no bar below
0.98 separates them, and a movement check does not help either (the wrong joins imply normal walking speeds).
The code's rule is that a duplicate record is better than two people merged into one, so security cameras now use
0.98: people are split into more records (33 for 19) but not mixed. The 2 remaining mixed records are the tracker
swapping two people who cross, not re-linking. Kiosk cameras (people close, faces visible) keep 0.90, measured
on webcam footage. Check another view with: `tools/relink_check.py <video> <gt.txt> --config <test config>`.

**Look-alikes and the timeline** use the same similarity. On this video every pair of records scored ≥ 0.80 (the
timeline's bar), but only 24 of the 561 pairs are the same person; the closest look-alike was the same person
10 times of 26 (14 of 26 among the closest 3). On far or overhead cameras treat them as hints; the timeline
already says "Same clothes can be a different person".

**A person re-identification model was measured and not adopted** (2026-10-12). OSNet-AIN x1.0 (Torchreid, MIT
code and weights, 8.9 MB, trained on research-only re-identification datasets; `labeler/reid.py`,
`tools/relink_check.py --reid`) against SigLIP 2 on the 7 MOT16 clips with true identities (street, mall,
square, night, a moving camera and a bus; walking people only), downloaded with the user's OK:

| Test | SigLIP 2 (used) | OSNet-AIN | Both, half each |
|---|---|---|---|
| Inside the pipeline: closest look-alike is the same person (74 records with a repeat) | **34** | 24 | 31 |
| MOT16's own boxes, 5 pictures early vs 5 late in each visit (200 people) | 61% | 60% | 66% |

Inside the pipeline (the pictures it really keeps) the re-identification model was worse, and together they were
no better than SigLIP alone, so nothing changed. It did better on far views (the bus clip: 18 vs 10 of 33 on
MOT16's boxes) and worse close up (9 vs 15 of 19): measure it again on footage of the user's own outdoor
cameras. On the same clips re-linking (security settings) made 11 joins, 2 right; 9 wrong joins came from the
short-gap rule (within 5 s, near the same spot, similarity 0.76-0.93), 7 of them on the moving-camera street
clip (MOT16-05). Fixed cameras gave too few joins (3) to change a setting on; check with the user's footage.

## Importing recorded videos

The **Import** page (admin) adds old footage: a phone video, a recorder's files, a whole folder. People in it are
found, labeled and stored like people seen live, with the **real time they were filmed**, so they can be
searched, asked about, reviewed and counted in Reports on their real dates. The videos stay where they are.

1. Type a video file or folder on this PC and press **Find videos**.
2. Check each file's start time. It comes from the file name (`VID_20261009_143012.mp4`, `20261009_143012.mp4`,
   recorder names like `ch01_20261009143012.avi`), else from the MP4/MOV header (creation time), else from the
   file's modified time minus its length. Every person gets this time plus their moment in the video.
3. Give the footage a camera name (not a live camera's name), a mode, how long to keep it (0 = until you delete
   it; counted from when it was filmed), the detector size and frames per second, and press **Start import**.

The import runs inside the app next to the live cameras, as fast as the GPU allows; the vision model still
labels everyone (the import waits while more than 2 people are queued, so live cameras are delayed by at most
two labeling jobs). Old footage sends no alerts and makes no alert clips. Cancel keeps the people found so far.
The page lists each import with **Delete all its people**; on the Live page the people list has a camera filter
(`/?camera=<name>`). Imports are in the Audit log.

Each imported person shows where they are in the original ("in door_cam.mp4 at 0:31"), and admins get
**▶ Original**: a short excerpt of the original file around that moment (from 2 s before, as long as they were
in view, at most 30 s), cut on first view and then kept until the person is deleted, so formats a browser cannot
play (an AVI from a recorder) still play. Only the file an import recorded for that person can be read; viewing is
in the Audit log. Measured: a 6 s excerpt of the 768×576 test video took 3.5 s to make the first time (2.4 MB) and
0.03 s after that. If the original was moved or deleted, the button says so.

Measured on a public test video (OpenCV's `vtest.avi`: 80 s, 768×576, 10 fps, people walking, seen from above):

| Detector size | Frames per second used | People found | Time (no vision model) |
|---|---|---|---|
| 640 | 10 | 3 | 15 s (5.2× real time) |
| 960 | 10 | 9 | 19 s (4.1×) |
| 1280 | 10 | 10 | 24 s (3.2×) |
| 1280 | 5 | 9 | 14 s (5.8×) |
| 1280 | 3.3 | 8 | 10 s (7.9×) |
| 1280 | 2.5 | 5 | 8 s (10.4×) |

With the vision model on (1280, 10 fps): 52 s for the 80 s video (1.5×), 10 people, 9 of them labeled by the
vision model; inside the app with the demo camera running at the same time: 1.3×, all 10 labeled, the live
camera kept labeling at 16 fps. The detector misses most people of this view at 640 (0.3 per frame against 3.3
at 1280), so the page defaults to 1280; use 640 for close-up phone videos (faster). Fewer frames per second split
tracks into more pieces (38 at 5 fps against 34 at 10, with re-linking off) and miss people, so 10 stays the
default; this video has only 10 fps, so more was not tested.

Correction (measured later against the true identities, see "One person, one record"): this video has 19
people, and 20 of the 24 joins above were wrong. Security cameras (and imports in security mode) now need
similarity 0.98 for such joins; the table above was measured with the old 0.90.

## Zones and counting lines

On the Live page (admin), **Zones** lets you draw on the camera picture:

- an **area** (click its corners): people in it are counted, with how long they stay (dwell time);
- a **counting line** (click 2 points across the way people walk): crossings are counted as *in* (the arrow's
  direction) and *out*. Click the points the other way round to flip the arrow.

Zones are drawn on the live picture, with how many people are in each area now. **Reports** shows, per zone,
how many entered and left and the average stay (areas) or how many crossed in and out (lines). On the
**Alerts** page, "entered zone" is a condition like any other, e.g. *entered zone = door* and *carrying =
backpack*; it also fires for crossing a line either way. Where a person "is" is the
middle of them on kiosk cameras (feet are usually out of view) and their feet on security cameras. A change
only counts after `zones.min_frames` (3) frames in a row, so someone standing on a border is not counted
again and again; very brief detections are never counted. Deleting a zone keeps its counts. Counts are removed
with the people they belong to (delete, retention).

**Reports by hour**: *Who came at which hour* (a table colored from light = few to dark = many: all people, by
apparent gender and age group; kiosk age ranges are grouped), *How long people stayed* (under 10 s ... 5 min or
more), and for a range of several days *A typical day* (average people per hour of the day), so the usual busy
hours stand out.

**Time in view and facing the camera**: for each person the app records how long they were actually in view
(a gap while still tracked counts at most 1 s per frame; time away between visits is not counted) and how much
of it they faced the camera: nose near the middle between the eyes and a straight enough face
(`engagement.facing_min`, 0.6). Reports shows the average time in view, the share of it spent facing the camera,
and how many people faced it at least `engagement.looked_s` (3) seconds; the CSV has both times per person.
The facing check was set from 6 people in two demo photos (it rejects a head turned about 50 degrees), so check
it on your own camera. People recorded before this version have no times.

After updating the app, press Ctrl+F5 once on each device so the browser loads the new pages; from this
version on, pages are always checked for updates.

### Loitering (stayed in an area too long)

An alert rule can ask how long someone stayed inside an **area** (not a counting line): on the Alerts page add the
condition **stayed in area (loitering)** and pick the area and the time (30 s, 1, 2, 5, 10 or 30 min, 1 h), e.g.
*stayed in gate 2 min, between 22:00 and 06:00, only when Away*. With a **vehicle** condition it is about vehicles:
*any vehicle, stayed in gate 10 min*. The camera checks everyone inside an area once a second; the rule fires once per
person (or vehicle), with a pop-up and a clip like other alerts. Time is measured from when the person's point (the
middle of the person on kiosk cameras, the feet on security cameras) was inside the area for `zones.min_frames` frames.
A person out of sight for more than `tracking.lost_after_s` (4 s) counts as having left, so their time starts again;
a parked vehicle hidden for a moment keeps its time. On the live picture, people and vehicles inside an area that has
such a rule show the time they have been in it, e.g. "gate 1:03" above a person's labels.

**Measured** (2026-10-14, `tools/loiter_check.py`: the real pipeline on the fixed-camera MOT16 street videos 02, 04 and
09, 17-35 s each, security mode, 10 frames a second, against the true per-person markings; two areas: the whole picture
and its middle third; counting people at least half visible on average, since MOT16 also marks people hidden behind
others):

| Rule | Found | False alarms | Missed because: never detected / the track broke |
|---|---|---|---|
| stayed 5 s | 51 of 93 (55%) | 6 of 75 alerts | 29 / 13 |
| stayed 10 s | 32 of 67 (48%) | 3 of 41 | 25 / 10 |
| stayed 15 s | 13 of 47 (28%) | 2 of 15 | 16 / 18 |

The alert comes about 1-2 s after the person reached the time (median; at worst 26 s, probably after the first track of that person broke). A false alarm is
an alert on a record that mostly follows someone who stayed less (causes not looked into one by one). What this says for real use: people the camera sees
clearly are found; people far away or half hidden in a crowd are not; and the longer the time, the more often a track
breaks (someone walks in front) and the time starts again, so a rule of minutes on a busy street will miss many. At a
quiet gate or door this matters much less, but it is **not measured on minutes-long stays** (no such marked footage
here). Tried and not kept: carrying the time over when a lost person is re-linked to their record (same found, 2 more
false alarms: most broken tracks are not re-linked on security cameras).

## Scene questions

A yes/no question about a camera's picture (or one area of it), which the vision model answers every few minutes: "Is the gate
open?", "Is there a car in the driveway?", "Are the lights on?". Admins write them on the Alerts page (**Scene
questions**: camera, question, every 1 minute to 1 hour, 5 minutes by default; alert when the answer is yes, or no for
a question like "Is the gate closed?"). A question is asked only when the vision model has nothing else to do (people
and vehicles first), one at a time, with the camera's latest picture without boxes or labels. Each answer is stored
with its picture (faces blurred for viewers) and shown next to the question.

An alert comes after **two matching answers in a row** (`scene.answers_in_a_row`), so one wrong answer does not alert;
with a question every 5 minutes it comes 5-10 minutes after the change. It does not come again until a different
answer has come in between. It pops up on every page with the picture, is listed on the Alerts page with a clip, and
keeps its own copy of the picture. Deleting a question deletes its answers and alerts. A database kept in an old
SQLite file cannot store these alerts (a warning is logged); PostgreSQL databases are updated at start.

**Measured** (2026-10-14, `tools/scene_check.py`, the app's own prompt and model, one picture every half second of
video):

| Question | Pictures | Right | Wrong | Wrong twice in a row |
|---|---|---|---|---|
| "Is someone lying on the floor?": after a fall (UR Fall) | 64 | 64 | 0 | 0 of 36 pairs |
| the same, people standing, sitting, bending (UR Fall) | 523 | 513 | 10 "yes" | 5 of 453 pairs |
| "Is there a car in the picture?": MOT16 04 and 13, a car clearly in view | 116 | 116 | 0 | 0 |
| the same, MOT16 02, 09, 11 (no cars) | 135 | 135 | 0 | 0 |

No "unsure" answers; 0.6 s per answer (median). The 10 wrong "yes" were all someone kneeling or on hands and knees
close to the floor. Someone lying down on purpose (on a bed or the floor; the marks do not say which) was answered
"yes" 63 of 94 times. "Twice in a row" is measured on pictures half a second apart; the app asks minutes apart, so
this overstates how often two wrong answers come together. Not measured (no such footage here): smoke or fire, an
open gate, lights, night pictures, your own cameras: try a question for a day and look at its answers on the Alerts
page before relying on it.

**One area of the picture.** A question can be about one area drawn on the Live page (a gate, a driveway) instead of
the whole picture: only that area, with a margin of a tenth of its size, goes to the vision model (the rest of the
cut-out darkened), and the stored picture is that cut-out. A gate is a small part of a wide picture, and a car parked
in the street must not answer "Is there a car in the driveway?". If the area is deleted, its questions go back to the
whole picture (the audit log says which). Choose it on the Alerts page: **Look at** (the whole picture, or one of the
camera's areas; draw areas with **Zones** on the Live page), when adding a question or later in its row.

Measured (2026-10-14, `tools/scene_area_check.py`, 784 pictures with a known answer for an area; nothing tuned on them):

| Pictures | Whole picture (before) | Area cut out (used) | Area outlined in red |
|---|---|---|---|
| COCO photos, area around a car: "Is there a car?" (150) | 149 right | 148 | 146 |
| COCO photos, area without a car, a car elsewhere (150) | 1 | 117 (10 unsure) | 124 |
| MOT16-04 street: the parked car's spot / the walkway / the bicycle stand (35 each) | 35 / 0 / 0 | 35 / 35 / 34 | 35 / 35 / 33 |
| MOT16-13: far, small cars, area 3x their size (28) | 28 | 28 | 28 |
| UR Fall, the floor: lying after a fall / not lying (37 / 277) | 37 / 271 | 37 / 269 | 37 / 269 |
| UR Fall, the wall, someone lying on the floor below it (37) | 0 | 37 | 0 |
| **All** | **66.5%** | **94.4%** | **90.2%** |

0.6 s per answer either way. The red outline was not kept: the model still answers about the whole picture when
something is outside the outline (the wall: 37 of 37 wrong). Most cut-out mistakes on COCO are road photos with a piece
of a car at the cut-out's edge or cars too small to be marked; a few are plain wrong (sky, signs).

**Only when Away, only at some hours.** Like an alert rule, a question can be limited to when the house is set to Away
(see Home / Away) and/or to some hours ("22-06" = 22:00 until 06:00): "Is the gate open?" at night only, so the family
going in and out by day gives no alerts. Outside its limits a question is not asked at all (the vision model stays free);
when its hours start or the house goes Away it is asked at once, and its answers in a row start afresh, so a gate left
open all day alerts again when the night starts. The Health page does not count the time outside the limits as late.
On the Alerts page: **Ask** (all day, or hours such as 22:00 to 06:00) and **only when Away**, when adding a question or
later in its row; the row says "not asked now: Home" or "not asked now: outside its hours".

**Settings and Health**: the Settings page has a **Scene questions** section (on/off, answers in a row before an alert,
1-5). The Health page has a "scene questions" line with the last answers, and a warning when a question has not been
answered for twice its interval (the vision model busy all the time, or the camera or the vision model down).

## What happened (Summary)

The **Summary** page (in every page's menu; admins and viewers) shows what happened in a period, from what is
already stored (nothing new is recorded for it):

- **While you were away**: from the last switch to Away until the switch back to Home (or until now, still Away);
  **Last night**: 18:00 yesterday until 08:00 today; **Today**; or **Choose…** any start and end (up to 31 days).
- Tiles: people, vehicles, animals, alerts, scene answers that changed ("Is the gate open?: no → yes"), camera problems.
- The first people with their pictures and a short description ("man, red t-shirt"), and people per camera.
- One timeline by day: time, kind, picture, text, "▶ Recording" at that moment, and the clip of an alert (admins);
  buttons show only one kind (only alerts, only people, ...).

After switching **Away → Home** with the button, a note under the menu says what happened meanwhile ("While you were
away: 3 people, 1 alert") with a link to the summary; it never opens anything by itself and ✕ closes it. The Live
page's Today tiles have a **Summary** tile too. Viewers (family) see the same, with faces blurred, without clips and
camera problems (as elsewhere). It takes about 30 ms for a busy day on this PC.

**This week** (Phase 40; the Summary page's **This week** button, `/summary?period=week`): the last 7 days (today so far and the 6 days
before), or the Monday-Sunday week of a chosen day (`&week=2026-10-12`). The same summary, plus one row per day:
people, vehicles, animals, alerts by kind (fall, help sign, no movement, unusual quiet, scene, vehicle, animal,
person), and for admins camera problems, PC problems and hours recorded; the falls, help signs and quiet alerts
listed one by one; recording storage; the week's checks ("was this right?", right / wrong per thing). Recordings are
kept only 3 days (`recording.keep_days`), so older days say "not kept" rather than 0 hours. Viewers get it without
problems, storage and checks. Measured: a week of 5,000 people in 0.13 s (SQLite, this PC).
On the page: a table "Day by day" (with a total), the falls, help signs and quiet alerts, recordings and checks,
then the people and the timeline as for other periods; **◀ / ▶** go a Monday-Sunday week back or forward,
**Last 7 days** comes back. **Print / Save as PDF** uses the browser's print: on paper only the title with the
dates, the tiles, the table, the alerts, storage and checks (no menus, no buttons, no pictures), dark on white also
in dark mode; a busy made-up week fits one A4 page. On a phone the table scrolls sideways inside its box.

## Home / Away

A switch at the top of every page says whether the house is empty: **🏠 Home** or **🔒 Away**. Alert rules can use
it: on the Alerts page add the condition **Home / Away → only when Away** (or only when Home), for people and
vehicles, e.g. *anyone, only when Away* or *any vehicle, 22:00-06:00, only when Away*. Rules without it work all the
time; the built-in **Fall** rule always works.

- **Who**: admins and viewers (family on their phones) may switch it; a tap asks "Switch to Away?" first. Every
  switch is in the Audit log ("home/away").
- **Weekly schedule** (Settings page, "Away every week"; or `arming.schedule` in config.yaml): one line per time
  the house is empty, e.g. `mon-fri 08:00-18:00`, `sat,sun 10:00-12:00`, `daily 23:00-06:00` (into the next
  morning). A button press holds until the schedule next changes: Home pressed at 07:00 is overruled at 08:00.
  A change due while the app was off is applied at the next start, logged with the time it was due.
- **Where to see it**: the button (hover: since when, by whom, the next change), a tile on the Live page, and a line
  on the Health page. A person already in view when it switches to Away is alerted at their next re-labeling
  (within about 15 s).
- Without a schedule and before the first press it is Home.

## Alert clips

Each alert gets a short video: the 5 seconds before it fired and the 5 after (`clips:` in config.yaml; a fall: the
15 s before), with the
boxes and labels as on the Live page. On the Alerts page, **Play clip** shows it (admins only; watching is in
the Audit log). Each camera keeps its last seconds in memory (about 5 MB); clips are WebM (VP8, built into
OpenCV, plays in every current browser), at most 960 pixels wide and 10 frames per second, about 1.5-3 MB each,
written in the background a few seconds after the alert. They are deleted with the person (Delete, retention)
or with their alert rule. With `storage.blur_faces: true` clips are made from the blurred picture.

## Recording

With `recording.enabled` (config.yaml, Settings page *Recording*) each live camera is recorded as H.264 video while
people are in view: from 10 seconds before the first person appears (kept in memory) to 10 seconds after the last
one leaves; someone returning within those 10 seconds continues the same recording (`recording.when: always`
records all the time). Files of at most 10 minutes in `data\recordings\<camera>\<day>\`, 1280 px wide, 10 frames
per second, with the camera name and the time written on every frame; kept 3 days and at most 100 GB in all, the
oldest deleted first every hour; recording pauses (Health page: problem) when the disk is below
`storage.min_free_gb`. FFmpeg makes the files (`tools\get_ffmpeg.py`, set up by install.bat: the LGPL build of
FFmpeg 8.1), on the graphics card (NVENC) when there is an NVIDIA one, else with Windows' encoder or OpenH264.

**Kept around alerts** (Phase 42): before a recording is deleted, the minutes around each alert in it, from 1 minute
before to 2 minutes after (`recording.alert_before_s`, `alert_after_s`; alerts close together make one part), are
copied into a part of their own (`<time>-alert.mp4`), kept **30 days** (`recording.alert_keep_days`; 0 = not
kept). A kept part is a recording like any other: the player, the blurred stream for viewers, Download, ▶ Recording
on the alert. Its frames are copied as they are, each with its time on the picture and its head boxes. When all
recordings are over `max_gb`, normal recordings are deleted before kept parts. A day's time-lapse still goes with
the day's last normal recording. Checked on a test copy with the real encoder: a 72-second recording with an alert
30 s in was kept whole (9.6 MB, 703 frames, plays: 70.3 s) and the original deleted, in 4.1 s. So a full 3-minute
part of a busy scene is about 25 MB; a quiet room at home is much less.

Where they show: the **Recordings** page draws kept parts on the day bar in their own color (legend "kept around an
alert") and marks them "kept: alert" in the list; an alert's ▶ Recording opens them; the week report's "Recorded"
column adds "+ 3 min kept" on days whose recordings are gone; the Health page's recording line counts them ("1 part
kept around alerts (10 MB, 30 days)"); the Settings page sets the days and the seconds before and after.

The time on the picture was hard to read before 2026-10-19: its dark outline drifted away from the white letters
along the line (OpenCV spaces thicker letters wider), so it looked doubled. Recordings made since then are clean.

The **Recordings** page shows each day as a bar (when it recorded, when people appeared); click it to play from
that moment. **▶ Recording** on a person (Live page, timeline) and on an alert opens the recording at that moment.
Admins play the file itself (the browser seeks), download it and delete whole recordings; viewers get a stream
made by the app with faces blurred (from the head positions saved for every frame) and never the file. Every
playback, download and delete is in the Audit log. Deleting a person does not cut them out of a recording.

**Search recordings by words**: the **Find in recordings** box at the top of the Recordings page. A description such as "woman in a green top this morning" or "white
car yesterday evening" is searched like the Ask box (people, vehicles, animals; time, camera and labels), and each
find becomes a **moment** in the recordings: from 3 seconds before it was seen to 3 seconds after, starting at the
right frame even when the camera ran unevenly. Things seen together on one camera are one moment ("2 people") while
it stays under a minute; a visit over two files plays as two parts. Finds without video (recording off, older than
`keep_days`, before recording existed) are counted ("9 found, 7 on video"). Admins and viewers may search (viewers'
pictures and video blurred as everywhere); each search is in the Audit log. Without the vision model only how people
look is searched (no times or vehicles). Each moment shows its picture, camera, time, length and what was found;
**▶ Play** opens its day and camera on the bar and plays from its start to its end (a moment over two files plays both
parts); **▶ Play all** plays the moments one after another; admins can **Download** a moment (cut without
re-encoding). Viewers get the blurred stream, which stops at the moment's end. Measured on a test copy with a MOT16
street video as the camera: an answer in
1.5-4.5 s (the language model's reading of the question takes most of it); the frame where a moment starts shows its
own start time in the clock the recorder writes (13:30:08 and 13:33:10 for two moments).

**Search recordings by photo**: **or with a photo** next to the search box (a JPEG or PNG; not stored): the person in a photo (the one in the middle, as in
Search by photo) is compared with everyone recorded, and the moments of the people at least 0.85 alike
(`photo_search.min_similarity`) are listed, the closest first, at most 20 people. Measured (2026-10-15,
`tools/photo_match_check.py`, MOT16 with true identities: a picture of each person against pictures of everyone at
other moments of the same video): the limit 0.85 was chosen on MOT16-02/04/05 (it keeps 60 of 62 people there); on
MOT16-09/10/11/13 it kept 20 of 28 and dropped 76% of the other people; the right person was the closest for 61-69%.
These are street videos with small people, where appearance alone is weak (as for Search by photo): a look-alike, not
a certain match. Real run on the test copy: a picture of one woman cut from the MOT16 shop-front video found her three
times (the video loops) and nobody else, in 0.35 s. The list shows how alike each person is ("95% alike"); Play,
Play all and Download work as for words.

**Faster playback**: under the picture **−10 s**, **+10 s** and the speed **1× 2× 4× 8×**; or the keys space
(pause), ← and → (10 s), 1, 2, 4, 8 (not while typing in the search box). The speed stays for the next file. Admins'
recordings and the time-lapse are a video, which changes speed itself. The viewers' blurred stream (and a recording
still being written) gets `speed` from the app (every 2nd, 4th or 8th frame at the same pace), and a skip or a new
speed starts it again from where it is: the page works out the place from the time it has played since its first
frame (the app paces the stream by the clock). Checked against the clock written on the picture: after 6 s at 1×
and 3 s at 4× the page's place matched it to the second (before counting from the first frame, 4× was 1 s ahead:
opening the file takes ~0.2 s). **Pause** (for the stream) keeps the last picture on screen. A search moment still stops at its end at any speed: checked on a test copy at
8×, three moments stopped 0.1-0.4 s after their end (the browser's own time report alone was 2 s late at 8×, so the
page checks every 50 ms); +10 s past a moment's end goes on to the next one.

Measured (2026-10-12): busy street footage (MOTChallenge) recorded all the time would be **9-13 GB per camera per
day** at quality 28 (`recording.quality`; 23 looks the same at normal size, 17-22 GB); OpenCV's own encoders would
have made 23-305 GB. FFmpeg uses 2-6% of one processor core; the camera loop kept its speed (18.5 frames/s without,
17.9-18.4 with, on a busy 1080p clip), because shrinking, blurring and encoding happen outside it. After a crash or
power cut the last ~3 seconds are lost and the file still plays; the next start closes it.

### Time-lapse

About a minute of video showing everything one camera recorded in a day (`labeler/timelapse.py`): **▶ Time-lapse**
on the Recordings page, next to the day, makes it (a progress line) and plays it, with the time of the moment shown
under it; **Open this moment** opens the real recording there. Today's time-lapse says when it was made and offers
**Make again** while the camera keeps recording. Admins and viewers (viewers get the blurred copy). Every n-th recorded frame is kept so the day fits 60 s at 25 frames a second, with
the camera's clock still written on every frame; a map of which recording and second each frame comes from lets a
click open the real recording. Made when asked for, kept next to the day's recordings (`timelapse.mp4`, and
`timelapse.blurred.mp4` with faces blurred for viewers: the user's choice) and made again when the day's recordings
change; deleted with the day's last recording. If a camera changed size during the day, later frames get black bars.

Measured (2026-10-15, the test copy's 12 recordings of one day, 8,178 frames, two cameras' sizes): 1,363 frames
(55 s), made in 8.3 s, 12 s with the blurred copy (+4 s: only the kept frames are blurred), 10.9 MB each; the clock on
the frames was within 1 s of the map. Reading the recordings is the work, about 1,000 recorded frames a second here:
a day with 3 hours recorded takes about 2 minutes, a full 24 hours about 15. (Joining the files with FFmpeg was 2x
faster but dropped half the frames when the size changed, so it was not used.) In the running test copy, with its
camera busy at the same time, the same day took 25 s from the button to playing.

## A video of each person

Every person seen live gets a short video of their visit: from 1 second before they appear to 1 second after
they leave, at most 20 seconds (their arrival; `person_videos:` in config.yaml, on/off on the Settings page). The
**▶ Video** button on a person (Live page people list, also in look-alikes, timeline and answers; and on the Review
page) plays it in a window above the page; a person who comes back gets a video per visit. Admins only (viewers
see none, as with alert clips); watching is in the Audit log. Deleted with the person (Delete, retention). Made
from the blurred picture with `storage.blur_faces`. Not for imported videos (their original file exists).

Size, measured on the test video as a live camera (2026-10-10): 480×360, 5 frames per second, median 1.3 MB per
person (0.9-2.3 MB); the camera kept its 10 fps. OpenCV's VP8 writer has no quality setting (it spends about
width × height ÷ 8 bytes on every frame), so the size is set by the picture size: at 640 px the median was 2.1 MB.
Animated WebP would be 4-6 times smaller but took 3-6 s of processor time per person and cannot be paused, so
it was not used. The Health page's **storage** line shows what pictures, person videos, alert clips, backups
and logs use, the average per person, people per day (last 7 days) and days of space left at that rate; a
warning below 30 days, unless retention deletes people before the disk fills.

## Camera picture check

What is wrong with a camera's picture, in plain words, with what to do (`labeler/picture_check.py`). From the frames
the camera already has, no extra model: **covered** (no detail: lens cover, another app), **dark**, **washed out** (bright), **blurry** (dirty lens
or out of focus: little fine detail for the picture's contrast, so a plain wall and a busy street compare), **light
from behind** (people much darker than the window behind them). On kiosk cameras, from the people recorded: **faces
rarely seen** (camera too high or to the side) or **too small for ages**.

Each running camera checks a frame every 30 seconds (`picture_check:` in config.yaml). A problem is reported only when
it was in at least 80% of the checks of the last 10 minutes, so a light switched off for a minute or someone close to
the lens does not count: then the **Live page** shows a banner with what to do, and the **Health page** line "camera
picture" says it with the time it started (and, for kiosk cameras, the faces numbers of the last 7 days). A covered
lens is the Health page's "blank picture" problem, as before. On the **Settings page**, **Check the picture** on a
camera's row checks it at once: the picture, what is wrong and what to do, the faces numbers and the brightness and
sharpness. Real run on a test copy with a darkened street video: the banner and the Health line came after the window
had filled (20 s in the test's settings), and Check the picture said "too dark" (brightness 12 of 255).

Measured (2026-10-15, `tools/picture_check_eval.py`): 400 COCO photos with people and 140 MOT16 street frames, each
as it is and made worse on purpose; the limits were set on the even-numbered pictures (the most caught with at most 1%
of good pictures flagged) and are reported on the odd-numbered ones only:

| Picture | COCO (200) | MOT16 (70) |
|---|---|---|
| as it is: flagged | 7 | 0 |
| slightly soft (blur 1 px) / dimmer (x 0.55): flagged | 10 / 9 | 0 / 0 |
| blurred (2.5 px at 640 wide): "blurry" | 181 | 70 |
| darkened (x 0.2): "dark" | 154 | 64 |
| washed out: "bright" | 172 | 60 |
| people darkened, background brightened: "light from behind" | 180 | 55 |
| lens cover: "covered" | 200 | 70 |

The 7 COCO photos flagged as they are were mostly right when looked at (night photos, a white sky, a person on a
bench against the sky). With a stricter 3% the dark and backlight catches were higher (99% / 90%) but 19 of 200 good
photos were flagged; the 1% limits were kept. Faces: on the MOT16 shop-front video run as a kiosk camera (102 people,
the camera at about head height) 26% of the time facing the camera and ages for 76%: no finding, as it should be; the
limits for faces (15% facing, 30% with ages, 10 people at least) are chosen, not measured (no public footage from a
camera mounted too high). Not measured: real dirty lenses, night infrared pictures, your webcam.

## Camera problems

The app watches each camera and reports **no picture** (10 s without a frame, e.g. a network camera that
went offline), a **blank picture** (10 s; lens covered or another app using the webcam) and **running slowly**
(below 5 frames per second for 30 s). The first 30 s after start are ignored while cameras warm up. A problem
shows as a pop-up and a banner above the live picture while it lasts; the Alerts page lists them with start
and end times (kept `storage.audit_retention_days`). Settings: `health:` in config.yaml.

## Where was this person? (timeline)

In the people list, **Timeline** (next to *Look-alikes*) shows where a person may have been seen, in time order:
their look-alikes from all cameras within `timeline.window_h` (24) hours, similarity at least
`timeline.min_similarity` (0.8), with how long after the previous sighting. Nobody can be in two places at
once, so a look-alike seen at the same time as an already chosen sighting (on another camera, or a second
person on the same one) is left out; the most similar ones are chosen first. It compares appearance, mostly
clothing, so it is a lead, not proof: two people in the same clothes look alike. The 0.8 threshold is not yet
measured across different cameras. Viewers may use it (it is part of searching); each use is in the Audit log.

## Night (infrared) pictures

When most of a person's frames are black and white (an infrared camera at night), top and bottom **colors are
stored as unknown** from every source: the vision model and the fast classifiers could only guess them from a
gray picture. Clothing type, carried items, headwear and the rest are labeled as usual. This is decided again
each time the person is labeled, so someone still in view when the camera switches back to color at dawn gets
real colors; guesses made at night never count. The live picture shows "LOW COLOR (IR/night)" meanwhile, and
the person is marked "low light" in the people list.

## Mobility aids

Each person also gets a **mobility aid** label: wheelchair, walking stick, crutches, walker, or none. A camera
cannot see a disability, only an aid the person is using. It is searchable ("person in a wheelchair", "old man
with a cane"), usable in alert rules (*mobility aid = wheelchair*), counted on the Reports page, and shown in the
people list ("using wheelchair"). It is health-related information: **Settings → "Label mobility aids"** turns
it off (no longer asked or stored; labels stored earlier stay until the person is deleted). On by default.

The vision model answers it as **one choice that includes look-alikes**: "ski or hiking poles", "other sports
equipment", "an ordinary chair or bench", "a trolley, pram or luggage", "pushing or standing next to a
wheelchair". A look-alike answer is stored as "none" (`schema.DECOYS`). Asked yes/no per aid instead, the model
called ski and hiking poles a walking stick: all 8 "aids" in the first 200 COCO people were wrong, and 5 still
after the prompt said poles are not walking sticks. The fast classifiers do not learn this label (too rare).

Measured 2026-10-13 (`tools/mobility_check.py`; answers checked by eye):

| | people | aids caught | false alarms |
|---|---|---|---|
| COCO people (everyday photos, almost no aids) | 1,532 | 2 of 2 (wheelchair, walking stick) | **0** |
| Open Images photos with a wheelchair or crutch: test half (held out) | 556 | wheelchair **211 of 218**, walking stick 3 of 3 | 3 |
| the same, validation half (the prompt was tuned on it) | 168 | wheelchair 75 of 82, walking stick 1, walker 1 | 0 |

- The Open Images photos (`tools/get_aid_photos.py`, 166 photos, downloaded with the user's OK; CC BY 2.0, boxes
  CC BY 4.0, measuring only) are mostly wheelchair sports, with helpers and bystanders next to the chairs. The
  "pushing or standing next to a wheelchair" answer was added after the validation half: helpers called
  "wheelchair" 3 of 7 → 0 of 7, at the cost of 3 hard crops (the chair barely visible). On the held-out half the
  3 false alarms are 2 people crouching or kneeling next to a wheelchair user (the cut-out shows both) and a
  blurry wheelchair player called "crutches". Missed: 7 wheelchair users, mostly with little of the chair in view.
- Crutches and walkers were too rare to measure (4 of Open Images' 5 "crutch" boxes were walking sticks, 1 unclear; one walker
  with a seat). COCO has almost no aids, so its row measures false alarms only.
- Cost: **+0.13 s per person** (2.12 → 2.26 s median over 80 COCO people, and 1.93 → 2.06 s in the other order).
  The 39-person COCO answer key without age: 89.5% → 90.4% (each step is one person; no loss). Age from the vision
  model went 57% → 46% there, but the app takes age from the age model.
- With mobility aids on, the fast classifiers can never cover every attribute, so the vision model always looks
  at each person (`student.skip_vlm_when_confident` was rare before: "rare with current data").
- People counted as "no aid" when no aid box touched them and the model said none; people too cut-off or
  blurry to tell were left out (69 of 793).

## Safety incidents (Phase 44: the base)

Twelve kinds of safety incident on security cameras (the user's list, 2026-10-20; plan:
[docs/PHASE44-49-PLAN.md](docs/PHASE44-49-PLAN.md)): a person needing help, a fall or collision, fire or smoke,
aggression, a dangerous object, a child in danger, a floor hazard, a blocked exit or aisle, unsafe equipment, animals or
pests, unusual crowding, suspicious behaviour (judged only by what a person does, never by how they look). The way it
works (the user's choices): **quick checks** every frame or second (body points, YOLO objects, simple rules: one phase
per group of kinds, from Phase 45) only raise a *suspect*; the **vision model confirms** it from up to 3 pictures of the
last seconds (`labeler/incidents.py`) and answers yes / no / unsure with a few words; only **yes** alerts.

- **Area roles** (Live page, Zones): walkway, exit, cooking, food preparation, dining, children's play area,
  restricted, bed or sofa (the old "lying is normal here"); changeable in the list. `zones.roles` in config.yaml limits
  which are offered. The same thing is normal in one area and an alarm in another (a knife in the cooking area).
- **Pictures from memory**: each safety camera (`cameras[].safety`, Settings column "Safety"; security cameras by
  default) keeps one picture a second for 10 s, shrunk to 960 px (~1.5 MB each).
- **One queue**: the vision model's work is one priority queue: critical checks first, then warnings, then people's
  labels, then the slow round (Phase 45: warnings behind a busy camera's labels were refused; the user moved them
  ahead). At most `safety.confirm_per_min` (6) checks a minute, one at a time; a critical check never waits for a lower
  one and is not held back by that limit; a check waiting over 30 s is dropped and counted; the same camera, kind, area
  and person is not checked again within 60 s.
- **Alerts**: one built-in rule per kind (turn off or limit by area, hours, Home / Away on the Alerts page; deleted, it
  comes back). Critical kinds (person needing help, fall, fire, aggression, dangerous object, child in danger) pop up
  red with a beep like a fall; the others as a plain pop-up. The alert keeps the newest picture (the area outlined;
  faces blurred for viewers) and a clip. Alerts page: "Show critical safety incidents"; the Check page measures each
  kind on its own ("fire or smoke alerts: 9 of 10 right"); Health line "safety checks" (checked since the start,
  yes / no / unsure, dropped, the latest answers).

Measured (`tools/incident_bench.py`, MOT16-09, qwen3-vl 4B, one request at a time): a check takes 0.6 s with 1 picture
and 1.4 s with 3; 4 pictures at 768 px are refused (4263 tokens, over Ollama's 4096; a bigger context per request
would reload the model between checks and labels), so at most 3. On a very busy camera (a new person every 4 s) people's
labels waited 3.6 s without checks, 3.9 s with 3 critical checks a minute and 7.7 s with 6 (the ceiling: checks come
only when a quick check sees something); a critical check waited ~3.7 s for the label being made. Real run on a test
copy: a stand-in suspect with 3 UR Fall pictures of a person lying was answered "yes, person lying on floor" in 1.8 s
and became a red pop-up; 3 MOT16 street pictures asked about fire were answered "no fire or smoke visible". No quick
check raises suspects yet, so no kind is measured; each later phase measures its own (found, false alarms per hour,
late).

### People: falls, lying still, slumped (Phase 45 step 1)

On safety cameras the quick checks for people (`labeler/safety.py`, one SafetyWatch per camera, every frame) raise
"person needing help" when someone is **lying still** 20 s (`safety.down_s`), **slumped** (the head below the
shoulders, not moving) 30 s (`slumped_s`), or has **not moved at all** for 5 min (`still_s`: the slow round for
distress, checked after people's labels with its own question). Never in an area with the role "bed"; once per still
period. The vision model sees the part of the picture around the person (a far person is too small in the whole
picture).

**Falls are checked by the vision model too** (the user's choice): when the fall check (above) confirms a fall on a
safety camera, the vision model is asked "Is a person lying or sitting on the floor (not on a bed, sofa or chair)?";
only a clear "no" stops the alert; no answer (no vision model, a failed check, one that waited too long, or one just
checked) alerts as before, so a fall is never lost to Ollama. On a camera without safety checks (kiosk) a fall alerts at
once as before. A fall is not checked again as "lying still". Tuning (`tools/incident_check.py`, UR Fall camera 0,
held 30 s): asked "does this person need help?", the model called one real fall "exercising" (1 of 15 lost); asked what
is visible, 15 of 15.

Report half (falls 16-30, everyday clips 21-40, 30 pictures a second, not looked at while tuning):

| | falls with an alert | false alerts (20 everyday clips) |
|---|---|---|
| Before (fall alerts at once) | 13 of 15 | 1 |
| Now (vision model + lying still / slumped) | **14 of 15** (13 falls, all confirmed; 1 more by lying still) | **3** |

The vision model stopped no real fall and no false one: the one false fall (adl-30) really had someone on the floor
("person on floor"). The 2 new false alerts are people sitting slumped on the floor (adl-33, adl-34) whose last
picture the measurement holds for 30 s; one person lying on a bed was answered "no" (a "bed" area skips it). MOT16
street videos (7, 3.5 min): no suspect. At 10 pictures a second the fall check caught only 9 of 15 there: measure
falls at the camera's own rate.

### Children (Phase 45 step 2)

"Child in danger" is raised for a child **alone** (no adult within 2.5 body heights) for 2 min (`safety.child_alone_s`;
0 turns it off: at home a child playing alone in a room is normal), a child **in a cooking, food preparation or
restricted area** for 3 s (`child_area_s`; draw the area and give it its role on the Live page), or a child
**climbing** (the feet a quarter of their height above where they were, the body not smaller: walking away from the
camera also lifts the feet in the picture but makes the body smaller). Who is a child: the age label "child" once
the labels came; before that (the user's choice: sooner, more mistakes) a small upright body, at most 65% of the height
of an upright person standing at the same depth (their feet on about the same line). A label "adult" wins.

Measured (`tools/incident_check.py`): **only false alarms**, as there is no video with children here (UR Fall and
MOT16 are adults; the 17 COCO photos with a labeled child have no kitchen). 77 videos (MOT16 7, UR Fall 70, 14 min,
with age labels): no child suspect. The size rule alone took 17 people for a child at some moment, judged by eye: 3 real
children (a toddler with a woman in MOT16-09, a child in MOT16-05) and 14 adults (half hidden in a crowd, reflections
in glass, one seen from high up) of roughly 500 people, all in the first seconds before their labels came. Whether a
real child is found in a kitchen or climbing needs footage with children (the user's own cameras).

### Crowding (Phase 45 step 3)

Give an area a people limit on the Live page (Zones: "at most N people"; each area's row shows "now N", the app's own
count, to set it by). More people than that for 30 s (`safety.crowd_s`) raises "unusual crowding"; **the vision model
decides** (the user's choice) whether it is dangerously crowded (packed, pushing, someone pressed against a barrier): a
busy but orderly area gives no alert. Also raised, for the whole picture: **running together**, at least 6 people in
view, most of them moving 1.2 body heights a second or faster the same way, for 2 s (walking is about half a body
height a second). Measured on a test copy (MOT16-04, a limit of 8): 16 people for 30 s, checked in 2.0 s, answer "no:
people walking with space between them".

**Counting** (`tools/crowd_check.py`, against the true markings of MOT16 and MOT20, a dense-crowd set downloaded with
the user's OK, non-commercial): the person detector with body points finds only 0-40% of the people in a crowd (none
in MOT20-03 at 640 px), so people in areas are counted once a second by a copy of the plain detector at 1280 px
(`safety.crowd_imgsz`; 14 ms a pass; its own copy: sharing the vehicle tracker's would reset it; the user's choice).
Median share counted: MOT16-02 94%, MOT16-04 82%, MOT20-01 88%, MOT20-02 85%, MOT20-03 58% (night, about 120 people),
MOT20-05 25% (about 180 people: too dense), MOT16-09 157% (it counts the reflections in the shop windows). Running
together: no false alarm on the 7 videos. Because the app counts fewer people than there are in dense crowds, set
the limit from the count it shows, not from a head count.

### Fire and smoke (Phase 46)

Our own detector, trained here on D-Fire (`tools/train_fire.py`: 21,527 pictures from surveillance cameras and the
web, public domain under CC0 by its license file; downloaded with the user's OK), looks at each safety camera once a
second (`safety.fire_every_s`; 9.5 ms a look). Fire or smoke on 3 of the last 5 looks (`fire_hold`) raises a critical
"fire or smoke" check: the vision model sees 3 pictures over the last 8 s (smoke spreads, flames flicker), cut around
what was found. A box mostly (80%) inside an area with the role "cooking" is normal (a stove, steam) and ignored; one
reaching past the area counts and the alert says so. Nothing again until nothing was seen for 60 s. Without the model
file (data/models/fire/fire_smoke.pt, git-ignored; `tools/models.py pack` copies it) the fire check is off and the
log says so.

The model on D-Fire's test part (4,302 pictures it never saw, single photos): mAP50 0.767, smoke 0.83 (precision
0.81, recall 0.77), fire 0.71 (0.71, 0.64). Real run on a test copy: a D-Fire test picture of a fire at night was seen
within 2 s and confirmed "yes: large fire with smoke" (10.5 s: the first request after start loads the model); a
daytime picture of the same kind of camera with no fire raised nothing in 40 s.

**On videos** (`tools/incident_check.py`, the real camera code and the vision model; D-Fire's sample surveillance
videos, downloaded with the user's OK, 0.44 GB): 50 with fire ("VP": mostly distant wildfire smoke on hillsides seen by
city cameras, 30 s each; marked as fire for the whole clip) and 50 without ("FP": fog and low clouds over hills, dust,
car headlights and city lights at night, a Christmas tree, red lights, sunset glare):

| | fire videos with an alert | how fast | no-fire videos with a false alert |
|---|---|---|---|
| Our detector + the vision model (the app) | **38 of 50** (43 raised; the vision model said no to 5) | median 3 s, worst 28 s | **5 of 50** (21 min; 15 raised, 10 stopped by the vision model) |
| The vision model alone, one picture per video ("Is there fire or smoke?", 0.6 s) | 36 of 50 | only as often as it is asked (a scene question: every 5 min) | 5 of 50 |

The false ones: a small orange spot at the end of a road taken for flames twice (FP13, FP29, the same camera), clouds
over hills taken for smoke (FP4, FP8, FP19). Missed: faint smoke at the edge of a hazy picture (VP7: "no visible
smoke"), 7 never raised by the detector. Everyday videos (MOT16 7, UR Fall 70, 10 min): nothing raised. Single photos
(COCO): 2.6% / 1.7% of 115 kitchens with an oven got a smoke / flame box and 3.4% / 1.8% of 500 other photos (a pizza
in an oven, a sandwich, a fridge, ceiling lights, a breaking wave): on a camera such a thing must be seen 3 of 5
seconds and pass the vision model, and a "cooking" area excuses stoves. Not measured yet: fires indoors (D-Fire's
videos are outdoor smoke), and the MIVIA videos (requested by the user; to add when they arrive).

### Dangerous objects (Phase 47 step 1)

On safety cameras, once a second: the plain detector's pass (the crowd count's) also looks for knives, scissors and
bats; **our own gun detector** (`tools/train_guns.py`, `data/models/guns/guns.pt`) for handguns, rifles and shotguns;
and **YOLOE prompt-free** (`data/models/yoloe/yoloe-11s-seg-pf.pt`, 27 MB from Ultralytics' release v8.4.0, AGPL-3.0,
downloaded with the user's OK; a fixed list of 4,585 things, so no text model: the one YOLOE needs for your own words,
Apple's MobileCLIP, is licensed for research only) for machetes and swords. One held at a person's wrist
(`weapon_reach`: 15% of their height) on 2 of the last 3 looks raises a critical "dangerous object" check. **Any real
gun in someone's hands alerts** (the user's choice): the vision model is only asked whether it is a real gun (not a
toy, a drawing or a screen); for knives, scissors and bats it decides whether they are held as a threat or used for
food or work. Not in an area with the role "cooking" or "food_prep". Again for the same person after 30 s without one.

Guns: YOLOE found only 10% of the true guns on Open Images' validation and test photos (439 photos, 827 guns; 36% at a
threshold of 0.05, with 4% of ordinary COCO photos then getting a gun box), so the user chose to train our own: Open
Images V6 training photos (2,236 with a handgun, rifle or shotgun and 1,000 people without; CC BY 2.0; 2.26 GB box list
read and not kept, 1.03 GB photos; downloaded with the user's OK), YOLO11s, 50 rounds in 19 min. On the 439 test photos
(never trained on): mAP50 0.64 (handgun 0.75, rifle 0.69, shotgun 0.47: confused with rifles), 76% of the guns found at
0.35, 11 ms a picture.

The whole check (`tools/weapon_check.py`, detectors + wrists + the vision model):

| | at a wrist | the vision model said yes (an alert) |
|---|---|---|
| Open Images: a person really holds a gun (217 photos) | 192 (88%) | **130** (60% of the 217) |
| COCO: everyday knives, scissors, bats (205 photos) | 89 | **0** |
| COCO: ordinary photos of people, no weapon (300) | 9 (false gun boxes) | **0** (8 no, 1 unsure) |

Many of the vision model's "no" were right: paintball guns, Lego toys, a gun on a tripod or a shelf. Questions asked
"threatening?" (before the user's choice) gave 14 alerts of 63 held guns. Everyday videos (MOT16, UR Fall): nothing
raised. Not measured: knives held as a threat (no such photos here), real camera footage.

### Floor hazards and blocked exits (Phase 47 step 2)

Draw an area on the Live page and give it the role **walkway** or **exit**. Each safety camera keeps an **empty
picture** (how it looks with nobody in it; saved as `<crop_dir>/empty/<camera>.png`, so it survives a restart) and
compares those areas with it once a second (`labeler/floor.py`, about 10 ms at 1080p):

- **Learning** (about 30 s): 15 looks 2 s apart; where people stand is left out, and each spot takes its middle value,
  so passers-by the detectors miss drop out. After that the picture follows slow light changes, but a spot that
  changed is held as it was, so a bag never fades into it. The Health page line "floor and exits" shows the progress.
- **Something new** that stays where nobody stands: **30 s on a walkway** = a "floor hazard" check, **2 minutes in an
  exit** = a "blocked exit or aisle" check. Someone standing on it does not reset the time.
- The vision model sees **the empty picture and now**, cut out closely around the change, and is asked to compare
  them. **Two "yes" about a minute apart** make the alert (the user's choice); a **"no"** (a chair moved, a shadow)
  puts the change into the empty picture so it is not asked again. After an alert the change is "reported" and not
  raised again while it stays; after 24 h it becomes part of the empty picture.
- **The lights** (most of the picture changes at once for 10 s): the empty picture is learned again, nothing is raised.
  A camera that **moves** (pans, zooms, turns between preset views) is handled the same way, again and again: this
  check is for still cameras only.
- A left bag counts as a floor hazard (walkway) or a blocked exit (exit) for now; who left it comes in Phase 49.

Measured with made-up videos (the user's choice, no download; `tools/make_floor_videos.py`): still-camera videos here
(MOT16-02, -04, -09; D-Fire's guard booth FP9 and FP10) looped to 4-5 minutes, and at 60 s a bag, a suitcase or a chair
(cut out of COCO photos with YOLOE, sized to the people there, with a soft shadow) or a spill (a dark or light blob) is
put down and stays. The MOT20 cameras and most of D-Fire's move, so they could not be used. Settings were chosen on
MOT16-02 and FP9 (the question "compare them" and the tight cut-out: 11 of 13 changes seen instead of 9), and the
numbers are from the other scenes (FP10 is the same booth as FP9 on another day). `tools/incident_check.py`, the
real camera code and the vision model:

| report half (MOT16-04, MOT16-09, FP10; 0.85 h of video) | put down | checked | **alerted** | alert after |
|---|---|---|---|---|
| bag or suitcase on a walkway | 3 | 3 | **2** | 1.5 min |
| suitcase or chair in an exit | 3 | 3 | **3** | 3 min |
| spill on a walkway (1 light, 2 dark) | 3 | 3 | **1** (the light one) | 1.5 min |
| nothing put down (3 videos, people walking) | 0 | 0 | **0** | |

On the tune half (MOT16-02, FP9), whole runs with the first question: 4 of 4 objects alerted, 0 of 2 spills, no false
alert; then each check alone with the final question and cut-out: objects 8 of 8 "yes", spills 3 of 5 (the dark one on
MOT16-02's dark stones 1 of 3), 0 of 8 spots people had walked over. On the report half the two dark spills were called
"shadow" on the first look and "no visible change" on the second; the missed bag got "yes, black bag" and then "no
objects added", so it was put into the empty picture (the agreed rule: a "no" ends it). A real run on a test copy
(MOT16-09 with a suitcase put down, a walkway drawn through the API) gave the alert 1 min 56 s after the suitcase
appeared, with the area outlined in the alert's picture; it also showed the same suitcase asked about again (spots a
passer-by had hidden began to differ later), so an alert or a "no" now covers the whole connected change, and the
report half was run again after that fix (the same numbers, one check pair per object).

Not measured: real spills (water is nearly invisible to a camera; these blobs are easier), broken glass, night or
infrared pictures, the user's own cameras.

### Unsafe equipment; pests left out (Phase 47 step 3)

**Unsafe equipment** (an overturned table or chair, a broken chair, cables across the floor, something hanging loose)
has no quick sign, so every 10 minutes each safety camera shows **every area with a role** to the vision model (the
newest picture, the area cut out; `labeler/equipment.py`, config `safety.equipment_every_s`). These checks come last
in the queue, after people's labels. A "yes" is looked at again a minute later, and the second "yes" alerts (the user's
choices). After an alert the area is "reported": the rounds go on but alert again only after a "no". The Health page
line "unsafe equipment" lists the areas.

Measured with `tools/equipment_check.py` (one picture per check, as the rounds send; the question as it was, not
tuned; the halves of the floor test):

| report half | "yes" |
|---|---|
| a chair put down on its side or upside down on a walkway (MOT16-04, -09, FP10; 9) | **6** |
| a cable drawn across a walkway (9) | **4** |
| ordinary scenes, whole and their areas (9), UR Fall offices with desks and chairs (20) | **0** |
| COCO photos of rooms (chair, couch, bed or dining table; 150) | **2** ("cables across floor") |

Tune half: chairs 5 of 6, cables 2 of 6, 3 of 150 COCO rooms. A real run on a test copy (a still picture of MOT16-04
with a chair on its side, rounds every 2 minutes): "yes, overturned chair" at the first round and again a minute later,
then the alert. Not measured: real broken or overturned furniture, cables on the user's floors, night pictures.

**Pests were left out** (the user's choice after measuring; the plan said: keep only if good). YOLOE prompt-free's
fixed word list has rat, mouse, rodent and roach, but on Open Images' 251 photos of mice and other small rodents
(validation and test, CC BY 2.0, 74 MB, downloaded with the user's OK; `tools/get_weapon_photos.py --kind pests`) it
found 3% (close-ups; at room distance 0-1%), mostly calling them "persian cat" or "raccoon", and its "mouse" is a
computer mouse: 34 of 46 COCO photos with one got a pest box. `tools/pest_check.py` and the photos are kept to try a
better model later (our own, as for guns). Open Images has no rat or cockroach class. The built-in "animals or pests"
rule stays; nothing raises it now.

## No movement

For someone living alone (an elderly parent): an alert when nobody has been seen on the camera for a long time while
someone should be in. The built-in rule **"No movement"** (made at start, like "Fall"): nobody seen for **4 hours,
between 08:00 and 22:00, only when the house is set to Home** (`labeler/quiet.py`). The time counts only while all of
that holds and the camera works: it starts again at 08:00 each day, when the house is switched to Home, and after a
camera problem (a covered lens or no picture gives its own alert instead). One alert per quiet stretch: an orange
pop-up on every page that stays until **OK** (no beep), in the Alerts list ("no movement") and on the Summary page;
when someone is seen again its message says when and after how long ("seen again at 12:30, after 4 h 30 min"). A rule
without a camera watches the **kiosk cameras** only (a gate camera sees nobody for hours); name a camera to watch it.

To change it, add your own rule on the Alerts page with **no movement (nobody seen for)** 1 to 12 hours, with the time
of day and Home / Away if wanted, and delete or turn off the built-in one (while a no-movement rule exists, the
built-in one is not made again). The Health page line "no movement" shows how long each camera has been quiet; the
Settings page turns it off (**No movement alerts**). Real run on a test copy (2026-10-15) with a video of an empty scene
for 11 minutes and then people, and a 10-minute rule: the alert came 10 minutes after the camera started ("nobody seen
on cam1 for 10 min (since 15:18)"), and when the people appeared its message became "...; seen again at 15:29, after 11
min"; nobody was found in the empty part. Not measured on real days yet (a long nap, someone reading in another room):
try it for a week and look at its alerts.

### Unusual quiet

The built-in rule **Unusual quiet** (`labeler/unusual.py`; only when Home) alerts when a kiosk camera has seen nobody
for a stretch of the day in which, in its own history, someone (nearly) always was: today's time since the last
sighting (or since the camera, Home or the day began), at least 3 whole hours (`routine.min_quiet_h`), in which
someone was seen on at least 97% of the counted days of that kind (`routine.quiet_share`: on 40 weekdays at most one
without). E.g. usually up at about 07:10 and nobody seen by 08:00: 00:00-08:00 had someone on every day, so it fires
at 08:00. Once a day per camera, with what is usual in the message ("on weekdays someone is seen between 00:00 and 08:00
on 40 of 40 days"); when someone is seen again it adds "seen at". A pop-up like No movement (stays until OK, no beep).
Nothing about time before the app started, during a camera problem or before switching to Home; nothing while the
camera is still learning. The rule can be turned off or made for one camera on the Alerts page ("unusual quiet", with
Home / Away only). The Health page line "usual day" shows each kiosk camera's usual hours today, or "still learning
(5 of 14 days with people)".

Measured with `tools/unusual_check.py` (made-up households, 8 weeks of history, then 4 weeks with the real watch
ticking every 5 minutes; every 7th test day nobody at all). The first version judged each usual hour (80%) on its own:
a household at home all day (seen in each hour on 90% of days) got an alert on 19 of 24 normal days. Comparing the
whole stretch, and at least 3 hours (chosen on seed 1: 2 hours still gave 3 of 24 there), on seeds 2-5 not used for
choosing (96 normal days each): regular 1 false alert, sleeps in 1, home all day 1, irregular 6, shift work 0; every
day with nobody up was found for the first three, 1-2 hours after the first usual hour began; the irregular and shift
households have nothing usual, so a quiet day is not found there (No movement still is).

People are deleted after 30 days (`storage.retention_days`), so the usual day keeps a small summary of each finished
day (table `routine_days`: the hours someone was seen and how many people; nothing about who), made before each
retention pass; it learns from 8 weeks. Checked on a test copy with 56 days of made-up people: at start 56 days were
summed up, then the people older than 30 days were deleted (73 of 146 left) and the Health line still said "from 56
days". The first real run (before summaries, 2026-10-18) showed the problem: with 29 days left, the weekend evenings
were 78%; with visits added to the test copy at 21:10 the alert (then per hour) fired by itself at 22:00:20 and the
pop-up showed on the Live page left open.

How the usual day is learned: the app learns when someone is usually seen on each kiosk camera
(`labeler/routine.py`): for each hour, separately for weekdays and the weekend, the share of days on which someone was
seen in that hour over the last 8 weeks (`routine.weeks`), from each person's first to last sight. An hour is "usual"
when it is at least 80% (`routine.usual_share`). Days with nobody at all on the camera are left out (the app was off,
or nobody was home: either way they say nothing about the usual day), and so is every hour the house was set to
Away. Until 14 such days (`routine.min_days`) the camera is still learning and no hour is usual. Checked on made-up
8-week histories: a regular household (up 07:10-08:40, lunch on weekdays, evenings missed 1 day in 20, the weekend
later) gave weekdays 07, 08, 12, 18-21 and Saturdays 09, 10, 18-21; an irregular one (1-4 visits at any hour) gave no
usual hour; 10 days of history gave "still learning". The user's own webcam has only a few days of people so far.
(The usual hours at 80% are shown on the Health page; the alert itself uses whole quiet stretches, above.)

**The usual day as a chart** (Phase 43; Health page, "The usual day", `/health#usual`, viewers too; the Alerts page's
"Unusual quiet" rule links to it): for each kiosk camera two small charts of the 24 hours, weekdays and weekends
(today's kind first). Each bar is the share of counted days on which someone was seen in that hour; the dashed line is
the "usual" share (80%); an hour with fewer counted days than half of that kind's days (Away on most of them) is drawn
hollow ("too few days to judge"); today's quiet stretch, in the whole hours the alert judges, is a gray band. Hover an
hour for "08:00-09:00: seen on 40 of 40 weekdays"; "Show as a table" lists every hour. While a camera is still learning
it says "still learning (5 of 14 days with people)" instead. The data: `GET /api/routine` (`Routine.chart`,
`UnusualWatch.today`: since when it is quiet, its whole hours, on how many history days someone was seen in them,
whether it is unusual, Away, and whether the alert fired today). Checked on a test copy with 56 days of made-up people
(Away 13:00-17:00 on most weekdays): at start 56 days were summed up, the chart showed weekdays 07, 08, 12, 18-21 and
14:00 hollow (5 of 5 days); today's quiet counts from the app's start (12:45), as the alert does.

## Help sign

A silent call for help: **one hand held above the head for 3 seconds** (`labeler/gestures.py`; the user's choice). From
the body points the detector already finds, no extra model: a hand is up when its wrist is above the top of the head
by 15% of the torso's length and its elbow above its shoulder (scratching the head is not), the body upright (someone
lying on the floor is the fall alert's case; not checked when the hips are out of view, as on a desk webcam), the
head found (not someone seen from behind). Held 3 s with short gaps allowed; one alert per sign; the hand down for
2 s before a new sign counts. Two hands up count too (no difference measured).

Measured (2026-10-15, `tools/gesture_check.py`):
- **False alarms: none** in 40 everyday videos (UR Fall: sitting, lying down, picking things up), 30 fall videos and 7
  MOT16 street videos. A hand counted as up in single frames now and then (4 of 2,983 frames in the falls, 12 of
  8,933 in the everyday videos), never for 3 seconds.
- **One frame, COCO photos** (judged by eye): on the first 1,500 photos, 6 of the 11 people the rule flagged in a
  random sample of 200 were wrong (lying, upside down, bent over, very dark), so the upright check was added after
  looking at them; on 1,500 other photos (not looked at before): 46 people flagged, about 38-40 of them really with a
  hand above the head (tennis serves, jumps, holding something up; wrong: a statue, hands on the head, a towel over
  the head), and in a random sample of 200 people 4 flagged, all right, 1 missed (a tennis serve).
- Not measured: a real sign held on video (no such public clip here). Real run on a test copy with a still photo of a
  tennis serve (one hand up) as the camera: the alert came 4 s after the camera started (3 s held once the person was
  tracked), once only while the hand stayed up, with a clip and the red pop-up. Try it yourself in front of the
  webcam once your app runs the new version.

The built-in alert rule **"Help sign"** (made at start, like "Fall"; turn it off on the Alerts page) gives a **red
pop-up with a beep on every page until OK**, like a fall, a clip from a few seconds before, and the Alerts list entry
"help sign". Settings page, Fall alerts section: **Help sign** on/off and **Hand up for (seconds)** (2-10; 3 by
default). Per camera in config.yaml: `help_sign: false` under a camera turns it off there.

## Fall alerts

A camera that sees a room from the side notices **a fall**: someone upright is down a moment later (within 1 s),
their head having come down at least 40% of their standing height, and they stay down for **10 seconds**
(Settings page: "Fall alerts", "Down for (seconds)", and per camera; `falls:` in config.yaml). The person's box turns red on the live picture ("FALL?" while waiting, "FALL" once
confirmed) and the app's log says `FALL [camera] person #N has been down for 10 s`. Lying down slowly (going to
bed) is not a fall, and neither is kneeling or crouching on the floor (the box must be clearly wider than tall).
Getting up within the 10 s cancels it. On the Live page, **Zones → Area → "Lying is normal here"** marks a bed or a
sofa: no fall starts there. Per camera, `fall_alert: false` turns it off (a camera that never sees the floor).
Built on the body points the detector already finds (`labeler/falls.py`): no extra model; about 0.02 ms per person
per frame (measured on 13,197 person-frames), against tens of milliseconds for the detector.

**Not a medical device.** Only falls in view are seen; a camera looking straight down from the ceiling does not
work (from above, standing and lying look alike: 0 of 30 on such a camera); the user's desk webcam probably does
not see the floor.

**The alert**: a built-in alert rule **"Fall"** (made at start when fall alerts are on; turn it off on the Alerts
page to stop fall alerts; deleted, it comes back at the next start). Every open page (this PC, another PC, a phone's
browser through Tailscale; admins and viewers, pictures blurred for viewers) shows a red pop-up **"FALL · camera ·
time"** that stays until **OK** is clicked and beeps every 2 seconds until then. Browsers play sound only after a
click on the page: until then the pop-up says "Click anywhere on this page to hear the alarm" (any click turns it on
for that page). Its clip (Alerts page, admins) starts 15 s before the alert, so it shows the fall itself, not only
the person lying (the camera keeps 15 s in memory then, about 9 MB). Other rules can ask for a fall too ("fell" as a
condition, e.g. *fell* + *age group = senior* on one camera). Unlike other rules, fall rules fire again when the same
person falls again: for that, the database's "one alert per rule and person" rule is dropped at start (PostgreSQL;
checked on a scratch database 2026-10-13), and the app keeps it for all other rules itself. Nothing is sent outside
this PC: when no page is open, nobody is told (the user chose no phone messages).

Measured 2026-10-13 on the **UR Fall Detection** videos (University of Rzeszow; 30 falls, 40 everyday activities;
CC BY-NC-SA 4.0, downloaded with the user's OK, 116 MB in `data\datasets\urfd`; `tools/fall_check.py`). The
videos' color picture is only 320×240; each clip's last picture is held 15 s, as if the person stays down:

| | falls caught | false alarms |
|---|---|---|
| Side camera, first half (falls 1-15, activities 1-20) | 15 of 15 | 0 of 20 |
| Side camera, second half (falls 16-30, activities 21-40) | **13 of 15** | **1 of 20** |
| 7 MOT16 street and mall videos (people walking) | - | 0 of 7 |
| Ceiling camera (falls 1-30) | 0 of 30 | - |

- A fall is confirmed **0.8 s after it began + the 10 s wait** (median).
- **Honest note**: the rules were first set on the first half, but the first look at the second half showed
  failures absent from the first half (people crouching at the camera's edge, the tracker's number changes), and
  the rules were changed after it. So the second half is no longer an untouched test; the walking videos and the
  ceiling camera were never tuned on. Rules found this way: the tracker gives a falling person a new number (the
  history moves along), loses people lying half cut off at the bottom edge ("fell out of sight" counts unless they
  were still upright), and can see one fallen person as two boxes (one alert).
- Missed: two people who ended up curled up or facing the camera (their box is not wider than tall). False alarm:
  someone kneeling right at the bottom edge until the clip ends.
- Settings near the chosen ones give similar results (head drop 30-50%, collapsed below 40-60% of standing height).

## Vehicles

Security cameras also see **cars, motorbikes, bicycles, buses and lorries** (a second detector, YOLO11s, 19 MB,
downloaded with the user's OK, AGPL-3.0 like the pose model; `labeler/vehicles.py`; `vehicles:` in config.yaml; per
camera `vehicles: true/false`, default on for security cameras and off for kiosk ones). Each vehicle's visit is one
record (table `vehicles`, picture `data\crops\vehicle_<id>.jpg`; viewers get a copy with the faces of people in
that frame blurred), drawn in purple on the live picture. A vehicle that
stands still for 60 s is **parked**: it stays one record however long it stands there, also when people walking past
hide it for a while, and is marked "left" when it drives off. Vehicles are kept as long as the camera's people
(retention), a moving vehicle starts a recording and wakes idle mode, a parked one does neither, and ignore zones
apply to them too. No number plates are read (the user's choice).

Measured 2026-10-13 (`tools/vehicle_check.py`):
- **COCO val2017** (5,000 photos, one look each, medium and large vehicles; detector size 640, confidence 0.35):
  cars 79% found, buses 85%, motorbikes 69%, lorries 65%, bicycles 59%; most mix-ups are lorry ↔ car (COCO's
  "truck" includes pickups, so a pickup is shown as "lorry/pickup"). Size 960 added little (cars 81%) and costs more.
- **Cost**: the vehicle detector doubles the detector time on every frame (9.7 → 19.9 ms on a 1080p frame); on
  every 2nd frame (the default) 14.8 ms on average. Looking at every frame found no more vehicles on MOT16.
- **MOT16-04** (a fixed camera over a street; 1 car, 1 motorbike and 9 bicycles marked, most standing): the
  standing car and motorbikes became parked and stayed one record over 2.5 minutes of the clip looping. Found:
  the tracker loses standing bicycles behind passing people for several seconds at a time, and each return was a new
  record that never became parked (19 records in 3 loops); now a vehicle that had stood still 5 s is remembered for
  2 minutes and continues its record when found again in the same place (11 records, the rack bicycles parked), and
  stillness is judged from the median position of the last second. 6 of the 11 marked vehicles were found at all
  (small bicycles far away, half hidden in a rack). A lit shop sign was taken for a car each loop (draw an ignore
  zone there; the vision model's check in the next step should catch most of these).
- **MOT16-13** (a street filmed from a moving bus): 17 of 22 cars found, 3 split into 2-3 records, 5 records not
  matching a marked vehicle. A fixed gate camera has no camera motion; there is no fixed street footage with marked
  vehicles here, so the user's own gate camera is the real test.

**Description**: once per vehicle (after 5 s in view, when it parks, or when it leaves, with its best picture) the
vision model answers whether a vehicle is in the picture, its type (car, SUV, van, pickup, lorry, bus, motorbike,
bicycle), color (the people's color list plus silver) and make and model, "unknown" when not clear. It waits behind
people in the same queue (vehicle jobs have negative numbers). The color estimator gives a first color at once (none
from black-and-white night pictures). A vehicle the model says is not in the picture is hidden from every list (kept
for checking). `vehicles.describe: false` turns the vision model off for vehicles.

Measured 2026-10-13 on **Stanford Cars** (196 US-market models from 2012 and earlier; one 513 MB file downloaded with
the user's OK, research use; `tools/vehicle_check.py makes`), 300 photos not used for tuning: **make right 88%** of
all (answered 98%, of those 90% right; many "wrong" ones are twins with another badge, e.g. GMC Savana / Chevrolet
Express), **model right 57%** (answered 90%, of those 63%: a guess), **type matches the body type 93%** (pickups
called lorry most often), 0.84 s per vehicle. Malaysian makes (Perodua, Proton) are not in that set: not measured.
Found while tuning on another 300 photos: with "not a vehicle" as one of the types, 8% of clear car photos (SUVs,
minivans, pickups) were answered so, 18% after adding "SUV" and a stricter sentence; a separate yes/no question first
brought it to 0, and a sentence telling the model what an SUV is made it call 72 ordinary cars SUVs (removed). On 60
COCO pictures of people, 57 were answered "no vehicle" (the other 3 may show one behind them); the lit shop sign on
MOT16-04 was still called a motorbike (an ignore zone there is the fix); its white car was named Nissan Skyline (it
looks like a Toyota Crown).

**Seeing and finding them**: the Live page lists **Vehicles seen** under the people (picture, color and type, make and
model marked "(guess)", time, passed / in view / parked / parked then left, ▶ Recording), filtered by kind, color and
day ("cars" also lists SUVs, vans and pickups; "gray" and "silver" find each other); admins delete selected ones;
viewers see the list (`/api/vehicles`). The **Ask box** answers questions about vehicles ("red car yesterday
evening", "motorbikes after 10 pm", "any Perodua today"): words for kinds, colors, makes (common ones and any the
vision model has named) pick the vehicles, the model's reading of the question the time and camera; the answer
appears in the Vehicles list. "Bike" finds motorbikes and bicycles (in Malaysia it often means a motorbike).
**Reports** show vehicles per hour (or day) and per kind; **counting lines and areas** count vehicles too, apart from
people (a vehicle's place is the bottom middle of its box, where the wheels touch the ground), in the zones table.
Checked on a test copy with MOT16-04: the list with pictures, "white car" in the Ask box found the parked white car,
Reports showed 8 vehicles by kind; no vehicle crossed the two lines drawn there (the clip's moving bicycles stay at
its edges), so line counting is checked by the tests only.

**Alerts**: an alert rule with the condition **"vehicle"** (any, car, SUV, van, pickup, lorry, bus, motorbike,
bicycle) is about vehicles; it can also ask for a **vehicle color**, an **entered zone** and a **time of day** (e.g.
*motorbike, entered zone gate, 22:00 to 06:00*). "Time of day" also works in rules about people. A vehicle rule
fires once per vehicle, checked when the vehicle is first counted (after its first picture), when the vision model
has described it and when it enters a zone; rules about people never fire for vehicles and the other way round; a
record the vision model called "not a vehicle" never alerts. The pop-up (every page), the Alerts list, the clip and
▶ Recording work as for people. For this, `alert_events` got a `vehicle_id`, and its `track_id` may now be empty:
PostgreSQL databases are changed at start (checked on a scratch database 2026-10-13; an old SQLite file skips vehicle
alerts with a warning). Checked on a test copy with a "Bicycle" rule: pop-ups with the picture on the Health page,
clips and the list on the Alerts page. Found: an alert fired before the vehicle's first picture was saved (now after
it), and a bicycle seen only half hidden got no picture at all (now the best unclear one is kept until a clear one
comes).

**Settings and Health**: the Settings page has a **Vehicles** section (on/off, which kinds, parked after) and a
"Vehicles" column per camera (default: ticked for security cameras; only a choice that differs from that is saved,
so a camera changed to security mode gets vehicles by itself). The Health page has a "vehicles" line: which cameras
look for them and how many came today, or a warning when `yolo11s.pt` is missing: the app then does not download it
at start (an offline PC would fail) but turns vehicles off for those cameras; `install.bat`, the offline package and
`python tools/models.py check/download/pack/unpack` include it.

## Animals

Dogs, cats and birds (`animals.kinds`; horses, sheep and cows can be added; monkeys and monitor lizards are not
something the detector knows) are found by the vehicle detector in the same pass, so they cost nothing extra. On
security cameras by default (`cameras[].animals`). Each visit is one record with a picture (people's faces blurred in
the viewers' copy) and a rough color; an animal lying still for a minute is **resting** and stays one record. They are
drawn in orange on the live picture and listed under **Animals seen** on the Live page (kind, color, day; admins can
delete). The Ask box understands dog, puppy, cat, kitten, bird, crow, pigeon, pet, animal ("black cat yesterday");
Reports shows animals per hour and per kind. Alert rules: **animal** = any / dog / cat / bird, with an area, the time
of day, Home / Away and "stayed in area", e.g. *dog, 22:00-06:00, only when Away*. Animals do not wake idle mode or
start recording (birds would keep the camera busy all day). Deleting an animal deletes its alerts and clips.

**Measured** (2026-10-14, `tools/animal_check.py`):

| COCO val2017 (5,000 photos) | Found, medium and large | Found, small (far) | Wrong boxes |
|---|---|---|---|
| Dogs | 156 of 207 (75%) | 5 of 11 | 17 |
| Cats | 166 of 200 (83%) | 0 of 2 | 17 |
| Birds | 131 of 193 (68%) | 33 of 234 (14%) | 22 |

That is at confidence 0.5, which the tracker needs to start following something. At 0.4 it would find 1-5% more but
with about 25-100% more wrong boxes (birds 22 → 43), so 0.5 stays. A photo is one chance; on a camera an animal gets many
frames, but must be found in most of them for a few seconds. Far-away birds are mostly missed.

False alarms on videos without animals: MOT16 street videos (7, 3.6 minutes, busy streets) **0**; UR Fall indoor
videos (100, 8.3 minutes, color half) **3**: each a person crawling on all fours towards the camera, taken for a dog.
Tried and not kept: leaving out an "animal" box that is the same box as a person the person detector found. It lost real
animals (boxes overlapping 50%: dogs 156 → 147, cats 166 → 159, birds 131 → 123; the person detector sometimes "sees a
person" on a close-up animal), and when nearly identical (70%: dogs 150, cats 160, birds 123) it removed only 1 of the 3
false dogs.

**Settings and Health**: the Settings page has an **Animals** section (on/off, which kinds) and an "Animals" column per
camera (default: ticked for security cameras; only a choice that differs is saved). The Health page has an "animals"
line: which cameras look for them and how many came today, or a warning when the detector file `yolo11s.pt` (shared
with vehicles) is missing. Not measured: the user's own camera, night (infrared) pictures, and animals common in Malaysia
(stray cats and dogs should be like COCO's; monitor lizards and monkeys are not found at all).

## Detection: how far can a camera see?

Each frame is shrunk to `detector.imgsz` (640) pixels on its long side before people are looked for. What
decides whether someone is found is how tall they are **in that shrunk picture** (COCO val2017, 10,777 people,
confidence high enough to start a new person, measured 2026-10-08):

| Person's height after shrinking | under 48 px | 48-64 | 64-96 | 96-128 | 128-192 | 192+ |
|---|---|---|---|---|---|---|
| Found | ~0-3% | 18% | 46% | 73% | 82% | 93% |

So plan for people at least about **100-130 pixels tall** in the picture the detector sees. For a Full-HD
(1920x1080) camera at 640, that is 300-400 pixels in the camera's own picture: about a third of its height. For a
wide camera with people further away, give that camera `imgsz: 1280` (per camera in config.yaml): the same
person is then twice as tall for the detector, from real camera pixels. Don't raise it for small pictures (a
640x480 webcam): enlarging adds no detail, and on COCO 1280 found more medium-sized but fewer large people
(48.2% vs 48.6% overall). Rejected after measuring: scanning 2x2 overlapping tiles: +5 points of people found
on COCO, but 13 times as many false alarms (people cut in half at tile edges) and 4.6 times slower.

## How many cameras?

Measured on this PC (RTX 4070 SUPER), the full app with N cameras, each showing a busy scene (the 4-person bus
photo at a camera's 15 frames per second), 2026-10-08:

| Cameras | Frames per second, each | Total | GPU busy | Labeling queue |
|---|---|---|---|---|
| 1 | 15.4 (full speed) | 15 | 43% | 0 |
| 2 | 14.2 | 28 | 76% | 0 |
| 4 | 9.6 | 38 | 65% | 2 |
| 6 | 6.6 | 40 | 75% | 10 |
| 8 | 5.5 | 44 | 79% | 19 |

So: **2 cameras at full speed, 4 at about 10 frames per second each** (enough for tracking people). From 6 on,
each camera slows to 5-7 fps (near the "running slowly" warning at 5) and people wait longer for their labels.
Quieter scenes need less. The GPU was never fully busy, so the limit is probably the Python side (all cameras
share one process for tracking, drawing and encoding pictures); batching cameras or one process per camera
would be the next step if more cameras are needed. GPU memory: about 7-8 GB in use, mostly the vision model.

## Things that look like people

Pictures of people (a monitor, reflection, poster, TV) are detected as people. In kiosk mode new people under
3% of the frame are ignored (`detector.min_box_area`); `ignore_zones` skips fixed areas. Ignored detections
are drawn as gray "ignored" boxes.

## Reliability

- **Crash recovery**: the supervisor restarts the app after a crash; a watchdog exits if a camera loop dies;
  after 5 crashes in 10 minutes it stops and shows the error.
- **Log files**: everything the window shows is also in `data\logs\visionlabeler.log` (with the date), and the
  supervisor's restarts in `data\logs\supervisor.log`, so a crash at night can be read after the window was
  closed. A file is renamed `.1`, `.2`, … at 10 MB and 5 old files are kept (`logs:` in config.yaml), so logs
  never use more than about 60 MB.
- **Health check**: the **Health** page (also "health OK" / "2 problems" in the Live page's status line) and
  `GET /api/health` check the database, Ollama (reachable, model pulled, and used by the app: it connects only
  at start), each camera (running, pictures arriving, current camera problems), free disk space, the newest
  database backup (a warning after 7 days), the last weekly retraining, and errors in the log in the last 24 h.
  `/api/health` answers **200** when nothing is broken and **503** when something is, with the reasons, so a
  monitoring tool (e.g. Uptime Kuma) can alert on the code. Viewers may use it; a tool without an account uses
  `?token=` with `web.health_token` (`VL_HEALTH_TOKEN` in `.env`) and gets only the problem list. Camera
  addresses are never shown (they can hold a password), and passwords in addresses are masked on every page.
- **Single copy**: a second copy refuses to start ("already using port 8000").
- **Clean stop**: Stop / `stop.bat` / Ctrl+C save the people in view first; an open Live page no longer
  blocks shutdown.
- **Ollama errors**: each request is retried once.
- **Security updates**: `.venv\Scripts\python.exe -m pip_audit` lists installed packages with known security
  problems. Checked 2026-10-15: **none**, after moving PyTorch 2.11 (CUDA 12.8) to **2.14.1 (CUDA 13.0)** and
  setuptools 81 to 84 (PyTorch 2.11 held it below 82). PyTorch's CUDA build is not checked by pip-audit; 2.14.1 has
  no advisory by version. The CUDA 13.0 build needs NVIDIA driver 580 or newer (the installer checks it). Measured
  before and after on the same inputs: the same people boxes (451 of 451 on 300 COCO photos, body points within
  0.2 px), the same appearance vectors (cosine 1.00000) and age answers (300 of 300), the same fall results
  (`tools/fall_check.py`, run fresh) and the same counting on MOT16-09 (`tools/relink_check.py`); speed the same or
  slightly faster (detector 11.9 -> 11.0 ms a frame, age model 9.4 -> 6.7 ms a face).
  GitHub's Dependabot (`.github/dependabot.yml`) opens a pull request when a package in requirements.txt gets a
  security fix; turn on "Dependabot alerts" and "security updates" once in the repository's Settings.
- **HTTPS certificate**: the Health page shows its end date (825 days after it was made) and warns 30 days before.
- **Backups**: the app backs up the database every night while it runs (`backup.hour`, 2:00; a PC that
  was off catches up later that day), and `stop.bat` makes one when it stops the app; at most one a day,
  the newest 7 kept in `data\backups` (`tools/backup_db.py`). `backup.copy_to` (Settings page: *Second copy
  in folder*) also copies each one to another disk or a USB drive. Once a week the newest backup is restored
  into a separate test database and its people counted, so a broken backup is noticed; the Health page shows
  the newest backup's age (a warning after 2 days), the second copy and the last restore check. By hand:
  `tools\backup_db.py --verify-only`. Measured here: backup and restore check of the 27-person database
  took 2 s (0.2 MB). Before 2026-10-10 only `stop.bat` made backups, so a PC that was never
  stopped had none.
- **Restore drill** (`tools\restore_drill.py`): the whole way back, as after a broken disk: a new, empty
  database container (port 5433, same image, nothing downloaded), the newest backup restored into it, a test copy
  of the app started on it (port 8002) with a copy of the pictures, every page and a few calls on restored people
  checked, then everything removed. The real database is not touched. Measured 2026-10-12: 27 of 27 people back,
  all pages, pictures, look-alike search, reports and the health check working; **22 s** in all (new database
  2 s, restore 1 s, app start 15 s). **The backups hold the database only**: with `--no-pictures` (as after a
  lost disk) people come back with labels, search and reports, but without pictures or person videos.
- **Long-run check** (`tools\soak.py`): samples a running copy every 10 minutes (the app's private memory,
  threads, handles and own GPU memory; Ollama; GPU power; FPS; errors in the log) and charts the trend after the
  first hour. 24 hours on the webcam (2026-10-11; the lens cover was closed, so no people were labeled): the app
  never stopped answering, no errors, threads and handles flat, private memory 4.93 → 4.99 GB (+60 MB in 23 h,
  levelling off). Its working set (RSS) fell from 2.1 GB to 0.13 GB: Windows trims a quiet program's memory,
  nothing was freed, so private memory is the number to watch. FPS fell from ~40 to ~18 for 11 hours while the PC
  was used for other things (GPU memory of all programs up to 6.5 GB) and came back by itself without a restart.
- **Disk space**: warning at startup and in the page header below `storage.min_free_gb` (5 GB).
- **Auto-start**: `install-autostart.bat` / `remove-autostart.bat`.

## Access from other devices

To allow other devices, create login accounts and set `web.host: 0.0.0.0`. This PC never needs to log in (it is
always admin); other devices see a login page. While there is no account, the app stays on this PC only and
says so in its window. A request passed on by a forwarding tool on this PC (reverse proxy, Cloudflare Tunnel,
Tailscale Serve/Funnel) looks local, so the app recognizes their headers and asks it to log in like any other
device.

```bash
.venv\Scripts\python.exe tools\users.py add ann --role admin
```

| Role | May |
|---|---|
| `admin` | everything: review, reports and CSV export, alert rules, answer key, delete, stop |
| `viewer` | Live page, search and questions, alert pop-ups; nothing else |

Other commands: `users.py list`, `password <name>`, `role <name> admin|viewer`, `remove <name>`. Passwords
need 10+ characters and are stored only as salted scrypt hashes. A login lasts `web.session_hours` (12);
changing a password or role logs that user out everywhere. Five wrong passwords from one device block it
for 5 minutes. A shared password from older setups (`VL_WEB_PASSWORD` in `.env`) becomes the first admin
account (user `admin`) at startup.

**HTTPS** (encrypted, recommended whenever other devices use the app): set `web.https: true`. Other devices then
open `https://<this PC's name or address>:8443`; plain HTTP stays on this PC only (`http://127.0.0.1:8000`,
used by start.bat and stop.bat), so nothing changes here. The app makes a self-signed certificate the first
time (`data/tls/`, valid 825 days, for this PC's name and addresses). Browsers warn about it once per device:
check that the fingerprint they show matches the "HTTPS certificate fingerprint" line in the app's window,
then accept. You can put a certificate of your own in `web.cert_file` / `web.key_file` instead. Not meant to be
exposed to the internet directly.

**From another site** (another house or office): use a private network app instead of opening a port on the
router. The step-by-step guide is [docs/SECOND-SITE.md](docs/SECOND-SITE.md); the Health page checks each
step (this PC's Tailscale address and how many other devices are in the network, the firewall rule for the
HTTPS port, the certificate fingerprint, the last login over Tailscale), only reading, every 5 minutes. With [Tailscale](https://tailscale.com) (free for personal use) on this PC and the other PC, signed in
to the same account (or the other person invited to your tailnet):

1. Give the other person an account: `tools\users.py add <name> --role viewer`.
2. Restart VisionLabeler. When Windows asks whether Python may use the network, allow **Private networks** only.
3. Let Tailscale devices reach the HTTPS port. Tailscale's network counts as "public" in Windows, so in an
   **administrator** PowerShell run once (only Tailscale addresses, only this port):
   `New-NetFirewallRule -DisplayName "VisionLabeler HTTPS (Tailscale)" -Direction Inbound -Protocol TCP -LocalPort 8443 -RemoteAddress 100.64.0.0/10 -Action Allow -Profile Any`
4. On the other PC open `https://<this PC's Tailscale name or 100.x address>:8443`, check the certificate
   fingerprint, and log in.

Don't use Tailscale Serve/Funnel, Cloudflare Tunnel or port forwarding for this, and don't give remote-desktop
access (AnyDesk, TeamViewer) to this PC: that person would be "this PC", which is always admin. The live video
uses a few Mbit/s of this site's upload speed while someone watches.

**Face blurring**: viewer accounts see faces pixelated in the live video, the "in view now" thumbnails and all
saved pictures (`web.blur_faces_for_viewers`, on by default); admins see faces. Each saved picture gets a
blurred copy in `data/crops/blurred/`; pictures saved before this existed are shown to viewers blurred all over.
For the strongest privacy, `storage.blur_faces: true` blurs faces in the saved pictures themselves (and never
saves face close-ups), so nobody can see faces later; labeling still uses the unblurred frames in memory, but
the Review and Answer key pages then show blurred faces. The blurred area comes from the head keypoints (larger
than the face, to cover profiles); without them, the top of the person is blurred.

**Privacy paperwork**: templates for a camera notice sign ([docs/camera-notice.md](docs/camera-notice.md)) and a
data-protection summary — what is stored, where, how long, who can see it
([docs/data-protection.md](docs/data-protection.md)). Turn on **BitLocker** on this PC so the pictures, database
and backups are encrypted on disk. Pictures and labels never leave this PC; at startup the app only asks
Hugging Face whether its models have updates (model names only); `HF_HUB_OFFLINE=1` in `.env` stops that once
the models are downloaded.

**How long people are kept**: `storage.retention_days` (30), or a camera's own `retention_days` (e.g. 7 for an
entrance camera). Older people, their pictures and labels are deleted at startup and every hour while running.

**Erasing one person's data** (e.g. someone asks to be removed): on the Live page, find them with the Ask box
(a description such as "woman in a red coat yesterday afternoon") or with an attribute filter; open a likely
match's look-alikes to find their other visits, also on other cameras; tick every record of them, then
*Delete selected*. Their pictures (and blurred copies), labels, review history, embeddings and alerts are
removed, and the Audit log records the deletion. Backups made by stop.bat (`data/backups`, the last 7, at most
one a day) still contain them until they are rotated out; the same holds for people past their retention time.

**Audit log** (admin page *Audit log*): who logged in (and failed), watched the live video, searched or asked,
looked up look-alikes, viewed a report, exported CSV, deleted people, changed alert rules, reviewed labels,
used the answer key, stopped the app, and account changes made with `users.py`, with time and address.
Someone at this PC is recorded as "this PC". The Live page's once-a-second updates are not recorded. Entries
older than `storage.audit_retention_days` (365) are deleted at startup.

## A camera at another site (a phone or another PC sends it here)

A webcam on another PC, or a phone, anywhere (another house, another country) can be one of this PC's cameras:
**the device sends its picture here** through Tailscale, and this PC does the detecting, labeling and recording as
for its own cameras. Sending is an outgoing connection from the device, so it works even where the firewall blocks
everything coming in (a company PC, a phone network); the device needs no open port and no server.

**On this PC (once per camera):** Settings page → Cameras → **Add camera sent from a phone or PC** → Save. The
page then shows the camera's **send addresses**, one for a PC (`rtsp://…:8554/…`) and one for a phone app
(`rtmp://…:1935/…`), with a new random password, **shown only that once** (only its SHA-256 hash is stored; the
Audit log and the app's log never contain it; **New password** makes another). Restart now. Once for all such
cameras: let Tailscale devices reach ports 8554 and 1935 (one line in PowerShell as administrator, shown on the
page under "How to send"; the Health page checks it). The receiver is **MediaMTX** (MIT, `data\mediamtx`, set up by
install.bat or `tools\get_mediamtx.py download`): it runs while VisionLabeler runs (started again within seconds if
it stops); each camera may send only to its own name, with its own password, from a Tailscale address
(`receiver.publish_from`); VisionLabeler reads it from 127.0.0.1 with a password made at each start; nothing else
of MediaMTX is on (no HLS, WebRTC, SRT, MoQ or API). The Health page's **cameras from other devices** line says
which are sending now.

**On another PC:** copy the folder `remote-camera\` there and run **setup-camera.bat** (its `README.txt` says what
to do in plain words): paste the PC send address; it downloads FFmpeg (LGPL, 170.5 MB) after asking, checked against
its published checksum, finds and tests the webcam (1280×720 at 15 fps when it offers it), picks an H.264 encoder
that works there, sends a 5-second test picture (clear messages for a wrong password, an unknown camera, or no
answer), then sends from every login with no window, starting again by itself after any drop. No firewall change
and no administrator on that PC. **remove-camera.bat** takes it all away. (The first version, 2026-10-12 morning,
served the picture instead and was blocked by a company firewall; running the new setup cleans it up.)

**On a phone:** install Tailscale (signed in to the same network) and a streaming app that sends RTMP, for example
Larix Broadcaster (free, Android and iPhone); add a connection with the phone send address, 1280×720, about
1.5 Mbit/s; keep the app open, the screen on if the phone asks, and the phone charging.

Measured (2026-10-12) on test copies: H.264 1280×720 at 15 fps is **about 1.5 Mbit/s** (0.64 GB an hour, 15 GB a
day of the device's upload if it sends all day); the test copy read it at ~20 frames/s and found and recorded the
people in it; a wrong password, another camera's name and anyone else reading were refused (tests, with MediaMTX
itself); sending stopped → "no picture from the camera", back within seconds when it returns; MediaMTX killed →
started again in 4 s, and the PC sender connected again by itself, picture back in ~35 s. Found on the way: FFmpeg
waits forever when the receiver dies halfway (now a 10 s limit), and MediaMTX 1.21's MoQ listens on a fixed port
(turned off). Not yet tested with a real phone or a real second PC (this PC's Tailscale was not connected).

## Database

PostgreSQL 17 with pgvector in Docker (`docker-compose.yml`), on 127.0.0.1:5432, data in the Docker volume
`visionlabeler_pgdata`, password in `.env` (keep it private). Tables: `cameras`, `tracks`, `attributes`,
`embeddings` (`vector(768)` with an HNSW index), `reviews`, `alert_rules`, `alert_events`, `answer_key`,
`users` (password hashes), `login_sessions` (hashed cookie tokens), `audit_log`.
For a quick test without Docker: `database.url: sqlite:///data/vision.db`.

## Training the fast classifiers

They are saved in `models/attributes.joblib` and loaded at startup; there is no retraining on start. They give
instant labels on the Live page until the vision model answers; the vision model is skipped only when they are
confident about every attribute (with the current data: nobody, since luggage is too rare to learn). While the
age model is loaded they give no age, because the age model measured better. To retrain after adding data
(teacher labels and embeddings are cached, so only new people are labeled):

```bash
.venv\Scripts\python.exe tools/teacher_label.py data/datasets/coco_people
```

```bash
.venv\Scripts\python.exe tools/train_attributes.py data/datasets/coco_people --no-balance
```

`export_dataset.py` exports human-verified labels from the database as a dataset.

**Spot checks in the Review queue**: every 5th item (`review.spot_check_every`) is a confident label nobody
has checked yet, from people of the last 30 days, spread over cameras, attributes and values, so your answers
also cover what the system is sure about (and the retraining learns from them). The selection stays the same
for a day. Spot checks are not counted in "to review".

**Weekly retraining on your own cameras** (`retrain:` in config.yaml): every 7 days, at night, the fast
classifiers are retrained on the COCO practice people plus the finished people from your cameras (their labels:
your Review corrections where you made one, otherwise the vision model's; never the classifiers' own guesses),
yours counting 3 times as much. Answer-key people are never trained on: they are the test. The new version
replaces the current one only if it scores at least as well on the answer key (otherwise it is thrown away),
and the cameras switch to it at once. It needs 30 finished people from your cameras outside the answer key and
10 answer-key people; until then each run is recorded as "not enough people" and nothing changes. The Answer
key page lists every run (trained on, score before and after, kept or not) and has **Retrain now**. Files:
`data/models/attributes.joblib` (current), `attributes.prev.joblib` (the one before); the shipped
`models/attributes.joblib` is used until the first kept retraining. The answer-key score of each camera is also
recorded once a day and shown as a chart (**Accuracy over time**), so you can see whether changes help.

**First kept retraining (2026-10-12)**: the user's webcam had given only 1 person outside the answer key, so 7
eye-level clips of the MOTChallenge MOT16 dataset (street and shopping mall; downloaded with the user's OK, 1.95 GB,
in `data\datasets\mot16`) were imported as the recorded camera `mot16` (kiosk mode, detector 640, 10 frames a
second, dated March 2016 so they stay out of today's reports): **127 people** (kiosk mode skips far-away people;
of about 300 in the clips). Retrained on 126 of them + 1,534 COCO people, the fast classifiers' score on the 26
webcam answer-key people went **61.8% → 62.1%**, kept. Per attribute: gender 67/67%, carrying 25/25%, eyewear 50 →
43%, face mask and headwear 100%, top color 9 → 18%, top type 82%. With 26 people one person moves an attribute
by 4-11%, so this is no measurable change; the final labels (74.6% on the Answer key page) come mostly from the
vision model, which the classifiers only stand in for until it answers. A larger answer key makes the check meaningful.

## PC watch

The app also watches the PC it runs on (`labeler/pcwatch.py`, read only): the graphics card's temperature, power and
memory once a minute (`nvidia-smi`; hot from 85 °C, `pcwatch.gpu_hot_c`), and each disk's health as Windows sees it
once an hour (`Get-PhysicalDisk`; anything but Healthy is a problem: copy the data off and replace the disk). While
it runs it writes a heartbeat (`data/heartbeat.json`) once a minute; a normal stop (Stop, stop.bat, Ctrl+C, Restart
now) marks it at the very end. At the next start a heartbeat without that mark means the app or the PC stopped
without warning, around its last beat; the PC's start time tells which: "the PC was turned off or restarted" (a power
cut, Windows updates, or shutting down without stop.bat) or "the app stopped without warning" (its window closed, the
process ended). Checked on a test copy: ended by force, the next start said so ("around 14:09"); after a normal stop,
nothing. On this PC: GPU 34 °C at rest, 9 W; both disks Healthy; reading the GPU takes 0.06 s, the disks 1.6 s.
Where it shows: the **Health** page has a line "PC" (the card's temperature, power and the hottest since start;
each disk) and "unexpected stops (7 days)" (how many, and the last one with when it was back). A problem going
on (hot card, unhealthy disk) shows like a camera problem: the Live page's banner, a pop-up there when it starts,
and the Alerts page's list "Camera and PC problems", where an unexpected stop is listed from the last sign of
life until the app ran again; admins also see them in the Summary (🖥️). A hot card counts as cool again only 5 °C
below the limit, so a temperature around it does not come and go every minute; a failed reading keeps what was
known. Real run on a test copy with the limit set to 30 °C (the card was at 37 °C), ended by force and started
again: the Live banner, both Health lines and the three list entries (hot, stopped 14:22-14:23, hot) were right.

## Updates

The Settings page's **Updates** box shows whether a new version is on your GitHub repository (checked a few minutes
after start, then once a day; **Check now** at once) with the titles of the changes. **Update now** (admins; asks first;
in the Audit log): backs up the database, adds the new version with `git pull --ff-only` (only when nothing in the
app's folder was changed by hand and the copy has no changes of its own), installs new Python packages only when
`requirements.txt` changed (if that fails, the files go back to the running version), then restarts (with start.bat;
otherwise it says to restart by hand). If anything goes wrong, the version running now stays. The Health page line
"updates" says the same. Git uses the GitHub sign-in this PC already has; it never asks for a password from the app.
A copy installed from the offline package (not a git copy) says it cannot check. Tested on scratch repositories and
on a copy of this repository set back 3 versions: listed the 3 changes and updated in under a second (no packages).

### Safe-update test

Before restarting the app on a new version, `tools/upgrade_check.py` tries the start-up database update on a copy:

```bash
.venv\Scripts\python.exe tools\upgrade_check.py
```

It makes a throwaway database (`visionlabeler_upgrade`) on the same PostgreSQL server from a dump of the real one,
inside the database container (the real database is only read, also while the old app runs; no copy of the data is
written to a file), opens it with today's code with a time limit (`--timeout`, 300 s), counts the rows of every table
before and after, and says how long it took, what it added, which old rules it removed, that nothing was lost, and
what What's new will list. Then it drops the copy and writes `data/upgrade_check.json`. Docker must be running
(start.bat starts it). Checked on 2026-10-19 on a database built by the pre-Phase-13 code with 25 people and 3
alerts: 0.9 s, 10 tables and 5 columns added, the old "once per rule and person" rule and two "may not be empty"
rules removed, nothing lost, What's new 24 entries. With the start-up code from before the Phase 36 fix it stopped
after 30 s and showed what the database was waiting on (the lock from pitfall 30) instead of hanging.

The **Update now** button runs it too, with the new version, after getting it and before restarting: if the new
version cannot update the copy (an error, rows lost, or longer than 5 minutes), the version running now is put back
(files and Python packages) and the reason is shown; if the test cannot run at all (Docker off), the update goes on
as before. The newest result shows in the Updates box and on the Health page ("safe-update test"). At start the
console says "opening the database...", then "database updated for this version in 0.9 s: 10 new tables, 5 new
columns" when there was anything to do, and every 30 s while it takes long ("still opening / updating the
database..."), so a slow or stuck update is never a silent window.

On the user's own database (2026-10-19, read only, with their OK; their app was not running): copied in 1.5 s,
updated in 1.1 s, 154 people, 9 tables and 5 columns added, the three old rules removed, nothing lost; What's new
will list 22 features (from Phase 17: the database shows their app last ran Phase 13-15 code). The real database
was left as it was, and the copy was dropped.

### What's new

After an update (by the button, by hand with git, or a new package), the Live page shows one line under the header,
**"New in this version: …"**, with a link to the **What's new** page (`/whats-new`; also linked from the Updates
box): each feature added since the version that ran before, in plain words, with where to find it, marked **New**,
then everything added earlier. Viewers see only what they can use. ✕ hides the line in that browser until the next
update (opening the page does too); restarting the same version keeps it. The app keeps the newest feature it ran in
`data/last_version.json` (`whats_new.state_file`). The first start with this feature has no such file, so it judges
by the database: tables or columns it had to add at that start belong to features that never ran here (a copy that
last ran before recording, Phase 13, lists everything since); a new database (a new install) lists nothing as new.
Checked on a test copy: a state file saying Phase 24 listed the 10 later features; no file and a database without
the Phase 24 table listed Phase 24 and the 10 after it.

### First-run check

The **First-run check** page (`/first-run`, admins; linked from the Live page's "New in this version" line, What's
new and the Updates box) says in one list whether everything works after an update, each line with what to do when
it does not: the Health page (all green, or its problems), the database (every table and column of this version;
what this start added), the built-in alert rules Fall, Help sign and No movement (on, off, or missing), a recording
from today (recording starts when someone is in view, so stand in front of a camera first), Home / Away, and each
camera's picture (the same check as Settings, "Check the picture"). **Check again** runs it once more.

**Try the help sign**: for 30 seconds the camera tells the page when it sees the sign (a bar shows how long a hand is
up), instead of making an alert: no pop-up, no beep, nothing in the alert list, but a line in the Audit log. A hand
already up when you press the button counts after 3 more seconds. Checked on a test copy with a COCO photo of a
tennis serve (one arm straight up): seen 3 s after pressing, no alert; without the restart of the count it was missed
(a still photo's sign had already been given, before pressing), which is why it is there.

## Tests

The tests need `pytest` and `httpx2` (Starlette's test client; with only the older `httpx` it warns):

```bash
.venv\Scripts\python.exe -m pip install pytest httpx2
```

```bash
.venv\Scripts\python.exe -m pytest -q
```

`tests/test_postgres.py` also updates an old PostgreSQL database (a throwaway one it makes and drops); it runs only
when `VL_TEST_POSTGRES_URL` names a server, e.g. the URL in `.env`, and is skipped otherwise (as on GitHub).

504 tests: vocabulary, colors, geometry, database rules, delete, search, retention, re-linking, label voting,
the VLM worker and retries, classifiers, age source, cameras, accounts and roles, audit log, HTTPS, face
blurring, supervisor, log files, the health check, the Settings page and restarts, the installer's model packing, backups, the counting check's scoring, person videos, the storage line, original-video excerpts, search by photo, today at a glance, idle mode, the long-run check and the restore drill, the re-identification wrapper, recording and the Recordings page, network-camera logging, receiving cameras from other devices, the second-site checks, video import (start times, MP4 headers, video time), alerts, reports, questions, answer key (also per camera), a syntax check of
every page's script, and a check that no text file holds stray control characters (one once broke `stop.bat`).
Changes are tracked with git.

## Files

| Path | What |
|---|---|
| `config.yaml` | all settings |
| `labeler/schema.py` | the attribute vocabulary |
| `labeler/pipeline.py` | camera loop, tracking, labeling, re-linking |
| `labeler/vlm.py`, `labeler/age.py`, `labeler/student.py` | vision model, age model, fast classifiers |
| `labeler/agent.py` | questions → search plans |
| `labeler/alerts.py`, `labeler/reports.py` | alert rules, report summaries |
| `labeler/db.py` | database tables and queries |
| `labeler/web.py`, `labeler/static/` | web UI |
| `labeler/supervisor.py`, `labeler/logs.py` | restart after crashes; log files |
| `labeler/healthcheck.py` | the Health page and `/api/health` |
| `labeler/settings.py` | the Settings page (`data/settings.yaml` over config.yaml) |
| `labeler/importer.py` | the Import page: recorded videos with real times |
| `labeler/recorder.py`, `labeler/recordings_web.py` | recording while people are in view; the Recordings page |
| `labeler/receiver.py` | receiving cameras sent from a phone or another PC (MediaMTX) |
| `labeler/falls.py` | fall detection from body points |
| `labeler/vehicles.py` | vehicles: tracking, parked vehicles, the `vehicles` table |
| `remote-camera/` | for another PC: send its webcam here through Tailscale (setup-camera.bat, remove-camera.bat) |
| `run.py`, `start.bat`, `stop.bat` | starting and stopping |
| `install.bat` / `install.ps1`, `tools/models.py`, `tools/pack_offline.ps1` | installing on a new PC, offline package |
| `tools/` | benchmarks, evaluation, training, backups, long-run check (`soak.py`), restore drill, FFmpeg setup, mobility-aid check (`mobility_check.py`, `get_aid_photos.py`), fall check (`fall_check.py`), vehicle check (`vehicle_check.py`) |
| `tests/` | automated tests |
| `data/` | person pictures, backups, log files, datasets (not in git) |

## Privacy

- Face crops are not saved (`storage.save_face_crops: false`); faces are used in memory for age only.
- Data older than `storage.retention_days` (30) is deleted on startup.
- Mobility aids (wheelchair, walking stick, crutches, walker) are health-related: Settings → "Label mobility aids"
  turns them off (not asked, not stored from then on).
- Gender and age are visual estimates, not facts. Put up visible notice signage where cameras run, and check
  local law (PDPA/GDPR or equivalent) before deploying.

## Licenses to check before commercial use

Ultralytics YOLO: AGPL-3.0 (commercial license needed for closed products, or swap in RF-DETR, Apache-2.0).
Qwen3-VL: Apache-2.0. Age model: Apache-2.0. COCO photos: various Flickr licenses. FairFace: CC BY 4.0.
FFmpeg (recording): the LGPL build, used as a separate program (`data\ffmpeg`, its LICENSE.txt alongside).
MediaMTX (receiving cameras from other devices): MIT (`data\mediamtx`, its license alongside).
MOT16 (MOTChallenge, imported as the recorded camera `mot16` and used by the retrained classifiers since
2026-10-12): CC BY-NC-SA 3.0, **non-commercial only**: before commercial use, delete that camera's people on the
Import page and retrain (or delete `data\models\attributes.joblib` to go back to the shipped classifiers).
Open Images photos with wheelchairs and crutches (`data\datasets\openimages_aids`, measuring only): CC BY 2.0
(each photo's author), boxes CC BY 4.0.
UR Fall Detection videos (`data\datasets\urfd`, measuring only): CC BY-NC-SA 4.0, non-commercial.
OSNet-AIN re-identification weights (`data\models\reid`, measuring tool only, not used by the app): MIT, trained on
MSMT17, DukeMTMC-reID and CUHK03 (research use; DukeMTMC was withdrawn by its authors in 2019).
