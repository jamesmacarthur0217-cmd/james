# VisionLabeler

Local camera system that detects people, tracks them, and labels each person with a fixed set of
attributes (apparent age range, clothing type and color, headwear, eyewear, carried items).
Uncertain labels go to a human review queue, and verified labels are exported as training data.
Everything runs on this PC: no cloud calls.

**Daily use: see [START-STOP-GUIDE.md](START-STOP-GUIDE.md).** `start.bat` after turning on the PC,
`stop.bat` before shutting it down.

## How it works

```
Camera / video / RTSP
  → YOLO11-pose + ByteTrack        detect and track every person (GPU, every frame)
  → best-frame selection           keep the sharpest, largest, most front-facing frames per person
  → color estimator                clothing colors from torso/leg regions found by pose keypoints
  → Qwen3-VL 4B (via Ollama)       structured attributes from the best frames, once per person
  → SigLIP 2 embedding             one appearance vector per person (mean of the best crops)
  → reconcile                      low confidence, or two sources disagreeing → review queue
  → PostgreSQL + pgvector          tracks, attributes (with source, model, confidence), embeddings, reviews
  → web UI                         live view, text search, look-alikes, review queue
  → export_dataset.py              human-verified labels → training dataset
```

Two modes, set in `config.yaml` (`mode:`) or with `--mode`:

| | kiosk | security |
|---|---|---|
| For | people 0.5–1.5 m away, facing the camera | people far away, often seen from above |
| Age | `age_range` (18-24, 25-34, …) | `age_group` (child / teen / adult / senior) |
| Face attributes | eyewear, face mask | none |

## Running it

Docker Desktop must be running (it holds the database). Then double-click `start.bat`: it starts the
database container, then the app, then opens the browser. Or from a terminal in this folder:

```bash
docker compose up -d --wait
```

```bash
.venv\Scripts\python.exe run.py
```

Then open http://127.0.0.1:8000. The camera source is set in `config.yaml`, or override it:

```bash
.venv\Scripts\python.exe run.py --source 0
```

```bash
.venv\Scripts\python.exe run.py --source "C:\videos\lobby.mp4"
```

```bash
.venv\Scripts\python.exe run.py --source "rtsp://user:pass@192.168.1.50:554/stream1" --mode security
```

Other options: `--no-vlm` (colors only), `--no-web` (headless), `--max-frames N` (stop after N frames),
`--port N` (web UI port), `--list-cameras` (show which webcam numbers give a picture).

### Webcam

`camera.source: 0` in `config.yaml` uses the first webcam. If you have several, run
`.venv\Scripts\python.exe run.py --list-cameras` and use the number that says "picture OK".
On Windows the app reads webcams through Media Foundation; DirectShow gave black frames on this PC.
If the live view says "NO PICTURE FROM CAMERA", check the lens cover and close other apps using the
camera (Teams, Discord, the Windows Camera app).

## Stopping

| Way | What happens |
|---|---|
| **Stop** button (top right of the web page) | Saves the people in view, finishes their labeling, closes the app. Database keeps running. |
| `stop.bat` | Same as Stop, waits until the app has fully exited, then stops the database container and frees the VLM's GPU memory. Full shutdown. |
| Ctrl+C in the app window | Same as Stop. If Windows then asks "Terminate batch job (Y/N)?", either answer is fine. |

Closing the app window with the X skips the saving step: the people in view at that moment lose their
final update. The database itself is not harmed.

Next start: `start.bat` starts the database again if it was stopped.

Ollama must be running for VLM labels (it starts automatically with Windows after install).
If it isn't, labeling falls back to colors only and the log says why.

## One person, one record

Two things keep a person from being split into several records:

1. **Tracker settings** (`labeler/trackers/people.yaml`). The detector passes weak detections (down to 0.1)
   to ByteTrack, which uses them only to keep an already-known person tracked through blurry, dark or
   half-visible frames. A new person still needs a clear detection (0.5). A lost person is remembered ~3 s.
2. **Re-linking** (`tracking.relink` in `config.yaml`). When someone leaves and a new person appears within
   30 s, their appearance is compared with the people who just left:
   - gap up to 5 s: similarity >= 0.75 and a box roughly where the person was lost;
   - gap up to 30 s: similarity >= 0.90.
   If it matches, the record continues (the log says "person #N is back"); otherwise it is a new person.

The thresholds come from measuring appearance similarity on webcam footage: the same person stayed around
0.93-0.95 (lowest 0.70) over gaps up to 10 s, while two different men in dark suits in one photo scored 0.86.
They deliberately favour a duplicate record over merging two different people. Re-linking has not yet been
tested with two different people taking turns in front of the camera; if it ever merges different people,
raise `long_gap_min_similarity`, or set `relink.enabled: false`.

Tested on a recorded webcam clip with an empty-room gap spliced in:

| Gap | Before | After |
|---|---|---|
| none | 1 record | 1 record |
| 2 s | 2 records | 1 record (tracker held on) |
| 10 s | 2 records | 1 record (re-linked, similarity 0.95) |
| 40 s | 2 records | 2 records (beyond the 30 s window, as intended) |

## Things that look like people

Pictures of people (a monitor showing the live view, a reflection, a poster, a TV) are detected as people.
Two settings handle them:

- `detector.min_box_area`: in kiosk mode, new people smaller than 3% of the frame are ignored (someone at
  a kiosk is much bigger). On this PC, an image of the user on a screen behind the desk covered 1.6%, the
  user 6-15%. Security mode has no minimum, since people there are far away.
- `camera.ignore_zones`: areas of the picture to skip, as `[left, top, right, bottom]` fractions of the
  frame. Use this for a fixed monitor, TV or poster.

Ignored detections are drawn as thin gray boxes labeled "ignored". A box-motion check and asking the VLM
"is this a real person?" were both tried and rejected: a picture's box jitters as much as a person sitting
still, and the VLM called a blurry real person a photo.

## Web UI

- **Live camera**: the moving video with the analysis drawn on it: pose skeleton, person number, time in view,
  status (observing / analyzing / re-analyzing) and current labels. The red dot pulses while frames arrive.
  **Full screen** shows only the video.
- **In view now**: one card per person on screen, updated several times a second: live thumbnail, time in
  view, status, when the labels were last updated, and the current labels.
- **Continuous analysis**: a person who stays in view is re-analyzed every `vlm.relabel_every_s` (15 s),
  so labels follow changes such as glasses on or a phone out. Labels are a vote over the last
  `vlm.history_frames` (4) answers, so one odd frame does not flip them. New people always go first.
- **People seen**: everyone recorded, newest first. Orange chips need review; outlined chips were verified
  by a person.
- **Describe a person**: free-text search over the embeddings, e.g. "man in a beige fur coat". This works
  for things that are not in the attribute list (patterns, logos, styles). Results are ranked, closest first;
  there is no hard yes/no cutoff, so the top results are the candidates to look at.
- **Look-alikes**: people whose appearance is closest to the selected one. Useful for finding the same
  person again later or on another camera, while they wear the same clothes. SigLIP is a general image model,
  not a trained re-identification model, so people in the same scene also score fairly high.
- **Filter by attribute**: exact match on a stored label, e.g. upper color = red.
- **Review queue**: labels where the color estimator and the VLM disagreed come first, then the least
  confident. Pick the right value and Save. Every decision is recorded in the `reviews` table.

## Database

PostgreSQL 17 with pgvector runs in Docker (`docker-compose.yml`), listening only on 127.0.0.1:5432.
The data lives in the Docker volume `visionlabeler_pgdata`, so it survives container restarts and updates.
The password and connection URL are in `.env`, generated randomly on setup; keep that file private.

Tables: `cameras`, `tracks`, `attributes`, `embeddings` (a `vector(768)` column with an HNSW index for
fast similarity search), `reviews`.

Back up the database:

```bash
docker exec visionlabeler-db pg_dump -U visionlabeler visionlabeler -f /tmp/backup.sql
```

```bash
docker cp visionlabeler-db:/tmp/backup.sql backup.sql
```

To run without Docker for a quick test, set `database.url: sqlite:///data/vision.db` in `config.yaml`.
Everything works the same; search is then brute force in memory.

## Speed with many people

Every person in the frame is detected, tracked and color-labeled on every frame (about 30 fps). The detailed
labels come from the VLM, which is the bottleneck. Measured on this PC (RTX 4070 SUPER), 24 people each run:

| Setting | Time per person |
|---|---|
| 1 request at a time | 4.4 s |
| **2 at once (`vlm.workers: 2`, the default)** | **2.3 s (1.9x faster)** |
| 3 or 4 at once | 2.3 s (no further gain: the GPU is full) |
| several people in one request | 2.6-2.8 s, but **rejected**: answers mixed people up (gender matched the one-person answer only 54% of the time with 4 per request) |

Ollama runs this model one request at a time; the gain comes from Ollama preparing the next request while
the GPU works on the current one. No Ollama setting needs changing.

In the 4-person test, everyone was fully labeled 11, 16, 21 and 25 s after appearing (before: 12-37 s).

## Fast attribute classifiers ("student")

Small classifiers predict labels from the SigLIP embedding the app already computes, in a few milliseconds.
They were trained from VLM labels ("teacher") of 1,534 people cut out of the public COCO 2017 photos.
Each answer has its own confidence threshold, chosen on 399 people the classifiers never saw: above it, the
student agrees with the VLM at least 90% of the time and clearly beats guessing. Only those answers are used.

What it does well, instantly: gender (80% of people, 91% agreement), lower clothing type, headwear,
and "no glasses / no mask / no bag" (96-99% agreement). What it cannot do yet: a real age range, top color
(only black/white), or confidently say someone *is* carrying something. Because of that it was confident
about every attribute for 0% of people, so **it does not reduce VLM work yet**; it gives the live view
partial labels within a second while the VLM finishes. Disagreements between it and the VLM go to review.

Spot-check of the teacher on 12 random people: upper clothing and glasses were good; gender was never wrong
when given; age leans towards "35-44"; lower clothing is sometimes guessed when legs are not visible; bags
have false positives. The student inherits these, so human-reviewed labels are the way to improve it.

The trained classifiers are saved in `models/attributes.joblib` and loaded at startup: no retraining on
start. Retrain only after adding data:

```bash
.venv\Scripts\python.exe tools/extract_people.py data/datasets/coco/val2017 data/datasets/coco_people
```

```bash
.venv\Scripts\python.exe tools/teacher_label.py data/datasets/coco_people --workers 2
```

```bash
.venv\Scripts\python.exe tools/train_attributes.py data/datasets/coco_people --no-balance
```

Teacher labels and embeddings are cached, so only new people are labeled. `tools/bench_vlm.py` repeats
the speed tests. COCO images are under various Flickr licenses; check them before commercial use.

## Training data

```bash
.venv\Scripts\python.exe export_dataset.py
```

Writes `data/export/<timestamp>/images/*.jpg` and `labels.jsonl` containing human-verified labels.
Add `--include-auto` to also include confident automatic labels.

## Files

| Path | What |
|---|---|
| `config.yaml` | all settings |
| `labeler/schema.py` | the attribute vocabulary; edit here to add or change attributes |
| `labeler/pipeline.py` | camera loop, tracking, best-frame selection |
| `labeler/vlm.py` | VLM prompts and the Ollama call |
| `labeler/colors.py` | rule-based clothing color estimator |
| `labeler/db.py` | database tables and queries |
| `labeler/web.py`, `labeler/static/` | web UI |
| `docker-compose.yml`, `.env` | database container and its private connection settings |
| `data/` | person crops and dataset exports (created on first run) |

## Measured on this PC (RTX 4070 SUPER)

- VLM: about 4 s per frame once warm, 2 frames per person, so a person is fully labeled about 8 s after
  labeling starts. The first call after startup takes longer while the model loads.
- GPU memory with detector, VLM and embedding model loaded: about 7.5 GB of 12 GB.
- Embedding a person or a search query: a few milliseconds. Search runs inside Postgres on the HNSW index.

## Privacy defaults

- Face crops are **not** saved (`storage.save_face_crops: false`). Face images are biometric data.
- Data older than `storage.retention_days` (30) is deleted on startup.
- `apparent_gender` and age are visual estimates, not facts. Put up visible notice signage where cameras run,
  and check local law (PDPA/GDPR or equivalent) before deploying.

## Licenses to check before commercial use

- Ultralytics YOLO: AGPL-3.0 (a commercial license is needed for closed-source products), or swap in
  RF-DETR (Apache-2.0).
- Qwen3-VL: Apache-2.0.

## Next steps

1. Real cameras: test with the actual camera position and lighting, then tune `detector.conf` and
   `tracking.*` in `config.yaml`.
2. Distillation: once a few hundred people are reviewed, train small attribute classifiers on the stored
   embeddings so the VLM is only called when they are unsure.
3. Natural-language query agent on top of attribute filters and embedding search.
