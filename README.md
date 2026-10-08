# VisionLabeler

Local camera system that detects people, tracks them, and labels each person with a fixed set of attributes
(apparent age, gender, clothing type and color, headwear, eyewear, carried items). Uncertain labels go to a
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

`cameras:` in `config.yaml` lists every camera; each runs its own detection loop, and the vision model,
embeddings and database are shared. The web page shows a tab per camera.

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
| **Alerts** | Rules like "red top + backpack" or "anyone" (per camera); each fires once per person |
| **Answer key** | Label recorded people yourself, blind, and see how accurate the system is on this camera |

### Asking questions

The Ask box understands questions like "who wore a red top after 3pm?", "women with a backpack yesterday",
"a 30 year old man on the entrance camera" or plain descriptions like "man in a beige fur coat". The local
model turns the question into a search plan (time, camera, conditions, appearance), shown above the results
so misunderstandings are visible. A condition is only used if the question mentions it, and a camera only if
it is named: the model otherwise invents filters. When few people match, the closest look-alikes whose labels
differ are added and marked, because labels can be wrong. Interpreting a question takes under a second.

### Deleting

Tick people in **People seen** (or **Select all shown**) and press **Delete selected**, or use
**Delete all…** (asks you to type DELETE ALL). Their pictures, labels, review history, embeddings and alerts
are removed. A deleted person still in view is shown as a gray "deleted" box and not recorded again.

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

## Speed

Measured on this PC (RTX 4070 SUPER): a person is fully labeled about **1.8–2.1 s** after labeling starts;
detection and colors run every frame for everyone. Things that turned out to matter:

- Ollama at `127.0.0.1`, not `localhost`: on Windows `localhost` added ~2 s to every request (4.4 s → 1.8 s).
- One request at a time (`vlm.workers: 1`): with that delay gone, 2 at once is no faster.
- Several people in one request was rejected: answers mixed people up (gender matched only 54%).
- The 8B model is slower and was not more accurate.

## One person, one record

- **Tracker** (`labeler/trackers/people.yaml`): weak detections only keep a known person tracked; a lost
  person is remembered ~3 s.
- **Re-linking** (`tracking.relink`): someone who reappears within 5 s near the same spot (similarity
  ≥ 0.75), or within 30 s looking very similar (≥ 0.90), continues the same record. Tested: 2 s and 10 s gaps
  → 1 record (before: 2); 40 s → 2 records, as intended.

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

## Alert clips

Each alert gets a short video: the 5 seconds before it fired and the 5 after (`clips:` in config.yaml), with the
boxes and labels as on the Live page. On the Alerts page, **Play clip** shows it (admins only; watching is in
the Audit log). Each camera keeps its last seconds in memory (about 5 MB); clips are WebM (VP8, built into
OpenCV, plays in every current browser), at most 960 pixels wide and 10 frames per second, about 1.5-3 MB each,
written in the background a few seconds after the alert. They are deleted with the person (Delete, retention)
or with their alert rule. With `storage.blur_faces: true` clips are made from the blurred picture.

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

## Things that look like people

Pictures of people (a monitor, reflection, poster, TV) are detected as people. In kiosk mode new people under
3% of the frame are ignored (`detector.min_box_area`); `ignore_zones` skips fixed areas. Ignored detections
are drawn as gray "ignored" boxes.

## Reliability

- **Crash recovery**: the supervisor restarts the app after a crash; a watchdog exits if a camera loop dies;
  after 5 crashes in 10 minutes it stops and shows the error.
- **Single copy**: a second copy refuses to start ("already using port 8000").
- **Clean stop**: Stop / `stop.bat` / Ctrl+C save the people in view first; an open Live page no longer
  blocks shutdown.
- **Ollama errors**: each request is retried once.
- **Backups**: `stop.bat` runs `tools/backup_db.py` (at most daily, keeps 7) into `data/backups`.
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
router. With [Tailscale](https://tailscale.com) (free for personal use) on this PC and the other PC, signed in
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

## Tests

```bash
.venv\Scripts\python.exe -m pytest -q
```

156 tests: vocabulary, colors, geometry, database rules, delete, search, retention, re-linking, label voting,
the VLM worker and retries, classifiers, age source, cameras, accounts and roles, audit log, HTTPS, face
blurring, supervisor, alerts,
reports, questions, answer key (also per camera), and a syntax check of every page's script. Changes are tracked with git.

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
| `labeler/supervisor.py` | restart after crashes |
| `run.py`, `start.bat`, `stop.bat` | starting and stopping |
| `tools/` | benchmarks, evaluation, training, backups |
| `tests/` | automated tests |
| `data/` | person pictures, backups, datasets (not in git) |

## Privacy

- Face crops are not saved (`storage.save_face_crops: false`); faces are used in memory for age only.
- Data older than `storage.retention_days` (30) is deleted on startup.
- Gender and age are visual estimates, not facts. Put up visible notice signage where cameras run, and check
  local law (PDPA/GDPR or equivalent) before deploying.

## Licenses to check before commercial use

Ultralytics YOLO: AGPL-3.0 (commercial license needed for closed products, or swap in RF-DETR, Apache-2.0).
Qwen3-VL: Apache-2.0. Age model: Apache-2.0. COCO photos: various Flickr licenses. FairFace: CC BY 4.0.
