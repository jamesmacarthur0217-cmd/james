# Phase 18 plan: cars, motorbikes, bicycles, buses and lorries (approved 2026-10-13)

## Context

The system only sees people (YOLO11s-pose). The user chose (2026-10-13) to also see **vehicles at the gate**:
**cars, motorbikes, bicycles, buses and lorries**; **no number plates** (other people's personal data, extra models);
described by **type, color and make/model** ("silver Perodua Myvi", the vision model; accuracy measured first).

The user has no camera at the gate yet (one desk webcam, kiosk mode): it is built and measured on public footage,
and works once a camera sees the street (an IP camera, or a phone sent in with the Phase 15 setup). Vehicles are on
by default for **security** cameras and off for kiosk cameras (a desk webcam never sees cars).

Already here for measuring, no download: COCO val2017 with boxes (about 1,900 cars, 370 motorbikes, 310 bicycles,
280 buses, 410 trucks), MOT16 with marked cars, bicycles and motorbikes (MOT16-13: street with cars; MOT16-04: one
car parked the whole video).

Design decision: vehicles get **their own table** (`vehicles`) and module (`labeler/vehicles.py`), not rows in
`tracks`: about 40 queries read `tracks` as people (reports, answer key, look-alikes, review, retraining), and every
one would need a filter.

## Step 1: detect, track and store vehicles

- Detector: **YOLO11s** (the detection model of the same family, COCO classes bicycle, car, motorcycle, bus,
  truck; ~19 MB from Ultralytics' GitHub release, AGPL-3.0 like the pose model; downloaded with the user's OK).
  A second model per camera with its own tracker (ByteTrack, `labeler/trackers/vehicles.yaml`), run on every
  `vehicles.every_n_frames` frame (cost measured first: frame rate and GPU with one and two cameras).
- `labeler/vehicles.py`: `VehicleWatch` per camera: tracks → records in a new table `vehicles` (camera, kind, started,
  ended, best picture, color, make/model, parked, moved), pictures in `data/crops/vehicles/`.
- **Parked vehicles**: one that has not moved (box centre within a share of its size) for `vehicles.parked_after_s`
  (60 s) is "parked": counted once, its record continues when it is hidden for a while (a person walking past) and
  found again at the same place, and "left" when it drives off. So a car parked all day is one record, not hundreds
  (MOT16-04).
- Idle mode (Phase 10) must not sleep through a car: a vehicle wakes it like a person (`_wake_on_detection`).
- Retention and Delete as for people (`cameras[].retention_days`, `storage.retention_days`); recording (Phase 13)
  also starts for a moving vehicle (not for a parked one).
- Measure: COCO val2017 detection (found / wrong, per kind; at 640 and 960); MOT16-13 and -04 counting against the
  marked identities (one record per vehicle, parked car once); frame rate and GPU memory with vehicles on and off.

**Step 1 results (2026-10-13)**: done (README "Vehicles"). yolo11s.pt downloaded with OK (19.3 MB). `labeler/vehicles.py`
(VehicleWatch + VehicleStore), table `vehicles`, retention, pipeline hook (every 2nd frame, own tracker
`trackers/vehicles.yaml`, purple boxes, recording and idle mode woken by moving vehicles, ignore zones), config
`vehicles:` and `cameras[].vehicles` (default: security cameras). Measured: COCO cars 79% found (medium/large),
cost 9.7 → 14.8 ms per 1080p frame at every 2nd frame; MOT16-04 looping: parked car/motorbikes one record; found and
fixed: standing bicycles lost now and then became new records and never parked (now remembered after 5 s still,
median position); MOT16-13 is from a moving bus (splits from camera motion, not tuned on). A shop sign was taken
for a car: the vision model should say "not a vehicle" in step 2. Not done yet: the offline package/installer and
Health for the new model (step 5); delete from the page (step 3).

## Step 2: describe them (type, color, make/model)

- Type from the detector: car, motorbike, bicycle, bus, lorry (COCO "truck" includes pickups: they are
  "lorry or pickup" on the page).
- Color: the existing color estimator (`labeler/colors.py`) on the middle of the body (not windows, wheels, road),
  plus the vision model's answer; the fixed color list as for people.
- Make/model: the vision model (Qwen3-VL 4B, the same Ollama model) once per vehicle from its best picture, answering
  JSON with "unknown" allowed (make only when the badge or shape makes it clear). Queued after people (people first).
  **Measured before keeping**: on Stanford Cars (196 makes/models; downloaded only with the user's OK; research use)
  how often the make is answered and right, and by eye on Malaysian-looking cars in COCO/MOT16 (Perodua, Proton are
  not in Stanford Cars: said honestly). If make/model is mostly wrong, only type and color are kept (asked first).

**Step 2 results (2026-10-13)**: done (README "Vehicles", Description). Stanford Cars test file downloaded with OK (513
MB). `vlm.PROMPT_VEHICLE` / `vehicle_schema` / `clean_vehicle` (a yes/no "vehicle_in_picture" first; types incl. SUV
and van; colors + silver; make/model free text cleaned), `VLMClient.label(..., "vehicle")`; `vehicles.body_color`
(first color), `apply_description` ("not a vehicle" hides: `vehicles.not_vehicle`); pipeline `_describe_vehicle`
(negative job ids routed back by `on_vlm_result`). Unseen 300 photos: make 88%, model 57%, type 93%, 0.84 s. Found:
"not a vehicle" as a type rejected 8-18% of clear cars; the separate question fixed it; an SUV definition sentence
made the model call ordinary cars SUVs. The shop sign is not caught (ignore zone).

## Step 3: see and find them

- Live page: a **Vehicles** list next to People (picture, type, color, make/model, arrived/left/parked, time,
  ▶ Recording), with filters (camera, type, color, day).
- Ask box: questions about vehicles ("red car yesterday evening", "motorbikes after 10 pm", "Perodua") search the
  vehicles table (type, color, time, make text) the way people are searched.
- Reports: vehicles per hour by type; zones and counting lines count vehicles separately from people (a gate line:
  cars in and out).

**Step 3 results (2026-10-13)**: done (README "Vehicles", Seeing and finding them). `/api/vehicles` (kind, color, day; viewers
too), `/api/vehicles/delete` (admins, audited), cameras say `vehicles`; Live page "Vehicles seen"; `agent.vehicle_query`
+ `QueryAgent.run` answers with `vehicles`; `reports.summarize_vehicles` + Reports panel; table
`vehicle_zone_events` (vehicles in the zones table); pictures `vehicle_<id>.jpg` with a face-blurred copy for viewers
(`VehicleStore.picture`, heads from the frame). Checked in the browser pane on MOT16-04; no vehicle crossed the lines
there (tests only for line counting).

## Step 4: alerts

- Alert rules can ask for a vehicle (type, color, entered zone; e.g. *motorbike entered zone gate after 11 pm*):
  `alert_events` gets a `vehicle_id` and `track_id` may be empty (PostgreSQL change at start, checked on a scratch
  database first, like Phase 17); pop-ups on every page and clips as for people.

**Step 4 results (2026-10-13)**: done (README "Vehicles", Alerts). Conditions `vehicle`, `vehicle_color`, `hours` (also
for people) in `labeler/alerts.py` (`is_vehicle_rule`, `in_hours`, `AlertEngine.check_vehicle`); `alert_events.vehicle_id`
and `track_id` nullable (`db.NULLABLE_NOW`, PostgreSQL at start; checked on a scratch database), `db.vehicle_attrs`,
alerts deleted with their vehicle; `VehicleStore.changed` -> `Pipeline._check_vehicle_alerts` (clip); rule validation
in web.py; Alerts page and pop-ups say "vehicle #N". Found in the real run: the alert came before the picture, and
some bicycles never had a clear picture (fallback kept now).

## Step 5: Settings, Health, docs

- Settings: vehicles on/off per camera (default: security cameras on), which kinds, parked after (s). Health: the
  vehicle detector loaded, frame rate. README "Vehicles" with all numbers; data-protection template (vehicle pictures
  can show number plates and people; kept as long as people's). HANDOFF/ROADMAP.

## Critical files

New: `labeler/vehicles.py`, `labeler/trackers/vehicles.yaml`, `tools/vehicle_check.py`, tests. Changed:
`labeler/pipeline.py` (second detector, wake, overlay boxes in another color), `labeler/db.py` (`vehicles` table,
retention, delete, `alert_events.vehicle_id`), `labeler/vlm.py` (vehicle prompt), `labeler/agent.py`,
`labeler/reports.py`, `labeler/zones.py`, `labeler/alerts.py`, `labeler/recorder.py` (trigger),
`labeler/web.py` + `static/index.html`, `reports.html`, `alerts.html`, `settings.html`, `labeler/settings.py`,
`labeler/config.py` + `config.yaml`, `install.ps1` / `tools/models.py` (the new model), README.

## Verification

- Tests: parked logic on made-up boxes (parked once, hidden and found again = same record, drives off = left), the
  vehicles table (store, list, retention, delete), VLM answer cleaning (unknown, off-list colors), Ask box words,
  rules with vehicle conditions and the alert_events change (SQLite and a PostgreSQL scratch database), Settings.
- Measurements above, reported honestly (README), including what did not work.
- Real run on a test copy (port 8001) with a MOT16 street clip as a security camera: vehicles listed with pictures,
  a parked car once, the overlay, a vehicle alert pop-up; no leftover `python.exe run.py`.
