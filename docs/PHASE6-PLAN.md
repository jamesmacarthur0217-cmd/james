# Phase 6 plan: operations (VisionLabeler)

## Context

Phases 0 and 2–5 are done (165 tests, see docs/HANDOFF.md). Phase 6 makes the system easier to run and keep
healthy. The user's choices (2026-10-09): **keep the console window** (start.bat + optional autostart, as
today — a real Windows service is not used anyway: it would start before login, where Docker Desktop is not
running and webcams usually cannot be opened); a **Settings page for the common settings**; and all three
extras: **health check**, **installer for a new PC**, **private GitHub backup with tests**. Each step below is
tested and committed on its own; README, START-STOP-GUIDE, HANDOFF and ROADMAP updated with each.

## Step 1 — Log files

- `run.py` (logging.basicConfig at line ~64): add a `RotatingFileHandler` → `data/logs/visionlabeler.log`
  (10 MB × 5), same format; the console keeps showing messages. Config `logs: {dir, max_mb, keep}`.
- `labeler/supervisor.py`: write its own lines (restart, give up) to `data/logs/supervisor.log` as well as the
  console, so crashes are visible after the window is closed.
- Tests: handler created, rotation settings applied (tmp dir); supervisor writes its file.

## Step 2 — Health check

- `GET /api/health` (web.py): app uptime and version (git commit if available), database (`SELECT 1` +
  time), Ollama (`/api/version`, model pulled — reuse `VLMClient.available()`), each camera (running, fps,
  seconds since last frame, current problems from `HealthMonitor.current()`), disk free (existing `_check_disk`
  logic), last database backup age (`data/backups`), last retraining (`db.list_model_history`), errors in the
  log in the last 24 h. HTTP 200 when everything is OK, 503 otherwise (for monitoring tools).
- Access: logged-in users (add to the viewer allowlist in `accounts.py`); optional `web.health_token` so a
  monitor can call `/api/health?token=…` without an account (status only, compare with `secrets.compare_digest`).
- Small `/health` page (linked from the header status) showing the same in plain words.
- Tests: all-OK → 200; database/Ollama/camera down (fakes) → 503 with the reason; token works, wrong token 401.

## Step 3 — Settings page (common settings) + restart

- **Storage**: never rewrite `config.yaml` (keeps its comments). Changes go to `data/settings.yaml`
  (git-ignored, BitLocker-protected), merged over config.yaml in `load_config` (`labeler/config.py`) — only for
  the default config, so test configs (`tools/make_test_config.py`) are unaffected. Written atomically.
- **Editable** (validated server-side with ranges/types): cameras — add, remove, source (webcam index, file
  that exists, rtsp/http URL), mode, `retention_days`, `imgsz` (640/960/1280); `storage.retention_days`,
  `storage.audit_retention_days`, `web.blur_faces_for_viewers`, `storage.blur_faces`, `web.https`,
  `retrain.enabled`. Camera **names cannot be changed** after creation (zones, rules, history refer to them);
  removing a camera keeps its recorded people until retention.
- **Passwords in camera URLs**: shown masked (`rtsp://user:•••@host`); an unchanged mask keeps the stored
  password; the audit log records changes with passwords masked.
- **Restart**: changes apply after a restart. "Restart now" button → `POST /api/restart` (admin, audited) →
  the app stops cleanly and exits with a new code **4 = restart requested**; `supervisor.py` restarts it at once
  and does not count it as a crash. Without the supervisor (run.py started directly) the page says to restart
  by hand.
- Page `/settings` (admin; nav link admin-only), styled like the other pages; "unsaved / restart needed" banner.
- Tests: overlay merge and precedence; validation (bad source, bad imgsz, unknown key rejected); masked
  password kept; viewer 403; audit entry; supervisor restarts on code 4 without counting it.

## Step 4 — Installer for a new PC (+ offline package)

- `install.ps1` (PowerShell, run as the user; asks before each big step, safe to run again): checks Windows
  and an NVIDIA GPU; installs missing Python 3.13, Docker Desktop, Ollama with `winget`; creates `.venv`, installs
  CUDA torch + `requirements.txt`; writes `.env` with a random database password; `docker compose up -d --wait`;
  `ollama pull qwen3-vl:4b-instruct`; pre-downloads YOLO, SigLIP 2 and the age model; creates the first admin
  account (`tools/users.py`); offers `install-autostart.bat`; ends with a check (`/api/health`).
- `tools/pack_offline.ps1`: builds a folder for PCs without internet (pip wheels via `pip download`, the Hugging
  Face cache entries, `yolo11s-pose.pt`, the Ollama model blobs, the Docker image via `docker save`);
  `install.ps1 -Offline <folder>` uses it.
- Testing on this PC (no new PC available): `install.ps1 -CheckOnly` (reports what is installed/missing, changes
  nothing); offline package built into a scratch folder and a fresh venv installed from it (`--no-index`), then
  the test suite run in that venv. Downloads/installs only with the user's OK.

## Step 5 — Private GitHub backup with tests (last; needs the user)

- Check nothing private is tracked (already checked: `.env`, `data/`, `*.pt` never committed).
- `.github/workflows/tests.yml`: ubuntu, Python 3.13, CPU torch + requirements, Node for page checks,
  `pytest -q` (all tests use SQLite and no GPU).
- The user creates an empty **private** repository on github.com (`gh` is not installed; installing it is
  optional) and gives the URL; after their explicit OK: add the remote and push. Check the first CI run.

## Critical files

`run.py`, `labeler/supervisor.py`, `labeler/config.py`, `labeler/web.py`, `labeler/accounts.py`,
`labeler/health.py`, `labeler/static/` (new `settings.html`, `health.html`; nav links via `admin-only`),
`config.yaml`; new `install.ps1`, `tools/pack_offline.ps1`, `.github/workflows/tests.yml`; tests in
`tests/test_*.py` (fixtures in `tests/conftest.py`).

Reuse: `privacy`-style atomic writes and `training.save` pattern (`tmp.replace`), `audit()` in web.py,
`HealthMonitor.current()`, `VLMClient.available()`, `run._check_disk`, `tools/make_test_config.py` for runs.

## Verification

- `pytest -q` green after every step (165 + new tests).
- Real runs on port 8001 with `tools/make_test_config.py` (never the user's data): logs appear and rotate;
  `/api/health` OK, then 503 with an unreachable camera (`--source rtsp://127.0.0.1:9/x`) and with a wrong
  Ollama host in the test config (never stop the user's Ollama); the Settings page saves `settings.yaml` in the
  scratch folder, Restart now restarts under the supervisor (`python -m labeler.supervisor --config …`), and the
  change is in effect; browser checks of both pages.
- Installer: `-CheckOnly` on this PC; offline package + fresh venv + tests in a scratch folder.
- GitHub: first Actions run green.
- Confirm no leftover `python.exe run.py` processes after each test; commit per step (`git commit -F`).
- Finally update HANDOFF/ROADMAP, then propose the next round (deferred items in HANDOFF section 8).
