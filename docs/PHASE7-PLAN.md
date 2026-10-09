# Phase 7 plan: daily backup, second site, import recorded videos (approved 2026-10-10)

## Context

Phase 6 is done (314 tests). The user chose three things for the next round (2026-10-10):

1. **Daily backup while running.** Today only `stop.bat` backs up the database, so a PC that is never stopped
   gets no backups (the new Health page warns after 7 days). Backups also sit on the same disk C:.
2. **Second site over Tailscale.** Tailscale is installed and signed in on this PC, but the other site's PC is
   not in the network yet, and the firewall rule from README "From another site" does not exist.
3. **Import recorded videos, fast and with real times** (the earlier proposal card). Today a video file plays
   at its own speed, loops, and every person gets *today's* time; there is no way to add old footage.

No phone alerts (the user said no). Order: smallest and safest first. One commit per step, tests, real run on
port 8001 with `tools/make_test_config.py`, README / START-STOP-GUIDE / HANDOFF / ROADMAP updated each time.

## Step 1: daily backup while the app runs

- `run.py`: new thread `_backup_loop` (like `_retention_loop`): once a day at `backup.hour` (default 2), if the
  newest backup is older than 20 h, run `tools/backup_db.py --keep <backup.keep>` as a subprocess (pg_dump via
  `docker exec` is a consistent snapshot while the app writes). PostgreSQL only. Failure → `log.error` (shows on
  the Health page through `labeler/logs.py` RecentErrors).
- `tools/backup_db.py`: options `--copy-to <folder>` (a second copy, e.g. a USB drive or another disk; keeps the
  same number) and `--verify` (restore the newest dump into a temporary database `visionlabeler_verify` in the
  same container with `pg_restore`, compare people counts with the live database, drop it). The daily run
  verifies once a week.
- Config `backup: {enabled: true, hour: 2, keep: 7, copy_to: ""}` in `labeler/config.py` + commented in
  `config.yaml`. Settings page (`labeler/settings.py` SIMPLE + `settings.html`): "Daily backup" on/off and
  "Second copy in folder" (must be an existing folder; validated like file sources in `check_source`).
- Health (`labeler/healthcheck.py` `check_backup`): warning after 2 days when daily backups are on (7 when off);
  shows the second copy and the last verify result.
- `stop.bat` unchanged (it already skips when a backup is younger than 20 h).
- Tests: due/not due by hour and age; failure logged; `--copy-to` keeps N; health thresholds; settings
  validation. Real check: needs Docker running (ask the user then), run once, verify restores.

## Step 2: second site (Tailscale), with a self-check

Most steps are done by the user on the other PC; the app helps and checks:

- Health page: new "Access from other devices" checks (`healthcheck.py`): HTTPS on, the app listening on
  `web.https_port`, at least one login account, Tailscale running and this PC's Tailscale address
  (`tailscale ip -4`, read-only), the Windows firewall rule for 8443 from 100.64.0.0/10 present
  (`Get-NetFirewallRule`, read-only), certificate fingerprint to compare. "off" when `web.host` is 127.0.0.1.
- A short plain-English guide `docs/SECOND-SITE.md` (the README steps, one action per line, what the user sees
  at each step), linked from the Health page and START-STOP-GUIDE.
- With the user, live: they add the other PC to their tailnet (they install Tailscale there themselves); they run
  the firewall command in an administrator PowerShell (I never change firewall settings); they open
  `https://<this PC's Tailscale address>:8443` on the other PC, check the fingerprint, log in as `TIXUAN` (viewer).
  I confirm from the Audit log that the login came from a 100.x address.
- Tests: each check with fakes (no Tailscale, no rule, no account, all good).

## Step 3: import recorded videos (fast, real times)

**Where:** inside the running app (the models are already loaded; a second process would load them again), as
a new admin page **Import** (`/import`, `labeler/static/import.html`, nav link `admin-only`). The user types a
file or folder path on this PC (no browser upload of big files), picks a camera name for the footage (new name,
e.g. `phone`; not a live camera's name) and mode, checks the start time of each file, and starts. One import at
a time; progress, cancel; every import in the Audit log.

**Real times:** a clock per pipeline instead of `time.monotonic()` / `datetime.now()`:
- `labeler/pipeline.py`: `self.clock` with `mono()` and `wall()`. Live = today's behavior. Import = time inside
  the video (`CAP_PROP_POS_MSEC`, right for variable-frame-rate phone videos) added to the file's start time.
  About 10 call sites (`process`, `_finish`, `_relabel`, `TrackState(... datetime.now() ...)`, zone events,
  `left_at`). Tracking, re-linking, time in view, facing time and zone dwell then all use video time.
- Start time per file, first that works, shown on the page and editable: typed by the user > the file name
  (`VID_20261009_143012`, `20261009_143012`, `IMG_…`, CCTV `…20261009143012…`) > MP4/MOV `mvhd` creation time
  (small parser, UTC → local) > file modified time minus duration. New `labeler/importer.py`.

**Fast:** `labeler/sources.py` `FrameSource(..., paced=False, stride=k)` for imports: no waiting between
frames, no looping, every k-th frame. Default stride chosen by measurement (see below), aiming at ~12 frames
per video second. The import waits while the VLM queue has more than 2 jobs, so live cameras are delayed by at
most ~2 import jobs and every imported person still gets labeled.

**What is different for imported footage:** no alerts and no alert clips (old footage must not pop up as live
alerts); no camera health checks; zones work if drawn for that camera name. Imported people are normal people:
search, Ask box, look-alikes, timeline, Reports (with their real dates), Review, Answer key.

**Retention:** old footage would be deleted within the hour by the 30-day rule. The import form has "Keep for
(days)" (default 0 = until deleted by hand), stored per camera: new column `cameras.retention_days`
(`db.NEW_COLUMNS`), used by `db.cleanup` for cameras that are not in the config. Removing an import = select its
camera on the Live page people list and Delete selected (as today).

**Camera lists:** `/api/cameras` adds recorded-only cameras (from the database) with `"live": false`; the Live
page tabs keep live cameras only; Reports and the people filter show both.

**Measure before choosing defaults:** on a test video (ask the user for a short phone video of people walking,
or their OK to download a free public test clip): people found and minutes of processing per hour of video at
stride 1, 2, 3, 4; pick the smallest stride with the same people count as stride 1. Also check one known time
(a phone video's name or metadata vs. its true recording time). Numbers go in README.

**Tests:** clock (video time drives lost-after, relink window, in-view time); start-time sources (names,
`mvhd` bytes, mtime fallback, user value wins); stride/paced source on a small generated video; VLM throttle;
no alerts for imports; retention per imported camera; `/api/cameras` live flag; page script parses; viewer 403.

## Critical files

`run.py` (backup thread, routing VLM answers to import pipelines: today `route_result` only knows the live
`pipelines` list), `tools/backup_db.py`, `labeler/healthcheck.py`, `labeler/settings.py` +
`labeler/static/settings.html`, `labeler/config.py` + `config.yaml`, `labeler/pipeline.py`, `labeler/sources.py`,
new `labeler/importer.py` + `labeler/static/import.html`, `labeler/web.py`, `labeler/db.py`, `docs/SECOND-SITE.md`.

Reuse: `_retention_loop` pattern, `Retrainer.due` (hour + age logic), `settings.check_source`,
`healthcheck._check`, `audit()` in web.py, `privacy.remove_picture`, `tools/make_test_config.py`.

## Verification

- `pytest -q` green after each step.
- Step 1: with the user's OK to start Docker: one backup made by the app's thread, `--verify` restores and the
  counts match, `--copy-to` a scratch folder; Health shows the backup line.
- Step 2: Health checks on this PC before and after the user adds the firewall rule; the other PC logs in.
- Step 3: real import on port 8001 (scratch config): page flow, progress, cancel, people with the video's real
  dates in the people list and Reports, live camera keeps labeling during the import, measured speed table.
- No leftover `python.exe run.py`; scratch files deleted; HANDOFF/ROADMAP updated; one commit per step.
