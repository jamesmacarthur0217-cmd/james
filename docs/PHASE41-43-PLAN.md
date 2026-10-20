# Phases 41-43 plan: safe-update test, recordings kept around alerts, the usual day as a chart (approved 2026-10-19; Phases 41-42 done)

## Context

On 2026-10-19 the Phases 36-40 plan was finished and Phase 28 closed (the tests ran once on ubuntu-26.04: the same
result as 24.04). The user chose this round from six ideas, in this order: **41 safe-update test, 42 keep recordings
around alerts longer, 43 the usual day as a chart**. Offered and not chosen this time: PC history chart, weekly
report saved by itself, a week in one minute (they may come back later; ask before offering them again). Still open
outside the plan: Phase 26 (real-camera week, once the user's app runs the new version), Phase 29 (more cameras), the
PostgreSQL check of the `routine_days` table (Phase 39; Docker was off).

Why 41 first: the user's own app last ran Phase 13-15 code. Phase 36 found, only by building a database with the old
code, that its next start would have hung for ever (fixed, pitfall 30). Their real database may hold other surprises
that made-up layouts do not.

## Phase 41: Safe-update test on a copy of the user's database

1. **`tools/upgrade_check.py`**: makes a throwaway database on the same PostgreSQL server from a fresh dump of the
   real one (`pg_dump` inside the container, as `tools/backup_db.py` does; the real database is only read, also while
   the old app is running), opens it with today's code in a child process with a time limit (5 min), and reports:
   how long the update took, the tables and columns it added (`Database.created_tables` / `added_columns`), the old
   constraints removed, the row counts before and after (nothing lost), and what What's new would list
   (`whatsnew.guess_before`). Then it drops the copy. A hang or an error is reported with the step it was in.
   Builds on `tools/restore_drill.py` (restore into a test database, checks, drop).
   **Done 2026-10-19**: on a pre-Phase-13 database: 0.9 s, 10 tables, 5 columns, old rules removed, nothing lost;
   with the start-up code from before the Phase 36 fix it reported the hang after 30 s with the waiting lock.
2. **Start-up message**: `run.py` logs "updating the database..." before `Database(...)` and the time taken after it
   when anything was added, so a slow first start is not a silent window. The Update button (Phase 35) and the
   Health page say when the last upgrade check ran and its result (stored in `data/upgrade_check.json`).
   **Done 2026-10-19**, and more: Update now runs the test with the new version before restarting and puts the
   running version back when it fails; while the start-up update is slow, the console says so every 30 s.
3. Real run: the check on the user's real database (read only; the user starts it, or says yes to it running),
   then docs. If it passes, the user restarts their app (stop.bat, start.bat) and opens the First-run check.
   **Done 2026-10-19**: 1.1 s, 154 people, 9 tables and 5 columns added, nothing lost. **Phase 41 is done.**

## Phase 42: Recordings kept around alerts

Recordings are deleted after 3 days (`recording.keep_days`); the week's report then says "not kept", and an alert's
own clip is only 20 s.
1. **Keep the stretch around each alert**: when `recorder.prune` is about to delete a file with alerts in it, it
   first cuts the minutes around each alert (`recording.alert_before_s` 60, `alert_after_s` 120; FFmpeg `-c copy`, at
   key frames, no re-encoding, as Download does) into a kept part (a `recordings` row marked `kept`, with its own
   frame list and blurred stream like any recording), kept `recording.alert_keep_days` (30) and counted within
   `max_gb`. Overlapping alerts make one part.
   **Done 2026-10-19** (the frames are copied with OpenCV and the same FFmpeg writer, not cut with `-c copy`, so
   each frame matches its frame list exactly). Found on the real run: the clock on all recordings looked doubled
   (outline thicker than the text); fixed.
2. **Pages**: the Recordings page shows kept parts on their day (a different color on the day bar, "kept: alert"),
   alerts' "▶ Recording" finds them, the week report counts their minutes ("kept around alerts"), Health's storage
   line lists them; Settings gets the two times and the days.
   **Done 2026-10-19**, and step 3 within steps 1-2 (tests, the real run with the encoder, ~25 MB for a full
   3-minute part of a busy scene). **Phase 42 is done.**
3. Tests (cutting, overlaps, deleting, max_gb), a real run on a test copy with an alert in a recording made "old",
   docs; measure the size of a kept part.

## Phase 43: The usual day as a chart

Reuse: `Routine.get(camera)` (shares per kind and hour, `asked`, `days`, `learning`, `per_day`), `Routine.state()`,
`routine.hour_ranges`, `UnusualWatch.quiet_since()` / `last_seen()` (`labeler/unusual.py`), the Reports page's hour
bar chart (`labeler/static/reports.html` `drawBars`, `.viz` tokens, "Show as a table"), `app.state.routine` /
`app.state.unusual` (made in `run.py`).

1. **API `GET /api/routine`** (`labeler/web.py`; add to `accounts.VIEWER` GET list): per kiosk camera: `learning`,
   `days` (weekday / weekend counts), `min_days`, `share` and `asked` per kind (24 values each), `usual_share` (0.8)
   and `quiet_share` (0.97), today's kind, today's quiet stretch from `UnusualWatch.quiet_since` (start time, whole
   hours so far, the history share for it via `Routine.stretch`), and whether "Unusual quiet" fired today
   (`UnusualWatch.fired`). `{"cameras": []}` without kiosk cameras or with `unusual.enabled` off. Tests with the
   made-up households of `tests/test_routine.py` (`_regular`, `_people`).
   **Done 2026-10-20**: `Routine.chart` (also `need` and `usual_hours` per kind), `UnusualWatch.today` (since, whole
   hours, seen on n of m days, `unusual`, `watched` (Home/Away), `fired`); `tests/test_usual_day.py`; 891 tests.
2. **Chart on the Health page** (`labeler/static/health.html`, under the "usual day" line, anchor `#usual`); read the
   dataviz skill first (its validator runs from a copy named `validate_palette.js` in a folder with
   `{"type":"module"}` in package.json): per camera two small 24-bar charts (weekday, weekend; today's kind first),
   one hue (categorical slot 1, `#2a78d6` light / `#3987e5` dark), a dashed 80% line labelled "usual", hours with too
   few days drawn hollow, today's quiet stretch as a gray band with a label; hover tooltip ("08:00-09:00: seen on 38
   of 40 weekdays"); "Show as a table"; "still learning (5 of 14 days)" instead of bars while learning; ResizeObserver
   redraw (pitfall 12); dark and light; phone width. Link "see the usual day" from the Alerts page's "Unusual quiet"
   rule row to `/health#usual`.
   **Done 2026-10-20**: Health panel `#usual` (`drawUsual`, `drawDay`, `usualTable`; the "usual" label sits in a
   right margin so it never covers the 23:00 bar on a phone; the chart loads after the checks so `#usual` lands
   right); `/api/routine` also gives `weeks`. Checked in the browser with made-up data (page-only server): dark,
   light, phone width (no sideways scroll), hover, table, the Alerts link; 892 tests.
3. **Tests, browser check, docs**: page test (ids, tokens); a test copy (`tools/make_test_config.py`) whose database is
   filled by a small script with 8 weeks of made-up people for a kiosk `cam1` (as `tests/test_routine.py` `_regular`)
   and run on an empty picture; browser check (admin and the viewer path, dark and light, phone width); README
   "Unusual quiet" section; HANDOFF; What's new entry `p43` (and fill in `p42`'s commit `6d7fe99`); this plan and the
   roadmap: "Phase 43 done; the 41-43 plan is finished".

## Way of working (unchanged)

Ask before each phase and each step; one commit per step; full suite green; push and check `gh run list`; real run on
a test copy (port 8001, `tools/make_test_config.py`), never the user's app (port 8000) or data except where a step says
"read only" and the user agreed; schema changes checked on a scratch PostgreSQL database (start Docker for it);
measure before claiming a gain; edit with the Write/Edit tools or Python scripts written with Write (pitfalls 1, 22,
25); tests never depend on the real date (pitfall 31); short, simple answers for the user. Each phase adds its entry
to `labeler/whatsnew.py`.

## Verification

- Each phase: its tests, the full suite (≥ 861), a real run on a test copy, browser check where pages change.
- Phase 41: the check run on a database built by the pre-Phase-13 code (it must report the 9 tables / 5 columns found
  in Phase 36) before it is run on the user's real one.
