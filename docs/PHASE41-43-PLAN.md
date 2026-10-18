# Phases 41-43 plan: safe-update test, recordings kept around alerts, the usual day as a chart (proposed 2026-10-19)

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
2. **Start-up message**: `run.py` logs "updating the database..." before `Database(...)` and the time taken after it
   when anything was added, so a slow first start is not a silent window. The Update button (Phase 35) and the
   Health page say when the last upgrade check ran and its result (stored in `data/upgrade_check.json`).
3. Real run: the check on the user's real database (read only; the user starts it, or says yes to it running),
   then docs. If it passes, the user restarts their app (stop.bat, start.bat) and opens the First-run check.

## Phase 42: Recordings kept around alerts

Recordings are deleted after 3 days (`recording.keep_days`); the week's report then says "not kept", and an alert's
own clip is only 20 s.
1. **Keep the stretch around each alert**: when `recorder.prune` is about to delete a file with alerts in it, it
   first cuts the minutes around each alert (`recording.alert_before_s` 60, `alert_after_s` 120; FFmpeg `-c copy`, at
   key frames, no re-encoding, as Download does) into a kept part (a `recordings` row marked `kept`, with its own
   frame list and blurred stream like any recording), kept `recording.alert_keep_days` (30) and counted within
   `max_gb`. Overlapping alerts make one part.
2. **Pages**: the Recordings page shows kept parts on their day (a different color on the day bar, "kept: alert"),
   alerts' "▶ Recording" finds them, the week report counts their minutes ("kept around alerts"), Health's storage
   line lists them; Settings gets the two times and the days.
3. Tests (cutting, overlaps, deleting, max_gb), a real run on a test copy with an alert in a recording made "old",
   docs; measure the size of a kept part.

## Phase 43: The usual day as a chart

1. **API**: `/api/routine` (admins and viewers): per kiosk camera the learned shares per hour for weekdays and the
   weekend (`routine.learn`), the days counted, still learning or not, today's quiet stretch so far (from
   `UnusualWatch`) and whether "Unusual quiet" fired today.
2. **Chart** on the Health page under the line "usual day" (and linked from the Alerts page's rule): 24 hourly bars per
   kind of day, the 80% line, today's quiet stretch marked; hover for the numbers, a table view, both color modes,
   phone width. Follows the dataviz skill (read it before writing the chart).
3. Tests, browser check (admin, viewer path, phone width, dark mode), docs.

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
