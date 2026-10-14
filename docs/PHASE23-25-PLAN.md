# Phases 23-25 (+16) plan: small fixes, "was this right?", "while you were away", search recordings (approved 2026-10-14; Phases 23-24 done, 25 started)

## Context

Phases 19-22 (Home/Away, loitering, animals, scene questions) are done. Their README sections all end with "not measured
on your own cameras", and three small problems were noticed on the way. The user chose (2026-10-14) these phases, in
this order: **23 small fixes, 24 "was this right?" on their own cameras, 25 "while you were away" summary, then 16
search recordings by words** (approved 2026-10-12, [PHASE16-PLAN.md](PHASE16-PLAN.md)). **Not chosen: phone alerts**
(the user keeps alerts on the pages only, as in Phase 7; don't offer again unless asked).

How to work (as before): ask the user before each phase and each step; one commit per step; tests, the full suite
green, `gh run list` green after pushing (tests must not need git-ignored files: HANDOFF pitfall 23); a real run on a
test copy (port 8001, `tools/make_test_config.py`) for anything touching the pipeline or pages; schema changes checked
on a scratch PostgreSQL database first (Docker may be off after a restart: start it); README/HANDOFF/ROADMAP.

Before Phase 24 the user restarts their own app (stop.bat, start.bat) so it runs Phases 13-22 and collects real footage.

## Phase 23: Small fixes

1. **Done 2026-10-14.** Settings page: the "unsaved changes" mark reacts to every field (today only the cameras table calls `changed()`);
   one input/change listener on the form. Test in `tests/test_pages.py` style plus a browser check.
2. **Not needed (checked 2026-10-14)**: the app writes to the real console window (the supervisor passes it on), which
   shows any text since Python 3.6; output sent to a file gets `\u5927`-style codes, no error; the log file is UTF-8 and
   keeps the text (tested). Planned as: Non-English text in the console: the console handler (`labeler/logs.py`) writes with the Windows code page, so a
   scene question or camera name in Chinese gives "--- Logging error ---" there (the log file is UTF-8 already).
   Write the console as UTF-8 with replacement (`sys.stderr.reconfigure(errors="replace")` or a handler with its own
   stream). Test with a Chinese question through the scene runner's log line.
3. **Done 2026-10-14** (one real transaction; a failure leaves the old table and the app starts anyway). **Phase 23 is
   done.** Old SQLite databases: `alert_events.track_id` / `rule_id` stay NOT NULL there (SQLite cannot drop NOT NULL), so
   vehicle, animal and scene alerts are skipped with a warning. At start, rebuild that one table on SQLite when it
   has the old layout (create the new table, copy the rows, swap, in one transaction; keep ids). Test with the old
   layout the tests already build (`test_old_sqlite_database_skips_vehicle_alerts...`, which then stores the alert).

## Phase 24: "Was this right?" on your own cameras

The answer key measures people's labels; nothing measures the newer detections on the user's cameras.
1. **Done 2026-10-14** (column `verdict`, not the SQL word `right`). Table `checks` (what: vehicle / animal / fall / loitering alert / scene answer, its id, right yes/no/not sure, by,
   when) and an admin **Check** page: recent items one at a time with their picture (and clip when there is one),
   keyboard Y / N / S like the Answer key page; mixed so every kind gets checked; items already checked skipped.
2. **Done 2026-10-14.** Results: per kind and camera, "right N of M" (and for vehicles/animals the kind: "it was a cat, not a dog" as an
   optional correction); shown on the Check page, a Health line ("checked: ... right"), and in Reports.
3. **Done 2026-10-14** (notes only: 10+ answered and under 70% right). Use the answers where measured gains are possible, only if measured: e.g. a scene question that is often wrong is
   flagged on the Alerts page; an animal kind often wrong on one camera can be turned off there. No automatic retuning.
4. **Done 2026-10-14** (README "Checking on your own cameras"). **Phase 24 is done.** Docs; with the user: a week of checks gives the first real numbers for README sections 17-22.

## Phase 25: "While you were away" summary

1. **Done 2026-10-14.** `labeler/summary.py`: for a period (from the last switch to Away until now, or "last night", "today", a chosen
   range): people (count, pictures of the first few), vehicles, animals, alerts (all kinds), scene answers that
   changed, camera problems; from the existing tables, no new data.
2. **Done 2026-10-14** (also for viewers, as the API). A **Summary** page (admins; viewers get the people-free parts and blurred pictures, as elsewhere) with a timeline
   per camera and the pictures; links to the Recordings page at each moment.
3. When someone switches Away -> Home, the button offers "What happened while you were away?" (a link, never
   automatic); the Live page's Today tiles link to it too.
4. Tests (periods over midnight, Away periods from the `arming` table, viewer view), real run, docs.

## Phase 16: Search recordings by words

As approved in [PHASE16-PLAN.md](PHASE16-PLAN.md): moments from all cameras for a typed description, on the
Recordings page; vehicles and animals ("white car this morning", "dog at night") added to it now that they exist.

## Critical files

`labeler/static/settings.html`, `labeler/logs.py`, `labeler/db.py` (SQLite rebuild, `checks`), new `labeler/checks.py`
and `labeler/static/check.html`, new `labeler/summary.py` and `labeler/static/summary.html`, `labeler/static/session.js`
(the Away -> Home link), `labeler/healthcheck.py`, `labeler/reports.py`, `labeler/web.py`, `labeler/accounts.py`;
Phase 16's files as in its plan. Tests: `tests/test_checks.py`, `tests/test_summary.py`, plus fixes in existing ones.

## Verification

- Unit tests for each table, endpoint and rule; full suite green; GitHub run green.
- PostgreSQL: new tables on a scratch database; the SQLite rebuild on a copy of an old-layout SQLite file.
- Real runs on a test copy in the browser pane (Check page with the keyboard, Summary after Away -> Home).
- With the user: real checks on their cameras (Phase 24) are the point of this round.
