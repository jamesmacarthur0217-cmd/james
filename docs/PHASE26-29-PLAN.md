# Phases 26-29 plan (+16): real-camera week, scene question areas, maintenance, more cameras (approved 2026-10-14, not started)

## Context

Today (2026-10-14) Phases 19-25 were finished and pushed (713 tests, GitHub green): Home/Away, loitering, animals,
scene questions, small fixes, the Check page ("was this right?"), and the Summary page. What is left of the current
plan is **Phase 16, search recordings by words** (approved 2026-10-12, `docs/PHASE16-PLAN.md`).

Proposed and approved by the user on 2026-10-14, to start after Phase 16; the user picks which phases to keep and
the order at the start of the session that begins them.

Important facts for the next session (all in `docs/HANDOFF.md`):
- The user's own app has **not run Phases 13-25 yet**: it needs a restart (`stop.bat`, `start.bat`), which updates the
  PostgreSQL database by itself (every change was checked on scratch databases). Several phases below need real
  footage from it.
- Not to be offered again unless the user asks: phone alerts (declined twice), backups of pictures/videos (only if a
  second disk), family vs strangers (not chosen), commercial licenses (Phase 1, only if use becomes commercial).
- Way of working: ask before each phase and each step; one commit per step; full suite green; `gh run list` green
  after pushing (pitfall 23); real run on a test copy (port 8001, `tools/make_test_config.py`); schema changes checked on
  a scratch PostgreSQL database (Docker may be off after a restart); README/HANDOFF/ROADMAP.

## Proposed order

### Phase 16: Search recordings by words (already approved, next)

As in `docs/PHASE16-PLAN.md`: type "man with a backpack yesterday evening" and get video moments from all cameras to
play on the Recordings page (admins and viewers), built on `labeler/agent.py` (`QueryAgent.run`), `db.list_recordings`,
`labeler/recordings_web.py` (`frame_list`, `frame_index`, stream). New since that plan: vehicles and animals already
answer Ask box questions (`agent.vehicle_query`, `agent.animal_query`), so "white car this morning" and "dog at night"
give moments too (their `started_at`/`ended_at` like people's).

### Phase 26: Real-camera week (with the user; needs their restarted app)

The point of Phases 17-25 is real use, but every README section still says "not measured on your cameras".
1. With the user: restart, Health page all green, recording on, a scene question or two, a week of Check page answers
   (`labeler/checks.py` results).
2. Read the numbers together (Check page, Health "checked by you", Reports) and change only what is measured to help,
   per camera: e.g. turn animals/vehicles off where the checks say they are mostly wrong (Settings column), a scene
   question's wording, `falls.after_s`, a loitering area's size. Each change re-checked the next days.
3. Write the real numbers into the README next to the public-video ones.

### Phase 27: Scene questions: only part of the picture, and only when Away

1. A scene question can look at **one area** (a zone drawn on the Live page, `labeler/zones.py`) instead of the whole
   picture: the area is cut out (with a margin) before it goes to the vision model (`labeler/scene.py`
   `SceneRunner.tick`, `vlm.PROMPT_SCENE`). A gate is a small part of a wide picture; measure whole vs cut-out with
   `tools/scene_check.py` (lying on the floor with the person's area; a car in a parking area on MOT16-04/13) and keep
   it only if it is better.
2. A scene question can be limited like alert rules: **only when Away** and **hours** (reusing `alerts.in_hours` and
   the Arming state, `labeler/arming.py`): "Is the gate open?" at night only, so no alert while the family is in and out.
3. Alerts page box: area and limits; tests; docs.

### Phase 28: Maintenance

1. GitHub: the `ubuntu-latest` runner moves to Ubuntu 26 from 2026-10-19: check the next runs after that date; fix if
   anything breaks (`.github/workflows/tests.yml`).
2. The test warning "Using httpx with starlette.testclient is deprecated": move to what Starlette asks for, if it is a
   small change; otherwise note it.
3. Security updates: `pip-audit` as in Phase 8; Dependabot PRs if the user turned them on.
4. The untracked `remote-camera.zip` in the project folder is the old (Phase 14) version: ask the user whether to
   delete it or make it again from the current `remote-camera/` folder.

### Phase 29: More cameras (waits until the user adds cameras)

Batching the detector across cameras or one process per camera (README "How many cameras?"), per-camera tuning, and
re-measuring re-identification on real outdoor footage (Phase 12, `tools/relink_check.py --reid`). Only when there are
more than ~3 cameras.

## Recommended order

16 -> 26 (as soon as the user has restarted and run a few days; can overlap with 16) -> 27 -> 28 (after 2026-10-19) ->
29 (when cameras are added). The user picks which to keep and the order at the start of the next session.

## Verification

- Docs-only commit: control-character scan on the changed files (pitfall 22), GitHub run green.
- Each future phase: its own tests, full suite green, real run on a test copy, PostgreSQL scratch checks for schema
  changes, numbers measured before claiming a gain (as in all earlier phases).
