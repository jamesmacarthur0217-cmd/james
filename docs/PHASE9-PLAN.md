# Phase 9 plan: small fixes (approved by the user's choice "Small fixes only", 2026-10-10)

## Context

Phases 0 and 2-8 are done (418 tests, pushed to the user's private GitHub). The user is not sure yet which
cameras they will add and chose "small fixes only" for the next round. These are the small, real issues found
during Phases 6-8, each with a test where it makes sense, the full suite green, one commit each, pushed.

## Fixes

1. **People list redraws every 3 s even when nothing changed** (`labeler/static/index.html`, the
   `setInterval(... refresh ..., 3000)`): buttons and checkboxes are replaced under the mouse, so a click can miss
   (it happened while testing the ▶ Video button). Redraw only when the list's data changed (compare the JSON of
   the people shown); same result, no flicker.
2. **Delete message counts videos as "picture files"** (`index.html` ~line 284 and `/api/tracks/delete`): say
   "files (pictures and videos)". Keep the `pictures_removed` field name (nothing else depends on the wording).
3. **Test copies read the real backups folder**: `labeler/backups.py` always uses `data/backups`, so a test app's
   Health page and Today strip showed the user's real backup age. Add `backup.dir` (default `data/backups`), used by
   `backups.folder()`, `tools/backup_db.py` (`--dir`, default from config) and the health check;
   `tools/make_test_config.py` points it into the scratch folder. Test: a test config never touches data/backups.
4. **GitHub Actions versions**: `actions/checkout@v4` and `actions/setup-python@v5` are old majors (latest: v7 for
   both); old ones run on Node versions GitHub is retiring, which would break CI one day. Read both v7 release notes
   for breaking changes, update `.github/workflows/tests.yml`, push, and ask the user to confirm the run is green.
5. **`tools/relink_check.py` tidy**: its help mentions a `--sheet` option that does not exist (remove the line) and
   it imports the age model for nothing (remove).
6. **Offline installer, database image** (the last untested part): with the user's OK to start Docker Desktop,
   `tools/pack_offline.ps1 -SkipWheels -SkipModels` builds only the image (`docker save`, ~0.4 GB into a temp folder),
   check `docker load` reads it back (`docker load -i` of the file into the same Docker is a no-op for an existing
   image, so verify with `docker image inspect` on the saved tarball's manifest instead), and run
   `install.bat -CheckOnly` with Docker running to exercise the database and account checks (read-only). Delete the
   temp files.
7. **Docs**: HANDOFF section 2/8 and ROADMAP status; README test count.

Not included (need a package install, or are not real problems): the Starlette test warning "install httpx2"
(only a warning in tests; would need a new package), pushed commit-message typos (history is not rewritten).

## Verification

- `pytest -q` green after each fix; page scripts parse (tests/test_pages.py).
- Fix 1: in the browser pane on port 8001 (scratch config), the list is not redrawn while unchanged (a marker on a
  DOM node survives 10 s), and still updates when a new person appears.
- Fix 3: a scratch run's Health page shows no backup from data/backups; `data/backups` untouched.
- Fix 4: the first Actions run after the push is green (the user checks; the browser pane is not signed in).
- Fix 6: the image tarball exists and lists `pgvector/pgvector:pg17`; `-CheckOnly` reports the database running
  and the account count, changing nothing.
- No leftover `python.exe run.py`; scratch files deleted; HANDOFF/ROADMAP updated; commits pushed.
