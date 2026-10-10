# Phase 10+ plan: further improvements (approved 2026-10-10)

## Context

Phases 0 and 2-9 are done: 420 tests, CI green on the user's private GitHub, daily backups, health check,
Settings page, installer, video import, person videos, photo search. The user uses one USB webcam at home (kiosk
mode) and is not sure yet which cameras they will add. This plan is saved here and linked from ROADMAP/HANDOFF. Work through it in order, one phase at a time, asking the user before
each phase starts (and before any download or install).

Measured weak spots found so far (README has the numbers): appearance on far/overhead views (wrong re-links,
look-alikes and photo search ~33-38% top-1), the retraining has not started learning (needs 30 more of the user's
own people), and nothing has ever run for days in a row under watch.

## Recommended order

### Phase 10 — Long-run check and an idle mode (small; for the current setup)
1. **24-hour soak test** on the user's real setup (with their OK; on port 8000 only if they agree, else a test copy
   on the webcam): sample every 10 min the app's memory, GPU memory, Ollama's memory, disk use, FPS, queue length,
   errors in the log (`tools/soak.py`, writes a CSV + a chart). Find leaks or slowdowns; fix what it finds.
2. **Idle mode**: when nobody has been in view for N minutes, run the detector at a few frames per second instead
   of every frame; back to full speed on the first person. Measure power with `nvidia-smi` (watts) idle before/
   after, and that no arriving person is missed (test video with long empty stretches). Config `idle:` + Settings.
3. A **restore drill** on a scratch database: restore the newest backup into a new container/database, start a test
   copy against it, check pages open (proves recovery, not only the dump).

### Phase 11 — Better labels on the user's own webcam (medium; biggest gain for the current setup)
1. Faster Answer key page: keyboard shortcuts, values pre-filled from the system's labels (confirm or change),
   "next person" without reload; measure seconds per person before/after with the user.
2. "Practice from my videos": import the user's own phone videos (Import page) as answer-key material.
3. Get the weekly retraining past its 30-person gate; then compare answer-key accuracy per attribute before/after
   (`/api/answer-key/score`, accuracy history chart). Keep only if it scores at least as well (gate exists).

### Phase 12 — When the user adds IP/outdoor cameras (decision point: ask first)
1. **Person re-identification model** (bodies, not faces) for re-linking, look-alikes, timeline and photo search on
   far views. Candidates: OSNet (torchreid, MIT code; check the training data's license), CLIP-ReID. License check
   first (Phase 1 rules). Measure with `tools/relink_check.py` on PETS09-S2L1 (MOTChallenge labels): goal 0 wrong
   joins with fewer splits than today (33 records for 19 people), look-alike top-1 above 10/26, photo search above
   6/18. Switch per mode only if it wins.
2. **More cameras** (if more than ~4): batched detection across cameras in one process; measure FPS and delay per
   camera (README "How many cameras?" has today's numbers).
3. Per-camera tuning helpers: detector size check on a sample of each camera (people found at 640/960/1280).

### Optional, only if the user asks
- **Family vs. strangers** (faces, opt-in, kept on this PC; face-model license check; strict privacy rules).
- **Phone alerts** (Telegram/ntfy/email; the user said no earlier).
- **Second site** over Tailscale (waits for the user; `docs/SECOND-SITE.md`).
- **Phase 1 licenses** before any commercial use.
- **Code health**: split `labeler/web.py` (~1,000 lines) into routers and `pipeline.py` helpers; no behaviour change,
  full suite as the safety net. Low user value; do when touching those files anyway.

## Working rules (from HANDOFF section 6-7, keep)

Test copies via `tools/make_test_config.py` on port 8001; never the user's data without asking; Edit/Write or
script files (no shell heredocs with backslashes); `git commit -F`/`-m` then `git push` (works from the shell,
`GIT_TERMINAL_PROMPT=0`); the user checks GitHub Actions (browser pane not signed in); measure before claiming a
gain, report honestly; README/HANDOFF/ROADMAP updated each step.

## Verification (each phase)

- `pytest -q` green, CI green after push (user confirms).
- Phase 10: soak CSV/chart with no growth trend over 24 h (or the leak found and fixed and re-measured); idle mode
  saves watts and misses no arriving person on the test video; the restored copy opens.
- Phase 11: seconds per answer-key person and per-attribute accuracy before/after, on the user's own labels.
- Phase 12: relink_check / look-alike / photo-search numbers before/after on PETS09-S2L1; FPS per camera.
