# Phase 16 plan: search recordings by words (approved 2026-10-12, not started)

## Context

Recording (Phase 13) keeps video while people are in view; the Recordings page plays a camera's day from a bar.
Finding *something* in it still means scrolling the bar, or searching people on the Live page and clicking
"▶ Recording" one person at a time. The user chose (2026-10-12): **type a description** ("man with a backpack
yesterday evening") **and get a list of video moments from all cameras to play**, for admins and viewers.

Almost everything exists: `labeler/agent.py` turns a question into a validated plan (time range, camera, vocabulary
conditions, free-text appearance) and returns matching people (`QueryAgent.run`, POST `/api/ask`); `/api/search`
ranks people by a text description (SigLIP 2) when the vision model is off; `db.list_recordings(camera, start, end)`
and `labeler/recordings_web.py` (`frame_list`, `frame_index`, `/recordings/file/{id}`, `/recordings/stream/{id}`)
find and play any moment. This phase connects them.

## Step 1: moments (`labeler/moments.py`, API)

- `moments(db, people, rec_folder, pad_s=3)`: for each person found, the recordings of their camera overlapping
  `[started_at - pad, ended_at + pad]`; a moment = camera, start, end, recording id, offset in it (`frame_index` on
  the frame list, so it is exact however unevenly the camera ran), the person(s), their picture and labels. People
  in the same scene (same camera, overlapping times) become **one moment** listing all of them; a person whose
  visit spans two files gets both parts in order. People without a recording are counted, not listed
  ("12 people match, 9 on video").
- POST `/api/recordings/search {question}`: `QueryAgent.run` (limit 100) when the vision model runs, else the text
  ranking of `/api/search` (no time words then: said on the page); returns the agent's explanation ("Looked for:
  …"), the moments newest first, and the counts. Audited ("search recordings", the question). Viewers allowed
  (add to `accounts.VIEWER`); their pictures come blurred as everywhere (`/crops/`), their playback stays the
  blurred stream.
- Stream end: `/recordings/stream/{id}?t=&d=` stops after `d` seconds, so a viewer's moment ends with the moment.

## Step 2: the Recordings page

- A search box above the day bar: "Find in recordings: e.g. *woman in a red top this morning*". Results: picture,
  camera, day and time, length, label chips, "2 people"; click → play from the moment's start (admins: the file,
  seeking as now; viewers: the stream) and stop at its end; **Play all** plays the moments one after another;
  admins also **Download** a moment (the existing cut download with start/end).
- The day bar and list jump to the moment's day and camera, so the user sees it in context.
- Plain messages: nothing found / found but not on video (recording off, older than `keep_days`, or the person
  was seen before recording existed) / the vision model is off, so only appearance words work.

## Critical files

New `labeler/moments.py`; `labeler/recordings_web.py` (search endpoint, stream `d`), `labeler/accounts.py`
(viewer allowlist), `labeler/static/recordings.html` (search box, results, play-all); reuse `labeler/agent.py`
(`QueryAgent.run`, `explain`), `db.list_recordings`, `recordings_web.frame_list/frame_index`, the `/api/search`
text ranking in `labeler/web.py`, `privacy`/`blur_for` as now. README (Recordings section), HANDOFF/ROADMAP, plan file
saved as `docs/PHASE16-PLAN.md`.

## Verification

- Tests (`tests/test_moments.py`, `tests/test_recordings_web.py` patterns with small mp4v files and a fake agent):
  merging people of one scene, a visit over two files, people without video counted, offsets from the frame list,
  newest first, viewer access (search yes, file/download no), the audit entry, stream stops after `d` seconds.
- Real run on a test copy (port 8001): a MOT16 clip as the camera with recording and the vision model on; ask
  "woman in a black top", "people with a backpack in the last hour"; check the moments play at the right second in
  the browser pane (admin and as a viewer), Play all, Download; measure the search time.
- With the user: a few searches on their own recordings once their app has recorded with the new code.
