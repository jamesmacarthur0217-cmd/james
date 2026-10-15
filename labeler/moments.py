"""Search recordings by words (Phase 16): what a search found, as moments in the recordings to play.

A search (the Ask box's QueryAgent, or the text ranking when the vision model is off) finds people, vehicles or
animals. Each was seen on one camera from `started_at` to `ended_at`; with a few seconds before and after (`pad_s`),
that is a stretch of that camera's recordings (labeler/recorder.py). Things seen on the same camera at overlapping
times are one moment ("2 people"), not one per person, as long as it stays under a minute (MAX_JOIN_S). A moment
can span two recording files (the recorder starts a new file after a gap or at the hour): it then has two parts,
played one after the other.

Where a part starts in a file comes from the file's frame list (recordings_web.frame_list: each frame's time), so it
is exact however unevenly the camera ran:
  offset_s / end_offset_s  seconds of real time from the file's start (the viewers' stream: ?t= and ?d=)
  video_s / video_end_s    the same frames' place in the MP4, which plays at a fixed frame rate (admins' player and
                           the cut download)
What has no recording (recording off, deleted after keep_days, before recording existed) is counted, not listed.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

PAD_S = 3.0   # seconds before and after each thing found
MAX_JOIN_S = 60.0   # overlapping things are joined only while the moment stays this short (a busy camera would
                    # otherwise chain everyone into one long moment); one thing seen longer is its own moment


def _when(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def describe(item: dict) -> str:
    """A few words for one thing found: "man, 30-39, red t-shirt", "white Toyota car", "black dog"."""
    if item["type"] == "person":
        from .summary import describe_person
        return describe_person(item)
    words = [item.get("color"), item.get("make") if item.get("make") not in (None, "unknown") else None, item.get("kind")]
    return " ".join(w for w in words if w) or item["type"]


def _scenes(items: list[dict], pad_s: float, now: datetime) -> list[dict]:
    """Things on the same camera with overlapping (padded) times, joined: [{camera, start, end, items}]."""
    spans = []
    for it in items:
        start = _when(it.get("started_at"))
        if start is None or not it.get("camera"):
            continue
        end = _when(it.get("ended_at")) or now   # still in view
        spans.append((it["camera"], start - timedelta(seconds=pad_s), max(end, start) + timedelta(seconds=pad_s), it))
    scenes: list[dict] = []
    for camera, start, end, it in sorted(spans, key=lambda s: (s[0], s[1])):
        last = scenes[-1] if scenes else None
        joins = last and last["camera"] == camera and start <= last["end"]
        if joins and (max(last["end"], end) - last["start"]).total_seconds() <= MAX_JOIN_S:
            last["end"] = max(last["end"], end)
            last["items"].append(it)
        else:
            scenes.append({"camera": camera, "start": start, "end": end, "items": [it]})
    return scenes


def _parts(db, base: Path, scene: dict, fps_default: float, frames: dict) -> list[dict]:
    """The recorded stretches of one scene, in time order (frames: frame lists already read, by recording)."""
    from .recordings_web import frame_index, frame_list
    parts = []
    for rec in db.list_recordings(scene["camera"], scene["start"], scene["end"]):
        if rec["id"] not in frames:
            frames[rec["id"]] = frame_list(base / rec["path"])
        header, times, _ = frames[rec["id"]]
        if not times:
            continue
        rec_start = datetime.fromisoformat(rec["started_at"])
        a = max(0.0, (scene["start"] - rec_start).total_seconds())
        b = (scene["end"] - rec_start).total_seconds()
        if a > times[-1] or b < times[0]:
            continue   # the file holds nothing of this stretch (it was cut short, or is still being written)
        i, j = frame_index(times, a), frame_index(times, b)
        if times[j] > b and j > i:
            j -= 1     # the last frame at or before the end
        fps = float(header.get("fps") or fps_default)
        parts.append({"recording": rec["id"], "offset_s": round(times[i], 2), "end_offset_s": round(times[j], 2),
                      "video_s": round(i / fps, 2), "video_end_s": round((j + 1) / fps, 2),
                      "start": (rec_start + timedelta(seconds=times[i])).isoformat(timespec="seconds"),
                      "end": (rec_start + timedelta(seconds=times[j])).isoformat(timespec="seconds")})
    return parts


def moments(db, items: list[dict], base: Path, fps: float, pad_s: float = PAD_S, now: datetime | None = None) -> dict:
    """{"moments": newest first, "found": things found, "on_video": of them in a moment, "not_on_video": the rest}.

    items: people, vehicles and animals as the database lists them, each with "type" (person / vehicle / animal).
    """
    now = now or datetime.now()
    out, on_video, frames = [], 0, {}
    for scene in _scenes(items, pad_s, now):
        parts = _parts(db, base, scene, fps, frames)
        if not parts:
            continue
        on_video += len(scene["items"])
        things = sorted(scene["items"], key=lambda it: it["started_at"])
        out.append({"camera": scene["camera"], "start": parts[0]["start"], "end": parts[-1]["end"],
                    "length_s": round(sum(p["end_offset_s"] - p["offset_s"] for p in parts), 1), "parts": parts,
                    "picture": next((it.get("crop") for it in things if it.get("crop")), None),
                    "items": [{"type": it["type"], "id": it["id"], "crop": it.get("crop"), "text": describe(it),
                               "started_at": it["started_at"], "labels_differ": bool(it.get("labels_differ"))}
                              for it in things]})
    out.sort(key=lambda m: m["start"], reverse=True)
    found = len(items)
    return {"moments": out, "found": found, "on_video": on_video, "not_on_video": found - on_video}
