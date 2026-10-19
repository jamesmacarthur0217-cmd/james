"""Recordings kept around alerts (Phase 42 step 1): the stretches, the copied parts (frames, frame list, table row),
pruning (kept parts last longer; normal recordings go first when the disk is full), the time-lapse rule, off."""
import copy
import json
from datetime import datetime, timedelta

import cv2
import numpy as np

from labeler import recorder, timelapse
from labeler.config import DEFAULTS
from labeler.db import AlertEvent

T0 = datetime(2026, 10, 5, 14, 0, 0)   # a recording from 14:00:00, 30 s at 10 frames a second


class CvWriter:
    """FFmpegWriter's interface on OpenCV's own writer (no FFmpeg on GitHub)."""

    def __init__(self, path, size, fps=10):
        self.path = path
        self.w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame):
        self.w.write(frame)

    def close(self):
        self.w.release()
        return True, ""


def _cfg(tmp_path, **rec):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["recording"].update({"dir": str(tmp_path / "rec"), "alert_before_s": 2, "alert_after_s": 3, **rec})
    return cfg


def _recording(db, cfg, start=T0, seconds=30.0, fps=10):
    """A real small video whose frame i has brightness i % 250, its frame list with one head box, and its row."""
    rel = f"cam1/{start:%Y-%m-%d}/{start:%H%M%S}.mp4"
    path = recorder.folder(cfg) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    n = int(seconds * fps)
    lines = [json.dumps({"w": 160, "h": 120, "fps": fps})]
    for i in range(n):
        w.write(np.full((120, 160, 3), i % 250, np.uint8))
        lines.append(json.dumps([round(i / fps, 2), [[10, 10, 40, 40 + i]]]))
    w.release()
    path.with_suffix(".jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rec_id = db.add_recording("cam1", rel, start, "mp4v")
    db.update_recording(rec_id, frames=n, size_bytes=path.stat().st_size, ended_at=start + timedelta(seconds=seconds))
    return db.get_recording(rec_id)


def _alert(db, camera, at, message="Red top"):
    rule = db.add_rule(message, [["upper_color", "red"]])["id"]
    e = db.record_alert(rule, db.create_track(camera, 1, at), "cam1", message, repeat=True)
    with db.Session.begin() as s:
        s.get(AlertEvent, e["id"]).created_at = at


def test_stretches_are_joined_and_kept_inside_the_file():
    assert recorder.alert_stretches([5, 8, 25], 29.9, 2, 3) == [(3, 11), (23, 28)]
    assert recorder.alert_stretches([1, 29], 29.9, 2, 3) == [(0, 4), (27, 29.9)]
    assert recorder.alert_stretches([10, 17], 29.9, 2, 3) == [(8, 20)]   # 5 s apart or less: one part
    assert recorder.alert_stretches([], 29.9, 2, 3) == []


def test_parts_are_copied_with_their_frames_and_frame_list(db, camera, tmp_path):
    cfg = _cfg(tmp_path)
    rec = _recording(db, cfg)
    for s in (5, 8, 25):
        _alert(db, camera, T0 + timedelta(seconds=s))
    _alert(db, camera, T0 + timedelta(minutes=5))   # outside this recording
    ids = recorder.keep_around_alerts(db, cfg, rec, writer_factory=lambda p, size: CvWriter(p, size))
    parts = [db.get_recording(i) for i in ids]
    assert [p["path"] for p in parts] == ["cam1/2026-10-05/140003-alert.mp4", "cam1/2026-10-05/140023-alert.mp4"]
    assert [p["kept"] for p in parts] == ["alert", "alert"] and [p["frames"] for p in parts] == [81, 51]
    assert parts[0]["started_at"] == "2026-10-05T14:00:03" and parts[0]["ended_at"].startswith("2026-10-05T14:00:11")
    header, rows = recorder.frame_rows(recorder.folder(cfg) / parts[0]["path"])
    assert header["kept"] == "alert" and header["from"] == rec["path"]
    assert rows[0] == (0.0, [[10, 10, 40, 70]]) and rows[-1][0] == 8.0   # times from the part's start, boxes kept
    # The frames copied are the original's frames 30-110 (3 s to 11 s), in order: compared without a second encoding.
    got = []

    class Keep(CvWriter):
        def write(self, frame):
            got.append(frame.copy())
            super().write(frame)

    rec2 = _recording(db, _cfg(tmp_path / "again"))
    recorder.keep_around_alerts(db, _cfg(tmp_path / "again"), rec2, writer_factory=lambda p, size: Keep(p, size))
    cap = cv2.VideoCapture(str(recorder.folder(_cfg(tmp_path / "again")) / rec2["path"]))
    original = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        original.append(f)
    cap.release()
    assert len(got) == 81 + 51 and np.array_equal(got[0], original[30]) and np.array_equal(got[80], original[110])
    assert np.array_equal(got[81], original[230])


def test_prune_keeps_parts_and_they_last_longer(db, camera, tmp_path):
    cfg = _cfg(tmp_path, keep_days=3, alert_keep_days=30, max_gb=100)
    rec = _recording(db, cfg)
    _alert(db, camera, T0 + timedelta(seconds=10))
    w = lambda p, size: CvWriter(p, size)   # noqa: E731
    assert recorder.prune(db, cfg, now=T0 + timedelta(days=4), writer_factory=w) == 1
    left = db.list_recordings("cam1")
    assert [r["kept"] for r in left] == ["alert"] and not (recorder.folder(cfg) / rec["path"]).exists()
    assert recorder.prune(db, cfg, now=T0 + timedelta(days=20), writer_factory=w) == 0   # 30 days for kept parts
    assert recorder.prune(db, cfg, now=T0 + timedelta(days=31), writer_factory=w) == 1
    assert db.list_recordings("cam1") == [] and not list((recorder.folder(cfg) / "cam1").rglob("*.mp4"))


def test_a_full_disk_deletes_normal_recordings_before_kept_parts(db, camera, tmp_path):
    cfg = _cfg(tmp_path)
    a = _recording(db, cfg, start=T0)
    _alert(db, camera, T0 + timedelta(seconds=10))
    recorder.prune(db, cfg, now=T0 + timedelta(days=4), writer_factory=lambda p, size: CvWriter(p, size))
    kept = db.list_recordings("cam1")[0]
    newer = _recording(db, cfg, start=T0 + timedelta(days=4))
    total = kept["size_bytes"] + newer["size_bytes"]
    out = db.recordings_to_prune(3, total - 1, now=T0 + timedelta(days=4), kept_days=30)
    assert [r["id"] for r in out] == [newer["id"]]   # the newer normal one goes before the older kept part
    assert a["id"] not in [r["id"] for r in db.list_recordings("cam1")]


def test_nothing_is_kept_when_turned_off_or_without_alerts(db, camera, tmp_path):
    cfg = _cfg(tmp_path, alert_keep_days=0)
    rec = _recording(db, cfg)
    _alert(db, camera, T0 + timedelta(seconds=10))
    assert recorder.keep_around_alerts(db, cfg, rec, writer_factory=lambda p, size: CvWriter(p, size)) == []
    cfg = _cfg(tmp_path / "other")
    rec = _recording(db, cfg, start=T0 + timedelta(hours=1))
    assert recorder.keep_around_alerts(db, cfg, rec, writer_factory=lambda p, size: CvWriter(p, size)) == []


def test_a_kept_part_does_not_hold_the_days_time_lapse(tmp_path):
    day = tmp_path / "cam1" / "2026-10-05"
    day.mkdir(parents=True)
    for name in ("timelapse.mp4", "timelapse.json", "140003-alert.mp4", "140003-alert.jsonl"):
        (day / name).write_bytes(b"x")
    timelapse.drop_if_alone(day)
    assert sorted(p.name for p in day.iterdir()) == ["140003-alert.jsonl", "140003-alert.mp4"]


# --- step 2: where kept parts show ---------------------------------------------------------------------------------

def test_kept_parts_in_the_day_list_storage_health_and_week(db, camera, tmp_path):
    from fastapi.testclient import TestClient
    from labeler import summary
    from labeler.healthcheck import check_recording
    from labeler.web import create_app
    cfg = _cfg(tmp_path, enabled=True, keep_days=3)
    _recording(db, cfg)
    _alert(db, camera, T0 + timedelta(seconds=10))
    recorder.prune(db, cfg, now=T0 + timedelta(days=4), writer_factory=lambda p, size: CvWriter(p, size))
    stats = db.recording_stats()
    assert stats["kept_files"] == 1 and stats["kept_bytes"] == stats["bytes"] > 0
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    cfg["cameras"] = [{"name": "cam1", "source": 0}]
    client = TestClient(create_app(cfg, db, None), client=("127.0.0.1", 5000))
    day = client.get("/api/recordings", params={"camera": "cam1", "day": "2026-10-05"}).json()
    assert [r["kept"] for r in day["recordings"]] == ["alert"] and day["alert_keep_days"] == 30
    found = client.get("/api/recordings/find", params={"camera": "cam1", "at": "2026-10-05T14:00:10"}).json()
    assert found["id"] == day["recordings"][0]["id"] and found["offset_s"] == 2.0   # ▶ Recording on the alert

    class Cam:
        name, recorder = "cam1", type("R", (), {"encoder": "mp4v", "paused": None, "dropped": 0})()

    (tmp_path / "ffmpeg.exe").write_bytes(b"")
    cfg["recording"]["ffmpeg"] = str(tmp_path / "ffmpeg.exe")
    line = check_recording(cfg, db, [Cam()])
    assert "1 part kept around alerts" in line["detail"] and "30 days" in line["detail"]
    # The week: a 2-minute recording kept whole around an alert shows as 2 minutes kept on a day "not kept".
    cfg2 = _cfg(tmp_path / "week", keep_days=3, alert_before_s=60, alert_after_s=120)
    _recording(db, cfg2, start=T0 + timedelta(hours=2), seconds=120)
    _alert(db, camera, T0 + timedelta(hours=2, seconds=60))
    recorder.prune(db, cfg2, now=T0 + timedelta(days=4), writer_factory=lambda p, size: CvWriter(p, size))
    week = summary.week(db, T0.replace(hour=0), T0.replace(hour=0) + timedelta(days=7), cfg=cfg2,
                        now=T0 + timedelta(days=6))
    day0 = week["days"][0]
    assert day0["day"] == "2026-10-05" and day0["recorded_h"] is None and day0["kept_min"] == 2
    assert all(d["kept_min"] == 0 for d in week["days"][1:])


def test_pages_show_kept_parts():
    from pathlib import Path
    static = Path(__file__).resolve().parent.parent / "labeler" / "static"
    page = (static / "recordings.html").read_text(encoding="utf-8")
    assert "--kept: #1baf7a" in page and "--kept: #199e70" in page and 'id="legend-kept"' in page
    assert "kept around an alert" in page and 'class="kept-tag"' in page
    week = (static / "summary.html").read_text(encoding="utf-8")
    assert "min kept" in week
    settings = (static / "settings.html").read_text(encoding="utf-8")
    assert 'id="recording_alert_keep_days"' in settings and 'id="recording_alert_before_s"' in settings
