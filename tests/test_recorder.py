"""Recording (labeler/recorder.py): when files start and stop, the memory buffer, splitting, the disk guard,
repairing after a crash, deleting old recordings, and (when FFmpeg is here) a real H.264 file."""
import copy
import json
import subprocess
from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from labeler import recorder
from labeler.config import DEFAULTS, ROOT

T0 = datetime(2026, 10, 12, 14, 0, 0)


class FakeWriter:
    made: list = []

    def __init__(self, path, size):
        self.path, self.size, self.frames = path, size, []
        FakeWriter.made.append(self)

    def write(self, frame):
        assert (frame.shape[1], frame.shape[0]) == self.size
        self.frames.append(frame)

    def close(self):
        self.path.write_bytes(b"x" * (100 * len(self.frames)))
        return True, ""


def _cfg(tmp_path, **rec):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["recording"].update(dir=str(tmp_path / "rec"), fps=10, before_s=2, after_s=2, max_width=640, **rec)
    cfg["storage"]["min_free_gb"] = 0
    return cfg


def _run(rec, seconds, people_at, heads=((100, 50, 140, 90),), w=1280, h=720):
    """Feed `seconds` of 25 fps video; people_at(t) says whether someone is in view at second t."""
    frame = np.zeros((h, w, 3), np.uint8)
    for i in range(int(seconds * 25)):
        t = i / 25
        person = people_at(t)
        rec.feed(frame, person, list(heads) if person else [], 1000 + t, T0 + timedelta(seconds=t))
    rec.close()


@pytest.fixture(autouse=True)
def _reset():
    FakeWriter.made = []


def _at(iso):
    """Seconds after T0, middle of the second (the dicts show whole seconds, cut down)."""
    return (datetime.fromisoformat(iso) - T0).total_seconds() + 0.5


def _sidecar(path):
    lines = [json.loads(x) for x in Path(path).with_suffix(".jsonl").read_text().splitlines()]
    return lines[0], lines[1:]


def test_records_from_before_to_after_people_with_heads_scaled(db, tmp_path):
    cfg = _cfg(tmp_path)
    rec = recorder.Recorder(cfg, "cam1", db, "h264_nvenc", writer_factory=FakeWriter, drop_when_full=False)
    _run(rec, 20, lambda t: 5 <= t < 8)        # someone in view from 5 s to 8 s
    rows = db.list_recordings()
    assert len(rows) == 1 and len(FakeWriter.made) == 1
    r = rows[0]
    assert _at(r["started_at"]) == pytest.approx(3, abs=0.6)                      # 2 s before (before_s)
    assert r["camera"] == "cam1" and r["path"] == "cam1/2026-10-12/140002.mp4"   # (2.96 s)
    assert _at(r["ended_at"]) == pytest.approx(10, abs=0.6)                       # 2 s after (after_s)
    assert 68 <= r["frames"] <= 72 and r["size_bytes"] > 0 and r["encoder"] == "h264_nvenc"
    head, frames = _sidecar(tmp_path / "rec" / r["path"])
    assert head == {"w": 640, "h": 360, "fps": 10.0}
    times = [f[0] for f in frames]
    assert times == sorted(times) and times[0] == 0 and 6.8 <= times[-1] <= 7.3
    assert [50, 25, 70, 45] in frames[25][1]                                      # head box at half size
    assert all(f.shape == (360, 640, 3) for f in FakeWriter.made[0].frames)


def test_a_person_returning_within_after_s_continues_the_same_file(db, tmp_path):
    rec = recorder.Recorder(_cfg(tmp_path), "cam1", db, "x", writer_factory=FakeWriter, drop_when_full=False)
    _run(rec, 30, lambda t: 5 <= t < 8 or 9 <= t < 11 or 20 <= t < 21)
    starts = [_at(r["started_at"]) for r in db.list_recordings()]
    assert starts == [pytest.approx(3, abs=0.6), pytest.approx(18, abs=0.6)]   # 8 -> 9 s within after_s; 11 -> 20 not


def test_long_recordings_are_split_into_files(db, tmp_path):
    rec = recorder.Recorder(_cfg(tmp_path, max_file_min=0.1), "cam1", db, "x", writer_factory=FakeWriter, drop_when_full=False)   # 6 s
    _run(rec, 20, lambda t: 2 <= t < 16)
    rows = db.list_recordings()
    assert len(rows) == 3
    for a, b in zip(rows, rows[1:]):
        gap = (datetime.fromisoformat(b["started_at"]) - datetime.fromisoformat(a["ended_at"])).total_seconds()
        assert abs(gap) <= 1                    # back to back
    assert sum(r["frames"] for r in rows) in range(175, 187)


def test_always_records_without_people(db, tmp_path):
    rec = recorder.Recorder(_cfg(tmp_path, when="always"), "cam1", db, "x", writer_factory=FakeWriter, drop_when_full=False)
    _run(rec, 5, lambda t: False)
    (r,) = db.list_recordings()
    assert r["started_at"] == "2026-10-12T14:00:00" and r["frames"] in range(49, 53)


def test_paused_when_the_disk_is_almost_full(db, tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg["storage"]["min_free_gb"] = 5
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(recorder.shutil, "disk_usage", lambda p: Usage(100e9, 98e9, 2e9))
    rec = recorder.Recorder(cfg, "cam1", db, "x", writer_factory=FakeWriter)
    _run(rec, 10, lambda t: t > 2)
    assert db.list_recordings() == [] and "disk almost full" in rec.paused


def test_a_failed_file_is_removed(db, tmp_path):
    class Broken(FakeWriter):
        def close(self):
            return False, "encoder failed"     # nothing written
    rec = recorder.Recorder(_cfg(tmp_path), "cam1", db, "x", writer_factory=Broken)
    _run(rec, 10, lambda t: 2 <= t < 4)
    assert db.list_recordings() == []
    assert not list((tmp_path / "rec").rglob("*.mp4")) and not list((tmp_path / "rec").rglob("*.jsonl"))


def test_repair_closes_files_left_open_by_a_crash(db, tmp_path):
    cfg = _cfg(tmp_path)
    cfg["recording"]["ffmpeg"] = str(tmp_path / "missing" / "ffmpeg.exe")   # no ffprobe either
    base = tmp_path / "rec" / "cam1" / "2026-10-12"
    base.mkdir(parents=True)
    (base / "140000.mp4").write_bytes(b"video")
    (base / "140000.jsonl").write_text('{"w": 640, "h": 360, "fps": 10}\n[0.0, []]\n[4.5, []]\n[4.6, [[1, 2', encoding="utf-8")
    kept = db.add_recording("cam1", "cam1/2026-10-12/140000.mp4", T0, "x")
    gone = db.add_recording("cam1", "cam1/2026-10-12/150000.mp4", T0 + timedelta(hours=1), "x")   # no file
    assert recorder.repair(db, cfg, run=lambda *a, **k: 1 / 0) == 1   # no ffprobe here: the list is trusted
    (r,) = db.list_recordings()
    assert r["id"] == kept and r["ended_at"] == "2026-10-12T14:00:04" and r["size_bytes"] == 5
    assert r["frames"] == 2   # the frame lines before the one cut short
    assert gone not in [x["id"] for x in db.list_recordings()]


def test_repair_trusts_the_video_when_it_lost_its_last_frames(db, tmp_path):
    cfg = _cfg(tmp_path)
    cfg["recording"]["ffmpeg"] = str(tmp_path / "ffmpeg.exe")
    (tmp_path / "ffprobe.exe").write_bytes(b"")
    base = tmp_path / "rec" / "cam1" / "2026-10-12"
    base.mkdir(parents=True)
    (base / "140000.mp4").write_bytes(b"video")
    (base / "140000.jsonl").write_text('{"w": 640, "h": 360, "fps": 10}\n' +
                                       "".join(f"[{i / 10}, []]\n" for i in range(50)), encoding="utf-8")
    db.add_recording("cam1", "cam1/2026-10-12/140000.mp4", T0, "x")
    probe = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "30\n", "")   # FFmpeg kept 30 of 50
    assert recorder.repair(db, cfg, run=probe) == 1
    (r,) = db.list_recordings()
    assert r["frames"] == 30 and r["ended_at"] == "2026-10-12T14:00:02"   # frame 30 is at 2.9 s


def test_prune_by_age_then_by_size(db, tmp_path):
    cfg = _cfg(tmp_path, keep_days=3, max_gb=2.5e-9 * 1000)   # 2.5 kB in all
    now = T0
    ids = []
    for days_ago, size in ((5, 1000), (2, 1000), (1, 1000), (0, 1000)):
        start = now - timedelta(days=days_ago)
        rel = f"cam1/{start:%Y-%m-%d}/{start:%H%M%S}.mp4"
        path = tmp_path / "rec" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        path.with_suffix(".jsonl").write_text("{}\n", encoding="utf-8")
        i = db.add_recording("cam1", rel, start, "x")
        db.update_recording(i, ended_at=start + timedelta(minutes=1), size_bytes=size)
        ids.append(i)
    assert recorder.prune(db, cfg, now) == 2          # 5 days old, then the oldest left until 2 kB fit
    assert [r["id"] for r in db.list_recordings()] == ids[2:]
    assert not (tmp_path / "rec" / f"cam1/{now - timedelta(days=5):%Y-%m-%d}").exists()   # empty day folder gone
    assert db.recording_stats()["bytes"] == 2000


def test_pick_encoder_takes_the_first_that_works(tmp_path):
    tried = []

    def run(cmd, **kw):
        enc = cmd[cmd.index("-c:v") + 1]
        tried.append(enc)
        return subprocess.CompletedProcess(cmd, 0 if enc == "h264_mf" else 1)
    recorder._encoder_cache.clear()
    assert recorder.pick_encoder(tmp_path / "ffmpeg.exe", run=run) == "h264_mf"
    assert tried == ["h264_nvenc", "h264_mf"]
    recorder._encoder_cache.clear()
    assert recorder.pick_encoder(tmp_path / "ffmpeg.exe", run=lambda c, **k: subprocess.CompletedProcess(c, 1)) is None
    recorder._encoder_cache.clear()


def test_the_time_is_written_on_a_copy():
    frame = np.zeros((360, 640, 3), np.uint8)
    out = recorder.stamp(frame, "cam1  2026-10-12 14:00:00")
    assert frame.max() == 0                              # the camera loop's picture is untouched
    assert out[300:360, :400].max() == 255               # white text near the bottom left


def test_the_outline_lines_up_with_the_text():
    """The dark outline ends within 2-3 px of the white text: it once drifted along the line (thicker letters are
    spaced wider in OpenCV), so the time looked doubled."""
    for h in (360, 720, 1080):
        out = recorder.stamp(np.full((h, int(h * 16 / 9), 3), 128, np.uint8), "cam1  2026-10-19 21:19:21")
        gray = out[:, :, 0].astype(int)
        white_cols = np.where((gray > 230).any(axis=0))[0]
        dark_cols = np.where((gray < 30).any(axis=0))[0]
        assert 0 < dark_cols[-1] - white_cols[-1] <= 3 and 0 < white_cols[0] - dark_cols[0] <= 3


def test_encoder_quality_settings():
    assert recorder.encoder_args("h264_nvenc", 28)[recorder.encoder_args("h264_nvenc", 28).index("-cq") + 1] == "28"
    assert recorder.encoder_args("h264_mf", 28)[-1] == "60"
    assert recorder.encoder_args("libopenh264", 28)[-1] == "2000k"
    assert recorder.encoder_args("libopenh264", 33)[-1] == "1000k"


FFMPEG = ROOT / "data" / "ffmpeg" / "ffmpeg.exe"


@pytest.mark.skipif(not FFMPEG.exists(), reason="FFmpeg is not on this PC (downloaded only with the user's OK)")
def test_a_real_h264_file_plays(db, tmp_path):
    cfg = _cfg(tmp_path)
    cfg["recording"]["ffmpeg"] = str(FFMPEG)
    recorder._encoder_cache.clear()
    enc = recorder.pick_encoder(FFMPEG)
    assert enc in recorder.ENCODERS
    rec = recorder.Recorder(cfg, "cam1", db, enc, drop_when_full=False)
    _run(rec, 6, lambda t: 2 <= t < 3, w=640, h=480)
    (r,) = db.list_recordings()
    cap = cv2.VideoCapture(str(tmp_path / "rec" / r["path"]))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n == r["frames"] and n >= 25


def test_health_line(db, tmp_path):
    from types import SimpleNamespace

    from labeler.healthcheck import check_recording
    cfg = _cfg(tmp_path)
    cfg["recording"]["enabled"] = False
    assert check_recording(cfg, db, [])["level"] == "off"
    cfg["recording"].update(enabled=True, ffmpeg=str(tmp_path / "nothing.exe"))
    assert "FFmpeg is not found" in check_recording(cfg, db, [])["detail"]
    (tmp_path / "ffmpeg.exe").write_bytes(b"")
    cfg["recording"]["ffmpeg"] = str(tmp_path / "ffmpeg.exe")
    rec = SimpleNamespace(encoder="h264_nvenc", paused="", dropped=0)
    cams = [SimpleNamespace(name="cam1", recorder=rec)]
    i = db.add_recording("cam1", "cam1/a.mp4", T0, "h264_nvenc")
    db.update_recording(i, ended_at=T0, size_bytes=2_500_000_000)
    ok = check_recording(cfg, db, cams)
    assert ok["level"] == "ok" and ok["detail"].startswith("1 files, 2.5 of 100 GB, kept 3 days, oldest 2026-10-12 14:00")
    rec.paused = "disk almost full (3.0 GB free)"
    assert check_recording(cfg, db, cams)["level"] == "problem"
    assert check_recording(cfg, db, [SimpleNamespace(name="cam1", recorder=None)])["level"] == "warning"
