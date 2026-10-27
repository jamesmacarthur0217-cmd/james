"""Made-up night videos (Phase 53, tools/make_night_videos.py): IR is black and white, softer in contrast and darker at
the edges; dark colour keeps some colour at about a quarter of the brightness; the grain is the same every run; paths
and truth files point into data/datasets/night_made. No dataset needed (GitHub has none)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import make_night_videos as night   # noqa: E402


def _scene():
    """A bright, colourful test picture: a red, a green and a blue band on a light gray background, some detail."""
    f = np.full((120, 160, 3), 170, np.uint8)
    f[20:40, :, 2] = 230
    f[50:70, :, 1] = 230
    f[80:100, :, 0] = 230
    cv2.putText(f, "test", (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    return f


def _rng():
    return np.random.default_rng(1)


def test_ir_is_black_and_white_flatter_and_darker_at_the_edges():
    day = _scene()
    ir = night.night("ir", day, _rng())
    assert ir.shape == day.shape and ir.dtype == np.uint8
    assert np.array_equal(ir[..., 0], ir[..., 1]) and np.array_equal(ir[..., 1], ir[..., 2])   # no colour
    gray = cv2.cvtColor(day, cv2.COLOR_BGR2GRAY).astype(float)
    even = night.ir(day, _rng(), np.ones(day.shape[:2], np.float32))    # without the darker edges
    smooth = cv2.GaussianBlur(even[..., 0].astype(float), (9, 9), 3)     # nor the grain
    assert smooth.std() < gray.std()                                     # flatter contrast
    lit = cv2.GaussianBlur(ir[..., 0].astype(float), (9, 9), 3)
    assert lit[60, 80] > lit[2, 2] + 20                                  # the middle brighter than a corner


def test_dark_keeps_some_colour_at_about_a_quarter_of_the_brightness():
    day = _scene()
    dk = night.night("dark", day, _rng())
    ratio = dk.mean() / day.mean()
    assert 0.15 < ratio < 0.4
    band = dk[25:35].reshape(-1, 3).mean(axis=0)                      # the red band: still reddest
    assert band[2] > band[0] + 5 and band[2] > band[1] + 5


def test_the_same_grain_every_run():
    day = _scene()
    a = night.night("dark", day, night._rng("v.mp4", "dark"))
    b = night.night("dark", day, night._rng("v.mp4", "dark"))
    c = night.night("dark", day, night._rng("w.mp4", "dark"))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_paths_and_truth_files(tmp_path, monkeypatch):
    assert night.out_path("ir", "data/datasets/urfd/fall-01-cam0.mp4") == \
        "data/datasets/night_made/ir/urfd/fall-01-cam0.mp4"
    data = tmp_path / "data" / "datasets"
    (data / "floor_made").mkdir(parents=True)
    (data / "floor_made" / "truth.csv").write_text(
        "video,kind,start_s,end_s,half\ndata/datasets/floor_made/a.mp4,floor,60,240,\n"
        "data/datasets/floor_made/b.mp4,,,,\n", encoding="utf-8")
    (data / "floor_made" / "areas.json").write_text('{"data/datasets/floor_made/a.mp4": [{"name": "w"}]}',
                                                    encoding="utf-8")
    monkeypatch.setattr(night, "DATA", data)
    monkeypatch.setattr(night, "OUT", data / "night_made")
    videos = night.write_set("dark", "floor_made/truth.csv", "floor_made/areas.json")
    assert videos == ["data/datasets/floor_made/a.mp4", "data/datasets/floor_made/b.mp4"]
    truth = (data / "night_made" / "dark" / "floor_made" / "truth.csv").read_text(encoding="utf-8")
    assert "data/datasets/night_made/dark/floor_made/a.mp4,floor,60,240," in truth
    areas = (data / "night_made" / "dark" / "floor_made" / "areas.json").read_text(encoding="utf-8")
    assert "night_made/dark/floor_made/a.mp4" in areas


def test_a_whole_video_is_made(tmp_path):
    src, dst = tmp_path / "day.mp4", tmp_path / "out" / "night.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _ in range(5):
        w.write(_scene())
    w.release()
    assert night.make_video(src, dst, "ir", "x") == 5 and dst.is_file()
    assert not (tmp_path / "out" / "night.part.mp4").exists()
    cap = cv2.VideoCapture(str(dst))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5 and round(cap.get(cv2.CAP_PROP_FPS)) == 10
    cap.release()


def test_the_day_and_night_table(tmp_path, monkeypatch, capsys):
    """tools/night_check.py: the same truth file by day and at night; one row per kind seen, found / false alarms."""
    import json
    import night_check as nc
    assert nc.base("day") == "data/datasets" and nc.base("ir") == "data/datasets/night_made/ir"
    monkeypatch.setattr(nc, "RESULTS", tmp_path)
    row = lambda kind, spans, found, fr, ff: {"kind": kind, "spans": spans, "found": found,   # noqa: E731
                                               "false_raised": fr, "false_found": ff}
    for cond, found in (("day", 14), ("ir", 3)):
        (tmp_path / f"falls_{cond}.json").write_text(json.dumps({"rows": [
            row("fall", 15, found, 0, 0), row("fire", 0, 0, 0, 0), row("suspicious", 0, 0, 4, 1 if cond == "ir" else 0)]}))
    nc.summary()
    out = capsys.readouterr().out
    assert "| falls | fall | 14 of 15 / 0 (0) | 3 of 15 / 0 (0) | - |" in out
    assert "| falls | suspicious | 0 (4) | 1 (4) | - |" in out and "| fire |" not in out


def test_fire_training_copies_are_black_and_white_and_vary():
    """tools/train_fire.py bw (Phase 53): each training picture's copy is black and white, the same for the same
    picture every run, and the night settings differ from picture to picture (not the one fixed look it is measured on)."""
    import zlib
    import train_fire
    day = _scene()
    seeded = lambda name: np.random.default_rng(zlib.crc32(name.encode()))   # noqa: E731
    a = train_fire.night_copy(day, seeded("AoF00001.jpg"))
    assert a.shape == day.shape and np.array_equal(a[..., 0], a[..., 2])
    assert np.array_equal(a, train_fire.night_copy(day, seeded("AoF00001.jpg")))
    looks = {round(float(train_fire.night_copy(day, seeded(f"p{i}.jpg")).mean())) for i in range(8)}
    assert len(looks) >= 5
    assert train_fire._label_of(Path("x/train/images/a.jpg")) == Path("x/train/labels/a.txt")
