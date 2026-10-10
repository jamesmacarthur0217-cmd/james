"""Long-run check (tools/soak.py): trends, log counting, the summary and chart."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import soak  # noqa: E402


def _rows(hours=24, every_h=1 / 6, **funcs):
    """Fake samples: column -> function of hours."""
    rows, t = [], 0.0
    while t <= hours + 1e-9:
        row = {"time": f"t{t:.2f}", "elapsed_h": round(t, 3), "up": 1, "health_ok": 1, "log_errors": 0,
               "log_warnings": 0}
        row.update({col: f(t) for col, f in funcs.items()})
        rows.append({k: str(v) for k, v in row.items()})
        t += every_h
    return rows


def _problem(rows, column):
    return {j["column"]: j["problem"] for j in soak.judge(rows)}.get(column)


def test_slope():
    assert abs(soak.slope_per_hour([0, 1, 2, 3], [5, 7, 9, 11]) - 2) < 1e-9
    assert soak.slope_per_hour([0, 1], [1, 2]) is None          # too few points
    assert soak.slope_per_hour([1, 1, 1], [1, 2, 3]) is None    # no time passed


def test_steady_growth_is_flagged_and_flat_is_not():
    rows = _rows(app_rss_mb=lambda t: 2000 + 20 * t,            # +480 MB a day: a leak
                 ollama_mb=lambda t: 5000 + (3 if int(t * 6) % 2 else -3),   # noise around a flat line
                 app_threads=lambda t: 60)
    assert _problem(rows, "app_rss_mb") == "GROWING"
    assert _problem(rows, "ollama_mb") == ""
    assert _problem(rows, "app_threads") == ""


def test_warmup_step_is_not_a_leak():
    # A model loaded in the first hour adds 1 GB at once; after that the memory is flat.
    rows = _rows(app_rss_mb=lambda t: 1500 if t < 0.5 else 2500)
    assert _problem(rows, "app_rss_mb") == ""


def test_small_growth_below_the_fixed_minimum_is_not_flagged():
    rows = _rows(app_threads=lambda t: 20 + 0.2 * t)     # +4.8 threads a day: over 10% of 20, under 10 threads
    assert _problem(rows, "app_threads") == ""


def test_falling_fps_is_flagged():
    assert _problem(_rows(fps=lambda t: 30 - 0.5 * t), "fps") == "FALLING"
    assert _problem(_rows(fps=lambda t: 30 + (1 if int(t) % 2 else -1)), "fps") == ""


def test_log_reader_counts_new_lines_and_follows_rollover(tmp_path):
    log = tmp_path / "visionlabeler.log"
    log.write_text("2026-10-11 10:00:00 ERROR old: before the start\n", encoding="utf-8")
    reader = soak.LogReader(log)
    assert reader.new_counts() == (0, 0)                         # lines from before the start are not counted
    with open(log, "a", encoding="utf-8") as f:
        f.write("2026-10-11 10:01:00 ERROR labeler.vlm: timeout\n"
                "2026-10-11 10:01:01 WARNING labeler.health: slow\n"
                "2026-10-11 10:01:02 INFO labeler: an ERROR word in a message is not a level\n")
    assert reader.new_counts() == (1, 1)
    assert reader.new_counts() == (0, 0)
    log.write_text("2026-10-11 11:00:00 CRITICAL x: rolled over\n", encoding="utf-8")   # smaller: a new file
    assert reader.new_counts() == (1, 0)
    assert soak.LogReader(None).new_counts() == (0, 0)


def test_summary_counts_downtime_and_errors():
    rows = _rows(hours=2, app_rss_mb=lambda t: 2000)
    rows[3]["up"] = "0"
    rows[4]["health_ok"] = "0"
    rows[5]["log_errors"] = "2"
    text = soak.summary(rows)
    assert "app not answering: 1 samples; health check not ok: 1 samples" in text
    assert "2 errors" in text
    assert "app down in 1 samples" in text


def test_report_writes_text_and_chart(tmp_path):
    rows = _rows(hours=3, app_rss_mb=lambda t: 2000 + t, fps=lambda t: 25, gpu_watts=lambda t: 40,
                 people_stored=lambda t: int(t))
    path = tmp_path / "soak.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=soak.COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    text = soak.report(path)
    assert "no growth or slowdown found" in text
    assert (tmp_path / "soak.txt").read_text(encoding="utf-8").startswith(f"{len(rows)} samples")
    assert (tmp_path / "soak.png").stat().st_size > 10_000


def test_sample_when_the_app_is_not_running(tmp_path):
    row = soak.sample(1, tmp_path, soak.LogReader(None), started=0)   # nothing listens on port 1
    assert row["up"] == 0 and "fps" not in row
    assert row["data_mb"] == 0
