"""Long-run check (soak test): watch a running copy of the app for hours and look for leaks or slowdowns.

    python tools/soak.py --port 8001 --hours 24 --every 600 --out <folder> [--data <folder>] [--log <file>]
    python tools/soak.py --report <folder>/soak.csv        # chart and trends again, from an earlier run

Every --every seconds it writes one row to <folder>/soak.csv: the app's memory (RSS and private bytes), threads,
handles and GPU memory; Ollama's memory; the whole GPU's memory, power and use; free disk and the size of --data (pictures, clips,
database); FPS, frames, people in view, VLM queue and people stored (/api/status); the health check result
(/api/health); new ERROR and WARNING lines in --log. It only reads: nothing in the app changes.

At the end (or with --report) it writes soak.png (one small chart per value) and soak.txt: for each value the
straight-line trend after the first hour (models loading, caches filling), and "GROWING" when it would add more
than 10% (and more than a small fixed amount) in 24 hours; FPS is flagged when it falls. A memory leak shows as
steady growth; one-off steps (a model loaded on first use) do not. Watch the private memory: Windows trims the
RSS (working set) of a quiet process, so it can fall a lot without anything being freed.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import psutil

COLUMNS = ["time", "elapsed_h", "up", "app_rss_mb", "app_private_mb", "app_threads", "app_handles", "app_gpu_mb", "ollama_mb",
           "gpu_mem_mb", "gpu_watts", "gpu_util", "disk_free_gb", "data_mb", "fps", "frames", "people_in_view",
           "vlm_queue", "people_stored", "health_ok", "log_errors", "log_warnings"]

# Values watched for steady growth: column -> (label, smallest change in 24 h worth flagging).
GROWTH = {
    "app_rss_mb": ("app memory (MB)", 200),
    "app_private_mb": ("app private memory (MB)", 200),
    "app_threads": ("app threads", 10),
    "app_handles": ("app handles", 500),
    "app_gpu_mb": ("app GPU memory (MB)", 300),
    "ollama_mb": ("Ollama memory (MB)", 200),
    "vlm_queue": ("VLM queue", 5),
}
# Shown in the chart and the summary, but growth is expected (people are stored) or the value is not a trend.
OTHER = {
    "fps": "FPS",
    "gpu_mem_mb": "GPU memory, all programs (MB)",
    "gpu_watts": "GPU power (W)",
    "data_mb": "stored data (MB)",
    "people_stored": "people stored",
    "disk_free_gb": "free disk (GB)",
    "log_errors": "new log errors",
}
WARMUP_H = 1.0


def find_app(port: int) -> psutil.Process | None:
    """The process listening on the app's port (run.py; with start.bat it is the supervisor's child)."""
    try:
        for c in psutil.net_connections(kind="tcp"):
            if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port and c.pid:
                return psutil.Process(c.pid)
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    for p in psutil.process_iter(["cmdline"]):   # fallback: run.py started with --port <port>
        cmd = p.info["cmdline"] or []
        if any(a.endswith("run.py") for a in cmd) and str(port) in cmd:
            return p
    return None


def process_numbers(p: psutil.Process | None) -> dict:
    if p is None:
        return {}
    try:
        with p.oneshot():
            mem = p.memory_info()
            out = {"app_rss_mb": mem.rss / 1e6, "app_threads": p.num_threads()}
            if hasattr(mem, "private"):
                out["app_private_mb"] = mem.private / 1e6
            if hasattr(p, "num_handles"):
                out["app_handles"] = p.num_handles()
            return out
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {}


def app_gpu_mb(pid: int) -> float | None:
    """The app's own GPU memory, from Windows' "GPU Process Memory" counters (nvidia-smi shows N/A per
    process on Windows, and the whole GPU's number also counts the desktop, browsers and games)."""
    if sys.platform != "win32":
        return None
    cmd = (f"(Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage' -ErrorAction Stop).CounterSamples"
           " | Measure-Object CookedValue -Sum | ForEach-Object Sum")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip()) / 1e6 if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def ollama_mb() -> float | None:
    """All Ollama processes together (the server, its model runner, the tray app)."""
    total, found = 0, False
    for p in psutil.process_iter(["name", "memory_info"]):
        if (p.info["name"] or "").lower().startswith("ollama") and p.info["memory_info"]:
            total += p.info["memory_info"].rss
            found = True
    return total / 1e6 if found else None


def gpu_numbers() -> dict:
    """The whole GPU (Windows does not give the memory per process): memory used, power, use."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,power.draw,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        mem, watts, util = (float(v) for v in out.stdout.strip().splitlines()[0].split(","))
        return {"gpu_mem_mb": mem, "gpu_watts": watts, "gpu_util": util}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {}


def folder_mb(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def get_json(url: str, timeout: float = 30) -> tuple[int | None, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:   # the health check answers 503 when something is wrong
        try:
            return e.code, json.loads(e.read())
        except ValueError:
            return e.code, {}
    except (OSError, ValueError):
        return None, {}


class LogReader:
    """Counts ERROR and WARNING lines added to a log file since the last call; follows the file when it is
    rolled over (renamed to .1 and started again)."""

    def __init__(self, path: Path | None):
        self.path = path
        self.offset = path.stat().st_size if path and path.exists() else 0

    def new_counts(self) -> tuple[int, int]:
        if self.path is None or not self.path.exists():
            return 0, 0
        size = self.path.stat().st_size
        if size < self.offset:   # rolled over
            self.offset = 0
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            text = f.read().decode("utf-8", "replace")
            self.offset = f.tell()
        # "2026-10-11 10:01:00 ERROR name: message": the level is the third word, not any word in the message.
        levels = [parts[2] for parts in (line.split(maxsplit=3) for line in text.splitlines()) if len(parts) > 2]
        return sum(lv in ("ERROR", "CRITICAL") for lv in levels), levels.count("WARNING")


def sample(port: int, data: Path | None, log: LogReader, started: float) -> dict:
    base = f"http://127.0.0.1:{port}"
    row: dict = {"time": datetime.now().isoformat(timespec="seconds"),
                 "elapsed_h": round((time.time() - started) / 3600, 3)}
    code, status = get_json(base + "/api/status")
    row["up"] = int(code == 200)
    if code == 200:
        row.update(fps=status.get("fps"), frames=status.get("frames"), people_in_view=status.get("active_people"),
                   vlm_queue=status.get("vlm_queue"), people_stored=status.get("tracks"),
                   disk_free_gb=status.get("disk_free_gb"))
        code, _ = get_json(base + "/api/health", timeout=60)
        row["health_ok"] = int(code == 200)
    app = find_app(port)
    row.update(process_numbers(app))
    if app is not None:
        row["app_gpu_mb"] = app_gpu_mb(app.pid)
    row["ollama_mb"] = ollama_mb()
    row.update(gpu_numbers())
    row["data_mb"] = folder_mb(data)
    row["log_errors"], row["log_warnings"] = log.new_counts()
    return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def series(rows: list[dict], column: str, after_h: float = 0.0) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for r in rows:
        try:
            x, y = float(r["elapsed_h"]), float(r[column])
        except (KeyError, TypeError, ValueError):
            continue
        if x >= after_h:
            xs.append(x)
            ys.append(y)
    return xs, ys


def slope_per_hour(xs: list[float], ys: list[float]) -> float | None:
    """Least-squares straight line through the points: change per hour."""
    if len(xs) < 3:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def judge(rows: list[dict], warmup_h: float = WARMUP_H) -> list[dict]:
    """One line per value: start, end, change per hour after the warm-up, and whether it is a problem."""
    out = []
    for col, (label, min_change) in GROWTH.items():
        xs, ys = series(rows, col, warmup_h)
        slope = slope_per_hour(xs, ys)
        if slope is None:
            continue
        per_day = slope * 24
        start = ys[0]
        flag = per_day > min_change and per_day > 0.10 * max(abs(start), 1e-9)
        out.append({"value": label, "column": col, "start": start, "end": ys[-1], "per_hour": slope,
                    "per_24h": per_day, "problem": "GROWING" if flag else ""})
    xs, ys = series(rows, "fps", warmup_h)
    slope = slope_per_hour(xs, ys)
    if slope is not None:
        first = sorted(ys[: max(3, len(ys) // 10)])
        flag = slope * 24 < -0.10 * max(first[len(first) // 2], 1e-9)   # vs. the median at the start
        out.append({"value": "FPS", "column": "fps", "start": ys[0], "end": ys[-1], "per_hour": slope,
                    "per_24h": slope * 24, "problem": "FALLING" if flag else ""})
    return out


def summary(rows: list[dict], warmup_h: float = WARMUP_H) -> str:
    if not rows:
        return "no samples"
    hours = float(rows[-1]["elapsed_h"])
    down = sum(r.get("up") in ("0", 0) for r in rows)
    unhealthy = sum(r.get("health_ok") in ("0", 0) for r in rows)
    errors = sum(int(float(r["log_errors"])) for r in rows if r.get("log_errors") not in (None, ""))
    warnings = sum(int(float(r["log_warnings"])) for r in rows if r.get("log_warnings") not in (None, ""))
    lines = [f"{len(rows)} samples over {hours:.1f} h ({rows[0]['time']} to {rows[-1]['time']})",
             f"app not answering: {down} samples; health check not ok: {unhealthy} samples",
             f"new log lines: {errors} errors, {warnings} warnings", "",
             f"trends after the first {warmup_h:g} h (straight line through the samples):",
             f"{'value':26} {'start':>9} {'end':>9} {'per hour':>10} {'per 24 h':>10}"]
    for j in judge(rows, warmup_h):
        lines.append(f"{j['value']:26} {j['start']:9.1f} {j['end']:9.1f} {j['per_hour']:10.2f} {j['per_24h']:10.1f}"
                     f"  {j['problem']}")
    for col in ("gpu_watts", "data_mb", "people_stored"):
        xs, ys = series(rows, col)
        if ys:
            lines.append(f"{OTHER[col]:26} {ys[0]:9.1f} {ys[-1]:9.1f}   (min {min(ys):.1f}, max {max(ys):.1f})")
    problems = [j for j in judge(rows, warmup_h) if j["problem"]]
    lines += ["", "result: " + ("no growth or slowdown found" if not problems and not down else
                                "look at: " + ", ".join(j["value"] for j in problems) +
                                (f"; app down in {down} samples" if down else ""))]
    return "\n".join(lines)


def chart(rows: list[dict], path: Path) -> None:
    """Small multiples: one line per value against hours, shared time axis (one axis per chart, no legends)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [(c, lbl) for c, (lbl, _) in GROWTH.items()] + list(OTHER.items())
    panels = [(c, lbl) for c, lbl in panels if series(rows, c)[1]]
    cols = 3
    nrows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(nrows, cols, figsize=(12, 2.4 * nrows), sharex=True, squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax, (col, label) in zip(axes.flat, panels):
        xs, ys = series(rows, col)
        ax.set_facecolor("#fcfcfb")
        ax.plot(xs, ys, color="#2a78d6", linewidth=1.6)
        ax.set_title(label, fontsize=10, color="#0b0b0b", loc="left")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.tick_params(colors="#52514e", labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#c9c8c2")
        ax.axvspan(0, min(WARMUP_H, max(xs or [0])), color="#efeeea", zorder=0)   # warm-up, not judged
    for col in range(cols):   # hours under the lowest chart of each column (the last row may be part empty)
        shown = [axes[r][col] for r in range(nrows) if r * cols + col < len(panels)]
        if shown:
            shown[-1].xaxis.set_tick_params(labelbottom=True)
            shown[-1].set_xlabel("hours", fontsize=9, color="#52514e")
    fig.suptitle(f"Long-run check: {rows[0]['time']} to {rows[-1]['time']} (grey: first hour, not judged)",
                 fontsize=11, color="#0b0b0b", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def report(csv_path: Path) -> str:
    rows = read_csv(csv_path)
    text = summary(rows)
    csv_path.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    if rows:
        chart(rows, csv_path.with_suffix(".png"))
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Long-run check: sample a running app and look for leaks.")
    ap.add_argument("--port", type=int, default=8001, help="the app's port on 127.0.0.1 (default 8001, a test copy)")
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--every", type=float, default=600, help="seconds between samples (default 600 = 10 min)")
    ap.add_argument("--out", type=Path, help="folder for soak.csv, soak.png, soak.txt")
    ap.add_argument("--data", type=Path, help="folder whose size is watched (the test copy's folder)")
    ap.add_argument("--log", type=Path, help="the app's log file (visionlabeler.log)")
    ap.add_argument("--stop-when-down", type=int, default=3,
                    help="end early after this many samples in a row without an answer (0 = never)")
    ap.add_argument("--report", type=Path, help="only make the chart and summary of an earlier soak.csv")
    args = ap.parse_args()
    if args.report:
        print(report(args.report))
        return
    if not args.out:
        ap.error("--out is needed")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "soak.csv"
    log, started, down_in_a_row = LogReader(args.log), time.time(), 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        while True:
            row = sample(args.port, args.data, log, started)
            writer.writerow(row)
            f.flush()   # a stopped soak run keeps what it measured
            print(" ".join(f"{k}={row.get(k)}" for k in ("time", "up", "app_rss_mb", "gpu_mem_mb", "fps",
                                                         "people_stored", "log_errors")), flush=True)
            down_in_a_row = 0 if row["up"] else down_in_a_row + 1
            if args.stop_when_down and down_in_a_row >= args.stop_when_down:
                print("the app stopped answering: ending early", flush=True)
                break
            if time.time() - started + args.every > args.hours * 3600 + 1:
                break
            time.sleep(args.every)
    print(report(path))


if __name__ == "__main__":
    sys.exit(main())
