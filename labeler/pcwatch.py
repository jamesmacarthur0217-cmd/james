"""PC watch (Phase 38): the PC itself, next to the cameras. Read only; nothing here changes the PC.

- GPU: temperature, power and memory from `nvidia-smi` once a minute (hot: at least `gpu_hot_c`, 85 °C).
- Disks: Windows' own health of each physical disk (`Get-PhysicalDisk`, PowerShell) once an hour; anything but
  "Healthy" is a problem (a disk that is failing says so here long before files go missing).
- Unexpected stops: a heartbeat file (`pcwatch.heartbeat_file`, data/heartbeat.json) written once a minute while the
  app runs. A normal stop (Stop button, stop.bat, Ctrl+C, Restart now) writes "clean" into it at the very end; a crash
  restart by the supervisor writes "crashed". At the next start a file without either means the app or the PC stopped
  without warning (power cut, the PC turned off or restarted without stop.bat, the process killed) around its last
  beat; the PC's boot time tells which of the two.
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

GPU_QUERY = "temperature.gpu,power.draw,memory.used,memory.total,name"
DISK_PS = ("Get-PhysicalDisk | Select-Object FriendlyName, MediaType, HealthStatus, OperationalStatus, Size "
           "| ConvertTo-Json -Compress")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CAMERA = "PC"   # the "camera" of its stored problems (table camera_events)


def _num(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:   # "[N/A]", "[Not Supported]"
        return None


def parse_gpu(out: str) -> list[dict]:
    """nvidia-smi's csv lines (no header, no units) -> one dict per GPU."""
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",", 4)]   # the name is last and may hold commas itself
        if len(parts) < 5:
            continue
        temp, power, used, total = (_num(p) for p in parts[:4])
        gpus.append({"name": parts[4], "temp_c": temp, "power_w": power,
                     "mem_used_mb": used, "mem_total_mb": total})
    return gpus


def parse_disks(out: str) -> list[dict]:
    """Get-PhysicalDisk as JSON (one object, or a list of them) -> one dict per disk."""
    out = out.strip()
    if not out:
        return []
    got = json.loads(out)
    if isinstance(got, dict):
        got = [got]
    disks = []
    for d in got:
        status = d.get("OperationalStatus")
        if isinstance(status, list):   # some disks report several, e.g. ["OK", "Predictive Failure"]
            status = ", ".join(map(str, status))
        disks.append({"name": str(d.get("FriendlyName") or "disk").strip(), "media": d.get("MediaType"),
                      "health": str(d.get("HealthStatus") or "Unknown"), "status": str(status or ""),
                      "size_gb": round((d.get("Size") or 0) / 1e9)})
    return disks


def boot_time() -> datetime | None:
    """When Windows started (now minus its uptime)."""
    try:
        tick = ctypes.windll.kernel32.GetTickCount64
        tick.restype = ctypes.c_ulonglong   # the default (a 32-bit int) would cut it after 25 days
        return datetime.now() - timedelta(milliseconds=tick())
    except (AttributeError, OSError):   # not Windows
        return None


def previous_stop(path: Path, booted: datetime | None = None) -> dict | None:
    """How the last run ended, read at start before the first beat: None when it stopped normally (or never ran),
    else {"kind": "unexpected" / "crashed", "last": its last sign of life, "pc_restarted": the PC started since}."""
    try:
        beat = json.loads(path.read_text(encoding="utf-8"))
        last = datetime.fromisoformat(beat["last"])
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as e:
        log.warning("PC watch: could not read %s (%s)", path, e)
        return None
    ended = beat.get("ended")
    if ended == "clean":
        return None
    return {"kind": "crashed" if ended == "crashed" else "unexpected", "last": last.isoformat(timespec="seconds"),
            "started": beat.get("started"), "pc_restarted": bool(booted and booted > last)}


def describe_stop(stop: dict) -> str:
    at = stop["last"][:16].replace("T", " ")
    if stop["kind"] == "crashed":
        return f"the app stopped after a problem around {at} and was started again"
    if stop["pc_restarted"]:
        return (f"the PC was turned off or restarted around {at} while the app was running (a power cut, Windows "
                "updates, or the PC shut down without stop.bat)")
    return f"the app stopped without warning around {at} (its window closed or the process was ended)"


class PCWatch:
    def __init__(self, cfg: dict, run=subprocess.run, clock=datetime.now, booted: datetime | None = None, db=None):
        c = cfg["pcwatch"]
        self.db = db   # problems and stops are stored as camera events of the camera "PC" (Alerts page, Summary)
        self.every_s, self.disk_every_s = float(c["every_s"]), float(c["disk_every_s"])
        self.gpu_hot_c = float(c["gpu_hot_c"])
        self.path = Path(c["heartbeat_file"])
        self.run, self.clock = run, clock
        self.booted = booted if booted is not None else boot_time()
        self.started = clock()
        self.gpus: list[dict] | None = None     # None: not read yet; []: no NVIDIA GPU or no nvidia-smi
        self.gpu_error: str | None = None
        self.disks: list[dict] | None = None
        self.disk_error: str | None = None
        self.gpu_at = self.disk_at = None
        self.gpu_max_c: float | None = None     # the hottest since start
        self.last_stop = previous_stop(self.path, self.booted)   # read before the first beat overwrites it
        self._lock = threading.Lock()
        self._open: dict[str, dict] = {}   # problem key ("hot:<gpu>", "disk:<name>") -> its stored event
        if db is not None:
            db.close_open_camera_events(CAMERA)   # left open by the last run: it ended then
        if self.last_stop:
            log.warning("PC watch: %s", describe_stop(self.last_stop))
            if db is not None:   # from the last sign of life until now: how long nothing was watched
                db.add_camera_event(CAMERA, "stopped", describe_stop(self.last_stop),
                                    datetime.fromisoformat(self.last_stop["last"]), self.started)

    # --- readings ------------------------------------------------------------------------------------------------
    def _cmd(self, args: list[str], timeout: float) -> str:
        r = self.run(args, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout or f"exit {r.returncode}").strip().splitlines()[-1][:200])
        return r.stdout

    def read_gpu(self) -> None:
        try:
            gpus = parse_gpu(self._cmd(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"], 20))
            error = None
        except FileNotFoundError:
            gpus, error = [], "nvidia-smi not found (no NVIDIA driver)"
        except (OSError, RuntimeError, subprocess.SubprocessError) as e:
            gpus, error = [], f"nvidia-smi failed: {e}"
        with self._lock:
            self.gpus, self.gpu_error, self.gpu_at = gpus, error, self.clock()
            temps = [g["temp_c"] for g in gpus if g["temp_c"] is not None]
            if temps:
                self.gpu_max_c = max(temps + ([self.gpu_max_c] if self.gpu_max_c is not None else []))

    def read_disks(self) -> None:
        try:
            disks = parse_disks(self._cmd(["powershell", "-NoProfile", "-NonInteractive", "-Command", DISK_PS], 60))
            error = None
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as e:
            disks, error = [], f"could not be read: {e}"
        with self._lock:
            self.disks, self.disk_error, self.disk_at = disks, error, self.clock()

    # --- heartbeat -----------------------------------------------------------------------------------------------
    def _write(self, ended: str | None) -> None:
        beat = {"pid": os.getpid(), "started": self.started.isoformat(timespec="seconds"),
                "last": self.clock().isoformat(timespec="seconds"), "ended": ended}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(beat), encoding="utf-8")
            os.replace(tmp, self.path)   # never half a file, even at a power cut
        except OSError as e:
            log.warning("PC watch: could not write %s (%s)", self.path, e)

    def beat(self) -> None:
        self._write(None)

    def mark_ended(self, crashed: bool = False) -> None:
        """The very end of a run: "clean", or "crashed" when the supervisor will start the app again."""
        self._write("crashed" if crashed else "clean")

    # --- results -------------------------------------------------------------------------------------------------
    def _current(self) -> dict[str, tuple[str, str]]:
        """What is wrong now: problem key -> (kind, plain words). A hot card stays hot until 5 °C below the limit,
        so a temperature around it does not open and close a problem every minute; a failed reading keeps what
        was known (no answer is not "fine")."""
        out = {}
        if self.gpu_error:
            out.update({k: (e["kind"], e["message"]) for k, e in self._open.items() if k.startswith("hot:")})
        for g in self.gpus or []:
            key, t = "hot:" + g["name"], g["temp_c"]
            if t is not None and (t >= self.gpu_hot_c or (key in self._open and t > self.gpu_hot_c - 5)):
                out[key] = ("hot", f"the graphics card is hot: {t:.0f} °C (check the PC's fans and air flow)")
        if self.disk_error:
            out.update({k: (e["kind"], e["message"]) for k, e in self._open.items() if k.startswith("disk:")})
        for d in self.disks or []:
            if d["health"] != "Healthy":
                out["disk:" + d["name"]] = ("disk", f"disk {d['name']}: Windows says {d['health']}"
                                            + (f" ({d['status']})" if d["status"] and d["status"] != "OK" else "")
                                            + "; copy your data off it and replace it soon")
        return out

    def problems(self) -> list[str]:
        """What is wrong now, in plain words (empty: fine, or not read yet)."""
        with self._lock:
            return [words for _, words in self._current().values()]

    def sync(self) -> list[dict]:
        """Open a stored problem for each new one, close those that are over; returns the ones just opened."""
        opened = []
        with self._lock:
            now = self._current()
            for key, (kind, words) in now.items():
                if key not in self._open:
                    log.warning("PC watch: %s", words)
                    self._open[key] = (self.db.open_camera_event(CAMERA, kind, words) if self.db is not None
                                       else {"id": None, "camera": CAMERA, "kind": kind, "message": words,
                                             "started_at": self.clock().isoformat(timespec="seconds"),
                                             "ended_at": None})
                    opened.append(self._open[key])
            for key in [k for k in self._open if k not in now]:
                e = self._open.pop(key)
                log.info("PC watch: over: %s", e["message"])
                if self.db is not None and e["id"] is not None:
                    self.db.close_camera_event(e["id"])
        return opened

    def current(self) -> list[dict]:
        """Problems going on now, as camera problems (the Live page's banner and pop-up)."""
        with self._lock:
            return list(self._open.values())

    def state(self) -> dict:
        with self._lock:
            return {"gpus": self.gpus, "gpu_error": self.gpu_error, "gpu_max_c": self.gpu_max_c,
                    "gpu_at": self.gpu_at.isoformat(timespec="seconds") if self.gpu_at else None,
                    "disks": self.disks, "disk_error": self.disk_error,
                    "disk_at": self.disk_at.isoformat(timespec="seconds") if self.disk_at else None,
                    "gpu_hot_c": self.gpu_hot_c, "last_stop": self.last_stop}

    def loop(self, stopping: threading.Event) -> None:
        """Beat and read the GPU every every_s, the disks every disk_every_s (first both at once)."""
        next_disks = 0.0
        while not stopping.is_set():
            try:
                self.beat()
                self.read_gpu()
                if time.monotonic() >= next_disks:
                    self.read_disks()
                    next_disks = time.monotonic() + self.disk_every_s
                self.sync()   # logged once when a problem starts, not every minute
            except Exception:
                log.exception("PC watch failed")
            stopping.wait(self.every_s)
