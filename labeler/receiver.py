"""Cameras sent from other devices (PHASE15-PLAN.md): MediaMTX on this PC receives them.

A camera with `source: push` is not read from an address: another device sends its picture here, a PC with FFmpeg
(remote-camera/setup-camera.bat) over RTSP, or a phone streaming app over RTMP, through Tailscale. Sending is an
outgoing connection from that device, so it works behind firewalls that block incoming ones (a company PC).

MediaMTX (`receiver.exe`, MIT license; tools/get_mediamtx.py) runs while the app runs, with a settings file written
here at each start (`receiver.dir`): each pushed camera may send only to its own path, with its own password
(`push_password`, stored as a SHA-256 hash; the plain one is shown once on the Settings page) and only from a
Tailscale address; VisionLabeler reads from 127.0.0.1 with a password made at each start; RTSP over TCP and RTMP,
everything else (HLS, WebRTC, SRT, its API) off. If MediaMTX stops, it is started again.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import string
import subprocess
import threading
import time
from pathlib import Path

from .config import resolve

log = logging.getLogger(__name__)

PUSH = "push"
READER = "visionlabeler"   # the user this app reads with (from 127.0.0.1 only)
TAILSCALE = "100.64.0.0/10"
ALPHABET = string.ascii_letters + string.digits


def is_push(cam: dict) -> bool:
    return str(cam.get("source", "")).strip().lower() == PUSH


def push_cameras(cfg: dict) -> list[dict]:
    return [c for c in cfg.get("cameras") or [] if is_push(c)]


def new_password(length: int = 24) -> str:
    """Letters and digits only: it goes into rtsp:// and rtmp:// addresses unchanged."""
    return "".join(secrets.choice(ALPHABET) for _ in range(length))


def hash_password(password: str) -> str:
    """MediaMTX's stored form: sha256: and the base64 of the SHA-256 of the password."""
    return "sha256:" + base64.b64encode(hashlib.sha256(password.encode()).digest()).decode()


def exe_path(cfg: dict) -> Path:
    return resolve(cfg["receiver"]["exe"])


def config_text(cfg: dict, read_password: str, publish_from: tuple[str, ...] = (TAILSCALE,)) -> str:
    """MediaMTX's settings for the pushed cameras (YAML written by hand: every value is ours, no quoting surprises)."""
    rc = cfg["receiver"]
    folder = resolve(rc["dir"]).as_posix()
    lines = ["# Written by VisionLabeler (labeler/receiver.py) at each start; changes here are overwritten.",
             "logLevel: warn", "logDestinations: [file]", f'logFile: "{folder}/mediamtx.log"',
             "authMethod: internal", "authInternalUsers:",
             f"  - user: {READER}", f'    pass: "{hash_password(read_password)}"', '    ips: ["127.0.0.1", "::1"]',
             "    permissions:", "      - action: read"]
    for cam in push_cameras(cfg):
        if not cam.get("push_password"):
            continue   # no password yet: nobody may send to it
        ips = ", ".join(f'"{ip}"' for ip in publish_from)
        lines += [f"  - user: {cam['name']}", f'    pass: "{cam["push_password"]}"', f"    ips: [{ips}]",
                  "    permissions:", "      - action: publish", f"        path: {cam['name']}"]
    lines += ["api: false", "metrics: false", "pprof: false", "playback: false",
              "rtsp: true", "rtspTransports: [tcp]", 'rtspEncryption: "no"', f"rtspAddress: :{int(rc['rtsp_port'])}",
              "rtmp: true", 'rtmpEncryption: "no"', f"rtmpAddress: :{int(rc['rtmp_port'])}",
              "hls: false", "webrtc: false", "srt: false", "moq: false", "paths:"]
    for cam in push_cameras(cfg):
        lines += [f"  {cam['name']}:", "    source: publisher"]
    return "\n".join(lines) + "\n"


CURRENT = None   # the running app's Receiver (run.py), for the Health page


class Receiver:
    """MediaMTX while the app runs. start() at startup, watch() on a thread, stop() at shutdown."""

    def __init__(self, cfg: dict, popen=subprocess.Popen, publish_from: tuple[str, ...] | None = None):
        self.cfg, self.popen = cfg, popen
        self.publish_from = tuple(publish_from or cfg["receiver"].get("publish_from") or (TAILSCALE,))
        self.rc = cfg["receiver"]
        self.folder = resolve(self.rc["dir"])
        self.exe = exe_path(cfg)
        self.read_password = new_password()
        self.proc = None
        self.restarts = 0
        self.problem = ""
        self.settle_s = 0.5   # how long a just-started MediaMTX must keep running to count as started

    def read_url(self, camera: str) -> str:
        return f"rtsp://{READER}:{self.read_password}@127.0.0.1:{int(self.rc['rtsp_port'])}/{camera}"

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> bool:
        global CURRENT
        CURRENT = self
        if not self.exe.exists():
            self.problem = f"MediaMTX not found at {self.exe} (tools\\get_mediamtx.py download)"
            log.warning("cameras sent from other devices: %s", self.problem)
            return False
        self.folder.mkdir(parents=True, exist_ok=True)
        cfg_file = self.folder / "mediamtx.yml"
        cfg_file.write_text(config_text(self.cfg, self.read_password, self.publish_from), encoding="utf-8")
        self._stop_leftovers()
        flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        self.proc = self.popen([str(self.exe), str(cfg_file)], cwd=str(self.folder), stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        time.sleep(self.settle_s)
        if not self.running():
            self.problem = f"MediaMTX stopped at once (see {self.folder / 'mediamtx.log'})"
            log.warning("cameras sent from other devices: %s", self.problem)
            return False
        self.problem = ""
        log.info("receiving cameras sent from other devices (%s) on ports %s (RTSP) and %s (RTMP)",
                 ", ".join(c["name"] for c in push_cameras(self.cfg)), self.rc["rtsp_port"], self.rc["rtmp_port"])
        return True

    def watch(self, stop: threading.Event, every_s: float = 5) -> None:
        """Start MediaMTX again if it stops (at most once a minute after the first try)."""
        last = 0.0
        while not stop.wait(every_s):
            if self.proc is not None and not self.running() and time.monotonic() - last >= 60:
                last = time.monotonic()
                self.restarts += 1
                log.warning("the receiver for cameras from other devices stopped; starting it again")
                self.start()

    def stop(self, wait_s: float = 10) -> None:
        if self.proc is None:
            return
        proc, self.proc = self.proc, None
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(wait_s)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _stop_leftovers(self) -> None:
        """A MediaMTX left by a crashed earlier start of this app (same program, same folder) holds the ports."""
        try:
            import psutil
        except ImportError:
            return
        for p in psutil.process_iter(["exe", "cwd"]):
            try:
                if p.info["exe"] and Path(p.info["exe"]) == self.exe and p.info["cwd"] and Path(p.info["cwd"]) == self.folder:
                    log.info("stopping a receiver left by an earlier run (pid %d)", p.pid)
                    p.kill()
                    p.wait(5)
            except (psutil.Error, OSError):
                continue
