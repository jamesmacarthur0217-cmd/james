"""Cameras sent from other devices (labeler/receiver.py): MediaMTX's settings, starting, restarting, stopping;
and, when MediaMTX and FFmpeg are on this PC, a real picture sent in and read out (and refused without the
password)."""
import copy
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

from labeler import receiver
from labeler.config import DEFAULTS, ROOT


def _cfg(tmp_path, cams=None):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["receiver"].update(dir=str(tmp_path / "mtx"), exe=str(tmp_path / "mediamtx.exe"))
    cfg["cameras"] = cams if cams is not None else [
        {"name": "cam1", "source": 0},
        {"name": "tixuan", "source": "push", "push_password": receiver.hash_password("secretpass")},
        {"name": "phone", "source": " PUSH "},                          # no password yet: nobody may send
    ]
    return cfg


def test_which_cameras_are_sent():
    assert receiver.is_push({"source": "push"}) and receiver.is_push({"source": " Push "})
    assert not receiver.is_push({"source": 0}) and not receiver.is_push({"source": "rtsp://x/push"})


def test_passwords():
    p = receiver.new_password()
    assert len(p) == 24 and p.isalnum() and receiver.new_password() != p
    # MediaMTX's form: sha256: + base64 of the SHA-256 (checked against MediaMTX itself below)
    assert receiver.hash_password("secret") == "sha256:K7gNU3sdo+OL0wNhqoVWhr3g6s1xYv72ol/pe/Unols="


def test_settings_file(tmp_path):
    text = receiver.config_text(_cfg(tmp_path), "readpass")
    assert f'pass: "{receiver.hash_password("readpass")}"' in text and 'ips: ["127.0.0.1", "::1"]' in text
    assert "  - user: tixuan" in text and 'ips: ["100.64.0.0/10"]' in text and "path: tixuan" in text
    assert "user: phone" not in text                                # no password: no sender account
    assert "  tixuan:\n    source: publisher" in text and "  phone:\n    source: publisher" in text
    assert "cam1" not in text and "all_others" not in text          # no other path can be sent to
    for line in ("rtspTransports: [tcp]", "rtspAddress: :8554", "rtmpAddress: :1935", "hls: false",
                 "webrtc: false", "srt: false", "moq: false", "api: false"):
        assert line in text
    assert "secretpass" not in text and "readpass" not in text      # only hashes


class FakeProc:
    def __init__(self, *a, **k):
        self.alive, self.terminated = True, False

    def poll(self):
        return None if self.alive else 1

    def terminate(self):
        self.alive, self.terminated = False, True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.alive = False


def test_start_watch_restart_stop(tmp_path):
    cfg = _cfg(tmp_path)
    Path(cfg["receiver"]["exe"]).write_bytes(b"")
    made = []
    rec = receiver.Receiver(cfg, popen=lambda *a, **k: made.append(FakeProc()) or made[-1])
    rec.settle_s = 0
    assert rec.start() and rec.running()
    assert (tmp_path / "mtx" / "mediamtx.yml").exists()
    assert rec.read_url("tixuan") == f"rtsp://visionlabeler:{rec.read_password}@127.0.0.1:8554/tixuan"
    made[0].alive = False                                           # it stopped by itself
    stop = threading.Event()
    t = threading.Thread(target=rec.watch, args=(stop, 0.01))
    t.start()
    for _ in range(200):
        if len(made) == 2:
            break
        time.sleep(0.01)
    stop.set()
    t.join()
    assert len(made) == 2 and rec.running() and rec.restarts == 1
    rec.stop()
    assert made[1].terminated and not rec.running()


def test_missing_mediamtx_is_reported(tmp_path):
    rec = receiver.Receiver(_cfg(tmp_path))
    assert not rec.start() and "not found" in rec.problem


MTX = ROOT / "data" / "mediamtx" / "mediamtx.exe"
FFMPEG = ROOT / "data" / "ffmpeg" / "ffmpeg.exe"
FFPROBE = ROOT / "data" / "ffmpeg" / "ffprobe.exe"


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(not (MTX.exists() and FFMPEG.exists()), reason="MediaMTX and FFmpeg are not on this PC")
def test_a_real_picture_in_and_out(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["receiver"].update(exe=str(MTX), rtsp_port=_free_port(), rtmp_port=_free_port())
    rec = receiver.Receiver(cfg, publish_from=("127.0.0.1",))       # the test sends from this PC itself
    assert rec.start(), rec.problem
    port, rtmp = cfg["receiver"]["rtsp_port"], cfg["receiver"]["rtmp_port"]
    try:
        def send(url, fmt="rtsp", extra=()):
            return subprocess.Popen([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-re", "-f", "lavfi", "-i",
                                     "testsrc=size=320x240:rate=10", "-t", "20", "-c:v", "libopenh264", "-g", "10",
                                     *extra, "-f", fmt, url], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        def probe(url):
            out = subprocess.run([str(FFPROBE), "-v", "error", "-rtsp_transport", "tcp", "-show_entries",
                                  "stream=width", "-of", "csv=p=0", url], capture_output=True, text=True, timeout=20)
            return out.stdout.strip()

        wrong = send(f"rtsp://tixuan:nope@127.0.0.1:{port}/tixuan", extra=("-rtsp_transport", "tcp"))
        other = send(f"rtsp://tixuan:secretpass@127.0.0.1:{port}/cam1", extra=("-rtsp_transport", "tcp"))
        assert wrong.wait(20) != 0 and other.wait(20) != 0             # wrong password; a path that is not its own
        good = send(f"rtsp://tixuan:secretpass@127.0.0.1:{port}/tixuan", extra=("-rtsp_transport", "tcp"))
        time.sleep(3)
        assert probe(rec.read_url("tixuan")) == "320"                  # this app reads it
        assert probe(f"rtsp://127.0.0.1:{port}/tixuan") == ""          # nobody else may
        good.kill()
        phone = send(f"rtmp://127.0.0.1:{rtmp}/tixuan?user=tixuan&pass=secretpass", fmt="flv")   # as a phone app does
        time.sleep(3)
        assert probe(rec.read_url("tixuan")) == "320"
        phone.kill()
    finally:
        rec.stop()


def test_health_line(tmp_path):
    from types import SimpleNamespace

    from labeler.healthcheck import check_receiver
    cfg = _cfg(tmp_path)
    cams = [SimpleNamespace(name=c["name"], cam=c, last_frame_at=-1e9, frames=0) for c in cfg["cameras"]]
    assert check_receiver(cfg, cams[:1]) is None                          # no camera is sent: no line
    stopped = SimpleNamespace(running=lambda: False, problem="", restarts=0)
    assert check_receiver(cfg, cams, rec=stopped)["level"] == "problem"
    rec = SimpleNamespace(running=lambda: True, problem="", restarts=0)
    cams[2].last_frame_at = 104.0                                         # the loop just started: no frame yet
    cams[1].last_frame_at, cams[1].frames = 100.0, 50
    line = check_receiver(cfg, cams, rec=rec, now=105.0)
    assert line["level"] == "ok" and line["detail"] == "sending now: tixuan; waiting for: phone"
    netsh = ("Rule Name:                            VisionLabeler cameras (Tailscale)\nEnabled:                              Yes\n"
             "Direction:                            In\nProfiles:                             Domain,Private,Public\n"
             "LocalIP:                              Any\nRemoteIP:                             100.64.0.0/255.192.0.0\n"
             "Protocol:                             TCP\nLocalPort:                            8554\nAction:                               Allow\n")
    line = check_receiver(cfg, cams, rec=rec, firewall=netsh, now=105.0)
    assert line["level"] == "warning" and "reach port 1935" in line["detail"]
