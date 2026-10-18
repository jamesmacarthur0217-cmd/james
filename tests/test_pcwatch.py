"""PC watch (Phase 38 step 1): GPU and disk readings from made-up command outputs, problems, the heartbeat and how the
last run ended."""
import copy
import json
import subprocess
import threading
from datetime import datetime, timedelta

from labeler import pcwatch
from labeler.config import DEFAULTS
from labeler.pcwatch import PCWatch, describe_stop, parse_disks, parse_gpu, previous_stop

NOW = datetime(2026, 10, 18, 12, 0, 0)
GPU_OK = "35, 9.80, 1472, 12282, NVIDIA GeForce RTX 4070 SUPER\n"
DISKS_OK = ('[{"FriendlyName":"KINGSTON SNV3S1000G","MediaType":"SSD","HealthStatus":"Healthy","OperationalStatus":"OK",'
            '"Size":1000204886016},{"FriendlyName":"WDC WD10EZEX","MediaType":"HDD","HealthStatus":"Healthy",'
            '"OperationalStatus":"OK","Size":1000204886016}]')


class Run:
    """subprocess.run stand-in: answers by program name; raises FileNotFoundError for missing programs."""

    def __init__(self, gpu=GPU_OK, disks=DISKS_OK, gpu_code=0):
        self.gpu, self.disks, self.gpu_code, self.calls = gpu, disks, gpu_code, []

    def __call__(self, args, **kw):
        self.calls.append(args[0])
        if args[0] == "nvidia-smi":
            if self.gpu is None:
                raise FileNotFoundError("nvidia-smi")
            return subprocess.CompletedProcess(args, self.gpu_code, self.gpu, "NVIDIA-SMI has failed" if self.gpu_code else "")
        return subprocess.CompletedProcess(args, 0, self.disks, "")


def _cfg(tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["pcwatch"]["heartbeat_file"] = str(tmp_path / "heartbeat.json")
    return cfg


def test_parse_gpu_and_disks():
    g = parse_gpu(GPU_OK + "[N/A], [N/A], 10, 4096, Some, Card\n")
    assert g[0] == {"name": "NVIDIA GeForce RTX 4070 SUPER", "temp_c": 35.0, "power_w": 9.8, "mem_used_mb": 1472.0,
                    "mem_total_mb": 12282.0}
    assert g[1]["temp_c"] is None and g[1]["name"] == "Some, Card"
    d = parse_disks(DISKS_OK)
    assert [x["name"] for x in d] == ["KINGSTON SNV3S1000G", "WDC WD10EZEX"] and d[0]["size_gb"] == 1000
    one = parse_disks('{"FriendlyName":"X","HealthStatus":"Warning","OperationalStatus":["OK","Predictive Failure"]}')
    assert one == [{"name": "X", "media": None, "health": "Warning", "status": "OK, Predictive Failure", "size_gb": 0}]
    assert parse_disks("") == []


def test_problems_hot_gpu_and_failing_disk(tmp_path):
    w = PCWatch(_cfg(tmp_path), run=Run(gpu="91, 220.5, 9000, 12282, RTX\n",
                                        disks='{"FriendlyName":"WDC","HealthStatus":"Warning","OperationalStatus":"Degraded"}'),
                clock=lambda: NOW, booted=NOW - timedelta(days=1))
    assert w.problems() == []   # nothing read yet
    w.read_gpu(), w.read_disks()
    got = w.problems()
    assert len(got) == 2 and "91 °C" in got[0] and "disk WDC: Windows says Warning (Degraded)" in got[1]
    s = w.state()
    assert s["gpu_max_c"] == 91.0 and s["gpus"][0]["power_w"] == 220.5 and s["disk_at"] == "2026-10-18T12:00:00"


def test_no_nvidia_tools_or_a_failing_one(tmp_path):
    w = PCWatch(_cfg(tmp_path), run=Run(gpu=None), clock=lambda: NOW, booted=NOW)
    w.read_gpu()
    assert w.gpus == [] and "not found" in w.gpu_error and w.problems() == []
    w.run = Run(gpu_code=9)
    w.read_gpu()
    assert w.gpu_error == "nvidia-smi failed: NVIDIA-SMI has failed"


def test_heartbeat_and_how_the_last_run_ended(tmp_path):
    cfg = _cfg(tmp_path)
    path = tmp_path / "heartbeat.json"
    assert previous_stop(path) is None   # never ran
    clock = [NOW]
    w = PCWatch(cfg, run=Run(), clock=lambda: clock[0], booted=NOW - timedelta(hours=5))
    assert w.last_stop is None
    w.beat()
    clock[0] = NOW + timedelta(minutes=3)
    w.beat()
    assert json.loads(path.read_text())["last"] == "2026-10-18T12:03:00"
    # Killed (no end mark), the PC still up since before: the app stopped.
    stop = previous_stop(path, booted=NOW - timedelta(hours=5))
    assert stop == {"kind": "unexpected", "last": "2026-10-18T12:03:00", "started": "2026-10-18T12:00:00",
                    "pc_restarted": False}
    assert "the app stopped without warning around 2026-10-18 12:03" in describe_stop(stop)
    # The PC started after the last beat: a power cut or a shutdown without stop.bat.
    stop = previous_stop(path, booted=NOW + timedelta(minutes=20))
    assert stop["pc_restarted"] and "the PC was turned off or restarted around 2026-10-18 12:03" in describe_stop(stop)
    again = PCWatch(cfg, run=Run(), clock=lambda: NOW + timedelta(minutes=25), booted=NOW + timedelta(minutes=20))
    assert again.last_stop["pc_restarted"] is True   # read at start, before its own first beat
    w.mark_ended()
    assert previous_stop(path) is None
    w.mark_ended(crashed=True)
    assert "after a problem" in describe_stop(previous_stop(path))
    path.write_text("half a fi", encoding="utf-8")
    assert previous_stop(path) is None   # unreadable: no report rather than a wrong one


def test_loop_beats_reads_and_logs_a_problem_once(tmp_path, caplog):
    run = Run(gpu="90, 100, 1, 2, RTX\n")
    w = PCWatch(_cfg(tmp_path), run=run, booted=None)
    w.every_s = 0.01
    stopping = threading.Event()
    t = threading.Thread(target=w.loop, args=(stopping,))
    with caplog.at_level("WARNING", logger="labeler.pcwatch"):
        t.start()
        threading.Event().wait(0.3)
        stopping.set()
        t.join(5)
    assert run.calls.count("powershell") == 1 and run.calls.count("nvidia-smi") > 3   # disks hourly, GPU each turn
    assert sum("graphics card is hot" in r.message for r in caplog.records) == 1
    assert (tmp_path / "heartbeat.json").exists()


def test_boot_time_is_in_the_past():
    b = pcwatch.boot_time()
    assert b is None or b < datetime.now()
