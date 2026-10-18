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


# --- step 2: stored problems, the Health page, the Live page ------------------------------------------------------

def test_problems_are_stored_as_pc_events_and_closed(db, tmp_path):
    run = Run(gpu="88, 200, 1, 2, RTX\n")
    w = PCWatch(_cfg(tmp_path), run=run, booted=None, db=db)
    w.read_gpu(), w.read_disks()
    opened = w.sync()
    assert [e["kind"] for e in opened] == ["hot"] and opened[0]["camera"] == "PC" and w.sync() == []
    assert [e["message"] for e in w.current()] == [opened[0]["message"]]
    run.gpu = "82, 200, 1, 2, RTX\n"   # cooler, but not 5 °C below the limit: still hot
    w.read_gpu()
    assert w.sync() == [] and len(w.current()) == 1
    run.gpu = None                     # nvidia-smi gone: no answer is not "fine"
    w.read_gpu()
    assert w.sync() == [] and len(w.current()) == 1
    run.gpu = "60, 50, 1, 2, RTX\n"
    w.read_gpu()
    w.sync()
    assert w.current() == []
    events = db.list_camera_events()
    assert len(events) == 1 and events[0]["ended_at"] is not None


def test_an_unexpected_stop_is_stored_from_the_last_beat_until_the_start(db, tmp_path):
    cfg = _cfg(tmp_path)
    (tmp_path / "heartbeat.json").write_text(json.dumps({"last": "2026-10-18T09:00:00", "started": "2026-10-18T08:00:00",
                                                         "ended": None}), encoding="utf-8")
    db.open_camera_event("PC", "hot", "left open by the last run")
    PCWatch(cfg, run=Run(), clock=lambda: NOW, booted=NOW - timedelta(hours=1), db=db)
    events = {e["kind"]: e for e in db.list_camera_events()}
    assert events["hot"]["ended_at"] is not None   # closed: the last run ended
    s = events["stopped"]
    assert s["camera"] == "PC" and s["started_at"] == "2026-10-18T09:00:00" and s["ended_at"] == "2026-10-18T12:00:00"
    assert "the PC was turned off or restarted around 2026-10-18 09:00" in s["message"]


def test_health_lines(db, tmp_path):
    from labeler.healthcheck import check_pc, check_stops
    w = PCWatch(_cfg(tmp_path), run=Run(), clock=lambda: NOW, booted=None, db=db)
    assert check_pc(w)["detail"].startswith("not read yet") and check_pc(None) is None
    w.read_gpu(), w.read_disks()
    line = check_pc(w)
    assert line["level"] == "ok" and "graphics card 35 °C, 10 W (hottest since start 35 °C)" in line["detail"]
    assert "disks: KINGSTON SNV3S1000G Healthy, WDC WD10EZEX Healthy" in line["detail"]
    w.run = Run(disks='{"FriendlyName":"WDC","HealthStatus":"Unhealthy","OperationalStatus":"OK"}')
    w.read_disks()
    line = check_pc(w)
    assert line["level"] == "problem" and line["detail"].startswith("disk WDC: Windows says Unhealthy;")
    assert check_stops(db, NOW) == {"name": "unexpected stops (7 days)", "level": "ok", "detail": "none"}
    db.add_camera_event("PC", "stopped", "the app stopped without warning around 2026-10-17 22:10",
                        datetime(2026, 10, 17, 22, 10), datetime(2026, 10, 17, 22, 15))
    db.add_camera_event("PC", "stopped", "too old", datetime(2026, 10, 1), datetime(2026, 10, 1, 1))
    line = check_stops(db, NOW)
    assert line["level"] == "warning" and line["detail"] == ("1; the last: the app stopped without warning around "
                                                             "2026-10-17 22:10, back at 22:15")


def test_status_and_health_api_show_the_pc(db, tmp_path):
    from fastapi.testclient import TestClient
    from labeler.web import create_app
    cfg = _cfg(tmp_path)
    cfg["storage"]["crop_dir"] = str(tmp_path / "crops")
    app = create_app(cfg, db, [])
    w = app.state.pcwatch = PCWatch(cfg, run=Run(gpu="95, 1, 1, 2, RTX\n"), booted=None, db=db)
    w.read_gpu(), w.read_disks(), w.sync()
    client = TestClient(app, client=("127.0.0.1", 5000))
    problems = client.get("/api/status").json()["camera_problems"]
    assert [(p["camera"], p["kind"]) for p in problems] == [("PC", "hot")]
    names = {c["name"]: c for c in client.get("/api/health").json()["checks"]}
    assert names["PC"]["level"] == "warning" and names["unexpected stops (7 days)"]["detail"] == "none"
    assert client.get("/api/camera-events").json()[0]["camera"] == "PC"
