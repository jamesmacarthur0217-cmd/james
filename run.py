"""Start the labeling pipeline(s) and the web UI.

    python run.py                          # use config.yaml (all cameras listed there)
    python run.py --source video.mp4       # one camera, this source
    python run.py --source 0 --mode security
    python run.py --source clip.mp4 --no-web --max-frames 300   # headless test run
    python run.py --list-cameras           # which webcam numbers give a picture

Stop with the Stop button in the web page, stop.bat, or Ctrl+C in this window.
Exit codes: 0 = stopped normally; 3 = a camera loop crashed (start.bat restarts the app); 1 = error.
"""
from __future__ import annotations

import os

from labeler.config import load_dotenv

# .env first: some libraries read their settings once, when imported (e.g. HF_HUB_OFFLINE=1, which stops the
# model update check at startup), and the imports below load them.
load_dotenv()

# Must be set before OpenCV opens a camera; otherwise many USB webcams take ~8 s to open.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import argparse  # noqa: E402
import logging  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from labeler.config import load_config, resolve  # noqa: E402
from labeler.db import Database  # noqa: E402
from labeler.pipeline import Pipeline  # noqa: E402
from labeler.vlm import VLMClient, VLMWorker  # noqa: E402

log = logging.getLogger("run")
EXIT_CRASHED = 3


def main() -> int:
    ap = argparse.ArgumentParser(description="Local camera labeling system")
    ap.add_argument("--config", help="path to config.yaml")
    ap.add_argument("--source", help="run one camera with this source: webcam index, file, or rtsp:// URL")
    ap.add_argument("--mode", choices=["kiosk", "security"])
    ap.add_argument("--no-vlm", action="store_true", help="colors only, no vision-language model")
    ap.add_argument("--no-web", action="store_true", help="run without the web UI")
    ap.add_argument("--max-frames", type=int, help="stop after this many frames")
    ap.add_argument("--port", type=int, help="web UI port (default from config.yaml)")
    ap.add_argument("--list-cameras", action="store_true", help="show which webcam numbers work, then exit")
    args = ap.parse_args()

    if args.list_cameras:
        from labeler.sources import list_cameras
        print("Checking cameras (about 3 s each)...")
        cams = list_cameras()
        for idx, info in cams:
            print(f"  source {idx}: {info}")
        if not cams:
            print("  no cameras found")
        return 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    if args.source is not None:
        cfg["cameras"] = [{**cfg["cameras"][0], "source": args.source}]
    if args.mode:
        cfg["mode"] = args.mode
        for cam in cfg["cameras"]:
            cam["mode"] = args.mode
    if args.no_vlm:
        cfg["vlm"]["enabled"] = False
    if args.port:
        cfg["web"]["port"] = args.port

    if not args.no_web and _port_in_use(cfg["web"]["host"], cfg["web"]["port"]):
        # On Windows a second copy could otherwise share the port silently: two copies fighting over
        # the camera and GPU, and stop.bat would only stop one of them.
        log.error("VisionLabeler (or another program) is already using port %s. Open http://127.0.0.1:%s, "
                  "or stop the running copy with stop.bat first.", cfg["web"]["port"], cfg["web"]["port"])
        return 0
    if not args.no_web and cfg["web"].get("https") and _port_in_use(cfg["web"]["host"], cfg["web"]["https_port"]):
        log.error("another program is already using the HTTPS port %s (web.https_port)", cfg["web"]["https_port"])
        return 1

    crop_dir = resolve(cfg["storage"]["crop_dir"])
    crop_dir.mkdir(parents=True, exist_ok=True)
    _check_disk(crop_dir, cfg["storage"]["min_free_gb"])

    db = Database(cfg["database"]["url"])
    stale = db.close_stale_tracks()
    if stale:
        log.info("marked %d people left active by an earlier unclean stop as finished", stale)
    _apply_retention(db, cfg, crop_dir)

    pipelines: list[Pipeline] = []

    def route_result(track_id: int, results: list) -> None:
        """Send a VLM answer to the camera that owns this person."""
        for p in pipelines:
            if track_id in p.owned_ids:
                p.on_vlm_result(track_id, results)
                return

    worker = None
    if cfg["vlm"]["enabled"]:
        client = VLMClient(cfg["vlm"], cfg["mode"])
        ok, reason = client.available()
        if ok:
            worker = VLMWorker(client, route_result, cfg["vlm"]["max_queue"], cfg["vlm"]["workers"])
            log.info("VLM ready: %s (%d at once)", cfg["vlm"]["model"], cfg["vlm"]["workers"])
        else:
            log.warning("VLM disabled: %s", reason)

    embedder = None
    if cfg["embeddings"]["enabled"]:
        from labeler.embed import Embedder
        embedder = Embedder(cfg["embeddings"])

    student = None
    if cfg["student"]["enabled"] and embedder is not None:
        from labeler.student import Student
        path = resolve(cfg["student"]["model_path"])
        if path.exists():
            student = Student(path, embedder.model_name)
        else:
            log.warning("fast classifiers enabled but %s not found; train them with tools/train_attributes.py", path)

    age_model = None
    if cfg["age_model"]["enabled"]:
        from labeler.age import AgeEstimator
        try:
            age_model = AgeEstimator(cfg["age_model"]["model"])
        except Exception as e:
            log.warning("age model unavailable (%s); age comes from the VLM", e)

    from labeler.alerts import AlertEngine
    alerts = AlertEngine(db)
    pipelines.extend(Pipeline(cfg, db, worker, embedder, student, camera=cam) for cam in cfg["cameras"])
    for p in pipelines:
        p.alerts = alerts
        p.age = age_model
    log.info("cameras: %s", ", ".join(f"{p.name} ({p.cam['source']}, {p.mode})" for p in pipelines))

    threads = [threading.Thread(target=p.run, kwargs={"max_frames": args.max_frames},
                                name=f"pipeline-{p.name}", daemon=True) for p in pipelines]

    if args.no_web:
        for t in threads:
            t.start()
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            for p in pipelines:
                p.stop()
        _finish_labeling(worker)
        return EXIT_CRASHED if any(p.crashed for p in pipelines) else 0

    from labeler.web import check_access, create_app, ensure_first_admin

    web = cfg["web"]
    ensure_first_admin(db, web)          # an old shared password becomes the first admin account
    check_access(web, db.count_users())  # while nobody can log in, stays on this PC only
    for t in threads:
        t.start()
    agent = None
    if worker is not None:   # questions use the same local model as labeling
        from labeler.agent import QueryAgent
        agent = QueryAgent(cfg, db, embedder, [p.name for p in pipelines])
    health = None
    if cfg["health"]["enabled"]:
        from labeler.health import HealthMonitor
        health = HealthMonitor(db, cfg["health"])
    app = create_app(cfg, db, pipelines, embedder, alerts, agent, health)
    server = _web_servers(app, web)

    def request_shutdown() -> None:
        app.state.stopping.set()   # ends live video streams
        # Short delay so the Stop request gets its reply before the web server closes.
        threading.Timer(0.5, setattr, (server, "should_exit", True)).start()

    app.state.request_shutdown = request_shutdown
    exit_code = [0]

    def watchdog() -> None:
        """If a camera loop crashes, stop the app with a code that makes start.bat restart it,
        instead of leaving a web page with a frozen picture."""
        while not server.should_exit:
            for p, t in zip(pipelines, threads):
                if p.crashed or (not t.is_alive() and not p.stop_event.is_set() and args.max_frames is None):
                    log.error("[%s] camera loop stopped unexpectedly; restarting the app", p.name)
                    exit_code[0] = EXIT_CRASHED
                    app.state.stopping.set()
                    server.should_exit = True
                    return
            time.sleep(2)

    threading.Thread(target=watchdog, name="watchdog", daemon=True).start()
    threading.Thread(target=_retention_loop, args=(db, cfg, crop_dir, app.state.stopping),
                     name="retention", daemon=True).start()
    if health is not None:
        from labeler.health import health_loop
        threading.Thread(target=health_loop, args=(health, pipelines, app.state.stopping),
                         name="health", daemon=True).start()
    # stop.bat waits for this process to exit before stopping the database, so final saves complete.
    pid_file = resolve("data/app.pid")
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    log.info("open http://127.0.0.1:%s on this PC  (stop: Stop button, stop.bat, or Ctrl+C)", web["port"])
    if web["host"] in ("127.0.0.1", "localhost", "::1"):
        pass   # this PC only
    elif web.get("https"):
        log.info("other devices: https://<this PC's name or address>:%s", web["https_port"])
    else:
        log.info("other devices: http://<this PC's name or address>:%s (plain HTTP; see web.https)", web["port"])
    try:
        server.run()
    finally:
        app.state.stopping.set()
        log.info("shutting down: saving the people still in view...")
        for p in pipelines:
            p.stop()
        for t in threads:
            t.join(timeout=15)
        _finish_labeling(worker)
        pid_file.unlink(missing_ok=True)
        log.info("stopped")
    return exit_code[0]


RETENTION_EVERY_S = 3600


def _apply_retention(db: Database, cfg: dict, crop_dir) -> None:
    """Delete people (and their pictures) older than their camera's retention_days, and old audit entries."""
    from labeler.privacy import remove_picture
    per_camera = {c["name"]: c["retention_days"] for c in cfg["cameras"] if c.get("retention_days") is not None}
    removed = db.cleanup(cfg["storage"]["retention_days"], per_camera)
    for name in removed:
        remove_picture(crop_dir, name)   # and its blurred copy
    if removed:
        log.info("retention: deleted %d old pictures", len(removed))
    old_audit = db.cleanup_audit(cfg["storage"]["audit_retention_days"])
    if old_audit:
        log.info("retention: deleted %d old audit log entries", old_audit)
    db.cleanup_camera_events(cfg["storage"]["audit_retention_days"])   # camera problems: kept as long


def _retention_loop(db: Database, cfg: dict, crop_dir, stop: threading.Event) -> None:
    """Also apply retention while running: a camera running for weeks must not keep people longer."""
    while not stop.wait(RETENTION_EVERY_S):
        try:
            _apply_retention(db, cfg, crop_dir)
        except Exception:
            log.exception("retention cleanup failed; trying again in an hour")


def _web_servers(app, web: dict):
    """The web server; with web.https also an HTTPS server for other devices. Plain HTTP then listens on this
    PC only (start.bat, stop.bat and the local browser use it, without certificate warnings)."""
    import contextlib

    import uvicorn
    # timeout_graceful_shutdown: don't wait forever for open connections (a browser showing the live video).
    common = {"log_level": "warning", "timeout_graceful_shutdown": 3}
    if not web.get("https"):
        return uvicorn.Server(uvicorn.Config(app, host=web["host"], port=web["port"], **common))

    class FollowerServer(uvicorn.Server):
        """Leaves Ctrl+C to the main server (signal handlers are per process); stops when it stops."""
        @contextlib.contextmanager
        def capture_signals(self):
            yield

    from labeler.tls import ensure_certificate
    cert, key = resolve(web["cert_file"]), resolve(web["key_file"])
    log.info("HTTPS certificate fingerprint (SHA-256): %s", ensure_certificate(cert, key))
    plain = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=web["port"], **common))
    secure = FollowerServer(uvicorn.Config(app, host=web["host"], port=web["https_port"],
                                           ssl_certfile=str(cert), ssl_keyfile=str(key), **common))
    return _ServerPair(plain, secure)


class _ServerPair:
    """Plain HTTP for this PC plus HTTPS for the network, started and stopped together. Has the same
    `should_exit` / `run()` as one uvicorn server, so the Stop button and the watchdog work unchanged."""

    def __init__(self, plain, secure):
        self.plain, self.secure = plain, secure

    @property
    def should_exit(self) -> bool:
        return self.plain.should_exit or self.secure.should_exit

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        self.plain.should_exit = self.secure.should_exit = value

    def run(self) -> None:
        import asyncio

        async def follow():
            while not self.should_exit:
                await asyncio.sleep(0.2)
            self.should_exit = True

        async def both():
            await asyncio.gather(self.plain.serve(), self.secure.serve(), follow())

        asyncio.run(both())


def _port_in_use(host: str, port: int) -> bool:
    target = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((target, port)) == 0


def _check_disk(path, min_free_gb: float) -> None:
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < min_free_gb:
        log.warning("only %.1f GB free on the disk holding %s (warning below %s GB): delete old people, "
                    "lower storage.retention_days, or free up space", free_gb, path, min_free_gb)


def _finish_labeling(worker: VLMWorker | None, wait_s: float = 120) -> None:
    """Let queued VLM jobs finish so the last people seen still get their labels."""
    if worker is None:
        return
    if worker.pending:
        log.info("finishing %d labeling jobs (up to %d s; Ctrl+C again to skip)...", worker.pending, wait_s)
    try:
        if not worker.close(wait_s=wait_s):
            log.warning("some labeling jobs did not finish; those people keep their color labels only")
    except KeyboardInterrupt:
        log.warning("skipped the remaining labeling jobs")


if __name__ == "__main__":
    sys.exit(main())
