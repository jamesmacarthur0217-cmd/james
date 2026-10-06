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

# Must be set before OpenCV opens a camera; otherwise many USB webcams take ~8 s to open.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import argparse  # noqa: E402
import logging  # noqa: E402
import shutil  # noqa: E402
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

    crop_dir = resolve(cfg["storage"]["crop_dir"])
    crop_dir.mkdir(parents=True, exist_ok=True)
    _check_disk(crop_dir, cfg["storage"]["min_free_gb"])

    db = Database(cfg["database"]["url"])
    stale = db.close_stale_tracks()
    if stale:
        log.info("marked %d people left active by an earlier unclean stop as finished", stale)
    removed = db.cleanup(cfg["storage"]["retention_days"])
    for name in removed:
        (crop_dir / name).unlink(missing_ok=True)
    if removed:
        log.info("retention: deleted %d old crop files", len(removed))

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

    from labeler.alerts import AlertEngine
    alerts = AlertEngine(db)
    pipelines.extend(Pipeline(cfg, db, worker, embedder, student, camera=cam) for cam in cfg["cameras"])
    for p in pipelines:
        p.alerts = alerts
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

    import uvicorn
    from labeler.web import check_access, create_app

    web = cfg["web"]
    check_access(web)   # refuses to open the UI to the network without a password
    for t in threads:
        t.start()
    app = create_app(cfg, db, pipelines, embedder, alerts)
    server = uvicorn.Server(uvicorn.Config(app, host=web["host"], port=web["port"], log_level="warning"))
    # Short delay so the Stop request gets its reply before the web server closes.
    app.state.request_shutdown = lambda: threading.Timer(0.5, setattr, (server, "should_exit", True)).start()
    exit_code = [0]

    def watchdog() -> None:
        """If a camera loop crashes, stop the app with a code that makes start.bat restart it,
        instead of leaving a web page with a frozen picture."""
        while not server.should_exit:
            for p, t in zip(pipelines, threads):
                if p.crashed or (not t.is_alive() and not p.stop_event.is_set() and args.max_frames is None):
                    log.error("[%s] camera loop stopped unexpectedly; restarting the app", p.name)
                    exit_code[0] = EXIT_CRASHED
                    server.should_exit = True
                    return
            time.sleep(2)

    threading.Thread(target=watchdog, name="watchdog", daemon=True).start()
    # stop.bat waits for this process to exit before stopping the database, so final saves complete.
    pid_file = resolve("data/app.pid")
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    log.info("open http://%s:%s  (stop: Stop button, stop.bat, or Ctrl+C)", web["host"], web["port"])
    try:
        server.run()
    finally:
        log.info("shutting down: saving the people still in view...")
        for p in pipelines:
            p.stop()
        for t in threads:
            t.join(timeout=15)
        _finish_labeling(worker)
        pid_file.unlink(missing_ok=True)
        log.info("stopped")
    return exit_code[0]


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
