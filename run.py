"""Start the labeling pipeline and the web UI.

    python run.py                          # use config.yaml
    python run.py --source video.mp4       # override the camera source
    python run.py --source 0 --mode security
    python run.py --source clip.mp4 --no-web --max-frames 300   # headless test run
    python run.py --list-cameras           # which webcam numbers give a picture

Stop with the Stop button in the web page, stop.bat, or Ctrl+C in this window.
"""
from __future__ import annotations

import os

# Must be set before OpenCV opens a camera; otherwise many USB webcams take ~8 s to open.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import argparse  # noqa: E402
import logging  # noqa: E402
import threading  # noqa: E402

from labeler.config import load_config, resolve
from labeler.db import Database
from labeler.pipeline import Pipeline
from labeler.vlm import VLMClient, VLMWorker

log = logging.getLogger("run")


def main() -> None:
    ap = argparse.ArgumentParser(description="Local camera labeling system")
    ap.add_argument("--config", help="path to config.yaml")
    ap.add_argument("--source", help="webcam index, video/image path, or rtsp:// URL")
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
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    if args.source is not None:
        cfg["camera"]["source"] = args.source
    if args.mode:
        cfg["mode"] = args.mode
    if args.no_vlm:
        cfg["vlm"]["enabled"] = False
    if args.port:
        cfg["web"]["port"] = args.port

    db = Database(cfg["database"]["url"])
    crop_dir = resolve(cfg["storage"]["crop_dir"])
    removed = db.cleanup(cfg["storage"]["retention_days"])
    for name in removed:
        (crop_dir / name).unlink(missing_ok=True)
    if removed:
        log.info("retention: deleted %d old crop files", len(removed))

    worker = None
    pipeline: Pipeline | None = None
    if cfg["vlm"]["enabled"]:
        client = VLMClient(cfg["vlm"], cfg["mode"])
        ok, reason = client.available()
        if ok:
            worker = VLMWorker(client, lambda tid, res: pipeline.on_vlm_result(tid, res), cfg["vlm"]["max_queue"],
                               cfg["vlm"]["workers"])
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

    pipeline = Pipeline(cfg, db, worker, embedder, student)

    if args.no_web:
        try:
            pipeline.run(max_frames=args.max_frames)
        except KeyboardInterrupt:
            pipeline.stop()
        _finish_labeling(worker)
        return

    import uvicorn
    from labeler.web import create_app

    thread = threading.Thread(target=pipeline.run, kwargs={"max_frames": args.max_frames},
                              name="pipeline", daemon=True)
    thread.start()
    web = cfg["web"]
    app = create_app(cfg, db, pipeline, embedder)
    server = uvicorn.Server(uvicorn.Config(app, host=web["host"], port=web["port"], log_level="warning"))
    # Short delay so the Stop request gets its reply before the web server closes.
    app.state.request_shutdown = lambda: threading.Timer(0.5, setattr, (server, "should_exit", True)).start()
    # stop.bat waits for this process to exit before stopping the database, so final saves complete.
    pid_file = resolve("data/app.pid")
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    log.info("open http://%s:%s  (stop: Stop button, stop.bat, or Ctrl+C)", web["host"], web["port"])
    try:
        server.run()
    finally:
        log.info("shutting down: saving the people still in view...")
        pipeline.stop()
        thread.join(timeout=15)
        _finish_labeling(worker)
        pid_file.unlink(missing_ok=True)
        log.info("stopped")


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
    main()
