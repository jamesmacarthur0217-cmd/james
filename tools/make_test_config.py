"""Write a test config for trying the app without touching the user's data.

    python tools/make_test_config.py <folder> [--https] [--network] [--two-cameras] [--fast-health]

Copies config.yaml with the database (SQLite), pictures, recordings, log files, Settings page file and HTTPS certificate
inside <folder>, and the app on
this PC only (web.host 127.0.0.1, web.https false): the real config opens the app to the network with HTTPS.
Then:  python run.py --config <folder>/test_config.yaml --source samples/bus.jpg --port 8001
Stop:  curl -X POST http://127.0.0.1:8001/api/shutdown   (and check no python.exe run.py is left running)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--https", action="store_true", help="HTTPS on port 8444, certificate in <folder>/tls")
    ap.add_argument("--network", action="store_true", help="web.host 0.0.0.0 (stays local without accounts)")
    ap.add_argument("--two-cameras", action="store_true", help="cam1 = zidane.jpg, cam2 = bus.jpg")
    ap.add_argument("--fast-health", action="store_true", help="camera problems after 5 s instead of 10-30 s")
    args = ap.parse_args()
    folder = args.folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["database"]["url"] = "sqlite:///" + (folder / "test.db").as_posix()
    cfg.setdefault("storage", {})["crop_dir"] = (folder / "crops").as_posix()
    cfg["web"].update({"host": "127.0.0.1", "https": False})
    cfg.setdefault("logs", {})["dir"] = (folder / "logs").as_posix()
    cfg["settings_file"] = (folder / "settings.yaml").as_posix()   # the Settings page writes here, not data/
    cfg.setdefault("backup", {})["dir"] = (folder / "backups").as_posix()   # never the user's backups
    cfg.setdefault("recording", {})["dir"] = (folder / "recordings").as_posix()   # nor their recordings
    cfg.setdefault("student", {})["retrained_path"] = (folder / "models" / "attributes.joblib").as_posix()
    if args.https:
        cfg["web"].update({"https": True, "https_port": 8444, "cert_file": (folder / "tls" / "cert.pem").as_posix(),
                           "key_file": (folder / "tls" / "key.pem").as_posix()})
    if args.network:
        cfg["web"]["host"] = "0.0.0.0"
    if args.two_cameras:
        cfg["cameras"] = [{"name": "cam1", "source": "samples/zidane.jpg"}, {"name": "cam2", "source": "samples/bus.jpg"}]
    if args.fast_health:
        cfg.setdefault("health", {}).update({"startup_grace_s": 5, "offline_after_s": 5, "blank_after_s": 5})
    out = folder / "test_config.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
