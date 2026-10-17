from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "camera": {"name": "cam1", "source": 0, "loop_files": True, "ignore_zones": []},
    "mode": "kiosk",
    "detector": {
        "model": "yolo11s-pose.pt",
        "conf": 0.1,
        "imgsz": 640,
        "device": 0,
        "tracker": "labeler/trackers/people.yaml",
        "min_box_area": {"kiosk": 0.03, "security": 0.0},
    },
    "zones": {"min_frames": 3},
    "engagement": {"facing_min": 0.6, "looked_s": 3.0},
    "timeline": {"min_similarity": 0.8, "window_h": 24},
    "clips": {"enabled": True, "before_s": 5, "after_s": 5, "fps": 10, "max_width": 960},
    "person_videos": {"enabled": True, "max_s": 20, "fps": 5, "max_width": 480, "before_s": 1, "after_s": 1},
    "idle": {"enabled": True, "after_s": 60, "fps": 3},
    # Home / Away (labeler/arming.py): Away times each week, e.g. "mon-fri 08:00-18:00"; the pages' button too.
    "arming": {"schedule": []},
    # Cameras sent from other devices (labeler/receiver.py): MediaMTX runs only when a camera has `source: push`.
    "receiver": {"exe": "data/mediamtx/mediamtx.exe", "dir": "data/mediamtx", "rtsp_port": 8554, "rtmp_port": 1935,
                 "publish_from": ["100.64.0.0/10"]},   # who may send: Tailscale addresses
    # Off unless config.yaml (or the Settings page) turns it on: it uses a lot of disk and stores everything seen.
    "recording": {"enabled": False, "when": "people", "before_s": 10, "after_s": 10, "fps": 10, "max_width": 1280,
                  "quality": 28, "encoder": "auto", "max_file_min": 10, "keep_days": 3, "max_gb": 100,
                  "dir": "data/recordings", "ffmpeg": "data/ffmpeg/ffmpeg.exe"},
    "health": {"enabled": True, "offline_after_s": 10, "blank_after_s": 10, "slow_after_s": 30, "min_fps": 5,
               "startup_grace_s": 30},
    "tracking": {
        "min_frames_keep": 8,
        "min_frames_for_label": 20,
        "best_frames": 5,
        "lost_after_s": 4.0,
        "color_every_n_frames": 2,
        "relink": {
            "enabled": True,
            "window_s": 30,
            "check_at_frames": [5, 12],
            "short_gap_s": 5,
            "short_gap_min_similarity": 0.75,
            "short_gap_min_overlap": 0.2,
            # Appearance alone (no position check): far or overhead views need a much higher bar (README).
            "long_gap_min_similarity": {"kiosk": 0.90, "security": 0.98},
        },
    },
    "vlm": {
        "enabled": True,
        "host": "http://127.0.0.1:11434",   # not "localhost": on Windows that adds ~2 s per request
        "model": "qwen3-vl:4b-instruct",
        "frames_per_track": 2,
        "workers": 1,
        "relabel_every_s": 15,
        "history_frames": 4,
        "timeout_s": 90,
        "max_queue": 32,
    },
    "labels": {"mobility_aids": True},
    # Vehicles (labeler/vehicles.py): a second detector, on security cameras by default (cameras[].vehicles).
    "vehicles": {"enabled": True, "model": "yolo11s.pt", "tracker": "labeler/trackers/vehicles.yaml", "imgsz": 640,
                 "every_n_frames": 2, "kinds": ["car", "motorbike", "bicycle", "bus", "lorry"], "min_frames": 5,
                 "lost_after_s": 4.0, "parked_after_s": 60.0, "parked_memory_s": 3600.0, "move_ratio": 0.2,
                 "still_s": 5.0, "still_memory_s": 120.0, "describe": True, "describe_after_s": 5.0},
    # Animals (labeler/animals.py): from the vehicle detector in the same pass, security cameras by default
    # (cameras[].animals). Followed like vehicles: parked_after_s = "resting" (one record however long it lies there).
    "animals": {"enabled": True, "kinds": ["dog", "cat", "bird"], "min_frames": 5, "lost_after_s": 4.0,
                "parked_after_s": 60.0, "parked_memory_s": 3600.0, "move_ratio": 0.2, "still_s": 5.0,
                "still_memory_s": 120.0, "describe_after_s": 5.0},
    # Scene questions (labeler/scene.py): checked every check_every_s; an alert after answers_in_a_row answers that
    # match (the user's choice: 2, against single wrong answers). New questions are asked every 5 minutes by default.
    "scene": {"enabled": True, "check_every_s": 10, "answers_in_a_row": 2, "frame_every_s": 2.0, "frame_max_age_s": 30,
              "retry_after_s": 60},
    # Camera picture check (labeler/picture_check.py): a frame every every_s; a finding is reported when it was in
    # lasting_share of the checks of the last window_s (Health line, Live banner).
    "picture_check": {"enabled": True, "every_s": 30.0, "window_s": 600.0, "lasting_share": 0.8},
    # Fall alerts (labeler/falls.py): upright -> lying within drop_s, the head down min_drop of standing height,
    # then down for after_s. Per camera: cameras[].fall_alert (default on). Measured: tools/fall_check.py.
    # Help sign (labeler/gestures.py): one hand above the head held hold_s; built-in rule "Help sign". Per camera:
    # cameras[].help_sign (default on).
    "help_sign": {"enabled": True, "hold_s": 3.0, "gap_s": 0.5, "rearm_s": 2.0, "margin": 0.15, "both_ok": True,
                  "max_tilt": 45.0},
    # Update button (labeler/updater.py): check GitHub for a new version once a day (Settings page; Health line).
    "updates": {"check": True},
    # What's new (labeler/whatsnew.py): the newest feature the app ran last time, to say what an update added.
    "whats_new": {"state_file": "data/last_version.json"},
    # Search recordings by photo (Phase 33): people at least this alike (cosine, SigLIP 2) are listed, closest first;
    # 0.85 chosen on MOT16-02/04/05 (tools/photo_match_check.py). At most `limit` people.
    "photo_search": {"min_similarity": 0.85, "limit": 20},
    # No movement (labeler/quiet.py): the built-in "No movement" rule is made at start; checked every check_every_s.
    "quiet": {"enabled": True, "check_every_s": 60.0},
    "falls": {"enabled": True, "after_s": 10.0, "drop_s": 1.0, "min_drop": 0.4, "getup_s": 1.0, "lying_angle": 60.0,
              "low_ratio": 0.5, "lost_s": 0.5, "min_width": 1.25},   # optional fields (schema.OPTIONAL_FIELDS); off = not asked, not stored
    "embeddings": {"enabled": True, "model": "google/siglip2-base-patch16-224", "device": "cuda"},
    "reid": {"weights": "data/models/reid/osnet_ain_x1_0.pt", "device": "cuda"},   # tools/relink_check.py --reid
    "age_model": {"enabled": True, "model": "dima806/fairface_age_image_detection"},
    "student": {"enabled": False, "model_path": "models/attributes.joblib", "every_n_frames": 15,
                "skip_vlm_when_confident": True, "retrained_path": "data/models/attributes.joblib"},
    "retrain": {"enabled": True, "every_days": 7, "hour": 3, "own_weight": 3.0, "min_own_people": 30,
                "min_answer_key": 10, "target": 0.9, "coco_dir": "data/datasets/coco_people"},
    "review": {"confidence_threshold": 0.6, "spot_check_every": 5},
    "database": {"url": "sqlite:///data/vision.db"},
    "logs": {"dir": "data/logs", "max_mb": 10, "keep": 5},
    "backup": {"enabled": True, "hour": 2, "keep": 7, "copy_to": "", "dir": "data/backups"},
    "storage": {"crop_dir": "data/crops", "save_face_crops": False, "retention_days": 30, "min_free_gb": 5,
                "audit_retention_days": 365, "blur_faces": False},
    "web": {"host": "127.0.0.1", "port": 8000, "username": "admin", "password": "", "session_hours": 12,
            "https": False, "https_port": 8443, "cert_file": "data/tls/cert.pem", "key_file": "data/tls/key.pem",
            "blur_faces_for_viewers": True, "health_token": ""},
}

MASK = "•••"
_URL_PASSWORD = re.compile(r"(\b[a-z][a-z0-9+.-]*://[^:/@\s]*:)([^@\s]+)(@)", re.I)


def mask_password(text: str) -> str:
    """rtsp://user:secret@host/... → rtsp://user:•••@host/... (for pages, the audit log and the health check)."""
    return _URL_PASSWORD.sub(lambda m: m.group(1) + MASK + m.group(3), str(text))


def _merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load KEY=value lines from .env into the environment (existing variables win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _expand(value):
    """Replace ${VAR} (or ${VAR:-default}) in config strings with environment variables,
    so secrets stay in .env."""
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            name, default = m.group(1), m.group(3)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise KeyError(f"config uses ${{{name}}} but it is not set (see .env)")
        return re.sub(r"\$\{(\w+)(:-([^}]*))?\}", sub, value)
    return value


def load_config(path: str | Path | None = None, use_settings: bool = True) -> dict:
    """config.yaml (or `path`) over the defaults, then the Settings page's changes (labeler/settings.py) over
    that. The settings file is data/settings.yaml for the project's own config.yaml; another config uses one only
    if it names it (`settings_file:`), so a test config never picks up the real one."""
    load_dotenv()
    cfg = copy.deepcopy(DEFAULTS)
    path = Path(path).resolve() if path else ROOT / "config.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    cfg = _expand(cfg)
    cfg["_config_path"] = str(path)
    settings_file = cfg.get("settings_file") or ("data/settings.yaml" if path == ROOT / "config.yaml" else None)
    if settings_file:
        from . import settings
        cfg["_settings_path"] = str(resolve(settings_file))
        if use_settings:
            # Not expanded: values from the page are stored as typed, never read as ${VARIABLES}. Tried on a copy
            # first: a broken (hand-edited) settings file must not stop the app from starting.
            try:
                overlay = settings.read(resolve(settings_file))
                trial = settings.apply(copy.deepcopy(cfg), overlay)
                normalize_cameras(trial)
                cfg = trial
                cfg["_settings_loaded"] = overlay   # to tell later whether the file changed (restart needed)
            except Exception as e:
                cfg["_settings_error"] = f"{settings_file} could not be used ({e}); using config.yaml only"
    cfg["cameras"] = normalize_cameras(cfg)
    if cfg["mode"] not in ("kiosk", "security"):
        raise ValueError(f"mode must be 'kiosk' or 'security', got {cfg['mode']!r}")
    from .arming import parse_schedule
    try:
        parse_schedule(cfg["arming"]["schedule"])
    except ValueError as e:
        raise ValueError(f"arming.schedule: {e}") from None
    return cfg


CAMERA_DEFAULTS = {"source": 0, "loop_files": True, "ignore_zones": []}


def normalize_cameras(cfg: dict) -> list[dict]:
    """The cameras to run: the `cameras:` list, or the single `camera:` block. Each gets a unique name
    and a mode (its own, or the global one)."""
    cams = cfg.get("cameras") or [cfg["camera"]]
    out, names = [], set()
    for i, cam in enumerate(cams):
        c = {**CAMERA_DEFAULTS, **cam}
        c.setdefault("name", f"cam{i + 1}")
        c["name"] = str(c["name"])
        if c["name"] in names:
            raise ValueError(f"two cameras are named {c['name']!r}; names must be unique")
        names.add(c["name"])
        c.setdefault("mode", cfg["mode"])
        if c["mode"] not in ("kiosk", "security"):
            raise ValueError(f"camera {c['name']}: mode must be 'kiosk' or 'security'")
        out.append(c)
    return out


def resolve(p: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def sqlite_url(url: str) -> str:
    """Make relative sqlite:/// paths relative to the project root."""
    prefix = "sqlite:///"
    if url.startswith(prefix) and not Path(url[len(prefix):]).is_absolute():
        db_path = resolve(url[len(prefix):])
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return prefix + db_path.as_posix()
    return url
