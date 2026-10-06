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
            "long_gap_min_similarity": 0.90,
        },
    },
    "vlm": {
        "enabled": True,
        "host": "http://localhost:11434",
        "model": "qwen3-vl:4b-instruct",
        "frames_per_track": 2,
        "workers": 2,
        "relabel_every_s": 15,
        "history_frames": 4,
        "timeout_s": 90,
        "max_queue": 32,
    },
    "embeddings": {"enabled": True, "model": "google/siglip2-base-patch16-224", "device": "cuda"},
    "student": {"enabled": False, "model_path": "models/attributes.joblib", "every_n_frames": 15,
                "skip_vlm_when_confident": True},
    "review": {"confidence_threshold": 0.6},
    "database": {"url": "sqlite:///data/vision.db"},
    "storage": {"crop_dir": "data/crops", "save_face_crops": False, "retention_days": 30},
    "web": {"host": "127.0.0.1", "port": 8000},
}


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
    """Replace ${VAR} in config strings with environment variables, so secrets stay in .env."""
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            if m.group(1) not in os.environ:
                raise KeyError(f"config uses ${{{m.group(1)}}} but it is not set (see .env)")
            return os.environ[m.group(1)]
        return re.sub(r"\$\{(\w+)\}", sub, value)
    return value


def load_config(path: str | Path | None = None) -> dict:
    load_dotenv()
    cfg = copy.deepcopy(DEFAULTS)
    path = Path(path) if path else ROOT / "config.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    cfg = _expand(cfg)
    if cfg["mode"] not in ("kiosk", "security"):
        raise ValueError(f"mode must be 'kiosk' or 'security', got {cfg['mode']!r}")
    return cfg


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
