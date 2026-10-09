"""Settings changed on the Settings page (admin), kept in data/settings.yaml on top of config.yaml.

config.yaml is never rewritten (it would lose its comments): it stays the commented list of defaults, and only
the values changed on the page go to data/settings.yaml, which load_config() merges over it. Delete that file
to go back to config.yaml. Changes apply after a restart (the page has a Restart now button).

Only the common settings are editable here, each validated: cameras (add, remove, source, mode, retention,
detector size), how long people and the audit log are kept, face blurring, HTTPS, weekly retraining. Zones and
alert rules have their own pages; accounts are managed with tools/users.py.

A camera keeps its name for good (zones, alert rules and recorded people refer to it); removing a camera keeps
its recorded people until their retention runs out. Passwords in camera addresses are shown masked
(rtsp://user:•••@host); sending the mask back unchanged keeps the stored password.
"""
from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .config import CAMERA_DEFAULTS, MASK, ROOT, mask_password, resolve

DEFAULT_FILE = "data/settings.yaml"
CAMERA_NAME = re.compile(r"[A-Za-z0-9_-]{1,32}")
URL_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://")
IMGSZ = (640, 960, 1280)
MAX_DAYS = 3650
HEADER = ("# Written by the Settings page (only the values changed there; config.yaml has the rest, with comments).\n"
          "# Delete this file to go back to config.yaml. Changes take effect after a restart.\n")


class CameraIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    source: str | int
    mode: str = "kiosk"
    retention_days: int | None = Field(default=None, ge=0, le=MAX_DAYS)
    imgsz: int | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not CAMERA_NAME.fullmatch(v):
            raise ValueError("a camera name is 1-32 letters, digits, - or _")
        return v

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        if v not in ("kiosk", "security"):
            raise ValueError("mode is kiosk or security")
        return v

    @field_validator("imgsz")
    @classmethod
    def _imgsz(cls, v: int | None) -> int | None:
        if v is not None and v not in IMGSZ:
            raise ValueError(f"detector size is one of {', '.join(map(str, IMGSZ))}")
        return v


class SettingsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cameras: list[CameraIn] = Field(min_length=1, max_length=16)
    retention_days: int = Field(ge=0, le=MAX_DAYS)
    audit_retention_days: int = Field(ge=0, le=MAX_DAYS)
    blur_faces: bool
    blur_faces_for_viewers: bool
    https: bool
    retrain_enabled: bool


# Page field -> (section, key) in the config.
SIMPLE = {
    "retention_days": ("storage", "retention_days"),
    "audit_retention_days": ("storage", "audit_retention_days"),
    "blur_faces": ("storage", "blur_faces"),
    "blur_faces_for_viewers": ("web", "blur_faces_for_viewers"),
    "https": ("web", "https"),
    "retrain_enabled": ("retrain", "enabled"),
}
CAMERA_FIELDS = ("name", "source", "mode", "retention_days", "imgsz")


def settings_path(cfg: dict) -> Path | None:
    """Where this app's settings file is (None: this config does not use one, e.g. a test config)."""
    p = cfg.get("_settings_path")
    return Path(p) if p else None


def read(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold settings")
    return data


def write(path: Path, overlay: dict) -> None:
    """Atomically: a half-written file must never stop the next start."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(HEADER + (yaml.safe_dump(overlay, sort_keys=False, allow_unicode=True) if overlay else ""),
                   encoding="utf-8")
    tmp.replace(path)


def overlay_keys(overlay: dict) -> list[str]:
    """Settings in the file that the page does not manage (hand-edited): reported, not applied."""
    allowed = {"cameras": None, **{s: set() for s, _ in SIMPLE.values()}}
    for section, key in SIMPLE.values():
        allowed[section].add(key)
    unknown = []
    for k, v in overlay.items():
        if k not in allowed:
            unknown.append(k)
        elif allowed[k] is not None:
            unknown += [f"{k}.{kk}" for kk in (v or {}) if kk not in allowed[k]] if isinstance(v, dict) else [k]
    return unknown


def apply(cfg: dict, overlay: dict) -> dict:
    """Merge the settings file over the loaded config.yaml (cameras replace the whole list)."""
    for k in overlay_keys(overlay):
        cfg.setdefault("_settings_ignored", []).append(k)
    if isinstance(overlay.get("cameras"), list):
        cfg["cameras"] = copy.deepcopy(overlay["cameras"])
    for section, key in SIMPLE.values():
        if isinstance(overlay.get(section), dict) and key in overlay[section]:
            cfg[section][key] = overlay[section][key]
    return cfg


def current(cfg: dict) -> dict:
    """The page's values from a loaded config (camera passwords masked)."""
    out = {field: cfg[s][k] for field, (s, k) in SIMPLE.items()}
    out["cameras"] = [{"name": c["name"], "source": mask_password(c["source"]) if isinstance(c["source"], str)
                       else c["source"], "mode": c.get("mode", cfg["mode"]),
                       "retention_days": c.get("retention_days"), "imgsz": c.get("imgsz")} for c in cfg["cameras"]]
    return out


def check_source(source: str | int, stored: str | int | None) -> str | int:
    """A webcam number, a file that exists, or a camera address. A masked password means "keep the stored one"."""
    if isinstance(source, int) or (isinstance(source, str) and source.strip().isdigit()):
        n = int(source)
        if not 0 <= n <= 9:
            raise ValueError("a webcam number is 0-9")
        return n
    s = source.strip()
    if not s:
        raise ValueError("the source is empty")
    if "${" in s:
        raise ValueError("${...} is for config.yaml; type the address itself")
    if s.lower().startswith(URL_SCHEMES):
        if MASK in s:
            if not isinstance(stored, str) or mask_password(stored) != s:
                raise ValueError("the address changed: type its password again")
            return stored
        if not re.match(r"^[a-z]+://([^/@\s]*@)?[^/:@\s]+", s, re.I):
            raise ValueError("this camera address has no host")
        return s
    if not resolve(s).is_file():
        raise ValueError(f"no file {s!r} (a path relative to {ROOT}, or a full path)")
    return s


def build(body: SettingsIn, base: dict, effective: dict) -> tuple[dict, list[str]]:
    """The new settings file (only what differs from config.yaml) and the changes in words, for the audit log.

    base: config.yaml without the settings file; effective: with it (what the next start would use now)."""
    names = [c.name for c in body.cameras]
    if len(set(names)) != len(names):
        raise ValueError("two cameras have the same name")
    stored = {c["name"]: c for c in effective["cameras"]}
    cameras = []
    for cam in body.cameras:
        old = stored.get(cam.name)
        try:
            source = check_source(cam.source, old["source"] if old else None)
        except ValueError as e:
            raise ValueError(f"camera {cam.name}: {e}") from None
        c = {"name": cam.name, **CAMERA_DEFAULTS, **copy.deepcopy(old or {}), "source": source, "mode": cam.mode}
        for key in ("retention_days", "imgsz"):
            if getattr(cam, key) is None:
                c.pop(key, None)
            else:
                c[key] = getattr(cam, key)
        cameras.append(c)

    overlay: dict = {}
    if _cams_key(cameras) != _cams_key(base["cameras"]):
        overlay["cameras"] = cameras
    values = body.model_dump()
    for field, (section, key) in SIMPLE.items():
        if values[field] != base[section][key]:
            overlay.setdefault(section, {})[key] = values[field]

    changes = []
    before = current(effective)
    for field in SIMPLE:
        if before[field] != values[field]:
            changes.append(f"{field}: {before[field]} → {values[field]}")
    old_cams = {c["name"]: c for c in before["cameras"]}
    new_cams = {c["name"]: c for c in current({**effective, "cameras": cameras})["cameras"]}
    for name in old_cams.keys() - new_cams.keys():
        changes.append(f"camera {name} removed")
    for name, c in new_cams.items():
        if name not in old_cams:
            changes.append(f"camera {name} added ({c['source']}, {c['mode']})")
        else:
            changes += [f"camera {name} {k}: {_shown(old_cams[name][k])} → {_shown(c[k])}" for k in CAMERA_FIELDS[1:]
                        if old_cams[name][k] != c[k]]
    return overlay, changes


def _shown(value) -> str:
    """A camera's own retention / detector size, or "general" when it uses the general value (as on the page)."""
    return "general" if value is None else str(value)


def _cams_key(cams: list[dict]) -> list:
    return [{k: c.get(k) for k in CAMERA_FIELDS} | {"other": {k: v for k, v in c.items() if k not in CAMERA_FIELDS}}
            for c in cams]


def validation_message(e: ValidationError) -> str:
    """Pydantic's error list in one plain sentence for the page."""
    parts = []
    for err in e.errors():
        where = ".".join(str(x) for x in err["loc"] if x != "body")
        msg = err["msg"].removeprefix("Value error, ")
        parts.append(f"{where}: {msg}" if where else msg)
    return "; ".join(parts)


def restart_available() -> bool:
    """Restart now needs the supervisor (start.bat), which starts the app again when it exits with code 4."""
    return os.environ.get("VL_SUPERVISED") == "1"
