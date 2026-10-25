"""The safety overview (Phase 52, GET /api/safety, admins): on one page, what the safety checks do on each safety camera.

Per safety camera: which checks run and why one does not (turned off on the Settings page, its model missing, no
area to watch), its areas with a role and what that role means for the checks, the floor check's empty picture (when
it was learned), and the latest checks with the vision model's answers. For all cameras: alerts per kind in the last
7 days, with the Check page's "right N of M" where there are checks.

The design agreed with the user (2026-10-25): the latest checks come from memory (ConfirmQueue.recent, the last 50
since the start, as the Health line), no new table; the page is for admins only (camera pictures and alerts).
Everything here only reads; the running app's settings are shown (a saved change counts after a restart).
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

from .zones import AREA, ROLES

# The kinds, in the Settings page's order: (key, words, the config switch (section, key)).
KINDS = [("falls", "falls", ("falls", "enabled")),
         ("down", "people down (lying still, slumped, not moved)", ("safety", "down_enabled")),
         ("children", "children", ("safety", "children_enabled")),
         ("crowd", "crowding", ("safety", "crowd_enabled")),
         ("fire", "fire and smoke", ("safety", "fire_enabled")),
         ("weapons", "dangerous objects", ("safety", "weapons_enabled")),
         ("floor", "floor and exits", ("safety", "floor_enabled")),
         ("equipment", "unsafe equipment", ("safety", "equipment_enabled")),
         ("fights", "fights", ("safety", "fights_enabled")),
         ("behaviour", "suspicious behaviour (climbing, restricted area, pacing, camera covered or turned)",
          ("behaviour", "enabled"))]
DAYS = 7


def role_meaning(role: str, cfg: dict) -> str:
    """What an area's role changes for the checks, in plain words with the running times."""
    s, b = cfg.get("safety") or {}, cfg.get("behaviour") or {}
    child = f"a child here {s.get('child_area_s', 3):g} s is checked"
    text = {
        "walkway": f"floor check: something new on the floor {s.get('floor_hold_s', 30):g} s is checked",
        "exit": f"floor check: something blocking the exit {s.get('exit_hold_s', 120):g} s is checked",
        "cooking": f"a knife or scissors held here is normal; fire or smoke inside it is ignored (a stove); {child}",
        "food_prep": f"a knife or scissors held here is normal; {child}",
        "restricted": f"someone inside {b.get('restricted_s', 3):g} s at night ({b.get('night_hours', '22-06')}) or "
                      f"when Away alerts; {child}",
        "bed": "lying here is normal: no fall or person-down check",
        "fence": "climbing over it is checked by the vision model (critical at night or when Away)",
    }.get(role, "")
    every = float(s.get("equipment_every_s", 600))
    if s.get("equipment_enabled", True):
        round_words = f"looked at for unsafe equipment every {every / 60:g} min"
        text = f"{text}; {round_words}" if text else round_words
    return text


def _kinds(p, cfg: dict) -> list[dict]:
    """Each kind on this camera: "on", "partly" (on, with a note) or "off", and why."""
    areas = [z for z in p.zone_counter.zones if z["kind"] == AREA]
    roles = {z.get("role") for z in areas}
    out = []
    for key, words, (section, option) in KINDS:
        on = bool((cfg.get(section) or {}).get(option, True))
        state, why = ("on", "") if on else ("off", "turned off on the Settings page")
        if key == "falls" and on and getattr(p, "falls", None) is None:
            state, why = "off", "turned off for this camera (Settings page, Cameras)"
        elif key == "fire" and on and getattr(p, "fire_watch", None) is None:
            state, why = "off", "no fire model on this PC (tools/train_fire.py)"
        elif key == "weapons" and on:
            missing = [w for w, path in (("guns", getattr(p, "_gun_path", None)),
                                         ("machetes and swords", getattr(p, "_yoloe_path", None))) if path is None]
            if missing:
                state, why = "partly", f"{' and '.join(missing)} not looked for (no model on this PC)"
        elif key == "floor" and on and not roles & {"walkway", "exit"}:
            state, why = "partly", "no walkway or exit area yet (draw one on the Live page)"
        elif key == "equipment" and on and not any(z.get("role") for z in areas):
            state, why = "partly", "no area with a role yet (Live page)"
        elif key == "crowd" and on and not any(z.get("max_people") for z in areas):
            state, why = "partly", "no area with a people limit: only people running together"
        out.append({"key": key, "words": words, "state": state, "why": why})
    return out


def _floor(p) -> dict | None:
    watch = getattr(p, "floor_watch", None)
    if watch is None:
        return None
    st = watch.state()
    return {"learned": st["learning"] is None, "learning": st["learning"], "learned_at": st["learned_at"],
            "saved": st["saved"], "relearned": st["relearned"], "picture": watch.picture() is not None}


def _alerts(db, now: datetime) -> list[dict]:
    """Safety alerts per kind in the last DAYS days (all cameras, then per camera), with the Check page's answers."""
    from . import checks
    from .db import incident_words
    from .incidents import CATEGORIES
    events = [e for e in db.list_alerts(limit=5000, start=now - timedelta(days=DAYS)) if e.get("category")]
    count = Counter(e["category"] for e in events)
    per_camera = Counter((e["category"], e["camera"]) for e in events)
    measured = {(r["what"], r["camera"]): r for r in checks.results(db.list_checks())}
    out = []
    for category, (level, words, _) in CATEGORIES.items():
        r = measured.get((f"{incident_words(category)} alerts", None))
        if not count[category] and r is None:
            continue
        out.append({"category": category, "words": words, "level": level, "alerts": count[category],
                    "cameras": {cam: n for (c, cam), n in per_camera.items() if c == category},
                    "checked": None if r is None else {k: r[k] for k in ("right", "wrong", "unsure", "share", "few")}})
    return out


def overview(cfg: dict, db, pipelines: list, incidents=None, now: datetime | None = None) -> dict:
    now = now or datetime.now()
    s = cfg.get("safety") or {}
    latest = incidents.latest() if incidents is not None else []
    cameras = []
    for p in pipelines:
        if getattr(p, "safety_watch", None) is None:
            continue
        areas = [{"name": z["name"], "role": z.get("role") or "", "role_words": ROLES.get(z.get("role") or "", ""),
                  "max_people": z.get("max_people"), "meaning": role_meaning(z.get("role") or "", cfg)
                  if z.get("role") else ""} for z in p.zone_counter.zones if z["kind"] == AREA]
        cameras.append({"name": p.name, "kinds": _kinds(p, cfg), "areas": areas, "floor": _floor(p),
                        "checks": [c for c in reversed(latest) if c["camera"] == p.name]})
    queue = None
    if incidents is not None:
        st = incidents.state()
        queue = {k: st[k] for k in ("sent", "yes", "no", "unsure", "failed", "dropped_old", "direct", "waiting",
                                    "wait_median_s")}
    return {"enabled": bool(s.get("enabled", True)), "cameras": cameras, "queue": queue,
            "night_hours": (cfg.get("behaviour") or {}).get("night_hours", "22-06"),
            "alerts": _alerts(db, now), "days": DAYS, "kept": incidents.recent.maxlen if incidents is not None else 0}
