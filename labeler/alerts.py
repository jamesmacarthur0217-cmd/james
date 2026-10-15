"""Alert rules: notify when a person matching a description appears.

A rule is a list of conditions ("upper_color = red", "carrying = backpack", "zone = door"), optionally limited
to one camera. "zone = X" means the person entered area X or crossed line X. A rule with no conditions matches
anyone (useful for a camera that should normally see nobody). Each rule fires at most once per person.
Rules are checked whenever a person's labels change or they enter a zone.

A rule with a "vehicle" condition is about vehicles (labeler/vehicles.py): "vehicle = any / car / motorbike ...",
"vehicle_color = red", "zone = gate"; it fires once per vehicle, checked when the vehicle is first counted, when it is
described and when it enters a zone. Person rules never fire for vehicles and the other way round. "hours = 22-06"
(people and vehicles) limits a rule to a time of day (from 22:00 until 06:00).

A rule with an "animal" condition is about animals (labeler/animals.py): "animal = any / dog / cat / bird", with a
zone, the hours, Home / Away and how long it stayed; once per animal visit, like vehicles.

"armed = away" (or "home"; people, vehicles and animals) limits a rule to when the house is set to Away (or Home) on the
pages' Home / Away switch or by its weekly schedule (labeler/arming.py). Rules without it work all the time, like the
built-in "Fall" rule.

"stayed = gate:120" (people and vehicles) is someone staying in area "gate" at least 120 s (loitering). The time inside
an area is kept by labeler/zones.py; the camera checks people and vehicles inside areas once a second with how long
they have been in each (`stayed`), and the rule fires once per person or vehicle like the others. A person out of sight
for more than tracking.lost_after_s leaves the area (their time starts again); a parked vehicle hidden a moment keeps it.

"quiet = 240" (minutes) is no movement (labeler/quiet.py): nobody seen on a camera that long, with "hours" and "armed"
counting only while they hold; a built-in rule "No movement" (4 h, 08:00-22:00, only when Home: the user's choice) is
made at start like "Fall". A rule without a camera is for kiosk cameras only (a gate camera sees nobody for hours).

"fallen = yes" is a fall (labeler/falls.py): rules with it are checked only when a fall is confirmed, and fire
for every fall, also a second one of the same person. A built-in rule "Fall" is made at start when fall alerts
are on and no rule asks for a fall (turn it off on the Alerts page to stop fall alerts; deleted, it comes back).
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime

from . import schema
from .db import Database

log = logging.getLogger(__name__)


ZONE = "zone"   # condition on where the person went, not on how they look
FALLEN = "fallen"   # condition on a fall having just been confirmed
FALL_RULE = [[FALLEN, "yes"]]
VEHICLE, VEHICLE_COLOR, HOURS, ARMED, STAYED = "vehicle", "vehicle_color", "hours", "armed", "stayed"
MAX_STAY_S = 24 * 3600
ANIMAL = "animal"
QUIET = "quiet"     # minutes with nobody seen (labeler/quiet.py)
QUIET_RULE = [[QUIET, "240"], ["hours", "08-22"], ["armed", "home"]]
CAR_GROUP = {"car", "SUV", "van", "pickup"}   # "car" in a rule, like in the Ask box
HOUR_PRESETS = ["22-06", "20-06", "00-06", "18-24", "06-18", "08-18", "09-17"]


def is_vehicle_rule(conditions: list[list[str]]) -> bool:
    return any(name == VEHICLE for name, _ in conditions)


def is_animal_rule(conditions: list[list[str]]) -> bool:
    return any(name == ANIMAL for name, _ in conditions)


def is_quiet_rule(conditions: list[list[str]]) -> bool:
    return any(name == QUIET for name, _ in conditions)


def rule_kind(conditions: list[list[str]]) -> str:
    """Who a rule is about: "person", "vehicle", "animal" or "quiet" (nobody: no movement)."""
    if is_quiet_rule(conditions):
        return "quiet"
    return "vehicle" if is_vehicle_rule(conditions) else "animal" if is_animal_rule(conditions) else "person"


def valid_hours(value: str) -> bool:
    parts = value.split("-")
    return len(parts) == 2 and all(p.isdigit() and 0 <= int(p) <= 24 for p in parts) and parts[0] != parts[1]


def in_hours(value: str, hour: int) -> bool:
    """"22-06" = from 22:00 until 06:00 (over midnight); "18-24" = from 18:00 until midnight."""
    start, end = (int(p) % 24 for p in value.split("-"))
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def parse_stayed(value: str) -> tuple[str, int]:
    """"gate:120" -> ("gate", 120). A zone name may itself hold a colon: the seconds are after the last one."""
    zone, sep, secs = str(value).rpartition(":")
    if not sep or not zone or not secs.isdigit() or not 1 <= int(secs) <= MAX_STAY_S:
        raise ValueError(f"stayed must look like gate:120 (an area and 1-{MAX_STAY_S} seconds), not {value!r}")
    return zone, int(secs)


def nice_seconds(s: float) -> str:
    s = int(s)
    if s % 3600 == 0:
        return f"{s // 3600} h"
    if s % 60 == 0:
        return f"{s // 60} min"
    return f"{s} s"


def is_fall_rule(conditions: list[list[str]]) -> bool:
    return any(name == FALLEN for name, _ in conditions)


def matches(conditions: list[list[str]], attrs: dict[str, str]) -> bool:
    for name, value in conditions:
        current = attrs.get(name)
        if name == HOURS:
            if not in_hours(value, int(attrs.get("hour", datetime.now().hour))):
                return False
        elif name == STAYED:   # attrs[STAYED]: {area name: seconds inside}, only from the once-a-second check
            zone, secs = parse_stayed(value)
            if (attrs.get(STAYED) or {}).get(zone, 0) < secs:
                return False
        elif name == VEHICLE:
            if not current or (value != "any" and current != value and not (value == "car" and current in CAR_GROUP)):
                return False
        elif name == ANIMAL:
            if not current or (value != "any" and current != value):
                return False
        elif name == VEHICLE_COLOR:
            if current != value and {current, value} != {"gray", "silver"}:
                return False
        elif name in schema.LIST_FIELDS or name in (ZONE, FALLEN):
            if value not in (current or "").split(","):
                return False
        elif current != value:
            return False
    return True


def describe(conditions: list[list[str]]) -> str:
    if not conditions:
        return "anyone"
    def one(n: str, v: str) -> str:
        if n == ZONE:
            return f"in zone {v}"
        if n == FALLEN:
            return "fell"
        if n == VEHICLE:
            return "any vehicle" if v == "any" else f"vehicle = {v}"
        if n == ANIMAL:
            return "any animal" if v == "any" else f"animal = {v}"
        if n == ARMED:
            return f"only when {v.capitalize()}"
        if n == QUIET:
            return f"nobody seen for {nice_seconds(int(v) * 60)}"
        if n == STAYED:
            zone, secs = parse_stayed(v)
            return f"stayed in {zone} {nice_seconds(secs)}"
        if n == HOURS:
            a, b = v.split("-")
            return f"between {int(a):02d}:00 and {int(b) % 24:02d}:00"
        return f"{n.replace('_', ' ')} = {v}"
    return ", ".join(one(n, v) for n, v in conditions)


def _asks_armed(rules: list[dict]) -> bool:
    return any(name == ARMED for r in rules for name, _ in r["conditions"])


class AlertEngine:
    def __init__(self, db: Database, arming=None):
        """arming: the app's one labeler.arming.Arming (the pages switch the same one); without it, always Home."""
        self.db = db
        self.arming = arming
        self._lock = threading.Lock()
        self._rules: list[dict] | None = None

    def invalidate(self) -> None:
        """Reload rules on the next check (call after rules change)."""
        with self._lock:
            self._rules = None

    def _enabled_rules(self) -> list[dict]:
        with self._lock:
            if self._rules is None:
                self._rules = [r for r in self.db.list_rules() if r["enabled"]]
            return self._rules

    def ensure_fall_rule(self) -> None:
        """The built-in "Fall" rule, unless a rule already asks for a fall."""
        if not any(is_fall_rule(r["conditions"]) for r in self.db.list_rules()):
            self.db.add_rule("Fall", FALL_RULE)
            self.invalidate()
            log.info('alert rule "Fall" added (turn it off on the Alerts page to stop fall alerts)')

    def ensure_quiet_rule(self) -> None:
        """The built-in "No movement" rule, unless a rule already asks for it (deleted, it comes back)."""
        if not any(is_quiet_rule(r["conditions"]) for r in self.db.list_rules()):
            self.db.add_rule("No movement", QUIET_RULE)
            self.invalidate()
            log.info('alert rule "No movement" added (4 h, 08:00-22:00, only when Home; Alerts page to change)')

    def quiet_rules(self) -> list[dict]:
        return [r for r in self._enabled_rules() if is_quiet_rule(r["conditions"])]

    def _armed(self) -> str:
        from .arming import HOME
        return self.arming.current() if self.arming is not None else HOME

    def stay_limits(self, camera: str, vehicles: bool = False, animals: bool = False) -> dict[str, int]:
        """{area name: the shortest stay a rule asks for} on this camera, for people (or vehicles, or animals): the
        camera only asks check() with `stayed` for someone inside such an area that long."""
        out: dict[str, int] = {}
        want = "vehicle" if vehicles else "animal" if animals else "person"
        for r in self._enabled_rules():
            if (r["camera"] and r["camera"] != camera) or rule_kind(r["conditions"]) != want:
                continue
            for name, value in r["conditions"]:
                if name == STAYED:
                    zone, secs = parse_stayed(value)
                    out[zone] = min(secs, out.get(zone, secs))
        return out

    def check(self, track_id: int, camera: str, fallen: bool = False, stayed: dict | None = None) -> list[dict]:
        """Fire every enabled rule this person now matches (once per rule and person). fallen: a fall was just
        confirmed: only the rules asking for a fall are checked, and they fire again for a later fall.
        stayed: {area name: seconds inside} (the camera's once-a-second check of people inside areas)."""
        rules = [r for r in self._enabled_rules() if (not r["camera"] or r["camera"] == camera)
                 and is_fall_rule(r["conditions"]) == fallen and rule_kind(r["conditions"]) == "person"]
        if not rules:
            return []
        attrs = self.db.resolved_values(track_id)
        if attrs is None:   # deleted meanwhile
            return []
        if any(name == ZONE for r in rules for name, _ in r["conditions"]):
            attrs = {**attrs, ZONE: ",".join(self.db.zones_entered(track_id))}
        if fallen:
            attrs = {**attrs, FALLEN: "yes"}
        attrs = {**attrs, "hour": datetime.now().hour, STAYED: stayed or {}}
        if _asks_armed(rules):
            attrs[ARMED] = self._armed()
        fired = []
        for rule in rules:
            if matches(rule["conditions"], attrs):
                event = self.db.record_alert(rule["id"], track_id, camera,
                                             f"{rule['name']}: {describe(rule['conditions'])}", repeat=fallen)
                if event is not None:
                    log.warning("ALERT [%s] %s -> person #%s", camera, rule["name"], track_id)
                    fired.append(event)
        return fired

    def check_animal(self, animal_id: int, camera: str, stayed: dict | None = None) -> list[dict]:
        """Fire every enabled animal rule this animal now matches (once per rule and animal visit)."""
        rules = [r for r in self._enabled_rules() if (not r["camera"] or r["camera"] == camera)
                 and is_animal_rule(r["conditions"])]
        if not rules:
            return []
        attrs = self.db.animal_attrs(animal_id)
        if attrs is None:
            return []
        attrs.update({"hour": datetime.now().hour, STAYED: stayed or {}})
        if _asks_armed(rules):
            attrs[ARMED] = self._armed()
        fired = []
        for rule in rules:
            if matches(rule["conditions"], attrs):
                event = self.db.record_alert(rule["id"], None, camera, f"{rule['name']}: {describe(rule['conditions'])}",
                                             animal_id=animal_id)
                if event is not None:
                    log.warning("ALERT [%s] %s -> animal #%s", camera, rule["name"], animal_id)
                    fired.append(event)
        return fired

    def check_vehicle(self, vehicle_id: int, camera: str, stayed: dict | None = None) -> list[dict]:
        """Fire every enabled vehicle rule this vehicle now matches (once per rule and vehicle). stayed: as check()."""
        rules = [r for r in self._enabled_rules() if (not r["camera"] or r["camera"] == camera)
                 and is_vehicle_rule(r["conditions"])]
        if not rules:
            return []
        attrs = self.db.vehicle_attrs(vehicle_id)
        if attrs is None:   # deleted, or the vision model saw no vehicle
            return []
        attrs["hour"] = datetime.now().hour
        attrs[STAYED] = stayed or {}
        if _asks_armed(rules):
            attrs[ARMED] = self._armed()
        fired = []
        for rule in rules:
            if matches(rule["conditions"], attrs):
                event = self.db.record_alert(rule["id"], None, camera, f"{rule['name']}: {describe(rule['conditions'])}",
                                             vehicle_id=vehicle_id)
                if event is not None:
                    log.warning("ALERT [%s] %s -> vehicle #%s", camera, rule["name"], vehicle_id)
                    fired.append(event)
        return fired
