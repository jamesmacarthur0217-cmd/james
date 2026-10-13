"""Alert rules: notify when a person matching a description appears.

A rule is a list of conditions ("upper_color = red", "carrying = backpack", "zone = door"), optionally limited
to one camera. "zone = X" means the person entered area X or crossed line X. A rule with no conditions matches
anyone (useful for a camera that should normally see nobody). Each rule fires at most once per person.
Rules are checked whenever a person's labels change or they enter a zone.

"fallen = yes" is a fall (labeler/falls.py): rules with it are checked only when a fall is confirmed, and fire
for every fall, also a second one of the same person. A built-in rule "Fall" is made at start when fall alerts
are on and no rule asks for a fall (turn it off on the Alerts page to stop fall alerts; deleted, it comes back).
"""
from __future__ import annotations

import logging
import threading

from . import schema
from .db import Database

log = logging.getLogger(__name__)


ZONE = "zone"   # condition on where the person went, not on how they look
FALLEN = "fallen"   # condition on a fall having just been confirmed
FALL_RULE = [[FALLEN, "yes"]]


def is_fall_rule(conditions: list[list[str]]) -> bool:
    return any(name == FALLEN for name, _ in conditions)


def matches(conditions: list[list[str]], attrs: dict[str, str]) -> bool:
    for name, value in conditions:
        current = attrs.get(name)
        if name in schema.LIST_FIELDS or name in (ZONE, FALLEN):
            if value not in (current or "").split(","):
                return False
        elif current != value:
            return False
    return True


def describe(conditions: list[list[str]]) -> str:
    if not conditions:
        return "anyone"
    return ", ".join(f"in zone {v}" if n == ZONE else "fell" if n == FALLEN else f"{n.replace('_', ' ')} = {v}"
                     for n, v in conditions)


class AlertEngine:
    def __init__(self, db: Database):
        self.db = db
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

    def check(self, track_id: int, camera: str, fallen: bool = False) -> list[dict]:
        """Fire every enabled rule this person now matches (once per rule and person). fallen: a fall was just
        confirmed: only the rules asking for a fall are checked, and they fire again for a later fall."""
        rules = [r for r in self._enabled_rules() if (not r["camera"] or r["camera"] == camera)
                 and is_fall_rule(r["conditions"]) == fallen]
        if not rules:
            return []
        attrs = self.db.resolved_values(track_id)
        if attrs is None:   # deleted meanwhile
            return []
        if any(name == ZONE for r in rules for name, _ in r["conditions"]):
            attrs = {**attrs, ZONE: ",".join(self.db.zones_entered(track_id))}
        if fallen:
            attrs = {**attrs, FALLEN: "yes"}
        fired = []
        for rule in rules:
            if matches(rule["conditions"], attrs):
                event = self.db.record_alert(rule["id"], track_id, camera,
                                             f"{rule['name']}: {describe(rule['conditions'])}", repeat=fallen)
                if event is not None:
                    log.warning("ALERT [%s] %s -> person #%s", camera, rule["name"], track_id)
                    fired.append(event)
        return fired
