"""Unsafe equipment (Phase 47 step 3): an overturned table or chair, a broken chair, cables across the floor, something
hanging loose. There is no quick sign for these, so (the user's choices, 2026-10-21) every `equipment_every_s` (10 min)
each safety camera shows **every area with a role** to the vision model: the newest picture with the area cut out, at
the lowest priority (a slow round, after people's labels). A "yes" is looked at again after `equipment_again_s` (1 min)
and the second "yes" alerts, as for floors (unsafe things stay; one look can be wrong). After an alert the area is
"reported": the rounds go on, but alert again only after a "no" (it was put right). The first round of each area comes
`equipment_first_s` after the start, the areas `equipment_gap_s` apart, so they do not all ask at once.
"""
from __future__ import annotations

from collections import deque

from .zones import AREA


class EquipmentRounds:
    def __init__(self, cfg: dict):
        self.every_s = float(cfg.get("equipment_every_s", 600))
        self.again_s = float(cfg.get("equipment_again_s", 60))
        self.first_s = float(cfg.get("equipment_first_s", 60))
        self.gap_s = float(cfg.get("equipment_gap_s", 20))
        self.areas: dict = {}       # area id -> {"next", "waiting", "yes_at", "reported"}
        self.answers: deque = deque()   # (area id, answer or None), from the confirm queue's thread

    def due(self, now: float, zones: list[dict]) -> list[tuple[dict, bool, str]]:
        """The areas to look at now: (area, alerts on "yes", reason)."""
        self._take_answers(now)
        watched = [z for z in zones if z.get("kind") == AREA and z.get("role")]
        for zid in set(self.areas) - {z["id"] for z in watched}:
            del self.areas[zid]   # the area was deleted or lost its role
        out = []
        for i, z in enumerate(watched):
            st = self.areas.setdefault(z["id"], {"next": now + self.first_s + self.gap_s * i, "waiting": False,
                                                 "yes_at": None, "reported": False})
            if st["waiting"] or now < st["next"]:
                continue
            st["waiting"] = True
            second = st["yes_at"] is not None and not st["reported"]
            reason = ("a second look, a minute after a yes" if second else
                      "the round every 10 min (reported before: alerts again only after a no)" if st["reported"] else
                      "the round every 10 min")
            out.append((z, second, reason))
        return out

    def answered(self, area_id, answer: dict | None) -> None:
        """The confirm queue's answer (None: none came, or the check was not sent). Any thread."""
        self.answers.append((area_id, answer))

    def _take_answers(self, now: float) -> None:
        while self.answers:
            zid, answer = self.answers.popleft()
            st = self.areas.get(zid)
            if st is None:
                continue
            st["waiting"] = False
            said = (answer or {}).get("answer")
            if said == "yes" and st["reported"]:
                st["next"] = now + self.every_s
            elif said == "yes" and st["yes_at"] is None:
                st["yes_at"], st["next"] = now, now + self.again_s   # a first look: once more in a minute
            elif said == "yes":
                st["yes_at"], st["reported"], st["next"] = None, True, now + self.every_s   # the second yes alerted
            elif said == "no":
                st["yes_at"], st["reported"], st["next"] = None, False, now + self.every_s
            else:   # unsure, failed, dropped (the vision model busy) or not sent: again in a minute
                st["next"] = now + self.again_s

    def state(self) -> dict:
        """For the Health page."""
        return {"areas": len(self.areas), "checking": sum(1 for s in self.areas.values() if s["yes_at"] is not None),
                "reported": sum(1 for s in self.areas.values() if s["reported"])}
