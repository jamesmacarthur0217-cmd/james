"""Dangerous objects (Phase 47 step 1): a knife, scissors, a bat, a gun, a machete or a sword held at a person's
wrist, outside a cooking or food preparation area, raises a critical "dangerous object" suspect; the vision model
then decides whether it is held in a threatening way or used for food or work.

Two looks a second on safety cameras (the user's choices, 2026-10-21):
- the plain detector's once-a-second pass (Pipeline._look_objects, the crowd count's) also asks for COCO's knife,
  scissors and baseball bat;
- YOLOE prompt-free (yoloe-11s-seg-pf.pt, Ultralytics, AGPL-3.0; 27 MB, a fixed list of 4,585 things, so no text
  model: MobileCLIP's weights are for research only) for guns, rifles, shotguns, machetes and swords, which COCO does
  not have.

"Held": a wrist point (body points 9, 10) inside the object's box grown by `weapon_reach` of the person's height, on
`weapon_hold` of the last `weapon_looks` looks. A person in an area with the role "cooking" or "food_prep" is left
out (knives are normal there). Raised again for the same person only after `weapon_rearm_s` without holding one.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .falls import MIN_CONF

COCO_WEAPONS = {34: "baseball bat", 43: "knife", 76: "scissors"}
YOLOE_WORDS = ("gun", "handgun", "rifle", "shotgun", "machete", "sword", "weapon")
L_WRIST, R_WRIST = 9, 10
SAFE_ROLES = ("cooking", "food_prep")


def at_wrist(obj: tuple, kpts: np.ndarray | None, person: tuple, reach: float) -> bool:
    """A wrist of this person inside the object's box, grown by `reach` x the person's height."""
    if kpts is None:
        return False
    grow = reach * max(person[3] - person[1], 1.0)
    x1, y1, x2, y2 = obj[0] - grow, obj[1] - grow, obj[2] + grow, obj[3] + grow
    return any(kpts[i, 2] >= MIN_CONF and x1 <= kpts[i, 0] <= x2 and y1 <= kpts[i, 1] <= y2 for i in (L_WRIST, R_WRIST))


class WeaponWatch:
    def __init__(self, cfg: dict):
        self.looks = int(cfg.get("weapon_looks", 3))
        self.hold = int(cfg.get("weapon_hold", 2))
        self.reach = float(cfg.get("weapon_reach", 0.15))
        self.rearm_s = float(cfg.get("weapon_rearm_s", 30))
        self.people: dict = {}   # key -> {"looks": deque of word or None, "last": time held, "fired": bool}

    def update(self, now: float, objects: list[tuple], people: dict) -> list[tuple]:
        """objects: (word, confidence, box) from this look; people: key -> (box, body points, area role) of the
        people in view. Returns (key, word) for each person to raise now."""
        held = {}
        for word, conf, box in sorted(objects, key=lambda o: -o[1]):
            for key, (pbox, kpts, role) in people.items():
                if role not in SAFE_ROLES and key not in held and at_wrist(box, kpts, pbox, self.reach):
                    held[key] = word
                    break
        out = []
        for key in set(people) | set(self.people):
            p = self.people.setdefault(key, {"looks": deque(maxlen=self.looks), "last": -1e9, "fired": False})
            word = held.get(key)
            p["looks"].append(word)
            if word:
                p["last"] = now
            elif p["fired"] and now - p["last"] >= self.rearm_s:
                p["fired"] = False
            words = [w for w in p["looks"] if w]
            if len(words) >= self.hold and not p["fired"]:
                p["fired"] = True
                out.append((key, max(set(words), key=words.count)))
        for key in [k for k, p in self.people.items() if k not in people and now - p["last"] > self.rearm_s]:
            del self.people[key]   # gone a while: forget
        return out
