"""Quick checks for people (Phase 45): signs in the body points and boxes the camera already has, every frame, that a
person may need help. They only raise a suspect; the vision model confirms it (labeler/incidents.py).

One SafetyWatch per safety camera (Pipeline.safety_watch), fed like FallWatch with each person's box and body points
and the role of the area they stand in. For each person it keeps the last few minutes and, once a frame, looks for:

- lying still: the posture is lying (labeler/falls.py posture: the torso near horizontal, or a box much wider than
  tall) on most frames for `down_s` seconds, and the person has not moved: someone who lay down slowly or collapsed
  without a fall the fall check would see. Question: does this person need help?
- slumped: not lying, the head below the line of the shoulders (bent forward or to the side) on most frames for
  `slumped_s` seconds, not moving: someone slumped in a chair or against a wall.
- still for long: not moved for `still_s` seconds in any posture: the slow round for distress (a person can be unwell
  without looking slumped). Sent at the lowest priority, after people's labels, with its own question.

"Not moved": the middle of the box stayed within `still_move` of the person's height of where it is now. Each sign is
raised once per still period (moving again re-arms it). Nothing is raised in an area with the role "bed" (lying,
sleeping and resting are normal there).

Children (Phase 45 step 2), each raising "child in danger":
- who is a child: the age label "child" (age group, or an age range 0-9) once the labels came; before that (the user's
  choice: sooner, more false alarms) a small upright body: at most `child_size` of the height of an upright person
  who is not a child standing at the same depth (feet within 15% of that person's height), on most frames of the last
  3 s. A label "adult", "teen" or "senior" wins over the size;
- alone: no adult within `child_near` x the taller one's height for `child_alone_s` (0 = not checked);
- in a cooking, food preparation or restricted area for `child_area_s`;
- climbing: the feet `child_climb` x the child's height above where they were (median of the last 10 s) while the
  body did not get smaller (walking away from the camera also lifts the feet in the picture, but shrinks the body),
  for 2 s.

Crowding (Phase 45 step 3), raising "unusual crowding" about an area or the whole picture:
- too many: more people inside an area than its limit ("at most N people", set on the Live page; where a person's
  feet are, as for counting) for `crowd_s`; once until the count is back within the limit for `crowd_s`. The count
  comes from the camera's once-a-second count with the plain detector (Pipeline._look_objects), else the people
  tracked here;
- running together: at least `rush_min` people in view, most of them (`rush_share`) moving at `rush_speed` body
  heights a second or more, the same way (their directions agree: `rush_agree`), for 2 s: people pushing or fleeing.
  Walking is about half a body height a second. Once until it calms down.
Someone pressed against a barrier has no quick sign: only the vision model can see it, when it checks a crowd.

Fights (Phase 48, the user's choices): for each two people close together (their middles within `fight_near` x the
taller one's height), each frame is a sign when one of them has a wrist moving fast (`fight_speed` body heights a
second, measured against their own body, so walking and turning do not count) while they touch (boxes overlapping) or
that wrist reaches into the other's head or upper body. On `fight_share` of the frames of the last `fight_window_s`
(at least 3 frames) the two raise one critical "aggression" suspect (fights(): both people, for a cut-out around
them); the vision model decides from 3 pictures over the last 2 s, and one "yes" alerts. Again for the same two only
after `fight_rearm_s` without a sign. Hugs and handshakes touch but are slow; a punch, a push or a kick is fast.
`fight_speed` 2.5 was chosen on half of the Surveillance Camera Fight clips and MOT16 (README "Fights"): at 2.0 the
check raised 64% of the fights but about 6 suspects a minute on busy streets (critical checks, ahead of people's
labels); at 2.5, 52% and 0.8 a minute. Asking for the two to stand at the same depth, a minimum size or surer wrist
points cost more fights than they saved and were left out.

Collisions (Phase 48, the user's choice): no check of their own; any fall alerts already ("fall or collision"), and
touched() tells who else's box touched the person's around the fall (each person's boxes are kept TOUCH_KEEP_S), so
the fall's message can say "after touching person #12".
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .falls import LYING, MIN_CONF, UPRIGHT, posture
from .geometry import L_EAR, L_EYE, L_SHOULDER, NOSE, R_EAR, R_EYE, R_SHOULDER

L_WRIST, R_WRIST = 9, 10
WRIST_GAP_S = (0.1, 0.4)   # a wrist's speed: from where it was this long ago (the newest look of at least 0.1 s)

DISTRESS_QUESTION = ("Does this person look unwell, collapsed, unconscious or in distress (not simply resting, "
                     "reading, working or sleeping in a bed)?")
SHARE = 0.8   # a sign must be there on this share of the frames of its time
CHILD_AREAS = ("cooking", "food_prep", "restricted")
CHILD_WINDOW_S = 3.0    # "is a child" by size: on most frames of this long
CLIMB_WINDOW_S = 10.0   # where the feet usually are: the median of this long
CLIMB_HOLD_S = 2.0      # climbing: feet up this long
RUSH_WINDOW_S = 1.0     # speed: from where a person was this long ago
RUSH_HOLD_S = 2.0       # running together this long
TOUCH_KEEP_S = 30.0     # each person's boxes are kept this long, for touched() (a fall is confirmed ~10 s after it)


def head_low(kpts: np.ndarray | None) -> bool | None:
    """The head below the line of the shoulders (image y grows downwards); None when not seen well enough."""
    if kpts is None:
        return None
    shoulders = [kpts[i, 1] for i in (L_SHOULDER, R_SHOULDER) if kpts[i, 2] >= MIN_CONF]
    head = [kpts[i, 1] for i in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR) if kpts[i, 2] >= MIN_CONF]
    if not shoulders or not head:
        return None
    return float(np.mean(head)) > float(np.mean(shoulders))


@dataclass
class _Person:
    hist: deque = field(default_factory=deque)   # (t, centre x, centre y, height, lying, head low)
    role: str = ""
    box: tuple = ()
    last: float = 0.0
    fired: set = field(default_factory=set)
    upright: bool = False
    age: str | None = None                        # "child", "adult" (any other age), or None: no label yet
    small: deque = field(default_factory=deque)   # (t, small next to an adult)
    feet: deque = field(default_factory=deque)    # (t, box bottom, height)
    alone_since: float | None = None
    area_since: float | None = None
    climb_since: float | None = None
    anchor: tuple | None = None                   # where the person is (fractions of the picture: the feet)
    wrists: deque = field(default_factory=deque)  # (t, centre x, centre y, height, left wrist, right wrist) for 1 s
    boxes: deque = field(default_factory=deque)   # (t, box) for TOUCH_KEEP_S


class SafetyWatch:
    def __init__(self, cfg: dict):
        # Each kind on or off (Phase 51, the Settings page). People are still followed when one is off: a fall's
        # "after touching person #12" needs their boxes.
        self.down_on = bool(cfg.get("down_enabled", True))
        self.children_on = bool(cfg.get("children_enabled", True))
        self.crowd_on = bool(cfg.get("crowd_enabled", True))
        self.fights_on = bool(cfg.get("fights_enabled", True))
        self.down_s = float(cfg.get("down_s", 20))
        self.slumped_s = float(cfg.get("slumped_s", 30))
        self.still_s = float(cfg.get("still_s", 300))
        self.still_move = float(cfg.get("still_move", 0.08))
        self.keep_s = max(self.down_s, self.slumped_s, self.still_s) + 5
        self.child_size = float(cfg.get("child_size", 0.65))
        self.child_near = float(cfg.get("child_near", 2.5))
        self.child_alone_s = float(cfg.get("child_alone_s", 120))
        self.child_area_s = float(cfg.get("child_area_s", 3))
        self.child_climb = float(cfg.get("child_climb", 0.25))
        self.crowd_s = float(cfg.get("crowd_s", 30))
        self.rush_min = int(cfg.get("rush_min", 6))
        self.rush_share = float(cfg.get("rush_share", 0.6))
        self.rush_speed = float(cfg.get("rush_speed", 1.2))
        self.rush_agree = float(cfg.get("rush_agree", 0.8))
        self.people: dict = {}
        self.areas: dict = {}   # area id -> {"over_since", "calm_since", "fired"}
        self.rush = {"since": None, "fired": False}
        self.fight_near = float(cfg.get("fight_near", 1.0))
        self.fight_speed = float(cfg.get("fight_speed", 2.5))
        self.fight_window_s = float(cfg.get("fight_window_s", 1.0))
        self.fight_share = float(cfg.get("fight_share", 0.5))
        self.fight_rearm_s = float(cfg.get("fight_rearm_s", 10))
        self.pairs: dict = {}   # frozenset of two keys -> {"signs": deque of (t, bool), "last_sign", "fired"}

    def update(self, key, t: float, box: tuple, kpts: np.ndarray | None, role: str = "", age: str | None = None,
               anchor: tuple | None = None) -> None:
        """age: "child", "adult" (any other known age) or None (no label yet); anchor: where the person is
        (zones.anchor, fractions of the picture), for counting people inside areas."""
        p = self.people.setdefault(key, _Person())
        x1, y1, x2, y2 = box[:4]
        pose = posture(kpts, box)
        p.hist.append((t, (x1 + x2) / 2, (y1 + y2) / 2, max(y2 - y1, 1.0), pose == LYING, head_low(kpts)))
        while p.hist and t - p.hist[0][0] > self.keep_s:
            p.hist.popleft()
        p.role, p.box, p.last, p.upright, p.age, p.anchor = role, tuple(box[:4]), t, pose == UPRIGHT, age, anchor
        p.feet.append((t, y2, max(y2 - y1, 1.0)))
        while p.feet and t - p.feet[0][0] > CLIMB_WINDOW_S:
            p.feet.popleft()
        wrist = [(float(kpts[i, 0]), float(kpts[i, 1])) if kpts is not None and kpts[i, 2] >= MIN_CONF else None
                 for i in (L_WRIST, R_WRIST)]
        p.wrists.append((t, (x1 + x2) / 2, (y1 + y2) / 2, max(y2 - y1, 1.0), *wrist))
        while p.wrists and t - p.wrists[0][0] > 1.0:
            p.wrists.popleft()
        p.boxes.append((t, tuple(box[:4])))
        while p.boxes and t - p.boxes[0][0] > TOUCH_KEEP_S:
            p.boxes.popleft()

    def forget(self, key) -> None:
        self.people.pop(key, None)
        for pair in [k for k in self.pairs if key in k]:
            del self.pairs[pair]

    def mark(self, key, sign: str = "down") -> None:
        """This sign is handled already for this still period (a fall: no second "lying still" check)."""
        if key in self.people:
            self.people[key].fired.add(sign)

    @staticmethod
    def _still_for(p: _Person, move: float) -> float:
        """How long the person has stayed where they are now (within `move` x their height)."""
        t, cx, cy, h, *_ = p.hist[-1]
        since = t
        for ti, x, y, hi, *_ in reversed(p.hist):
            if abs(x - cx) > move * h or abs(y - cy) > move * h:
                break
            since = ti
        return t - since

    @staticmethod
    def _share(p: _Person, since: float, index: int) -> float:
        frames = [f[index] for f in p.hist if f[0] >= since and f[index] is not None]
        return sum(bool(v) for v in frames) / len(frames) if frames else 0.0

    def crowd(self, t: float, areas: list[dict], counts: dict | None = None) -> list[tuple]:
        """(area or None for the whole picture, reason) for each crowding sign now. areas: the camera's areas with a
        people limit (zones with "max_people"); counts: {area id: people} from the camera's own count, if any."""
        from .zones import inside
        if not self.crowd_on:
            return []
        out = []
        present = [p for p in self.people.values() if p.hist and t - p.last <= 1.0]
        for z in areas:
            limit = z.get("max_people")
            if not limit:
                continue
            n = counts[z["id"]] if counts is not None and z["id"] in counts else \
                sum(p.anchor is not None and inside(p.anchor, z["points"]) for p in present)
            a = self.areas.setdefault(z["id"], {"over_since": None, "calm_since": None, "fired": False})
            if n > limit:
                a["calm_since"] = None
                a["over_since"] = t if a["over_since"] is None else a["over_since"]
                if t - a["over_since"] >= self.crowd_s and not a["fired"]:
                    a["fired"] = True
                    out.append((z, f"{n} people in {z['name']} (at most {limit}) for {int(t - a['over_since'])} s"))
            else:
                a["over_since"] = None
                a["calm_since"] = t if a["calm_since"] is None else a["calm_since"]
                if t - a["calm_since"] >= self.crowd_s:
                    a["fired"] = False
        rushing = self._rushing(t, present)
        if rushing:
            self.rush["since"] = t if self.rush["since"] is None else self.rush["since"]
            if t - self.rush["since"] >= RUSH_HOLD_S and not self.rush["fired"]:
                self.rush["fired"] = True
                out.append((None, f"{rushing} people moving fast the same way"))
        else:
            self.rush["since"] = None
            self.rush["fired"] = False
        return out

    def _rushing(self, t: float, present: list) -> int:
        """How many people are moving fast the same way now (0 when it is not most of at least rush_min people)."""
        if len(present) < self.rush_min:
            return 0
        moves = []
        for p in present:
            old = next((f for f in p.hist if t - f[0] <= RUSH_WINDOW_S + 0.05), None)
            now = p.hist[-1]
            if old is None or now[0] - old[0] < 0.5 * RUSH_WINDOW_S:
                continue
            dx, dy, dt = now[1] - old[1], now[2] - old[2], now[0] - old[0]
            speed = (dx * dx + dy * dy) ** 0.5 / now[3] / dt   # body heights a second
            if speed >= self.rush_speed:
                moves.append((dx, dy))
        if len(moves) < max(self.rush_min, self.rush_share * len(present)):
            return 0
        units = np.array([(dx, dy) / np.hypot(dx, dy) for dx, dy in moves])
        return len(moves) if float(np.linalg.norm(units.mean(axis=0))) >= self.rush_agree else 0

    def touched(self, key, since: float, until: float) -> list:
        """The other people whose box touched this person's in the same frame between since and until, the one
        touched most first (for a fall's message: a collision with someone)."""
        p = self.people.get(key)
        if p is None:
            return []
        mine = {round(t, 3): b for t, b in p.boxes if since <= t <= until}
        counts = {}
        for other, q in self.people.items():
            if other == key:
                continue
            n = 0
            for t, b in q.boxes:
                a = mine.get(round(t, 3))
                if a is not None and min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                    n += 1
            if n:
                counts[other] = n
        return sorted(counts, key=lambda k: -counts[k])

    @staticmethod
    def _fast_wrists(p: _Person) -> list[tuple[float, float, float]]:
        """(x, y now, speed) of each wrist seen now and 0.1-0.4 s ago; speed in body heights a second against the
        body's own middle; [] when there is no look to measure from."""
        t, cx, cy, h, *now = p.wrists[-1]
        # The newest look at least 0.1 s ago: a punch goes out and back in 0.2-0.3 s, so a longer gap can find the
        # fist where it started and see no movement (found by the tests at 10 frames a second).
        old = max((f for f in p.wrists if WRIST_GAP_S[0] <= t - f[0] <= WRIST_GAP_S[1]), key=lambda f: f[0],
                  default=None)
        if old is None:
            return []
        out = []
        for i in (0, 1):
            w, w0 = now[i], old[4 + i]
            if w is None or w0 is None:
                continue
            dx = (w[0] - cx) - (w0[0] - old[1])
            dy = (w[1] - cy) - (w0[1] - old[2])
            out.append((w[0], w[1], float(np.hypot(dx, dy)) / h / (t - old[0])))
        return out

    def _fight_sign(self, a: _Person, b: _Person) -> bool:
        """This frame: are these two close, and does a fast wrist of one touch or reach the other?"""
        ha, hb = a.box[3] - a.box[1], b.box[3] - b.box[1]
        ax, ay = (a.box[0] + a.box[2]) / 2, (a.box[1] + a.box[3]) / 2
        bx, by = (b.box[0] + b.box[2]) / 2, (b.box[1] + b.box[3]) / 2
        if np.hypot(ax - bx, ay - by) > self.fight_near * max(ha, hb):
            return False
        touch = min(a.box[2], b.box[2]) > max(a.box[0], b.box[0]) and min(a.box[3], b.box[3]) > max(a.box[1], b.box[1])
        for me, other in ((a, b), (b, a)):
            oh = other.box[3] - other.box[1]
            upper = (other.box[0], other.box[1], other.box[2], other.box[1] + 0.6 * oh)   # head and upper body
            for x, y, speed in self._fast_wrists(me):
                if speed < self.fight_speed:
                    continue
                if touch or (upper[0] <= x <= upper[2] and upper[1] <= y <= upper[3]):
                    return True
        return False

    def fights(self, t: float) -> list[tuple]:
        """(key, other key, reason) for each two people who look like fighting now (see the module text)."""
        if not self.fights_on:
            return []
        present ={k: p for k, p in self.people.items() if p.wrists and t - p.last <= 0.2}
        keys = sorted(present, key=str)
        out = []
        for i, ka in enumerate(keys):
            for kb in keys[i + 1:]:
                pair = frozenset((ka, kb))
                sign = self._fight_sign(present[ka], present[kb])
                st = self.pairs.get(pair)
                if st is None:
                    if not sign:
                        continue
                    st = self.pairs[pair] = {"signs": deque(), "last_sign": t, "fired": False}
                st["signs"].append((t, sign))
                while st["signs"] and t - st["signs"][0][0] > self.fight_window_s:
                    st["signs"].popleft()
                if sign:
                    st["last_sign"] = t
                elif st["fired"] and t - st["last_sign"] >= self.fight_rearm_s:
                    st["fired"] = False
                n, hits = len(st["signs"]), sum(s for _, s in st["signs"])
                if not st["fired"] and n >= 3 and hits >= self.fight_share * n:
                    st["fired"] = True
                    out.append((ka, kb, f"two people close, a fast arm reaching the other ({hits} of {n} frames)"))
        for pair in [k for k, st in self.pairs.items() if t - st["last_sign"] > self.fight_rearm_s + 5]:
            del self.pairs[pair]   # calm a while: forget
        return out

    def _children(self, t: float, present: dict) -> dict:
        """The people seen now who count as children (by label, else by size next to an adult)."""
        for key, p in present.items():
            small = False
            if p.age is None and p.upright:
                h, bottom = p.box[3] - p.box[1], p.box[3]
                for other_key, q in present.items():
                    if other_key == key or not q.upright or q.age == "child":
                        continue
                    qh = q.box[3] - q.box[1]
                    if abs(q.box[3] - bottom) <= 0.15 * qh and h <= self.child_size * qh:
                        small = True
                        break
            p.small.append((t, small))
            while p.small and t - p.small[0][0] > CHILD_WINDOW_S:
                p.small.popleft()
        return {k: p for k, p in present.items()
                if p.age == "child" or (p.age is None and sum(s for _, s in p.small) >= SHARE * len(p.small)
                                        and len(p.small) > 1)}

    def _near_adult(self, p: _Person, present: dict, children: dict) -> bool:
        cx, cy, h = (p.box[0] + p.box[2]) / 2, (p.box[1] + p.box[3]) / 2, p.box[3] - p.box[1]
        for key, q in present.items():
            if key in children or q is p:
                continue
            qx, qy, qh = (q.box[0] + q.box[2]) / 2, (q.box[1] + q.box[3]) / 2, q.box[3] - q.box[1]
            if ((qx - cx) ** 2 + (qy - cy) ** 2) ** 0.5 <= self.child_near * max(h, qh):
                return True
        return False

    def _child_signs(self, t: float, key, p: _Person, present: dict, children: dict) -> list[tuple]:
        out = []
        if self.child_alone_s > 0:
            if self._near_adult(p, present, children):
                p.alone_since = None
                p.fired.discard("alone")
            else:
                p.alone_since = t if p.alone_since is None else p.alone_since
                if t - p.alone_since >= self.child_alone_s and "alone" not in p.fired:
                    p.fired.add("alone")
                    out.append((key, "child", f"child with no adult near for {int(t - p.alone_since)} s", None, None))
        if p.role in CHILD_AREAS:
            p.area_since = t if p.area_since is None else p.area_since
            if t - p.area_since >= self.child_area_s and "area" not in p.fired:
                p.fired.add("area")
                where = {"cooking": "the cooking area", "food_prep": "the food preparation area"}.get(p.role,
                                                                                                    "a restricted area")
                out.append((key, "child", f"child in {where}", None, None))
        else:
            p.area_since = None
            p.fired.discard("area")
        if len(p.feet) >= 5:
            bottoms = sorted(b for _, b, _ in p.feet)
            heights = sorted(h for _, _, h in p.feet)
            usual_bottom, usual_h = bottoms[len(bottoms) // 2], heights[len(heights) // 2]
            _, bottom, h = p.feet[-1]
            up = usual_bottom - bottom >= self.child_climb * usual_h and h >= 0.85 * usual_h
            if up:
                p.climb_since = t if p.climb_since is None else p.climb_since
                if t - p.climb_since >= CLIMB_HOLD_S and "climb" not in p.fired:
                    p.fired.add("climb")
                    out.append((key, "child", "child climbing (feet raised)", None, None))
            else:
                p.climb_since = None
                if usual_bottom - bottom < 0.1 * usual_h:
                    p.fired.discard("climb")
        return out

    def tick(self, t: float) -> list[tuple]:
        """(key, category, reason, level or None, question or None) for each sign seen now."""
        out = []
        present = {k: p for k, p in self.people.items() if p.hist and t - p.last <= 1.0}
        if self.children_on:
            children = self._children(t, present)
            for key, p in children.items():
                out += self._child_signs(t, key, p, present, children)
        if not self.down_on:
            return out
        for key, p in self.people.items():
            if not p.hist or t - p.last > 1.0 or p.role == "bed":
                continue
            still = self._still_for(p, self.still_move)
            if still < 2.0:
                p.fired.clear()   # moving: every sign can be raised again
                continue
            lying = self._share(p, t - self.down_s, 4) >= SHARE
            low = self._share(p, t - self.slumped_s, 5) >= SHARE
            if still >= self.down_s and lying and "down" not in p.fired:
                p.fired.add("down")
                out.append((key, "person_down", f"lying still {int(still)} s", None, None))
            elif still >= self.slumped_s and low and not lying and "slumped" not in p.fired:
                p.fired.add("slumped")
                out.append((key, "person_down", f"slumped, head down, still {int(still)} s", None, None))
            elif still >= self.still_s and "still" not in p.fired:
                p.fired.add("still")
                out.append((key, "person_down", f"not moved for {int(still // 60)} min", "round", DISTRESS_QUESTION))
        return out
