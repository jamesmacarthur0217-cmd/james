"""Vehicles: cars, motorbikes, bicycles, buses and lorries, from a second detector (YOLO11s, COCO classes).

One record per vehicle visit (table `vehicles`; picture vehicle_<id>.jpg next to people's, with a copy for viewers in
which the faces of people in that frame are blurred, like people's pictures). A vehicle that stays where it is for `parked_after_s` is
"parked": it stays one record however long it stands there, also when someone walking past hides it for a while
(the tracker then loses it; a vehicle found again where a parked one was, within `parked_memory_s`, continues that
record), and it is marked "left" when it drives off. Without this a car parked all day would become a new record
every time it is hidden. A vehicle not yet parked but already standing still (`still_s`) is remembered the same way
for `still_memory_s`: the tracker loses standing bicycles behind passing people for several seconds at a time
(MOT16-04), and each return was a new record whose "still" clock started again, so they never became parked.

The watcher only decides; storing goes through a `store` (VehicleStore for the database, a fake one in tests).
Measured with tools/vehicle_check.py (README "Vehicles").
"""
from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# COCO class -> our name. COCO's "truck" includes pickups and vans with an open back: shown as "lorry or pickup".
KINDS = {1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "lorry"}
EDGE = 0.01   # a box this close to the picture's border is cut off: not used as the vehicle's picture


def _iou(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


@dataclass
class VehicleState:
    key: int
    first_t: float
    first_wall: datetime
    last_t: float = 0.0
    last_wall: datetime | None = None
    box: tuple = (0.0, 0.0, 0.0, 0.0)
    kinds: Counter = field(default_factory=Counter)   # COCO class -> summed confidence
    n: int = 0
    db_id: int | None = None
    anchor: tuple | None = None       # (cx, cy) where it last stood still
    recent: deque = field(default_factory=deque)   # (t, cx, cy, size) of the last second
    still_since: float = 0.0
    parked: bool = False
    left: bool = False
    best_area: float = 0.0
    best_crop: np.ndarray | None = None
    best_heads: list = field(default_factory=list)   # heads of people in the best picture (crop coordinates)
    best_clear: bool = False           # the best picture is confident and not cut off by the border
    best_version: int = 0              # counts picture changes
    picture_version: int = 0           # the version stored
    described: bool = False            # sent to be described (color, make/model) once
    zones: object = None               # zones.PersonZones: where it is relative to the camera's zones
    zone_log: list = field(default_factory=list)   # zone events not stored yet (no record yet)

    @property
    def kind(self) -> str:
        return KINDS[self.kinds.most_common(1)[0][0]] if self.kinds else "car"

    def size(self) -> float:
        return max(self.box[2] - self.box[0], self.box[3] - self.box[1], 1.0)


class VehicleWatch:
    """Vehicles on one camera. Call update() with each frame's vehicle detections (tracked)."""

    def __init__(self, cfg: dict, store, kinds: list[str] | None = None):
        self.store = store
        self.kinds = set(kinds or KINDS.values())
        self.min_frames = int(cfg["min_frames"])
        self.lost_after_s = float(cfg["lost_after_s"])
        self.parked_after_s = float(cfg["parked_after_s"])
        self.parked_memory_s = float(cfg["parked_memory_s"])
        self.move_ratio = float(cfg["move_ratio"])       # moved = the centre went this share of its size away
        self.still_s = float(cfg.get("still_s", 5.0))
        self.describe_after_s = float(cfg.get("describe_after_s", 5.0))
        self.still_memory_s = float(cfg.get("still_memory_s", 120.0))
        self.tracks: dict[int, VehicleState] = {}
        self.zone_counter = None   # zones.ZoneCounter of the camera (set by the pipeline): counting lines count vehicles
        self.parked_lost: dict[int, tuple[VehicleState, float]] = {}   # db id -> (state, lost at): standing ones

    # --- per frame ---------------------------------------------------------------------------------------

    def update(self, t: float, wall: datetime, frame: np.ndarray | None, dets: list[tuple],
               heads: list | None = None) -> None:
        """dets: [(tracker id, box, COCO class, confidence)] for this frame; heads: people's head boxes in it (their
        faces are blurred in the viewers' copy of a vehicle's picture)."""
        h, w = (frame.shape[:2] if frame is not None else (0, 0))
        for key, box, cls, conf in dets:
            if KINDS.get(cls) not in self.kinds:
                continue
            box = tuple(float(v) for v in box[:4])
            st = self.tracks.get(key)
            if st is None:
                st = self.tracks[key] = self._found_again(key, t, wall, box, cls) or VehicleState(key, t, wall)
            st.kinds[cls] += conf
            st.n += 1
            st.box, st.last_t, st.last_wall = box, t, wall
            self._movement(st, t, wall)
            if frame is not None:
                self._keep_picture(st, frame, box, conf, w, h, heads or [])
            if self.zone_counter is not None and self.zone_counter.zones and w:
                from .zones import PersonZones
                st.zones = st.zones or PersonZones()
                # Where the wheels touch the ground: the bottom middle of the box.
                st.zone_log += self.zone_counter.update(st.zones, ((box[0] + box[2]) / 2 / w, box[3] / h), wall)
            if st.db_id is None and st.n >= self.min_frames:
                st.db_id = self.store.create(st.kind, st.first_wall, key)
                self._store_picture(st)
                self.store.changed(st.db_id)   # after the picture: an alert pop-up shows it
            if st.db_id is not None and st.zone_log:
                self.store.zone_events(st.db_id, st.zone_log)
                st.zone_log = []
            if st.db_id is not None and not st.described and (st.parked or t - st.first_t >= self.describe_after_s):
                self._describe(st)
        for key, st in list(self.tracks.items()):
            if t - st.last_t > self.lost_after_s:
                self._finish(key)
        self.parked_lost = {k: v for k, v in self.parked_lost.items()
                            if t - v[1] <= (self.parked_memory_s if v[0].parked else self.still_memory_s)}

    def moving(self, t: float) -> bool:
        """A vehicle in view that is not parked (for recording and idle mode)."""
        return any(not st.parked and t - st.last_t < 1.0 and st.n >= self.min_frames for st in self.tracks.values())

    def shown(self, t: float) -> list[tuple[tuple, str]]:
        """(box, label) of the vehicles in view now, for the live picture."""
        out = []
        for st in self.tracks.values():
            if t - st.last_t < 0.5 and st.n >= self.min_frames:
                label = st.kind if st.kind != "lorry" else "lorry/pickup"
                out.append((st.box, f"#{st.db_id} {label}" + (" parked" if st.parked else "")))
        return out

    def close(self) -> None:
        for key in list(self.tracks):
            self._finish(key)

    # --- helpers -----------------------------------------------------------------------------------------

    def _found_again(self, key: int, t: float, wall: datetime, box: tuple, cls: int) -> VehicleState | None:
        """A parked (or standing) vehicle the tracker lost (hidden for a while) and finds again in the same place."""
        best, best_iou = None, 0.5
        for db_id, (st, _) in self.parked_lost.items():
            v = _iou(st.box, box)
            if v >= best_iou:
                best, best_iou = db_id, v
        if best is None:
            return None
        st, _ = self.parked_lost.pop(best)
        st.key = key
        self.store.update(st.db_id, ended_at=None)   # in view again
        return st

    def _movement(self, st: VehicleState, t: float, wall: datetime) -> None:
        # The median of the last second, not the box itself: people walking past a parked bicycle make its box
        # jump (MOT16-04: two bicycles in a rack never counted as parked from single boxes).
        st.recent.append((t, (st.box[0] + st.box[2]) / 2, (st.box[1] + st.box[3]) / 2, st.size()))
        while t - st.recent[0][0] > 1.0:
            st.recent.popleft()
        cx, cy, size = (float(np.median([r[i] for r in st.recent])) for i in (1, 2, 3))
        if st.anchor is None or math.hypot(cx - st.anchor[0], cy - st.anchor[1]) > self.move_ratio * size:
            if st.parked and not st.left:
                st.left = True      # it drives off: the visit continues until it is out of view
                st.parked = False
                if st.db_id is not None:
                    self.store.update(st.db_id, left_at=wall)
            st.anchor, st.still_since = (cx, cy), t
        elif not st.parked and not st.left and t - st.still_since >= self.parked_after_s and st.db_id is not None:
            st.parked = True
            self.store.update(st.db_id, parked=True, parked_at=wall)

    def _keep_picture(self, st: VehicleState, frame: np.ndarray, box: tuple, conf: float, w: int, h: int,
                      heads: list) -> None:
        x1, y1, x2, y2 = box
        clear = conf >= 0.5 and x1 > EDGE * w and y1 > EDGE * h and x2 < (1 - EDGE) * w and y2 < (1 - EDGE) * h
        area = (x2 - x1) * (y2 - y1)
        # A clear picture (confident, not cut off by the border) beats any unclear one; among the same, clearly
        # bigger is better (closer to the camera). Unclear ones are kept only while no clear one came: a bicycle
        # half hidden all the time still gets a picture (MOT16-04).
        if (clear and not st.best_clear) or (clear == st.best_clear and area > st.best_area * 1.15):
            pad_x, pad_y = 0.05 * (x2 - x1), 0.05 * (y2 - y1)
            ox, oy = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
            crop = frame[oy:min(h, int(y2 + pad_y)), ox:min(w, int(x2 + pad_x))]
            if crop.size:
                st.best_area, st.best_crop, st.best_clear = area, crop.copy(), clear
                st.best_version += 1
                st.best_heads = [(hx1 - ox, hy1 - oy, hx2 - ox, hy2 - oy) for hx1, hy1, hx2, hy2 in
                                 (tuple(b[:4]) for b in heads if b is not None)
                                 if hx2 > ox and hy2 > oy and hx1 < ox + crop.shape[1] and hy1 < oy + crop.shape[0]]

    def _describe(self, st: VehicleState) -> None:
        """Once per vehicle, with its best picture: after describe_after_s in view, when parked, or when it leaves."""
        if st.best_crop is not None:
            st.described = True
            self._store_picture(st)
            self.store.update(st.db_id, kind=st.kind)   # the detector's vote so far; the vision model may refine it
            self.store.describe(st.db_id, st.best_crop)

    def _store_picture(self, st: VehicleState) -> None:
        if st.db_id is not None and st.best_crop is not None and st.best_version > st.picture_version:
            self.store.picture(st.db_id, st.best_crop, st.best_heads)
            st.picture_version = st.best_version

    def _finish(self, key: int) -> None:
        st = self.tracks.pop(key)
        if st.db_id is None:
            return   # seen too briefly to count
        if not st.described:
            self._describe(st)
        self._store_picture(st)
        self.store.update(st.db_id, ended_at=st.last_wall, n_frames=st.n)
        standing = not st.left and st.last_t - st.still_since >= self.still_s
        if (st.parked and not st.left) or standing:
            self.parked_lost[st.db_id] = (st, st.last_t)   # hidden or gone: found again here = the same vehicle


class VehicleStore:
    """VehicleWatch's storage: the `vehicles` table and pictures in <crop_dir>/vehicles/."""

    def __init__(self, db, camera_id: int, crop_dir, describe=None, blur_all: bool = False, camera: str = "",
                 changed=None):
        self.db, self.camera_id, self.crop_dir, self.camera = db, camera_id, crop_dir, camera
        self._changed = changed   # (vehicle id) -> None: check the alert rules again
        self._describe = describe   # (vehicle id, picture) -> None: color now, the vision model's answer later
        self.blur_all = blur_all    # storage.blur_faces: faces blurred in the stored picture too
        (crop_dir / "blurred").mkdir(parents=True, exist_ok=True)

    def create(self, kind: str, started_at: datetime, tracker_id: int) -> int:
        return self.db.create_vehicle(self.camera_id, tracker_id, kind, started_at)

    def changed(self, vehicle_id: int) -> None:
        """Counted (with its first picture) or entered a zone: check the alert rules."""
        if self._changed is not None:
            self._changed(vehicle_id)

    def update(self, vehicle_id: int, **fields) -> None:
        self.db.update_vehicle(vehicle_id, **fields)

    def picture(self, vehicle_id: int, crop: np.ndarray, heads: list | None = None) -> None:
        import cv2
        from . import privacy
        name = f"vehicle_{vehicle_id}.jpg"
        blurred = privacy.blur_boxes(crop, heads or [])
        cv2.imwrite(str(self.crop_dir / name), blurred if self.blur_all else crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(privacy.blurred_path(self.crop_dir, name)), blurred, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.db.update_vehicle(vehicle_id, crop_path=name)

    def zone_events(self, vehicle_id: int, events: list[dict]) -> None:
        self.db.add_vehicle_zone_events(vehicle_id, self.camera, events)
        self.changed(vehicle_id)

    def describe(self, vehicle_id: int, crop: np.ndarray) -> None:
        if self._describe is not None:
            self._describe(vehicle_id, crop)


def body_color(crop: np.ndarray) -> str | None:
    """A first color from the color estimator: the middle of the vehicle (not wheels, windows at the top, road)."""
    from . import colors
    h, w = crop.shape[:2]
    if colors.is_low_color(crop):
        return None   # black and white (infrared at night): no color
    hit = colors.dominant(colors.color_histogram(crop[int(0.35 * h):int(0.75 * h), int(0.15 * w):int(0.85 * w)]))
    return hit[0] if hit else None


def apply_description(db, vehicle_id: int, answer: dict[str, str], night: bool = False) -> None:
    """Store the vision model's answer. "not a vehicle" hides the record (kept for checking, never listed)."""
    if answer.get("type") == "not a vehicle":
        db.update_vehicle(vehicle_id, not_vehicle=True)
        return
    fields = {k: answer[k] for k in ("make", "model") if answer.get(k, "unknown") != "unknown"}
    if answer.get("type") in ("car", "SUV", "van", "pickup", "lorry", "bus", "motorbike", "bicycle"):
        fields["kind"] = answer["type"]   # finer than the detector's (van, pickup)
    if answer.get("color", "unknown") != "unknown" and not night:
        fields["color"] = answer["color"]
    if fields:
        db.update_vehicle(vehicle_id, **fields)
