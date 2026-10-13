"""Animals: dogs, cats and birds, from the vehicle detector (YOLO11s knows them as COCO classes), in the same pass.

The user chose dogs, cats and birds (2026-10-14); COCO also has horses, sheep and cows (ANIMAL_KINDS lists what can
be chosen); monkeys and monitor lizards are not in it. On security cameras by default (`cameras[].animals`).

Followed like vehicles (labeler/vehicles.py: the same VehicleWatch, with these names): one record per visit (table
`animals`, picture animal_<id>.jpg with a copy for viewers in which people's faces are blurred); an animal that stays
where it is for `parked_after_s` is "resting" (a dog asleep on the porch) and stays one record however long it lies
there, also when it is hidden for a while. The color comes from the color estimator (no vision model): rough, e.g. a
black dog. Animals do not wake idle mode or start recording (birds fly past all day).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

# COCO class -> our name.
ALL_KINDS = {14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow"}


def kinds_map(kinds: list[str]) -> dict[int, str]:
    """The COCO classes of the chosen kinds."""
    return {cls: name for cls, name in ALL_KINDS.items() if name in kinds}


class AnimalStore:
    """The watcher's storage for animals: the `animals` table and pictures in <crop_dir>/animal_<id>.jpg."""

    # The watcher speaks of parking (labeler/vehicles.py); an animal is resting.
    FIELDS = {"parked": "resting", "parked_at": "resting_at"}

    def __init__(self, db, camera_id: int, crop_dir, blur_all: bool = False, camera: str = "", changed=None):
        self.db, self.camera_id, self.crop_dir, self.camera = db, camera_id, crop_dir, camera
        self.blur_all = blur_all    # storage.blur_faces: faces blurred in the stored picture too
        self._changed = changed     # (animal id) -> None: check the alert rules again
        (crop_dir / "blurred").mkdir(parents=True, exist_ok=True)

    def create(self, kind: str, started_at: datetime, tracker_id: int) -> int:
        return self.db.create_animal(self.camera_id, tracker_id, kind, started_at)

    def changed(self, animal_id: int) -> None:
        if self._changed is not None:
            self._changed(animal_id)

    def update(self, animal_id: int, **fields) -> None:
        self.db.update_animal(animal_id, **{self.FIELDS.get(k, k): v for k, v in fields.items()})

    def picture(self, animal_id: int, crop: np.ndarray, heads: list | None = None) -> None:
        import cv2
        from . import privacy
        name = f"animal_{animal_id}.jpg"
        blurred = privacy.blur_boxes(crop, heads or [])
        cv2.imwrite(str(self.crop_dir / name), blurred if self.blur_all else crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(privacy.blurred_path(self.crop_dir, name)), blurred, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.db.update_animal(animal_id, crop_path=name)

    def zone_events(self, animal_id: int, events: list[dict]) -> None:
        self.db.add_animal_zone_events(animal_id, self.camera, events)
        self.changed(animal_id)

    def describe(self, animal_id: int, crop: np.ndarray) -> None:
        """Its color from the color estimator (the same as a vehicle's body color)."""
        from .vehicles import body_color
        color = body_color(crop)
        if color:
            self.db.update_animal(animal_id, color=color)
