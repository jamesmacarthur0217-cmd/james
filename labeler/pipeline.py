"""Camera loop: detect and track people, pick their best frames, and label them."""
from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

from . import colors, geometry, privacy, schema
from .config import resolve
from .db import Database
from .embed import Embedder
from .sources import FrameSource, is_blank
from .student import Student
from .vlm import VLMWorker

log = logging.getLogger(__name__)

# COCO pose skeleton (keypoint index pairs) for the live overlay.
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
HIDDEN_VALUES = ("unknown", "none", "no")


@dataclass
class Candidate:
    score: float
    person: np.ndarray
    face: np.ndarray | None
    head: tuple | None = None     # where to blur in `person` for privacy (geometry.head_box, crop coordinates)


@dataclass
class TrackState:
    tracker_id: int
    started_at: datetime
    last_seen: float
    db_id: int | None = None
    n_frames: int = 0
    low_light_frames: int = 0
    box: tuple = (0, 0, 0, 0)
    first_box: tuple | None = None
    vector: np.ndarray | None = None      # appearance embedding, used to re-link a returning person
    candidates: list[Candidate] = field(default_factory=list)
    upper_hist: Counter = field(default_factory=Counter)
    lower_hist: Counter = field(default_factory=Counter)
    vlm_requested: bool = False
    labels: dict[str, tuple[str, float]] = field(default_factory=dict)   # name -> (value, confidence)
    # Live analysis while the person stays in view.
    seen_since: float = 0.0                                   # when they (re)appeared, for "time in view"
    kpts: np.ndarray | None = None                            # latest pose, for the overlay skeleton
    recent: list[Candidate] = field(default_factory=list)     # best frames since the last labeling
    thumb: np.ndarray | None = None                           # latest small crop for the live panel
    thumb_blurred: np.ndarray | None = None                   # the same with the face blurred, for viewers
    vlm_pending: bool = False
    last_labeled: float | None = None
    vlm_history: list[dict[str, str]] = field(default_factory=list)   # recent per-frame VLM answers
    student_labels: dict[str, tuple[str, float]] = field(default_factory=dict)  # trusted fast predictions
    deleted: bool = False          # the user deleted this person's record; stop recording them
    age_label: tuple[str, str, float] | None = None   # (attribute, value, confidence) from the age model

    def colors(self) -> dict[str, tuple[str, float]]:
        out = {}
        for name, hist in (("upper_color", self.upper_hist), ("lower_color", self.lower_hist)):
            result = colors.dominant(hist)
            if result:
                out[name] = result
        return out


class Pipeline:
    def __init__(self, cfg: dict, db: Database, vlm: VLMWorker | None, embedder: Embedder | None = None,
                 student: Student | None = None, camera: dict | None = None):
        self.cfg = cfg
        self.cam = camera or cfg.get("cameras", [cfg.get("camera")])[0]
        self.name = self.cam["name"]
        self.mode = self.cam.get("mode", cfg["mode"])
        self.db = db
        self.vlm = vlm
        self.embedder = embedder
        self.student = student if embedder is not None else None
        self.alerts = None   # AlertEngine, set by run.py
        self.age = None      # AgeEstimator, set by run.py
        self.age_ids: set[int] = set()   # people whose age came from the age model
        self.vlm_model_name = cfg["vlm"]["model"]
        self.threshold = cfg["review"]["confidence_threshold"]
        self.crop_dir = resolve(cfg["storage"]["crop_dir"])
        self.crop_dir.mkdir(parents=True, exist_ok=True)

        det = dict(cfg["detector"])
        self.model = YOLO(det["model"])
        tracker_path = resolve(det["tracker"])
        if tracker_path.exists():  # project tracker settings; otherwise an Ultralytics built-in name
            det["tracker"] = str(tracker_path)
        self.det_cfg = det
        cam = self.cam
        self.camera_id = db.get_or_create_camera(cam["name"], str(cam["source"]), self.mode)
        self.owned_ids: set[int] = set()   # records created by this camera, for routing VLM results

        self.tracks: dict[int, TrackState] = {}
        self.by_db_id: dict[int, TrackState] = {}
        # People who recently left the view, kept briefly so a returning person continues
        # the same record instead of becoming a new one: db_id -> (state, time lost).
        self.recently_lost: dict[int, tuple[TrackState, float]] = {}
        self.relink_cfg = cfg["tracking"]["relink"]
        min_area = det.get("min_box_area", 0.0)
        self.min_box_area = min_area.get(self.mode, 0.0) if isinstance(min_area, dict) else float(min_area)
        self.ignore_zones = [tuple(z) for z in cam.get("ignore_zones") or []]
        self._ignored: list[tuple] = []
        self.stop_event = threading.Event()
        self._jpeg_lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._jpeg_blurred: bytes | None = None
        self.viewer_watching_at = -1e9    # a viewer's live video wants blurred frames (privacy.py)
        self.fps = 0.0
        self.frames = 0
        self.blank = False
        self.running = False
        self.crashed = False
        self.deleted_ids: set[int] = set()
        self._live_lock = threading.Lock()
        self._live: list[dict] = []
        self._thumbs: dict[int, bytes] = {}
        self._thumbs_blurred: dict[int, bytes] = {}

    # --- public -------------------------------------------------------------------

    @property
    def latest_jpeg(self) -> bytes | None:
        with self._jpeg_lock:
            return self._jpeg

    def latest_jpeg_blurred(self) -> bytes | None:
        """The live frame with faces blurred, for viewers. Made only while a viewer is watching, so each
        call keeps it coming for a few more seconds."""
        self.viewer_watching_at = time.monotonic()
        with self._jpeg_lock:
            return self._jpeg_blurred

    def live(self) -> list[dict]:
        """People in view right now, with their current labels (updated every frame)."""
        with self._live_lock:
            return list(self._live)

    def live_thumb(self, tracker_id: int, blurred: bool = False) -> bytes | None:
        with self._live_lock:
            return (self._thumbs_blurred if blurred else self._thumbs).get(tracker_id)

    def forget(self, db_ids: list[int]) -> None:
        """Called when the user deletes records. A deleted person still in view is not recorded again
        (shown as a gray "deleted" box); if they leave and come back, they are a new person."""
        ids = set(db_ids)
        self.deleted_ids |= ids
        for st in list(self.tracks.values()):
            if st.db_id in ids:
                st.deleted = True
        for db_id in ids:
            self.by_db_id.pop(db_id, None)
            self.recently_lost.pop(db_id, None)

    def status(self) -> dict:
        return {
            "camera": self.name,
            "running": self.running,
            "mode": self.mode,
            "fps": round(self.fps, 1),
            "frames": self.frames,
            "active_people": len(self.tracks),
            "blank_picture": self.blank,
            "vlm_queue": self.vlm.pending if self.vlm else None,
            "vlm_enabled": self.vlm is not None,
        }

    def stop(self) -> None:
        self.stop_event.set()

    def run(self, max_frames: int | None = None) -> None:
        cam = self.cam
        source = FrameSource(cam["source"], loop=cam["loop_files"] and max_frames is None)
        log.info("[%s] pipeline started: source=%s (%s), mode=%s", self.name, cam["source"], source.kind, self.mode)
        t_last = time.monotonic()
        self.running = True
        try:
            while not self.stop_event.is_set():
                frame = source.read()
                if frame is None:
                    if source.ended:
                        break
                    continue
                self.process(frame)
                self.frames += 1
                now = time.monotonic()
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - t_last, 1e-6))
                t_last = now
                if max_frames and self.frames >= max_frames:
                    break
        except Exception:
            self.crashed = True
            log.exception("[%s] pipeline crashed", self.name)
            raise
        finally:
            self.running = False
            source.release()
            for tid in list(self.tracks):
                self._finish(tid)
            log.info("pipeline stopped after %d frames", self.frames)

    # --- per frame ----------------------------------------------------------------

    def process(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        tcfg = self.cfg["tracking"]
        result = self.model.track(
            frame, persist=True, tracker=self.det_cfg["tracker"], classes=[0],
            conf=self.det_cfg["conf"], imgsz=self.det_cfg["imgsz"], device=self.det_cfg["device"],
            verbose=False,
        )[0]
        low_color = colors.is_low_color(frame)
        if self.frames % 30 == 0:
            blank = is_blank(frame)
            if blank and not self.blank:
                log.warning("the camera is sending a blank picture: check the lens cover, "
                            "and close other apps that use the camera")
            elif self.blank and not blank:
                log.info("camera picture is back")
            self.blank = blank

        seen = set()
        self._ignored = []
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().tolist()
            kpts_all = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None
            for i, tid in enumerate(ids):
                if tid not in self.tracks and self._should_ignore(tuple(xyxy[i]), frame.shape):
                    self._ignored.append(tuple(xyxy[i]))
                    continue
                if tid in self.tracks and self.tracks[tid].deleted:
                    self.tracks[tid].last_seen = now
                    self._ignored.append((*tuple(xyxy[i]), "deleted"))
                    continue
                seen.add(tid)
                kpts = kpts_all[i] if kpts_all is not None else None
                st = self.tracks.get(tid)
                if st is None:
                    st = self.tracks[tid] = TrackState(tid, datetime.now(), now, seen_since=now)
                self._update(st, frame, tuple(xyxy[i]), kpts, now, low_color, tcfg)
                if st.db_id is None and st.n_frames in self.relink_cfg["check_at_frames"]:
                    st = self._try_relink(tid, st, now)

                if st.n_frames >= tcfg["min_frames_for_label"] and not st.vlm_requested:
                    self._ensure_db(st)
                    self._request_vlm(st)
                elif self._due_for_relabel(st, now):
                    self._relabel(st)

        for tid, st in list(self.tracks.items()):
            if tid not in seen and now - st.last_seen > tcfg["lost_after_s"]:
                self._finish(tid, lost_at=st.last_seen)
        if self.recently_lost:
            window = self.relink_cfg["window_s"]
            self.recently_lost = {k: v for k, v in self.recently_lost.items() if now - v[1] <= window}

        blurred = None
        if self.cfg["web"].get("blur_faces_for_viewers", True) and now - self.viewer_watching_at < 5:
            # Faces are blurred before the labels are drawn, so the labels stay readable.
            heads = [geometry.head_box(self.tracks[t].kpts, self.tracks[t].box) for t in seen]
            heads += [geometry.head_box(None, tuple(b[:4])) for b in self._ignored]
            blurred = self._draw(privacy.blur_boxes(frame, heads), seen, low_color, now)
        self._publish(self._draw(frame, seen, low_color, now), blurred)
        self._publish_live(seen, now)

    def _should_ignore(self, box: tuple, shape: tuple) -> bool:
        """Skip detections that cannot be a person we care about: inside an ignore zone (a monitor,
        poster or TV), or, in kiosk mode, too small to be someone standing at the kiosk.
        Only checked when a person first appears, so someone leaning back is never dropped."""
        h, w = shape[:2]
        x1, y1, x2, y2 = box
        if self.min_box_area and (x2 - x1) * (y2 - y1) / (w * h) < self.min_box_area:
            return True
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        return any(zx1 <= cx <= zx2 and zy1 <= cy <= zy2 for zx1, zy1, zx2, zy2 in self.ignore_zones)

    def _update(self, st: TrackState, frame: np.ndarray, box: tuple, kpts: np.ndarray | None,
                now: float, low_color: bool, tcfg: dict) -> None:
        st.n_frames += 1
        st.last_seen = now
        st.box = box
        st.kpts = kpts
        if st.first_box is None:
            st.first_box = box
        if low_color:
            st.low_light_frames += 1
        elif st.n_frames % tcfg["color_every_n_frames"] == 0:
            st.upper_hist += colors.color_histogram(geometry.crop(frame, geometry.torso_box(kpts, box)))
            st.lower_hist += colors.color_histogram(geometry.crop(frame, geometry.legs_box(kpts, box)))

        score = geometry.quality(frame, box, kpts, self.mode)
        k = tcfg["best_frames"]
        best_wanted = len(st.candidates) < k or score > st.candidates[-1].score
        recent_wanted = len(st.recent) < 2 or score > st.recent[-1].score
        thumb_wanted = st.n_frames % 5 == 1
        if not (best_wanted or recent_wanted or thumb_wanted):
            return
        person = geometry.crop(frame, box, pad=0.05)
        if person is None:
            return
        ox, oy = geometry.crop_origin(frame, box, pad=0.05)
        hx1, hy1, hx2, hy2 = geometry.head_box(kpts, box)
        head = (hx1 - ox, hy1 - oy, hx2 - ox, hy2 - oy)
        if thumb_wanted:
            h, w = person.shape[:2]
            size = (max(1, int(w * 160 / h)), 160)
            st.thumb = cv2.resize(person, size, interpolation=cv2.INTER_AREA)
            if self.cfg["web"].get("blur_faces_for_viewers", True):
                st.thumb_blurred = cv2.resize(privacy.blur_boxes(person, [head]), size, interpolation=cv2.INTER_AREA)
        if self.student is not None and st.n_frames % self.cfg["student"]["every_n_frames"] == 1:
            self._student_update(st, person)
        if best_wanted or recent_wanted:
            face = geometry.crop(frame, geometry.face_box(kpts), pad=0.1)
            cand = Candidate(score, person.copy(), face.copy() if face is not None else None, head)
            for pool, size, wanted in ((st.candidates, k, best_wanted), (st.recent, 2, recent_wanted)):
                if wanted:
                    pool.append(cand)
                    pool.sort(key=lambda c: c.score, reverse=True)
                    del pool[size:]

    # --- track lifecycle ----------------------------------------------------------

    def _ensure_db(self, st: TrackState) -> int:
        if st.db_id is None:
            st.db_id = self.db.create_track(self.camera_id, st.tracker_id, st.started_at)
            self.owned_ids.add(st.db_id)
            self.by_db_id[st.db_id] = st
        return st.db_id

    def _save_crops(self, st: TrackState) -> None:
        if not st.candidates:
            return
        best = st.candidates[0]
        name = f"{st.db_id}.jpg"
        blurred = (privacy.blur_boxes(best.person, [best.head]) if best.head is not None
                   else privacy.blur_whole(best.person))
        blur_on_disk = self.cfg["storage"].get("blur_faces", False)
        cv2.imwrite(str(self.crop_dir / name), blurred if blur_on_disk else best.person)
        if self.cfg["web"].get("blur_faces_for_viewers", True) and not blur_on_disk:
            path = privacy.blurred_path(self.crop_dir, name)
            path.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(path), blurred)
        fields = {"crop_path": name}
        # A face close-up would defeat blurred pictures, so it is never saved then.
        if self.cfg["storage"]["save_face_crops"] and best.face is not None and not blur_on_disk:
            face_name = f"{st.db_id}_face.jpg"
            cv2.imwrite(str(self.crop_dir / face_name), best.face)
            fields["face_path"] = face_name
        self.db.update_track(st.db_id, **fields)

    def _check_alerts(self, db_id: int | None) -> None:
        if self.alerts is not None and db_id is not None and db_id not in self.deleted_ids:
            try:
                self.alerts.check(db_id, self.name)
            except Exception:
                log.exception("checking alert rules failed")

    def _store_colors(self, st: TrackState) -> None:
        items = [(name, value, round(conf, 3)) for name, (value, conf) in st.colors().items()]
        if items:
            self.db.add_attributes(st.db_id, items, source="color_estimator", model="hsv-rules-v1",
                                   threshold=self.threshold)
        self._check_alerts(st.db_id)

    def _store_embedding(self, st: TrackState) -> None:
        if self.embedder is None or not st.candidates:
            return
        try:
            vector = self.embedder.track_vector([c.person for c in st.candidates])
            st.vector = vector
            self.db.upsert_embedding(st.db_id, vector, self.embedder.model_name)
            if self.student is not None:
                self._store_student(st, vector)
        except Exception:
            log.exception("embedding track %s failed", st.db_id)

    def _student_update(self, st: TrackState, person: np.ndarray) -> None:
        """Instant labels for the live view from the latest crop (trusted predictions only)."""
        try:
            preds = self.student.predict(self.embedder.images([person])[0], self.mode)
        except Exception:
            log.exception("student prediction failed")
            return
        st.student_labels = self._student_trusted(preds)

    def _student_trusted(self, preds: dict[str, tuple[str, float, bool]]) -> dict[str, tuple[str, float]]:
        """The student's trusted answers. While an age model is loaded the student gives no age: it learned
        age from the VLM, and the age model measured better. Otherwise its confident age would outrank the
        age model's (the highest confidence wins) and every disagreement would go to review."""
        return {n: (v, c) for n, (v, c, ok) in preds.items()
                if ok and not (self.age is not None and n in ("age_range", "age_group"))}

    def _store_student(self, st: TrackState, vector: np.ndarray) -> None:
        """Record the student's trusted labels for this person (from their best crops)."""
        trusted = self._student_trusted(self.student.predict(vector, self.mode))
        st.student_labels = trusted
        if trusted:
            self.db.add_attributes(st.db_id, [(n, v, round(c, 3)) for n, (v, c) in trusted.items()],
                                   source="student", model=self.student.name, threshold=self.threshold)
            self._check_alerts(st.db_id)

    def _student_covers_all(self, st: TrackState) -> bool:
        """True when the VLM has nothing left to add: the student is confident about every attribute,
        with age coming from the age model when it found a face."""
        if self.student is None:
            return False
        covered = set(st.student_labels) | ({st.age_label[0]} if st.age_label else set())
        return all(n in covered for n in schema.MODE_FIELDS[self.mode])

    def _store_age(self, st: TrackState) -> None:
        """Apparent age from the person's best face crops (the age model beats the VLM on age)."""
        if self.age is None:
            return
        try:
            result = self.age.estimate([c.face for c in st.candidates if c.face is not None][:3])
        except Exception:
            log.exception("age estimation failed")
            return
        if result is None:
            return
        from .age import TO_GROUP
        value, conf = result
        name = "age_range" if self.mode == "kiosk" else "age_group"
        if name == "age_group":
            value = TO_GROUP[value]
        st.age_label = (name, value, conf)
        self.age_ids.add(st.db_id)
        self.db.add_attributes(st.db_id, [(name, value, round(conf, 3))], source="age_model",
                               model=self.age.name, threshold=self.threshold)

    def _request_vlm(self, st: TrackState) -> None:
        st.vlm_requested = True
        st.recent = []
        self._save_crops(st)
        self._store_colors(st)
        self._store_embedding(st)
        self._store_age(st)
        if self.vlm is None:
            return
        if self.cfg["student"]["skip_vlm_when_confident"] and self._student_covers_all(st):
            log.info("person #%s labeled by the fast classifiers alone (all attributes confident)", st.db_id)
            return
        n = self.cfg["vlm"]["frames_per_track"]
        frames = [(c.person, c.face) for c in st.candidates[:n]]
        st.vlm_pending = self.vlm.submit(st.db_id, frames, self.mode)

    def _due_for_relabel(self, st: TrackState, now: float) -> bool:
        """Re-analyze a person who stays in view, so labels follow changes (glasses on, phone out).
        New people always go first: re-labeling waits until the VLM queue is empty."""
        every = self.cfg["vlm"]["relabel_every_s"]
        return (every > 0 and self.vlm is not None and st.vlm_requested and not st.vlm_pending
                and not (self.cfg["student"]["skip_vlm_when_confident"] and self._student_covers_all(st))
                and st.last_labeled is not None and now - st.last_labeled >= every
                and bool(st.recent) and self.vlm.pending == 0)

    def _relabel(self, st: TrackState) -> None:
        frames = [(c.person, c.face) for c in st.recent[:1]]
        st.recent = []
        st.vlm_pending = self.vlm.submit(st.db_id, frames, self.mode)

    def _try_relink(self, tid: int, new: TrackState, now: float) -> TrackState:
        """If this new track is a person who just left, continue their old record instead.

        Short gaps (detection dropouts, leaning out of view) need moderate appearance similarity
        plus a box near where the person was lost. Longer gaps need high similarity alone.
        Thresholds come from measurements on webcam footage; when in doubt, keep them separate:
        a duplicate record is better than merging two different people."""
        rc = self.relink_cfg
        if not rc["enabled"] or self.embedder is None or not self.recently_lost or not new.candidates:
            return new
        vec = self.embedder.track_vector([c.person for c in new.candidates])
        best = None
        for db_id, (old, lost_at) in self.recently_lost.items():
            if old.vector is None:
                continue
            gap = now - lost_at
            sim = float(vec @ old.vector)
            overlap = geometry.box_iou(old.box, new.first_box or new.box)
            if gap <= rc["short_gap_s"]:
                ok = sim >= rc["short_gap_min_similarity"] and overlap >= rc["short_gap_min_overlap"]
            else:
                ok = sim >= rc["long_gap_min_similarity"]
            if ok and (best is None or sim > best[1]):
                best = (db_id, sim, overlap, gap)
        if best is None:
            return new

        db_id, sim, overlap, gap = best
        old, _ = self.recently_lost.pop(db_id)
        old.tracker_id = tid
        old.n_frames += new.n_frames
        old.low_light_frames += new.low_light_frames
        old.last_seen, old.box = new.last_seen, new.box
        old.seen_since, old.kpts, old.thumb, old.recent = new.seen_since, new.kpts, new.thumb, new.recent
        old.upper_hist += new.upper_hist
        old.lower_hist += new.lower_hist
        k = self.cfg["tracking"]["best_frames"]
        old.candidates = sorted(old.candidates + new.candidates, key=lambda c: c.score, reverse=True)[:k]
        self.tracks[tid] = old
        self.by_db_id[db_id] = old
        self.db.update_track(db_id, status="active", ended_at=None)
        log.info("person #%d is back after %.1f s (similarity %.2f, box overlap %.2f): same record",
                 db_id, gap, sim, overlap)
        return old

    def _finish(self, tid: int, lost_at: float | None = None) -> None:
        st = self.tracks.pop(tid)
        if st.deleted:
            return
        if st.n_frames < self.cfg["tracking"]["min_frames_keep"]:
            if st.db_id is not None:
                self.by_db_id.pop(st.db_id, None)
            return
        self._ensure_db(st)
        if not st.vlm_requested:
            self._request_vlm(st)
        else:
            # The best frame and color estimate keep improving after labeling starts.
            self._save_crops(st)
            self._store_colors(st)
            self._store_embedding(st)
            self._store_age(st)
        self.db.update_track(
            st.db_id, status="finished", ended_at=datetime.now(), n_frames=st.n_frames,
            low_light=st.low_light_frames > st.n_frames / 2,
        )
        # A VLM result arriving after this point is still written to the database.
        self.by_db_id.pop(st.db_id, None)
        if lost_at is not None and st.vector is not None:
            self.recently_lost[st.db_id] = (st, lost_at)

    def on_vlm_result(self, db_id: int, results: list[dict[str, str]]) -> None:
        """Called on the VLM thread. Labels are a majority vote over the person's most recent
        answers, so they follow real changes without flickering on a single odd frame."""
        if db_id in self.deleted_ids:
            return
        st = self.by_db_id.get(db_id)
        if st is not None:
            st.vlm_pending = False
            st.last_labeled = time.monotonic()
        if not results:
            return
        if st is not None:
            st.vlm_history = (st.vlm_history + results)[-self.cfg["vlm"]["history_frames"]:]
            pool = st.vlm_history
        else:
            pool = results
        items = []
        for name in schema.MODE_FIELDS[self.mode]:
            if name in ("age_range", "age_group") and db_id in self.age_ids:
                continue   # the age model's answer is used instead (more accurate, see labeler/age.py)
            votes = Counter(r[name] for r in pool if r.get(name) not in (None, "unknown"))
            if votes:
                value, count = votes.most_common(1)[0]
                items.append((name, value, round(count / len(pool), 3)))
            else:
                items.append((name, "unknown", 1.0))
        changed = st is not None and st.labels and any(
            st.labels.get(n, (None,))[0] != v for n, v, _ in items)
        self.db.add_attributes(db_id, items, source="vlm", model=self.vlm_model_name, threshold=self.threshold)
        self._check_alerts(db_id)
        if st is not None:
            st.labels = {name: (value, conf) for name, value, conf in items}
        summary = ", ".join(f"{n}={v}" for n, v, _ in items if v not in ("unknown", "none"))
        log.info("person #%s %s: %s", db_id, "updated" if changed else "labeled", summary)

    # --- live view ----------------------------------------------------------------

    def _current_labels(self, st: TrackState) -> dict[str, tuple[str, float, str]]:
        """name -> (value, confidence, source): VLM labels first, then the fast classifiers,
        then live colors filling the gaps."""
        out = {n: (v, c, "vlm") for n, (v, c) in st.labels.items()}
        if st.age_label:
            name, value, conf = st.age_label
            out[name] = (value, round(conf, 2), "age_model")
        for name, (value, conf) in st.student_labels.items():
            if out.get(name, ("unknown",))[0] == "unknown":
                out[name] = (value, round(conf, 2), "student")
        for name, (value, conf) in st.colors().items():
            if out.get(name, ("unknown",))[0] in ("unknown",):
                out[name] = (value, round(conf, 2), "color_estimator")
        return out

    def _status(self, st: TrackState) -> str:
        if self._student_covers_all(st) and not st.vlm_pending:
            return "labeled"
        if not st.vlm_requested:
            return "observing"
        if st.vlm_pending:
            return "updating" if st.labels else "labeling"
        return "labeled" if st.labels else ("colors only" if self.vlm is None else "labeling")

    def _publish_live(self, seen: set[int], now: float) -> None:
        people = []
        for tid in sorted(seen):
            st = self.tracks[tid]
            people.append({
                "tracker_id": tid,
                "id": st.db_id,
                "in_view_s": round(now - st.seen_since, 1),
                "status": self._status(st),
                "labeled_ago_s": round(now - st.last_labeled, 1) if st.last_labeled else None,
                "attributes": {n: {"value": v, "confidence": c, "source": src, "needs_review": False}
                               for n, (v, c, src) in self._current_labels(st).items()},
            })
        thumbs = thumbs_blurred = None
        if self.frames % 10 == 0:
            thumbs, thumbs_blurred = {}, {}
            for tid in seen:
                for image, out in ((self.tracks[tid].thumb, thumbs), (self.tracks[tid].thumb_blurred, thumbs_blurred)):
                    if image is not None:
                        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok:
                            out[tid] = buf.tobytes()
        with self._live_lock:
            self._live = people
            if thumbs is not None:
                self._thumbs, self._thumbs_blurred = thumbs, thumbs_blurred

    # --- drawing ------------------------------------------------------------------

    def _label_lines(self, st: TrackState, now: float) -> list[str]:
        secs = int(now - st.seen_since)
        name = f"#{st.db_id}" if st.db_id else "new"
        status = self._status(st)
        lines = [f"{name}  {secs // 60}:{secs % 60:02d}" + ("" if status == "labeled" else f"  {status}...")]
        labels = {n: v for n, (v, _, _) in self._current_labels(st).items()}
        words = [labels[n] for n in schema.DISPLAY_ORDER if labels.get(n) not in (None, *HIDDEN_VALUES)]
        if words:
            lines.append(" ".join(words[:3]))
            if len(words) > 3:
                lines.append(" ".join(words[3:]))
        return lines

    def _draw(self, frame: np.ndarray, seen: set[int], low_color: bool, now: float) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        for zx1, zy1, zx2, zy2 in self.ignore_zones:
            cv2.rectangle(out, (int(zx1 * w), int(zy1 * h)), (int(zx2 * w), int(zy2 * h)), (90, 90, 90), 1)
        for x1, y1, x2, y2, *why in self._ignored:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (140, 140, 140), 1)
            cv2.putText(out, why[0] if why else "ignored", (int(x1) + 2, int(y1) + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (170, 170, 170), 1, cv2.LINE_AA)
        for tid in seen:
            st = self.tracks[tid]
            x1, y1, x2, y2 = (int(v) for v in st.box)
            color = (80, 200, 120) if st.labels else (0, 180, 255)
            if st.kpts is not None:
                for a, b in SKELETON:
                    if st.kpts[a, 2] > 0.5 and st.kpts[b, 2] > 0.5:
                        cv2.line(out, (int(st.kpts[a, 0]), int(st.kpts[a, 1])),
                                 (int(st.kpts[b, 0]), int(st.kpts[b, 1])), (255, 200, 80), 2, cv2.LINE_AA)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            y = max(y1, 20)
            for line in self._label_lines(st, now):
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(out, (x1, y - th - 6), (x1 + tw + 6, y + 2), color, -1)
                cv2.putText(out, line, (x1 + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
                y += th + 8
        if self.blank:
            for i, line in enumerate(["NO PICTURE FROM CAMERA", "Check the lens cover, and close other apps",
                                      "that use the camera (Teams, Discord, ...)"]):
                cv2.putText(out, line, (20, 50 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8 if i else 1.0,
                            (0, 200, 255), 2, cv2.LINE_AA)
        elif low_color:
            cv2.putText(out, "LOW COLOR (IR/night): color labels paused", (10, out.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
        return out

    def _publish(self, frame: np.ndarray, blurred: np.ndarray | None = None) -> None:
        def encode(img: np.ndarray) -> bytes | None:
            h, w = img.shape[:2]
            if w > 1280:
                img = cv2.resize(img, (1280, int(h * 1280 / w)))
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buf.tobytes() if ok else None

        jpeg = encode(frame)
        jpeg_blurred = encode(blurred) if blurred is not None else None
        with self._jpeg_lock:
            if jpeg is not None:
                self._jpeg = jpeg
            self._jpeg_blurred = jpeg_blurred
