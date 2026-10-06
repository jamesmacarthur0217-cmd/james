"""Database models and queries. SQLite by default; the same models work on PostgreSQL."""
from __future__ import annotations

import json
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text, TypeDecorator, UniqueConstraint,
    create_engine, delete, event, select, text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, selectinload, sessionmaker

from .config import sqlite_url

UNKNOWN = ("unknown", "none")

# Size of the stored embeddings (SigLIP 2 base). A model with a different size needs a new column.
EMBED_DIM = 768


class VectorColumn(TypeDecorator):
    """A float32 vector: a pgvector column on PostgreSQL, raw bytes on SQLite.
    Reads always return a numpy array."""
    impl = LargeBinary
    cache_ok = True

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector
            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(LargeBinary())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        return arr if dialect.name == "postgresql" else arr.tobytes()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return np.asarray(value, dtype=np.float32)
        return np.frombuffer(value, dtype=np.float32)


def _conflict(items: list[Attribute]) -> bool:
    """True when two sources give different confident answers for the same attribute."""
    return len({a.value for a in items if a.value not in UNKNOWN and a.confidence >= 0.5}) > 1


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "cameras"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    source: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(20))


class Track(Base):
    """One person, from first to last appearance in one camera."""
    __tablename__ = "tracks"
    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"), index=True)
    tracker_id: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20), default="active")
    n_frames: Mapped[int] = mapped_column(Integer, default=0)
    crop_path: Mapped[str | None] = mapped_column(String(300))
    face_path: Mapped[str | None] = mapped_column(String(300))
    low_light: Mapped[bool] = mapped_column(Boolean, default=False)
    attributes: Mapped[list[Attribute]] = relationship(back_populates="track", cascade="all, delete-orphan")
    embeddings: Mapped[list[Embedding]] = relationship(cascade="all, delete-orphan")


class Attribute(Base):
    """One label for a track from one source (color estimator, VLM, ...)."""
    __tablename__ = "attributes"
    id: Mapped[int] = mapped_column(primary_key=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"), index=True)
    name: Mapped[str] = mapped_column(String(50), index=True)
    value: Mapped[str] = mapped_column(String(200), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(50))
    model: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    # auto | needs_review | confirmed | corrected
    review_status: Mapped[str] = mapped_column(String(20), default="auto", index=True)
    corrected_value: Mapped[str | None] = mapped_column(String(200))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime)
    track: Mapped[Track] = relationship(back_populates="attributes")

    @property
    def final_value(self) -> str:
        return self.corrected_value if self.review_status == "corrected" else self.value


class Embedding(Base):
    """Appearance embedding of a track (mean of its best crops), for text search and look-alike search."""
    __tablename__ = "embeddings"
    id: Mapped[int] = mapped_column(primary_key=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"), index=True)
    model: Mapped[str] = mapped_column(String(100))
    dim: Mapped[int] = mapped_column(Integer)
    vector: Mapped[np.ndarray] = mapped_column(VectorColumn(EMBED_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class AlertRule(Base):
    """Notify when a person matching these conditions appears (see labeler/alerts.py)."""
    __tablename__ = "alert_rules"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    camera: Mapped[str | None] = mapped_column(String(100))          # None = any camera
    conditions: Mapped[str] = mapped_column(Text)                    # JSON list of [attribute, value]
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class AlertEvent(Base):
    __tablename__ = "alert_events"
    __table_args__ = (UniqueConstraint("rule_id", "track_id"),)       # each rule fires once per person
    id: Mapped[int] = mapped_column(primary_key=True)
    rule_id: Mapped[int] = mapped_column(ForeignKey("alert_rules.id", ondelete="CASCADE"), index=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), index=True)
    camera: Mapped[str] = mapped_column(String(100))
    message: Mapped[str] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)


class Review(Base):
    """Audit trail of human verification; also the source of training labels."""
    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(primary_key=True)
    attribute_id: Mapped[int] = mapped_column(ForeignKey("attributes.id", ondelete="CASCADE"), index=True)
    old_value: Mapped[str] = mapped_column(String(200))
    new_value: Mapped[str] = mapped_column(String(200))
    reviewer: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


def resolve(attrs: list[Attribute]) -> dict[str, dict]:
    """Pick one value per attribute name: human-verified first, then highest confidence."""
    by_name: dict[str, list[Attribute]] = defaultdict(list)
    for a in attrs:
        by_name[a.name].append(a)
    out: dict[str, dict] = {}
    for name, items in by_name.items():
        verified = [a for a in items if a.review_status in ("confirmed", "corrected")]
        best = verified[0] if verified else max(items, key=lambda a: a.confidence)
        out[name] = {
            "value": best.final_value,
            "confidence": 1.0 if verified else round(best.confidence, 2),
            "source": "human" if verified else best.source,
            "needs_review": any(a.review_status == "needs_review" for a in items),
        }
    return out


class Database:
    def __init__(self, url: str):
        url = sqlite_url(url)
        is_sqlite = url.startswith("sqlite")
        self.is_postgres = url.startswith("postgresql")
        self.engine = create_engine(
            url, pool_pre_ping=True,
            connect_args={"check_same_thread": False, "timeout": 30} if is_sqlite else {},
        )
        if is_sqlite:
            @event.listens_for(self.engine, "connect")
            def _pragmas(conn, _record):
                cur = conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()
        if self.is_postgres:
            with self.engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(self.engine)
        if self.is_postgres:
            # Approximate nearest-neighbour index, so search stays fast with millions of people.
            with self.engine.begin() as conn:
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_embeddings_vector_hnsw "
                                  "ON embeddings USING hnsw (vector vector_cosine_ops)"))
        self.Session = sessionmaker(self.engine, expire_on_commit=False)
        # SQLite only: an in-memory copy of all embeddings for brute-force search.
        self._vec_lock = threading.Lock()
        self._vectors: dict[int, np.ndarray] | None = None
        self._vec_model: str | None = None

    # --- writes from the pipeline -------------------------------------------------

    def get_or_create_camera(self, name: str, source: str, mode: str) -> int:
        with self.Session.begin() as s:
            cam = s.scalar(select(Camera).where(Camera.name == name))
            if cam is None:
                cam = Camera(name=name, source=source, mode=mode)
                s.add(cam)
                s.flush()
            cam.source, cam.mode = source, mode
            return cam.id

    def create_track(self, camera_id: int, tracker_id: int, started_at: datetime) -> int:
        with self.Session.begin() as s:
            t = Track(camera_id=camera_id, tracker_id=tracker_id, started_at=started_at, status="active")
            s.add(t)
            s.flush()
            return t.id

    def update_track(self, track_id: int, **fields) -> None:
        with self.Session.begin() as s:
            t = s.get(Track, track_id)
            if t is None:  # deleted by the user meanwhile
                return
            for k, v in fields.items():
                setattr(t, k, v)

    def add_attributes(self, track_id: int, items: list[tuple[str, str, float]], source: str,
                       model: str, threshold: float) -> None:
        """Insert (name, value, confidence) labels, replacing earlier unreviewed ones from the same source."""
        with self.Session.begin() as s:
            if s.get(Track, track_id) is None:  # deleted by the user meanwhile
                return
            names = [n for n, _, _ in items]
            for old in s.scalars(select(Attribute).where(
                    Attribute.track_id == track_id, Attribute.source == source,
                    Attribute.name.in_(names), Attribute.review_status.in_(("auto", "needs_review")))):
                s.delete(old)
            for name, value, conf in items:
                s.add(Attribute(track_id=track_id, name=name, value=value, confidence=conf,
                                source=source, model=model))
            s.flush()
            self._reconcile(s, track_id, threshold)

    @staticmethod
    def _reconcile(s: Session, track_id: int, threshold: float) -> None:
        """Flag low-confidence labels, and labels where two sources disagree."""
        attrs = s.scalars(select(Attribute).where(Attribute.track_id == track_id)).all()
        by_name: dict[str, list[Attribute]] = defaultdict(list)
        for a in attrs:
            by_name[a.name].append(a)
        for items in by_name.values():
            if any(a.review_status in ("confirmed", "corrected") for a in items):
                continue
            conflict = _conflict(items)
            for a in items:
                low = a.value not in UNKNOWN and a.confidence < threshold
                a.review_status = "needs_review" if (low or conflict) else "auto"

    # --- reads for the web UI -----------------------------------------------------

    def _camera_names(self, s: Session) -> dict[int, str]:
        return {c.id: c.name for c in s.scalars(select(Camera))}

    def list_tracks(self, limit: int = 50, name: str | None = None, value: str | None = None,
                    camera: str | None = None) -> list[dict]:
        with self.Session() as s:
            q = select(Track).order_by(Track.id.desc())
            if camera:
                q = q.where(Track.camera_id.in_(select(Camera.id).where(Camera.name == camera)))
            if name and value:
                match = select(Attribute.track_id).where(
                    Attribute.name == name,
                    ((Attribute.review_status != "corrected") & (Attribute.value == value))
                    | ((Attribute.review_status == "corrected") & (Attribute.corrected_value == value)),
                )
                q = q.where(Track.id.in_(match))
            tracks = s.scalars(q.limit(limit)).all()
            names = self._camera_names(s)
            return [self._track_dict(t, names) for t in tracks]

    @staticmethod
    def _track_dict(t: Track, camera_names: dict[int, str] | None = None) -> dict:
        return {
            "id": t.id,
            "camera_id": t.camera_id,
            "camera": (camera_names or {}).get(t.camera_id),
            "started_at": t.started_at.isoformat(timespec="seconds"),
            "ended_at": t.ended_at.isoformat(timespec="seconds") if t.ended_at else None,
            "status": t.status,
            "n_frames": t.n_frames,
            "crop": t.crop_path,
            "low_light": t.low_light,
            "attributes": resolve(t.attributes),
        }

    def review_queue(self, limit: int = 50) -> list[dict]:
        """One item per (track, attribute name) that needs a human decision, most uncertain first:
        disagreements between models, then the lowest confidence."""
        with self.Session() as s:
            flagged = s.execute(select(Attribute.track_id, Attribute.name)
                                .where(Attribute.review_status == "needs_review").distinct()).all()
            items = []
            for track_id, name in flagged:
                track = s.get(Track, track_id)
                # Show every source's answer, not only the flagged ones.
                attrs = [a for a in track.attributes if a.name == name]
                best = max(attrs, key=lambda a: a.confidence)
                conflict = _conflict(attrs)
                weakest = min((a for a in attrs if a.review_status == "needs_review"), key=lambda a: a.confidence)
                reason = ("models disagree" if conflict
                          else f"low confidence: {weakest.source} {weakest.confidence:.0%}")
                items.append({
                    "attribute_id": best.id,
                    "track_id": track_id,
                    "crop": track.crop_path,
                    "name": name,
                    "value": best.value,
                    "reason": reason,
                    "priority": round((1.0 if conflict else 0.0) + (1.0 - weakest.confidence), 3),
                    "candidates": [{"source": a.source, "value": a.value, "confidence": round(a.confidence, 2)}
                                   for a in attrs],
                })
            items.sort(key=lambda i: (-i["priority"], -i["track_id"]))
            return items[:limit]

    def apply_review(self, attribute_id: int, value: str, reviewer: str = "reviewer") -> None:
        """Record the human answer for this attribute and every other source's label of it."""
        with self.Session.begin() as s:
            target = s.get(Attribute, attribute_id)
            if target is None:
                raise KeyError(attribute_id)
            siblings = s.scalars(select(Attribute).where(
                Attribute.track_id == target.track_id, Attribute.name == target.name)).all()
            now = datetime.now()
            for a in siblings:
                s.add(Review(attribute_id=a.id, old_value=a.final_value, new_value=value, reviewer=reviewer))
                if a.value == value:
                    a.review_status, a.corrected_value = "confirmed", None
                else:
                    a.review_status, a.corrected_value = "corrected", value
                a.reviewed_at = now

    # --- embeddings ---------------------------------------------------------------

    def upsert_embedding(self, track_id: int, vector: np.ndarray, model: str) -> None:
        vector = np.asarray(vector, dtype=np.float32)
        with self.Session.begin() as s:
            if s.get(Track, track_id) is None:  # deleted by the user meanwhile
                return
            row = s.scalar(select(Embedding).where(Embedding.track_id == track_id, Embedding.model == model))
            if row is None:
                row = Embedding(track_id=track_id, model=model)
                s.add(row)
            row.dim, row.vector, row.created_at = vector.shape[0], vector, datetime.now()
        if self.is_postgres:
            return
        with self._vec_lock:
            if self._vectors is not None and self._vec_model == model:
                self._vectors[track_id] = vector

    def _all_vectors(self, model: str) -> tuple[np.ndarray, np.ndarray]:
        with self._vec_lock:
            if self._vectors is None or self._vec_model != model:
                with self.Session() as s:
                    rows = s.execute(select(Embedding.track_id, Embedding.vector)
                                     .where(Embedding.model == model)).all()
                self._vectors = {tid: v for tid, v in rows}
                self._vec_model = model
            if not self._vectors:
                return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
            ids = np.fromiter(self._vectors.keys(), dtype=np.int64)
            return ids, np.stack(list(self._vectors.values()))

    def nearest(self, query: np.ndarray, model: str, limit: int = 20,
                exclude: int | None = None) -> list[tuple[int, float]]:
        """(track_id, cosine similarity) of the closest tracks to a normalized query vector."""
        if self.is_postgres:
            dist = Embedding.vector.op("<=>", return_type=Float)(np.asarray(query, dtype=np.float32))
            q = select(Embedding.track_id, dist).where(Embedding.model == model)
            if exclude is not None:
                q = q.where(Embedding.track_id != exclude)
            with self.Session() as s:
                rows = s.execute(q.order_by(dist).limit(limit)).all()
            return [(tid, 1.0 - float(d)) for tid, d in rows]
        ids, mat = self._all_vectors(model)
        if len(ids) == 0:
            return []
        sims = mat @ np.asarray(query, dtype=np.float32)
        order = np.argsort(-sims)
        out = [(int(ids[i]), float(sims[i])) for i in order if int(ids[i]) != exclude]
        return out[:limit]

    def get_embedding(self, track_id: int, model: str) -> np.ndarray | None:
        with self.Session() as s:
            v = s.scalar(select(Embedding.vector).where(Embedding.track_id == track_id, Embedding.model == model))
        return v

    def get_tracks(self, ids: list[int]) -> list[dict]:
        """Tracks in the given order."""
        with self.Session() as s:
            by_id = {t.id: t for t in s.scalars(select(Track).where(Track.id.in_(ids)))}
            names = self._camera_names(s)
            return [self._track_dict(by_id[i], names) for i in ids if i in by_id]

    def stats(self) -> dict:
        from sqlalchemy import func
        with self.Session() as s:
            flagged = s.execute(select(Attribute.track_id, Attribute.name)
                                .where(Attribute.review_status == "needs_review").distinct()).all()
            return {
                "tracks": s.scalar(select(func.count(Track.id))) or 0,
                "needs_review": len(flagged),
            }

    def delete_tracks(self, ids: list[int] | None = None, everything: bool = False) -> tuple[list[int], list[str]]:
        """Delete people and everything stored about them (labels, reviews, embeddings).
        Returns (deleted ids, picture file names to remove). Bulk statements, so deleting thousands is quick."""
        if not everything and not ids:
            return [], []
        with self.Session.begin() as s:
            which = select(Track.id) if everything else select(Track.id).where(Track.id.in_(list(ids)))
            rows = s.execute(select(Track.id, Track.crop_path, Track.face_path).where(Track.id.in_(which))).all()
            deleted = [r.id for r in rows]
            if not deleted:
                return [], []
            paths = [p for r in rows for p in (r.crop_path, r.face_path) if p]
            for start in range(0, len(deleted), 1000):
                chunk = deleted[start:start + 1000]
                attr_ids = select(Attribute.id).where(Attribute.track_id.in_(chunk))
                s.execute(delete(Review).where(Review.attribute_id.in_(attr_ids)))
                s.execute(delete(Attribute).where(Attribute.track_id.in_(chunk)))
                s.execute(delete(Embedding).where(Embedding.track_id.in_(chunk)))
                s.execute(delete(AlertEvent).where(AlertEvent.track_id.in_(chunk)))
                s.execute(delete(Track).where(Track.id.in_(chunk)))
        with self._vec_lock:
            if self._vectors is not None:
                for i in deleted:
                    self._vectors.pop(i, None)
        return deleted, paths

    # --- alerts -----------------------------------------------------------------------

    @staticmethod
    def _rule_dict(r: AlertRule) -> dict:
        return {"id": r.id, "name": r.name, "camera": r.camera, "conditions": json.loads(r.conditions),
                "enabled": r.enabled, "created_at": r.created_at.isoformat(timespec="seconds")}

    def list_rules(self) -> list[dict]:
        with self.Session() as s:
            return [self._rule_dict(r) for r in s.scalars(select(AlertRule).order_by(AlertRule.id))]

    def add_rule(self, name: str, conditions: list[list[str]], camera: str | None = None) -> dict:
        with self.Session.begin() as s:
            r = AlertRule(name=name, camera=camera or None, conditions=json.dumps(conditions))
            s.add(r)
            s.flush()
            return self._rule_dict(r)

    def update_rule(self, rule_id: int, enabled: bool) -> bool:
        with self.Session.begin() as s:
            r = s.get(AlertRule, rule_id)
            if r is None:
                return False
            r.enabled = enabled
            return True

    def delete_rule(self, rule_id: int) -> bool:
        with self.Session.begin() as s:
            s.execute(delete(AlertEvent).where(AlertEvent.rule_id == rule_id))
            return s.execute(delete(AlertRule).where(AlertRule.id == rule_id)).rowcount > 0

    def resolved_values(self, track_id: int) -> dict[str, str] | None:
        """name -> current value for one person, or None if they no longer exist."""
        with self.Session() as s:
            t = s.get(Track, track_id)
            if t is None:
                return None
            return {n: a["value"] for n, a in resolve(t.attributes).items()}

    def record_alert(self, rule_id: int, track_id: int, camera: str, message: str) -> dict | None:
        """Store an alert unless this rule already fired for this person."""
        with self.Session.begin() as s:
            exists = s.scalar(select(AlertEvent.id).where(AlertEvent.rule_id == rule_id,
                                                          AlertEvent.track_id == track_id))
            if exists is not None or s.get(Track, track_id) is None:
                return None
            e = AlertEvent(rule_id=rule_id, track_id=track_id, camera=camera, message=message)
            s.add(e)
            s.flush()
            return {"id": e.id, "rule_id": rule_id, "track_id": track_id, "camera": camera, "message": message,
                    "created_at": e.created_at.isoformat(timespec="seconds")}

    def list_alerts(self, since_id: int = 0, limit: int = 50) -> list[dict]:
        with self.Session() as s:
            rows = s.execute(select(AlertEvent, Track.crop_path).join(Track, Track.id == AlertEvent.track_id)
                             .where(AlertEvent.id > since_id).order_by(AlertEvent.id.desc()).limit(limit)).all()
            return [{"id": e.id, "rule_id": e.rule_id, "track_id": e.track_id, "camera": e.camera,
                     "message": e.message, "crop": crop, "created_at": e.created_at.isoformat(timespec="seconds")}
                    for e, crop in rows]

    def latest_alert_id(self) -> int:
        from sqlalchemy import func
        with self.Session() as s:
            return s.scalar(select(func.max(AlertEvent.id))) or 0

    # --- reports ----------------------------------------------------------------------

    def people_between(self, start: datetime, end: datetime, camera: str | None = None,
                       limit: int | None = None, newest_first: bool = False) -> list[dict]:
        """Everyone who appeared in [start, end) with resolved attributes (oldest first by default)."""
        with self.Session() as s:
            q = (select(Track).options(selectinload(Track.attributes))
                 .where(Track.started_at >= start, Track.started_at < end)
                 .order_by(Track.started_at.desc() if newest_first else Track.started_at))
            if camera:
                q = q.where(Track.camera_id.in_(select(Camera.id).where(Camera.name == camera)))
            if limit:
                q = q.limit(limit)
            names = self._camera_names(s)
            return [self._track_dict(t, names) for t in s.scalars(q)]

    def vectors_for(self, ids: list[int], model: str) -> dict[int, np.ndarray]:
        """Embeddings of the given people (those that have one)."""
        out: dict[int, np.ndarray] = {}
        with self.Session() as s:
            for start in range(0, len(ids), 1000):
                rows = s.execute(select(Embedding.track_id, Embedding.vector).where(
                    Embedding.model == model, Embedding.track_id.in_(ids[start:start + 1000]))).all()
                out.update({tid: v for tid, v in rows})
        return out

    def close_stale_tracks(self) -> int:
        """Mark people left "active" by an app that was killed (power cut, forced close) as finished."""
        with self.Session.begin() as s:
            stale = s.scalars(select(Track).where(Track.status == "active")).all()
            for t in stale:
                t.status = "finished"
                t.ended_at = t.ended_at or t.started_at
            return len(stale)

    def cleanup(self, retention_days: int) -> list[str]:
        """Delete tracks older than the retention window; returns their crop paths."""
        if retention_days <= 0:
            return []
        cutoff = datetime.now() - timedelta(days=retention_days)
        with self.Session.begin() as s:
            old = s.scalars(select(Track).where(Track.started_at < cutoff)).all()
            paths = [p for t in old for p in (t.crop_path, t.face_path) if p]
            if old:
                s.execute(delete(AlertEvent).where(AlertEvent.track_id.in_([t.id for t in old])))
            for t in old:
                s.delete(t)
        if old:
            with self._vec_lock:
                self._vectors = None
        return paths
