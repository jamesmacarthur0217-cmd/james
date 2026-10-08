"""Database models and queries. SQLite by default; the same models work on PostgreSQL."""
from __future__ import annotations

import hashlib
import json
import secrets
import threading
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import (
    Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text, TypeDecorator, UniqueConstraint,
    create_engine, delete, event, select, text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, selectinload, sessionmaker

from . import schema
from .clips import clip_name
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
    in_view_s: Mapped[float | None] = mapped_column(Float)   # time actually in view (gaps not counted)
    facing_s: Mapped[float | None] = mapped_column(Float)    # of which facing the camera
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


class AnswerKey(Base):
    """Labels a person entered for a recorded person without seeing the system's labels.
    Used to measure how accurate the system really is on this camera."""
    __tablename__ = "answer_key"
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), primary_key=True)
    labels: Mapped[str] = mapped_column(Text)          # JSON {attribute: value}; "not sure" attributes left out
    labeled_by: Mapped[str] = mapped_column(String(100), default="reviewer")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class Review(Base):
    """Audit trail of human verification; also the source of training labels."""
    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(primary_key=True)
    attribute_id: Mapped[int] = mapped_column(ForeignKey("attributes.id", ondelete="CASCADE"), index=True)
    old_value: Mapped[str] = mapped_column(String(200))
    new_value: Mapped[str] = mapped_column(String(200))
    reviewer: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class User(Base):
    """A login for the web pages (see labeler/accounts.py). Only a salted hash of the password is stored."""
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True)
    password_hash: Mapped[str] = mapped_column(String(300))
    role: Mapped[str] = mapped_column(String(20))      # admin | viewer
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class Zone(Base):
    """An area or counting line drawn on one camera's picture (see labeler/zones.py)."""
    __tablename__ = "zones"
    id: Mapped[int] = mapped_column(primary_key=True)
    camera: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(String(100))
    kind: Mapped[str] = mapped_column(String(10))       # area | line
    points: Mapped[str] = mapped_column(Text)           # JSON [[x, y], ...], fractions of the frame
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class ZoneEvent(Base):
    """A person entering/exiting an area or crossing a line. Keeps the zone's name, so counts survive
    deleting or redrawing the zone."""
    __tablename__ = "zone_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id", ondelete="CASCADE"), index=True)
    camera: Mapped[str] = mapped_column(String(100))
    zone_id: Mapped[int] = mapped_column(Integer)
    zone_name: Mapped[str] = mapped_column(String(100), index=True)
    kind: Mapped[str] = mapped_column(String(10))       # enter | exit (areas), in | out (lines)
    at: Mapped[datetime] = mapped_column(DateTime, index=True)
    dwell_s: Mapped[float | None] = mapped_column(Float)   # on exit: time spent inside


class CameraEvent(Base):
    """A camera problem (no picture, blank picture, slow), from when it started until it ended."""
    __tablename__ = "camera_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    camera: Mapped[str] = mapped_column(String(100), index=True)
    kind: Mapped[str] = mapped_column(String(20))          # offline | blank | slow (labeler/health.py)
    message: Mapped[str] = mapped_column(String(300))
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime)


class AuditEntry(Base):
    """Who did what in the web pages: logins, viewing, searches, exports, deletes, rule changes, reviews.
    Frequent background requests (the Live page's polling) are not logged."""
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)
    username: Mapped[str] = mapped_column(String(100), index=True)
    role: Mapped[str] = mapped_column(String(20))
    address: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(50), index=True)
    detail: Mapped[str] = mapped_column(Text, default="")


class LoginSession(Base):
    """A logged-in browser. The cookie holds a random token; only its SHA-256 is stored here."""
    __tablename__ = "login_sessions"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)


def resolve(attrs: list[Attribute]) -> dict[str, dict]:
    """Pick one value per attribute name: human-verified first, then highest confidence.

    The fast classifiers ("student") only imitate the VLM, so once the VLM has answered they can only add
    their own mistakes: on the user's webcam their confident glasses answers were right 5 of 11 times,
    the VLM's 10 of 13. They still count until the VLM answers."""
    by_name: dict[str, list[Attribute]] = defaultdict(list)
    for a in attrs:
        by_name[a.name].append(a)
    out: dict[str, dict] = {}
    for name, items in by_name.items():
        verified = [a for a in items if a.review_status in ("confirmed", "corrected")]
        if any(a.source == "vlm" and a.value != "unknown" for a in items):
            candidates = [a for a in items if a.source != "student"]
        else:
            candidates = items
        best = verified[0] if verified else max(candidates, key=lambda a: a.confidence)
        out[name] = {
            "value": best.final_value,
            "confidence": 1.0 if verified else round(best.confidence, 2),
            "source": "human" if verified else best.source,
            "needs_review": any(a.review_status == "needs_review" for a in items),
        }
    return out


# Columns added to existing tables after the first release. create_all() makes new tables but never changes
# existing ones, so these are added at startup when missing (FLOAT works on SQLite and PostgreSQL).
NEW_COLUMNS = {"tracks": {"in_view_s": "FLOAT", "facing_s": "FLOAT"}}


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _score(rows: list[tuple[AnswerKey, Track]]) -> dict[str, dict]:
    """Per attribute: answer-key labels, how many the system answered, how many it got right."""
    stats: dict[str, dict] = defaultdict(lambda: {"n": 0, "answered": 0, "right": 0})
    for key_row, track in rows:
        current = {n: a["value"] for n, a in resolve(track.attributes).items()}
        # The Answer key page asks for an age group; a kiosk camera only stores an age range.
        if "age_group" not in current and "age_range" in current:
            current["age_group"] = schema.AGE_RANGE_TO_GROUP.get(current["age_range"], "unknown")
        for name, truth in json.loads(key_row.labels).items():
            st = stats[name]
            st["n"] += 1
            guess = current.get(name, "unknown")
            if guess != "unknown":
                st["answered"] += 1
                st["right"] += guess == truth
    return {n: {**v, "accuracy": round(v["right"] / v["answered"], 3) if v["answered"] else None,
                "overall": round(v["right"] / v["n"], 3) if v["n"] else None}
            for n, v in sorted(stats.items())}


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
        self._add_missing_columns()
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

    def _add_missing_columns(self) -> None:
        from sqlalchemy import inspect
        insp = inspect(self.engine)
        with self.engine.begin() as conn:
            for table, columns in NEW_COLUMNS.items():
                have = {c["name"] for c in insp.get_columns(table)}
                for name, sql_type in columns.items():
                    if name not in have:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"))

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
            "in_view_s": t.in_view_s,
            "facing_s": t.facing_s,
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
        """Delete people and everything stored about them (labels, reviews, embeddings, alerts, zone events).
        Returns (deleted ids, file names to remove: pictures and alert clips). Bulk statements, so deleting
        thousands is quick."""
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
                paths += [clip_name(i) for i in s.scalars(select(AlertEvent.id).where(AlertEvent.track_id.in_(chunk)))]
                s.execute(delete(AlertEvent).where(AlertEvent.track_id.in_(chunk)))
                s.execute(delete(AnswerKey).where(AnswerKey.track_id.in_(chunk)))
                s.execute(delete(ZoneEvent).where(ZoneEvent.track_id.in_(chunk)))
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

    def rule_clips(self, rule_id: int) -> list[str]:
        """Clip file names of this rule's alerts (removed with the rule)."""
        with self.Session() as s:
            return [clip_name(i) for i in s.scalars(select(AlertEvent.id).where(AlertEvent.rule_id == rule_id))]

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

    # --- users and logins ---------------------------------------------------------------

    def count_users(self) -> int:
        from sqlalchemy import func
        with self.Session() as s:
            return s.scalar(select(func.count(User.id))) or 0

    def list_users(self) -> list[dict]:
        with self.Session() as s:
            return [{"username": u.username, "role": u.role, "created_at": u.created_at.isoformat(timespec="seconds")}
                    for u in s.scalars(select(User).order_by(User.username))]

    def get_user(self, username: str) -> dict | None:
        with self.Session() as s:
            u = s.scalars(select(User).where(User.username == username)).first()
            return None if u is None else {"id": u.id, "username": u.username, "role": u.role,
                                           "password_hash": u.password_hash}

    def add_user(self, username: str, password_hash: str, role: str) -> int:
        from .accounts import ROLES
        if role not in ROLES:
            raise ValueError(f"role must be one of {', '.join(ROLES)}")
        if not username.strip():
            raise ValueError("the user name is empty")
        with self.Session.begin() as s:
            if s.scalars(select(User).where(User.username == username)).first() is not None:
                raise ValueError(f"user {username!r} already exists")
            u = User(username=username, password_hash=password_hash, role=role)
            s.add(u)
            s.flush()
            return u.id

    def update_user(self, username: str, password_hash: str | None = None, role: str | None = None) -> bool:
        """Change a password or role. The user is logged out everywhere, so the change applies at once."""
        from .accounts import ROLES
        if role is not None and role not in ROLES:
            raise ValueError(f"role must be one of {', '.join(ROLES)}")
        with self.Session.begin() as s:
            u = s.scalars(select(User).where(User.username == username)).first()
            if u is None:
                return False
            if password_hash is not None:
                u.password_hash = password_hash
            if role is not None:
                u.role = role
            s.execute(delete(LoginSession).where(LoginSession.user_id == u.id))
            return True

    def delete_user(self, username: str) -> bool:
        with self.Session.begin() as s:
            u = s.scalars(select(User).where(User.username == username)).first()
            if u is None:
                return False
            s.execute(delete(LoginSession).where(LoginSession.user_id == u.id))
            s.delete(u)
            return True

    def create_session(self, user_id: int, hours: float) -> str:
        """Log a user in; returns the token for the cookie. Expired sessions are cleaned up here."""
        token = secrets.token_urlsafe(32)
        now = datetime.now()
        with self.Session.begin() as s:
            s.execute(delete(LoginSession).where(LoginSession.expires_at < now))
            s.add(LoginSession(token_hash=_token_hash(token), user_id=user_id, created_at=now,
                               expires_at=now + timedelta(hours=hours)))
        return token

    def session_user(self, token: str) -> dict | None:
        """The logged-in user for a cookie token, or None (unknown or expired)."""
        if not token:
            return None
        with self.Session() as s:
            row = s.execute(select(LoginSession, User).join(User, User.id == LoginSession.user_id)
                            .where(LoginSession.token_hash == _token_hash(token))).first()
            if row is None or row[0].expires_at < datetime.now():
                return None
            return {"username": row[1].username, "role": row[1].role}

    def end_session(self, token: str) -> None:
        with self.Session.begin() as s:
            s.execute(delete(LoginSession).where(LoginSession.token_hash == _token_hash(token)))

    # --- zones --------------------------------------------------------------------------

    @staticmethod
    def _zone_dict(z: Zone) -> dict:
        return {"id": z.id, "camera": z.camera, "name": z.name, "kind": z.kind,
                "points": [tuple(p) for p in json.loads(z.points)]}

    def list_zones(self, camera: str | None = None) -> list[dict]:
        q = select(Zone).order_by(Zone.camera, Zone.name)
        if camera is not None:
            q = q.where(Zone.camera == camera)
        with self.Session() as s:
            return [self._zone_dict(z) for z in s.scalars(q)]

    def add_zone(self, camera: str, name: str, kind: str, points: list) -> dict:
        from .zones import check_points
        pts = check_points(kind, points)
        name = name.strip()[:100]
        if not name:
            raise ValueError("the zone needs a name")
        with self.Session.begin() as s:
            if s.scalars(select(Zone).where(Zone.camera == camera, Zone.name == name)).first():
                raise ValueError(f"camera {camera} already has a zone named {name!r}")
            z = Zone(camera=camera, name=name, kind=kind, points=json.dumps([list(p) for p in pts]))
            s.add(z)
            s.flush()
            return self._zone_dict(z)

    def delete_zone(self, zone_id: int) -> dict | None:
        """Delete a zone (its counted events stay, under its name). Returns the deleted zone."""
        with self.Session.begin() as s:
            z = s.get(Zone, zone_id)
            if z is None:
                return None
            out = self._zone_dict(z)
            s.delete(z)
            return out

    def add_zone_events(self, track_id: int, camera: str, events: list[dict]) -> None:
        with self.Session.begin() as s:
            s.add_all(ZoneEvent(track_id=track_id, camera=camera, zone_id=e["zone_id"], zone_name=e["zone_name"],
                                kind=e["kind"], at=e["at"], dwell_s=e.get("dwell_s")) for e in events)

    def zones_entered(self, track_id: int) -> list[str]:
        """Names of the areas this person entered and the lines they crossed (either way)."""
        with self.Session() as s:
            return sorted(s.scalars(select(ZoneEvent.zone_name).distinct().where(
                ZoneEvent.track_id == track_id, ZoneEvent.kind.in_(("enter", "in", "out")))))

    def zone_counts(self, start: datetime, end: datetime, camera: str | None = None) -> list[dict]:
        """Per camera and zone in [start, end): entries and average dwell (areas), ins and outs (lines)."""
        q = select(ZoneEvent).where(ZoneEvent.at >= start, ZoneEvent.at < end)
        if camera:
            q = q.where(ZoneEvent.camera == camera)
        stats: dict[tuple[str, str], dict] = {}
        with self.Session() as s:
            for e in s.scalars(q):
                st = stats.setdefault((e.camera, e.zone_name), {"camera": e.camera, "zone": e.zone_name,
                                                               "enter": 0, "exit": 0, "in": 0, "out": 0,
                                                               "_dwell": []})
                st[e.kind] += 1
                if e.kind == "exit" and e.dwell_s is not None:
                    st["_dwell"].append(e.dwell_s)
        out = []
        for key in sorted(stats):
            st = stats[key]
            dwell = st.pop("_dwell")
            st["avg_dwell_s"] = round(sum(dwell) / len(dwell), 1) if dwell else None
            out.append(st)
        return out

    # --- camera health ------------------------------------------------------------------

    @staticmethod
    def _camera_event_dict(e: CameraEvent) -> dict:
        return {"id": e.id, "camera": e.camera, "kind": e.kind, "message": e.message,
                "started_at": e.started_at.isoformat(timespec="seconds"),
                "ended_at": e.ended_at.isoformat(timespec="seconds") if e.ended_at else None}

    def open_camera_event(self, camera: str, kind: str, message: str) -> dict:
        with self.Session.begin() as s:
            e = CameraEvent(camera=camera, kind=kind, message=message[:300], started_at=datetime.now())
            s.add(e)
            s.flush()
            return self._camera_event_dict(e)

    def close_camera_event(self, event_id: int) -> None:
        with self.Session.begin() as s:
            e = s.get(CameraEvent, event_id)
            if e is not None and e.ended_at is None:
                e.ended_at = datetime.now()

    def close_open_camera_events(self) -> int:
        """At startup: problems still open were ended by the app stopping."""
        from sqlalchemy import update
        with self.Session.begin() as s:
            return s.execute(update(CameraEvent).where(CameraEvent.ended_at.is_(None))
                             .values(ended_at=datetime.now())).rowcount

    def list_camera_events(self, limit: int = 50) -> list[dict]:
        with self.Session() as s:
            return [self._camera_event_dict(e) for e in
                    s.scalars(select(CameraEvent).order_by(CameraEvent.id.desc()).limit(limit))]

    def cleanup_camera_events(self, retention_days: int) -> int:
        if retention_days <= 0:
            return 0
        with self.Session.begin() as s:
            return s.execute(delete(CameraEvent).where(
                CameraEvent.started_at < datetime.now() - timedelta(days=retention_days))).rowcount

    # --- audit log ----------------------------------------------------------------------

    def add_audit(self, username: str, role: str, address: str, action: str, detail: str = "") -> None:
        with self.Session.begin() as s:
            s.add(AuditEntry(username=username[:100], role=role[:20], address=address[:64], action=action[:50],
                             detail=detail[:1000]))

    def list_audit(self, limit: int = 100, username: str | None = None, action: str | None = None,
                   before_id: int | None = None) -> list[dict]:
        """Newest first. `before_id` pages back through older entries."""
        q = select(AuditEntry).order_by(AuditEntry.id.desc()).limit(limit)
        if username:
            q = q.where(AuditEntry.username == username)
        if action:
            q = q.where(AuditEntry.action == action)
        if before_id:
            q = q.where(AuditEntry.id < before_id)
        with self.Session() as s:
            return [{"id": e.id, "time": e.created_at.isoformat(timespec="seconds"), "username": e.username,
                     "role": e.role, "address": e.address, "action": e.action, "detail": e.detail}
                    for e in s.scalars(q)]

    def audit_choices(self) -> dict[str, list[str]]:
        """The user names and actions that appear in the log, for the page's filters."""
        with self.Session() as s:
            return {"users": sorted(s.scalars(select(AuditEntry.username).distinct())),
                    "actions": sorted(s.scalars(select(AuditEntry.action).distinct()))}

    def cleanup_audit(self, retention_days: int) -> int:
        if retention_days <= 0:
            return 0
        with self.Session.begin() as s:
            return s.execute(delete(AuditEntry).where(
                AuditEntry.created_at < datetime.now() - timedelta(days=retention_days))).rowcount

    # --- answer key ---------------------------------------------------------------------

    def next_for_answer_key(self, exclude: list[int] | None = None) -> dict | None:
        """A finished person with a picture who is not in the answer key yet (newest first)."""
        with self.Session() as s:
            q = (select(Track).where(Track.status == "finished", Track.crop_path.is_not(None),
                                     Track.id.not_in(select(AnswerKey.track_id)))
                 .order_by(Track.id.desc()))
            if exclude:
                q = q.where(Track.id.not_in(exclude))
            t = s.scalars(q.limit(1)).first()
            return None if t is None else {"id": t.id, "crop": t.crop_path,
                                           "camera": self._camera_names(s).get(t.camera_id)}

    def save_answer_key(self, track_id: int, labels: dict[str, str], labeled_by: str = "reviewer") -> bool:
        with self.Session.begin() as s:
            if s.get(Track, track_id) is None:
                return False
            row = s.get(AnswerKey, track_id) or AnswerKey(track_id=track_id)
            row.labels, row.labeled_by, row.created_at = json.dumps(labels), labeled_by, datetime.now()
            s.add(row)
            return True

    def answer_key_score(self, camera: str | None = None) -> dict:
        """How often the system's current labels match the answer key, per attribute (optionally for one
        camera). `cameras` gives each camera's mean score: kiosk and security cameras differ a lot, so
        each gets its own baseline."""
        with self.Session() as s:
            names = self._camera_names(s)
            rows = s.execute(select(AnswerKey, Track).join(Track, Track.id == AnswerKey.track_id)
                             .options(selectinload(Track.attributes))).all()
            if camera is not None:
                rows = [(k, t) for k, t in rows if names.get(t.camera_id) == camera]
            by_camera: dict[str, list] = defaultdict(list)
            for k, t in rows:
                by_camera[names.get(t.camera_id, "?")].append((k, t))
            cameras = {}
            for name, cam_rows in sorted(by_camera.items()):
                overall = [v["overall"] for v in _score(cam_rows).values() if v["overall"] is not None]
                cameras[name] = {"people": len(cam_rows),
                                 "mean_overall": round(sum(overall) / len(overall), 3) if overall else None}
            return {"people": len(rows), "attributes": _score(rows), "cameras": cameras}

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

    def cleanup(self, retention_days: int, per_camera: dict[str, int] | None = None) -> list[str]:
        """Delete people older than their camera's retention window (`per_camera`: camera name -> days,
        otherwise `retention_days`; 0 = keep forever). Returns their picture and alert clip names."""
        per_camera = per_camera or {}
        now = datetime.now()
        with self.Session.begin() as s:
            old = []
            for cam in s.scalars(select(Camera)):
                days = per_camera.get(cam.name, retention_days)
                if days and days > 0:
                    old += s.scalars(select(Track).where(Track.camera_id == cam.id,
                                                         Track.started_at < now - timedelta(days=days))).all()
            paths = [p for t in old for p in (t.crop_path, t.face_path) if p]
            if old:
                paths += [clip_name(i) for i in s.scalars(select(AlertEvent.id).where(
                    AlertEvent.track_id.in_([t.id for t in old])))]
                s.execute(delete(AlertEvent).where(AlertEvent.track_id.in_([t.id for t in old])))
                s.execute(delete(AnswerKey).where(AnswerKey.track_id.in_([t.id for t in old])))
                s.execute(delete(ZoneEvent).where(ZoneEvent.track_id.in_([t.id for t in old])))
            for t in old:
                s.delete(t)
        if old:
            with self._vec_lock:
                self._vectors = None
        return paths
