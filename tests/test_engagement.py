"""Time in view and facing the camera; adding new columns to an existing database."""
import sqlite3
from datetime import datetime

import numpy as np
from sqlalchemy import inspect

from labeler import geometry, reports
from labeler.db import Database
from labeler.pipeline import TrackState
from test_pipeline_logic import make_pipeline


def _face(facing: bool):
    k = np.zeros((17, 3), np.float32)
    k[geometry.NOSE] = (50, 50, 0.95)
    k[geometry.L_EYE] = (40, 40, 0.95)
    k[geometry.R_EYE] = (60, 40, 0.95) if facing else (46, 40, 0.95)   # side view: both eyes left of the nose
    return k


def test_side_view_is_not_facing_even_with_a_high_frontality_score():
    assert geometry.frontality(_face(False)) > 0.5                     # the old score alone would count it
    assert not geometry.facing_camera(_face(False)) and geometry.facing_camera(_face(True))
    assert not geometry.facing_camera(None)


def test_head_turned_halfway_is_not_facing():
    k = _face(True)
    k[geometry.NOSE, 0] = 57.4                                         # nose 0.37 eye widths off the middle
    assert not geometry.facing_camera(k)
    k[geometry.NOSE, 0] = 52.0                                         # 0.1 off: a slight turn still counts
    assert geometry.facing_camera(k)


def test_time_in_view_and_facing(db):
    p = make_pipeline(db)
    st = TrackState(1, datetime.now(), 0.0)
    for t, facing in [(0.5, True), (1.0, True), (1.5, False), (2.0, False)]:
        p._count_time(st, _face(facing), t)
        st.last_seen = t
    assert st.in_view_s == 2.0 and st.facing_s == 1.0
    p._count_time(st, None, 9.0)                       # tracked through a 7 s gap: counts 1 s at most
    assert st.in_view_s == 3.0 and st.facing_s == 1.0


def test_engagement_summary_and_csv():
    people = [{"in_view_s": 10.0, "facing_s": 5.0}, {"in_view_s": 30.0, "facing_s": 1.0},
              {"in_view_s": None, "facing_s": None}]      # still in view: no times yet
    g = reports.engagement(people, looked_s=3)
    assert g == {"people": 2, "avg_in_view_s": 20.0, "facing_share": 0.15, "looked": 1, "looked_s": 3}
    row = {"id": 1, "camera": "cam1", "started_at": "2026-10-08T10:00:00", "ended_at": None, "status": "finished",
           "in_view_s": 12.5, "facing_s": 0.0, "attributes": {}, "crop": None}
    lines = reports.to_csv([row]).splitlines()
    assert lines[0].startswith("id,camera,started_at,ended_at,status,in_view_s,facing_s,")
    assert lines[1].startswith("1,cam1,2026-10-08T10:00:00,,finished,12.5,0.0,")


def _person(hour, gender=None, age_range=None, age_group=None, in_view_s=None):
    attrs = {}
    for name, value in (("apparent_gender", gender), ("age_range", age_range), ("age_group", age_group)):
        if value:
            attrs[name] = {"value": value}
    return {"started_at": f"2026-10-08T{hour:02d}:15:00", "attributes": attrs, "in_view_s": in_view_s}


def test_who_came_at_which_hour():
    people = [_person(9, "man", age_range="30-39"), _person(9, "woman", age_range="60-69"),
              _person(9, "unknown"), _person(14, "woman", age_group="teen")]      # security cameras store a group
    rows = reports.hours_of_day(people)
    assert len(rows) == 24
    assert rows[9] == {"hour": 9, "people": 3, "man": 1, "woman": 1, "child": 0, "teen": 0, "adult": 1, "senior": 1}
    assert rows[14]["teen"] == 1 and rows[0]["people"] == 0


def test_how_long_people_stayed():
    people = [_person(9, in_view_s=s) for s in (4, 9.9, 10, 45, 61, 400)] + [_person(9)]   # last: no time yet
    assert [r["count"] for r in reports.dwell_histogram(people)] == [2, 1, 1, 1, 0, 1]


def test_summary_counts_the_days():
    from datetime import datetime, timedelta
    start = datetime(2026, 10, 1)
    assert reports.summarize([], start, start + timedelta(days=7))["days"] == 7
    assert reports.summarize([], start, start + timedelta(days=1))["days"] == 1


def test_new_columns_are_added_to_an_existing_database(tmp_path):
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)          # the tracks table as the first release made it
    con.execute("CREATE TABLE tracks (id INTEGER PRIMARY KEY, camera_id INTEGER, tracker_id INTEGER, "
                "started_at DATETIME, ended_at DATETIME, status VARCHAR(20), n_frames INTEGER, "
                "crop_path VARCHAR(300), face_path VARCHAR(300), low_light BOOLEAN)")
    con.commit()
    con.close()
    db = Database("sqlite:///" + path.as_posix())
    assert {"in_view_s", "facing_s"} <= {c["name"] for c in inspect(db.engine).get_columns("tracks")}
    cam = db.get_or_create_camera("c", "0", "kiosk")
    t = db.create_track(cam, 1, datetime.now())
    db.update_track(t, in_view_s=4.2, facing_s=1.0)
    assert db.list_tracks()[0]["in_view_s"] == 4.2
    Database("sqlite:///" + path.as_posix())          # second start: nothing to add, no error
