"""The help sign (Phase 32): labeler/gestures.py on made-up body points."""
import numpy as np

from labeler.gestures import GestureWatch, hands_up

# A person standing, facing the camera (image y grows downwards): x, y, confidence for the 17 COCO points.
STAND = {0: (100, 50), 1: (95, 45), 2: (105, 45), 3: (90, 48), 4: (110, 48), 5: (80, 80), 6: (120, 80),
         7: (75, 120), 8: (125, 120), 9: (75, 160), 10: (125, 160), 11: (85, 170), 12: (115, 170),
         13: (85, 230), 14: (115, 230), 15: (85, 290), 16: (115, 290)}


def person(**moved):
    k = np.zeros((17, 3), np.float32)
    for i, (x, y) in STAND.items():
        k[i] = (x, y, 0.9)
    for name, (x, y) in moved.items():
        k[{"lw": 9, "rw": 10, "le": 7, "re": 8}[name]] = (x, y, 0.9)
    return k


LEFT_UP = person(le=(70, 60), lw=(70, 20))            # left elbow above the shoulder, wrist above the head
BOTH_UP = person(le=(70, 60), lw=(70, 20), re=(130, 60), rw=(130, 20))


def test_one_frame():
    assert hands_up(person()) == []
    assert hands_up(LEFT_UP) == ["left"] and hands_up(BOTH_UP) == ["left", "right"]
    assert hands_up(person(lw=(80, 40), le=(70, 70))) == []                  # at the head (scratching it): not up
    assert hands_up(person(lw=(70, 20), le=(70, 100))) == []                 # elbow below the shoulder: not up
    hidden_head = LEFT_UP.copy()
    hidden_head[[0, 1, 2, 3, 4], 2] = 0.1
    assert hands_up(hidden_head) == [] and hands_up(None) == []              # no head found: not checked
    lying = LEFT_UP.copy()
    lying[[11, 12], 0], lying[[11, 12], 1] = lying[[11, 12], 0] + 120, 80     # hips beside the shoulders: lying
    assert hands_up(lying) == [] and hands_up(lying, max_tilt=None) == ["left"]
    desk = LEFT_UP.copy()
    desk[11:, 2] = 0.1                                                        # a desk webcam: no hips in view
    assert hands_up(desk) == ["left"]
    far = LEFT_UP.copy()
    far[:, :2] = far[:, :2] / 4                                               # the same person far away
    assert hands_up(far) == ["left"]


def test_held_three_seconds_once_per_sign():
    w = GestureWatch({"hold_s": 3.0})
    t = 0.0
    events = []
    for _ in range(40):                                                       # 4 s up, at 10 frames a second
        events.append(w.update(1, t, LEFT_UP))
        t += 0.1
    fired = [e for e in events if e]
    assert len(fired) == 1 and fired[0] == {"since": 0.0, "hand": "left"}
    assert [i for i, e in enumerate(events) if e] == [30]                     # at 3.0 s
    for _ in range(10):                                                       # down 1 s, up again: the same sign
        w.update(1, t, person())
        t += 0.1
    assert not any(w.update(1, t + i * 0.1, LEFT_UP) for i in range(40))
    t += 4
    for _ in range(25):                                                       # down 2.5 s: a new sign may come
        w.update(1, t, person())
        t += 0.1
    assert any(w.update(1, t + i * 0.1, LEFT_UP) for i in range(31))


def test_short_raises_and_gaps():
    w = GestureWatch({"hold_s": 3.0, "gap_s": 0.5})
    t = 0.0
    for _ in range(25):                                                       # 2.5 s up, then down: nothing
        assert w.update(1, t, LEFT_UP) is None
        t += 0.1
    for _ in range(10):
        w.update(1, t, person())
        t += 0.1
    out = []
    for i in range(35):                                                       # a frame missed now and then
        out.append(w.update(1, t, None if i in (10, 20) else LEFT_UP))
        t += 0.1
    assert any(out)
    assert GestureWatch({"both_ok": False}).update(2, 0, BOTH_UP) is None
    w2 = GestureWatch({"both_ok": False})
    assert not any(w2.update(2, i * 0.1, BOTH_UP) for i in range(40))        # stretching both arms: not a sign


def test_check_tool_counts_signs_and_leaves_out_the_frozen_tail():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import gesture_check as gc
    dets = [(i / 10, [(1, (0, 0, 1, 1), LEFT_UP)]) for i in range(40)]      # 4 s with a hand up
    signs, up, n = gc.video_signs(dets, both_ok=True)
    assert len(signs) == 1 and up == 40 and n == 40 and signs[0]["at"] == 3.0
    assert gc.video_signs(dets, both_ok=True, tail_from=2.0)[0] == []        # the frozen picture is not the video


# --- step 3: the alert ---

def test_rule_fires_for_every_sign_and_only_for_signs(db, camera):
    from labeler.alerts import AlertEngine, describe, HELP_RULE
    alerts = AlertEngine(db)
    alerts.ensure_help_rule()
    alerts.ensure_help_rule()
    assert [r["name"] for r in db.list_rules()] == ["Help sign"] and describe(HELP_RULE) == "made the help sign"
    from datetime import datetime
    tid = db.create_track(camera, 1, datetime.now())
    assert alerts.check(tid, "test-cam") == []                               # labels changed: not a sign
    assert len(alerts.check(tid, "test-cam", helped=True)) == 1
    assert len(alerts.check(tid, "test-cam", helped=True)) == 1              # a second sign alerts again
    events = db.list_alerts()
    assert events[0]["help"] is True and events[0]["fall"] is False
    from labeler.checks import alert_what
    assert alert_what(events[0], HELP_RULE) == "help"


def test_pipeline_sign_goes_to_the_rules_with_a_clip(db, camera):
    from datetime import datetime
    from labeler.pipeline import TrackState
    from test_falls import make_pipeline
    p = make_pipeline(db)
    p.gestures, p.help_clip_s = GestureWatch(), 8.0
    asked = []

    class Alerts:
        def check(self, track_id, camera_name, fallen=False, stayed=None, helped=False):
            asked.append((track_id, helped))
            return [{"id": 5}]

    class Clips:
        def request(self, event_id, before_s=None):
            asked.append(("clip", event_id, before_s))

    p.alerts, p.clips = Alerts(), Clips()
    st = TrackState(3, datetime.now(), 0.0, db_id=db.create_track(camera, 3, datetime.now()))
    p.tracks[3] = st
    p._on_help(3)
    assert asked == [(st.db_id, True), ("clip", 5, 8.0)]


def test_pages_show_the_help_sign():
    from pathlib import Path
    static = Path(__file__).resolve().parent.parent / "labeler" / "static"
    js = (static / "session.js").read_text(encoding="utf-8")
    assert 'className = "toast fall help"' in js and "HELP SIGN" in js
    assert 'help: "made the help sign"' in (static / "alerts.html").read_text(encoding="utf-8")
    assert 'id="help_hold_s"' in (static / "settings.html").read_text(encoding="utf-8")
